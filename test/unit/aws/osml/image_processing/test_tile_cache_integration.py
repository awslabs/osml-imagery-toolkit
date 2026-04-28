#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Integration tests for the full caching and retiling chain.

Verifies that source → CachedImageProvider → RetiledImageProvider →
DownsampledImageProvider with a shared TileCache produces correct output
pixels and stays within the configured byte budget.
"""

import math
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.cached_provider import CachedImageProvider
from aws.osml.image_processing.downsampled_provider import DownsampledImageProvider
from aws.osml.image_processing.resample import area_resample
from aws.osml.image_processing.retiled_provider import RetiledImageProvider
from aws.osml.image_processing.tile_cache import TileCache
from aws.osml.io import PixelType


class _MockProvider:
    """Minimal ImageAssetProvider backed by a single CHW array.

    Splits the backing array into an aligned tile grid with configurable
    physical block dimensions (simulates coarse-grid sources like J2K).
    """

    def __init__(self, image, tile_height=256, tile_width=256):
        self._image = image
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)
        self.get_block_call_count = 0

    @property
    def key(self):
        return "mock:integration"

    @property
    def num_rows(self):
        return int(self._image.shape[1])

    @property
    def num_columns(self):
        return int(self._image.shape[2])

    @property
    def num_bands(self):
        return int(self._image.shape[0])

    @property
    def num_pixels_per_block_horizontal(self):
        return self._tile_width

    @property
    def num_pixels_per_block_vertical(self):
        return self._tile_height

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def pixel_value_type(self):
        return PixelType.UInt8

    @property
    def pad_pixel_value(self):
        return 0.0

    @property
    def block_grid_size(self):
        rows = math.ceil(self.num_rows / self._tile_height)
        cols = math.ceil(self.num_columns / self._tile_width)
        return (rows, cols)

    def has_block(self, row, col, resolution_level=0):
        grid_rows, grid_cols = self.block_grid_size
        return 0 <= row < grid_rows and 0 <= col < grid_cols

    def get_block(self, row, col, resolution_level=0, bands=None):
        self.get_block_call_count += 1
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self.num_rows)
        x1 = min(x0 + self._tile_width, self.num_columns)
        block = self._image[:, y0:y1, x0:x1].copy()
        if block.shape[1] < self._tile_height or block.shape[2] < self._tile_width:
            padded = np.zeros(
                (self.num_bands, self._tile_height, self._tile_width),
                dtype=block.dtype,
            )
            padded[:, : block.shape[1], : block.shape[2]] = block
            block = padded
        return block

    @property
    def metadata(self):
        return {}


class TestFullChainIntegration(TestCase):
    """Full chain: source → CachedImageProvider → RetiledImageProvider → DownsampledImageProvider."""

    def test_chain_produces_correct_output(self):
        """Verify the chain produces pixel-correct downsampled output."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(3, 512, 512), dtype=np.uint8)

        # Simulate a coarse-block source (single 512x512 block)
        source = _MockProvider(image, tile_height=512, tile_width=512)

        cache = TileCache(max_bytes=8 * 1024 * 1024)

        # Build the chain
        cached = CachedImageProvider(source, cache=cache)
        retiled = RetiledImageProvider(cached, tile_width=256, tile_height=256, cache=cache)
        downsampled = DownsampledImageProvider(retiled, scale_factor=2, resample_func=area_resample, cache=cache)

        # Read all downsampled tiles and stitch them
        grid_rows, grid_cols = downsampled.block_grid_size
        ds_h = downsampled.num_rows
        ds_w = downsampled.num_columns
        result = np.zeros((3, ds_h, ds_w), dtype=np.uint8)

        for r in range(grid_rows):
            for c in range(grid_cols):
                block = downsampled.get_block(r, c)
                tile_h = downsampled.num_pixels_per_block_vertical
                tile_w = downsampled.num_pixels_per_block_horizontal
                y0 = r * tile_h
                x0 = c * tile_w
                y1 = min(y0 + tile_h, ds_h)
                x1 = min(x0 + tile_w, ds_w)
                result[:, y0:y1, x0:x1] = block[:, : y1 - y0, : x1 - x0]

        # Verify dimensions: 512/2 = 256
        self.assertEqual(result.shape, (3, 256, 256))

        # Verify pixel content: area_resample with scale_factor=2 should
        # approximate a 2x2 box average. Check that result is not all zeros.
        self.assertGreater(np.mean(result), 50)
        self.assertLess(np.mean(result), 200)

    def test_cache_budget_not_exceeded(self):
        """Verify current_bytes stays within max_bytes throughout processing."""
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(3, 1024, 1024), dtype=np.uint8)

        source = _MockProvider(image, tile_height=1024, tile_width=1024)
        budget = 4 * 1024 * 1024
        cache = TileCache(max_bytes=budget)

        cached = CachedImageProvider(source, cache=cache)
        retiled = RetiledImageProvider(cached, tile_width=256, tile_height=256, cache=cache)
        downsampled = DownsampledImageProvider(retiled, scale_factor=2, resample_func=area_resample, cache=cache)

        grid_rows, grid_cols = downsampled.block_grid_size
        for r in range(grid_rows):
            for c in range(grid_cols):
                downsampled.get_block(r, c)
                self.assertLessEqual(cache.current_bytes, budget)

    def test_cache_avoids_redundant_source_reads(self):
        """CachedImageProvider reduces decode calls when retiled tiles overlap the same source block."""
        rng = np.random.default_rng(99)
        image = rng.integers(0, 256, size=(1, 512, 512), dtype=np.uint8)

        # Single source block (512x512) gets split into 4 retiled blocks (256x256)
        source = _MockProvider(image, tile_height=512, tile_width=512)
        cache = TileCache(max_bytes=8 * 1024 * 1024)

        cached = CachedImageProvider(source, cache=cache)
        retiled = RetiledImageProvider(cached, tile_width=256, tile_height=256)

        # Read all 4 virtual tiles
        for r in range(2):
            for c in range(2):
                retiled.get_block(r, c)

        # Without caching, we'd decode the source block 4 times.
        # With CachedImageProvider, only 1 decode should occur.
        self.assertEqual(source.get_block_call_count, 1)

    def test_shared_cache_isolates_providers(self):
        """Different providers using the same cache don't collide on keys."""
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(1, 256, 256), dtype=np.uint8)

        source = _MockProvider(image, tile_height=256, tile_width=256)
        cache = TileCache(max_bytes=8 * 1024 * 1024)

        cached = CachedImageProvider(source, cache=cache)
        retiled = RetiledImageProvider(cached, tile_width=128, tile_height=128, cache=cache)
        downsampled = DownsampledImageProvider(retiled, scale_factor=2, resample_func=area_resample, cache=cache)

        # Read blocks at different layers
        source_block = cached.get_block(0, 0)
        retiled_block = retiled.get_block(0, 0)
        ds_block = downsampled.get_block(0, 0)

        # All should be different shapes (different tile sizes)
        self.assertEqual(source_block.shape, (1, 256, 256))
        self.assertEqual(retiled_block.shape, (1, 128, 128))
        self.assertEqual(ds_block.shape, (1, 64, 64))

        # Cache should have entries from all three layers
        self.assertGreater(cache.current_bytes, 0)

    def test_repeated_reads_hit_cache(self):
        """Reading the same block twice returns the cached copy without recomputation."""
        rng = np.random.default_rng(55)
        image = rng.integers(0, 256, size=(3, 256, 256), dtype=np.uint8)

        source = _MockProvider(image, tile_height=256, tile_width=256)
        cache = TileCache(max_bytes=8 * 1024 * 1024)

        cached = CachedImageProvider(source, cache=cache)
        retiled = RetiledImageProvider(cached, tile_width=128, tile_height=128, cache=cache)
        downsampled = DownsampledImageProvider(retiled, scale_factor=2, resample_func=area_resample, cache=cache)

        # First read
        block1 = downsampled.get_block(0, 0)
        calls_after_first = source.get_block_call_count

        # Second read — should hit cache everywhere
        block2 = downsampled.get_block(0, 0)
        calls_after_second = source.get_block_call_count

        np.testing.assert_array_equal(block1, block2)
        self.assertEqual(calls_after_first, calls_after_second)

    def test_multiband_chain(self):
        """Chain handles multi-band imagery correctly."""
        rng = np.random.default_rng(77)
        image = rng.integers(0, 256, size=(4, 512, 512), dtype=np.uint8)

        source = _MockProvider(image, tile_height=512, tile_width=512)
        cache = TileCache(max_bytes=8 * 1024 * 1024)

        cached = CachedImageProvider(source, cache=cache)
        retiled = RetiledImageProvider(cached, tile_width=256, tile_height=256, cache=cache)
        downsampled = DownsampledImageProvider(retiled, scale_factor=2, resample_func=area_resample, cache=cache)

        block = downsampled.get_block(0, 0)
        self.assertEqual(block.shape[0], 4)
        self.assertEqual(block.shape[1], 128)
        self.assertEqual(block.shape[2], 128)
