#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for RetiledImageProvider."""

import math
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.retiled_provider import RetiledImageProvider
from aws.osml.image_processing.tile_cache import TileCache


class _MockProvider:
    """Multi-block mock ImageAssetProvider with configurable grid."""

    def __init__(
        self,
        image,
        tile_height,
        tile_width,
        key="mock-source",
        num_resolution_levels=1,
        pad_pixel_value=0.0,
    ):
        self._image = image
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._key = key
        self._num_resolution_levels = num_resolution_levels
        self.pad_pixel_value = pad_pixel_value
        self.read_count = 0

        num_bands, img_h, img_w = image.shape
        self._grid_rows = math.ceil(img_h / tile_height)
        self._grid_cols = math.ceil(img_w / tile_width)

    @property
    def key(self):
        return self._key

    @property
    def num_rows(self):
        return self._image.shape[1]

    @property
    def num_columns(self):
        return self._image.shape[2]

    @property
    def num_pixels_per_block_horizontal(self):
        return self._tile_width

    @property
    def num_pixels_per_block_vertical(self):
        return self._tile_height

    @property
    def num_resolution_levels(self):
        return self._num_resolution_levels

    @property
    def block_grid_size(self):
        return (self._grid_rows, self._grid_cols)

    @property
    def num_bands(self):
        return self._image.shape[0]

    @property
    def pixel_value_type(self):
        return "uint8"

    @property
    def metadata(self):
        return {"format": "test"}

    def has_block(self, row, col, resolution_level=0):
        return 0 <= row < self._grid_rows and 0 <= col < self._grid_cols

    def get_block(self, row, col, resolution_level=0, bands=None):
        self.read_count += 1
        scale = 2**resolution_level
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self._image.shape[1])
        x1 = min(x0 + self._tile_width, self._image.shape[2])

        # Scale coordinates for resolution levels
        block_h = math.ceil((y1 - y0) / scale)
        block_w = math.ceil((x1 - x0) / scale)

        if resolution_level == 0:
            block = self._image[:, y0:y1, x0:x1].copy()
        else:
            # Simulate lower resolution by simple nearest-neighbor downscale
            full_block = self._image[:, y0:y1, x0:x1]
            block = full_block[:, ::scale, ::scale][:, :block_h, :block_w].copy()

        if bands is not None:
            block = block[list(bands), :, :]
        return block


class TestRetiledProviderKey(TestCase):
    """Tests for .key property."""

    def test_key_includes_source_and_dimensions(self):
        image = np.zeros((3, 100, 100), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=100, key="asset-1")
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)
        self.assertEqual(retiled.key, "asset-1:retiled:64x64")

    def test_key_changes_with_tile_dims(self):
        image = np.zeros((3, 100, 100), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=100, key="asset-1")
        r1 = RetiledImageProvider(source, tile_width=64, tile_height=64)
        r2 = RetiledImageProvider(source, tile_width=128, tile_height=128)
        self.assertNotEqual(r1.key, r2.key)


class TestRetiledProviderGridComputation(TestCase):
    """Tests for virtual grid computation."""

    def test_exact_division(self):
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)
        self.assertEqual(retiled.block_grid_size, (4, 4))

    def test_non_exact_division(self):
        image = np.zeros((1, 100, 150), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=150)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)
        # ceil(100/64)=2, ceil(150/64)=3
        self.assertEqual(retiled.block_grid_size, (2, 3))

    def test_fine_source_larger_virtual(self):
        """Source has small tiles (32x32), virtual is larger (128x128)."""
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=32, tile_width=32)
        retiled = RetiledImageProvider(source, tile_width=128, tile_height=128)
        # ceil(256/128)=2
        self.assertEqual(retiled.block_grid_size, (2, 2))


class TestRetiledProviderDimensions(TestCase):
    """Tests for num_rows/num_columns delegation."""

    def test_no_pad_delegates_source_dims(self):
        image = np.zeros((1, 100, 150), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=150)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, pad_edges=False)
        self.assertEqual(retiled.num_rows, 100)
        self.assertEqual(retiled.num_columns, 150)

    def test_pad_reports_grid_rounded_dims(self):
        image = np.zeros((1, 100, 150), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=150)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, pad_edges=True)
        # 2 rows * 64 = 128, 3 cols * 64 = 192
        self.assertEqual(retiled.num_rows, 128)
        self.assertEqual(retiled.num_columns, 192)


class TestRetiledProviderHasBlock(TestCase):
    """Tests for has_block."""

    def test_valid_positions(self):
        image = np.zeros((1, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=128, tile_width=128)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)
        # Grid is 2x2
        self.assertTrue(retiled.has_block(0, 0))
        self.assertTrue(retiled.has_block(0, 1))
        self.assertTrue(retiled.has_block(1, 0))
        self.assertTrue(retiled.has_block(1, 1))

    def test_out_of_bounds(self):
        image = np.zeros((1, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=128, tile_width=128)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)
        self.assertFalse(retiled.has_block(2, 0))
        self.assertFalse(retiled.has_block(0, 2))
        self.assertFalse(retiled.has_block(-1, 0))


class TestRetiledProviderResolutionLevels(TestCase):
    """Tests for num_resolution_levels."""

    def test_capped_by_tile_size(self):
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256, num_resolution_levels=10)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)
        # floor(log2(64)) + 1 = 7
        self.assertEqual(retiled.num_resolution_levels, 7)

    def test_capped_by_source(self):
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256, num_resolution_levels=3)
        retiled = RetiledImageProvider(source, tile_width=1024, tile_height=1024)
        # Source only has 3 levels
        self.assertEqual(retiled.num_resolution_levels, 3)

    def test_non_square_tiles(self):
        image = np.zeros((1, 256, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=512, num_resolution_levels=10)
        retiled = RetiledImageProvider(source, tile_width=128, tile_height=32)
        # floor(log2(min(128, 32))) + 1 = floor(log2(32)) + 1 = 6
        self.assertEqual(retiled.num_resolution_levels, 6)


class TestRetiledProviderGetBlockCoarse(TestCase):
    """Tests for get_block when slicing from coarser source blocks."""

    def test_slice_from_single_source_block(self):
        """Source has one 256x256 block, virtual tiles are 64x64."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)

        # Virtual tile (1, 2) → pixels [64:128, 128:192]
        tile = retiled.get_block(1, 2)
        expected = image[:, 64:128, 128:192]
        np.testing.assert_array_equal(tile, expected)

    def test_all_tiles_cover_image(self):
        """Stitching all virtual tiles recovers the full image."""
        np.random.seed(123)
        image = np.random.randint(0, 255, (1, 100, 100), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=100)
        retiled = RetiledImageProvider(source, tile_width=32, tile_height=32)

        rows, cols = retiled.block_grid_size
        reconstructed = np.zeros_like(image)
        for r in range(rows):
            for c in range(cols):
                tile = retiled.get_block(r, c)
                y0 = r * 32
                x0 = c * 32
                h = tile.shape[1]
                w = tile.shape[2]
                reconstructed[:, y0 : y0 + h, x0 : x0 + w] = tile

        np.testing.assert_array_equal(reconstructed, image)


class TestRetiledProviderGetBlockFine(TestCase):
    """Tests for get_block when stitching from finer source blocks."""

    def test_stitch_multiple_source_blocks(self):
        """Source has 32x32 tiles, virtual tiles are 64x64."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (1, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=32, tile_width=32)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)

        tile = retiled.get_block(0, 0)
        expected = image[:, 0:64, 0:64]
        np.testing.assert_array_equal(tile, expected)

    def test_stitch_corner_tile(self):
        """Bottom-right virtual tile stitches from multiple source blocks."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (2, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=32, tile_width=32)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64)

        tile = retiled.get_block(1, 1)
        expected = image[:, 64:128, 64:128]
        np.testing.assert_array_equal(tile, expected)


class TestRetiledProviderEdgeTiles(TestCase):
    """Tests for edge tile handling."""

    def test_partial_edge_tile_no_pad(self):
        """Edge tile is smaller than tile_width/tile_height."""
        image = np.ones((1, 100, 100), dtype=np.uint8) * 7
        source = _MockProvider(image, tile_height=100, tile_width=100)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, pad_edges=False)

        # Last column tile: only 36 pixels wide (100 - 64)
        tile = retiled.get_block(0, 1)
        self.assertEqual(tile.shape, (1, 64, 36))

        # Last row tile: only 36 pixels tall
        tile = retiled.get_block(1, 0)
        self.assertEqual(tile.shape, (1, 36, 64))

        # Bottom-right corner: 36x36
        tile = retiled.get_block(1, 1)
        self.assertEqual(tile.shape, (1, 36, 36))

    def test_padded_edge_tile(self):
        """Edge tile is padded to full tile dimensions."""
        image = np.ones((1, 100, 100), dtype=np.uint8) * 7
        source = _MockProvider(image, tile_height=100, tile_width=100, pad_pixel_value=99.0)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, pad_edges=True)

        # Last column tile: padded to 64x64
        tile = retiled.get_block(0, 1)
        self.assertEqual(tile.shape, (1, 64, 64))
        # First 36 cols have pixel data, rest are pad
        np.testing.assert_array_equal(tile[:, :, :36], 7)
        np.testing.assert_array_equal(tile[:, :, 36:], 99)

    def test_pad_value_from_source(self):
        """Pad value comes from source's pad_pixel_value."""
        image = np.ones((1, 50, 50), dtype=np.uint8) * 10
        source = _MockProvider(image, tile_height=50, tile_width=50, pad_pixel_value=42.0)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, pad_edges=True)

        tile = retiled.get_block(0, 0)
        self.assertEqual(tile.shape, (1, 64, 64))
        # Source pixels
        np.testing.assert_array_equal(tile[:, :50, :50], 10)
        # Padded area
        np.testing.assert_array_equal(tile[:, 50:, :], 42)
        np.testing.assert_array_equal(tile[:, :50, 50:], 42)


class TestRetiledProviderResolutionLevelGetBlock(TestCase):
    """Tests for get_block at higher resolution levels."""

    def test_level1_returns_half_size(self):
        """At level=1, tile dimensions are halved."""
        image = np.random.RandomState(42).randint(0, 255, (1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256, num_resolution_levels=3)
        retiled = RetiledImageProvider(source, tile_width=128, tile_height=128)

        tile = retiled.get_block(0, 0, resolution_level=1)
        # 128 / 2^1 = 64
        self.assertEqual(tile.shape, (1, 64, 64))

    def test_level2_returns_quarter_size(self):
        """At level=2, tile dimensions are quartered."""
        image = np.random.RandomState(42).randint(0, 255, (1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256, num_resolution_levels=3)
        retiled = RetiledImageProvider(source, tile_width=128, tile_height=128)

        tile = retiled.get_block(0, 0, resolution_level=2)
        # 128 / 2^2 = 32
        self.assertEqual(tile.shape, (1, 32, 32))


class TestRetiledProviderCache(TestCase):
    """Tests for TileCache integration."""

    def test_cache_hit_avoids_source_reads(self):
        image = np.random.RandomState(42).randint(0, 255, (1, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=128, tile_width=128)
        cache = TileCache(max_bytes=10 * 1024 * 1024)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, cache=cache)

        tile1 = retiled.get_block(0, 0)
        reads_after_first = source.read_count
        tile2 = retiled.get_block(0, 0)

        np.testing.assert_array_equal(tile1, tile2)
        self.assertEqual(source.read_count, reads_after_first)

    def test_no_cache_computes_every_time(self):
        image = np.random.RandomState(42).randint(0, 255, (1, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=128, tile_width=128)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, cache=None)

        retiled.get_block(0, 0)
        reads_1 = source.read_count
        retiled.get_block(0, 0)
        reads_2 = source.read_count
        self.assertGreater(reads_2, reads_1)

    def test_cache_stores_under_correct_key(self):
        image = np.zeros((1, 64, 64), dtype=np.uint8)
        source = _MockProvider(image, tile_height=64, tile_width=64, key="src")
        cache = TileCache(max_bytes=10 * 1024 * 1024)
        retiled = RetiledImageProvider(source, tile_width=32, tile_height=32, cache=cache)

        retiled.get_block(0, 0)
        expected_key = ("src:retiled:32x32", 0, 0, 0, None)
        self.assertIsNotNone(cache.get(expected_key))


class TestRetiledProviderBandSelection(TestCase):
    """Tests for band selection in get_block."""

    def test_bands_parameter(self):
        np.random.seed(42)
        image = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        source = _MockProvider(image, tile_height=64, tile_width=64)
        retiled = RetiledImageProvider(source, tile_width=32, tile_height=32)

        tile = retiled.get_block(0, 0, bands=(0, 2))
        self.assertEqual(tile.shape[0], 2)
        expected = image[(0, 2), :32, :32]
        np.testing.assert_array_equal(tile, expected)


class TestRetiledProviderInterface(TestCase):
    """Tests that all ImageAssetProvider properties are implemented."""

    def setUp(self):
        self.image = np.zeros((3, 200, 300), dtype=np.uint8)
        self.source = _MockProvider(self.image, tile_height=200, tile_width=300, key="test-asset")
        self.retiled = RetiledImageProvider(self.source, tile_width=64, tile_height=64)

    def test_num_pixels_per_block(self):
        self.assertEqual(self.retiled.num_pixels_per_block_horizontal, 64)
        self.assertEqual(self.retiled.num_pixels_per_block_vertical, 64)

    def test_num_bands(self):
        self.assertEqual(self.retiled.num_bands, 3)

    def test_pixel_value_type(self):
        self.assertEqual(self.retiled.pixel_value_type, "uint8")

    def test_metadata_delegates(self):
        self.assertEqual(self.retiled.metadata, {"format": "test"})


class TestRetiledProviderFastPath(TestCase):
    """Tests for the fast path when virtual grid matches source grid."""

    def test_matching_dims_is_passthrough(self):
        """When virtual and source tile dims match, no stitching needed."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (1, 128, 128), dtype=np.uint8)
        source = _MockProvider(image, tile_height=64, tile_width=64)
        retiled = RetiledImageProvider(source, tile_width=64, tile_height=64, pad_edges=False)

        tile = retiled.get_block(0, 0)
        expected = image[:, 0:64, 0:64]
        np.testing.assert_array_equal(tile, expected)

        tile = retiled.get_block(1, 1)
        expected = image[:, 64:128, 64:128]
        np.testing.assert_array_equal(tile, expected)
