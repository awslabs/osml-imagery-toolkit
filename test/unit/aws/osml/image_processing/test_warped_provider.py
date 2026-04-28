#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import math
from typing import Optional, Tuple
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.mapped_provider import MappedImageProvider
from aws.osml.image_processing.tile_cache import TileCache
from aws.osml.image_processing.warp_grid import GridBuilder, WarpGrid, WarpGridOptions
from aws.osml.image_processing.warped_provider import WarpedImageProvider


class _MockSource:
    """Minimal duck-typed ImageAssetProvider for testing."""

    def __init__(self, num_bands=3, width=512, height=512, block_w=256, block_h=256, dtype="uint8"):
        self._num_bands = num_bands
        self._width = width
        self._height = height
        self._block_w = block_w
        self._block_h = block_h
        self._dtype = dtype
        self._data = np.arange(num_bands * height * width, dtype=np.uint8).reshape(num_bands, height, width)

    @property
    def key(self):
        return "test_source"

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def num_rows(self):
        return self._height

    @property
    def num_columns(self):
        return self._width

    @property
    def num_pixels_per_block_horizontal(self):
        return self._block_w

    @property
    def num_pixels_per_block_vertical(self):
        return self._block_h

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def pixel_value_type(self):
        return self._dtype

    @property
    def block_grid_size(self):
        import math

        return (math.ceil(self._height / self._block_h), math.ceil(self._width / self._block_w))

    def get_block(self, row, col, resolution_level=0, bands=None):
        y0 = row * self._block_h
        x0 = col * self._block_w
        y1 = min(y0 + self._block_h, self._height)
        x1 = min(x0 + self._block_w, self._width)
        block = self._data[:, y0:y1, x0:x1]
        if bands is not None:
            block = block[list(bands), :, :]
        # Pad to full block size if at edge
        if block.shape[1] < self._block_h or block.shape[2] < self._block_w:
            nb = block.shape[0]
            padded = np.zeros((nb, self._block_h, self._block_w), dtype=block.dtype)
            padded[:, : block.shape[1], : block.shape[2]] = block
            return padded
        return block.copy()

    def has_block(self, row, col, resolution_level=0):
        return True


class _MockGridBuilder(GridBuilder):
    """Grid builder returning an identity-like warp grid (maps output directly to source)."""

    def __init__(self, output_w=512, output_h=512, block_w=256, block_h=256, return_none_for=None):
        super().__init__(WarpGridOptions.FAST)
        self._output_w = output_w
        self._output_h = output_h
        self._block_w = block_w
        self._block_h = block_h
        self._return_none_for = return_none_for or set()
        self.build_call_count = 0

    @property
    def tile_limits(self) -> Tuple[int, int, int, int]:
        max_row = math.ceil(self._output_h / self._block_h) - 1
        max_col = math.ceil(self._output_w / self._block_w) - 1
        return (0, 0, max_row, max_col)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self._block_w, self._block_h)

    def build(self, row: int, col: int) -> Optional[WarpGrid]:
        self.build_call_count += 1
        if (row, col) in self._return_none_for:
            return None

        h = self._block_h
        w = self._block_w

        # Identity mapping: output pixel (r, c) maps to source pixel at same position
        y_offset = row * h
        x_offset = col * w

        # map_x/map_y are relative to source_bbox origin
        map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
        map_y = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)
        valid_mask = np.ones((h, w), dtype=np.bool_)
        source_bbox = (x_offset, y_offset, w, h)

        return WarpGrid(
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid_mask,
            source_bbox=source_bbox,
            source_resolution_level=0,
        )


class TestWarpedProviderProperties(TestCase):
    """Tests for WarpedImageProvider property derivation."""

    def test_key(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.key, "test_source:warped")

    def test_num_rows(self):
        source = _MockSource()
        builder = _MockGridBuilder(output_h=1024)
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.num_rows, 1024)

    def test_num_columns(self):
        source = _MockSource()
        builder = _MockGridBuilder(output_w=2048)
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.num_columns, 2048)

    def test_block_dimensions(self):
        source = _MockSource()
        builder = _MockGridBuilder(block_w=128, block_h=64)
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.num_pixels_per_block_horizontal, 128)
        self.assertEqual(provider.num_pixels_per_block_vertical, 64)

    def test_num_bands_from_source(self):
        source = _MockSource(num_bands=4)
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.num_bands, 4)

    def test_num_bands_override(self):
        source = _MockSource(num_bands=4)
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder, num_bands=1)
        self.assertEqual(provider.num_bands, 1)

    def test_pixel_value_type(self):
        source = _MockSource(dtype="uint16")
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.pixel_value_type, "uint16")

    def test_num_resolution_levels(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.num_resolution_levels, 1)

    def test_block_grid_size(self):
        source = _MockSource()
        builder = _MockGridBuilder(output_w=1000, output_h=500, block_w=256, block_h=256)
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.block_grid_size, (2, 4))

    def test_metadata_is_none(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        self.assertIsNone(provider.metadata)


class TestWarpedProviderGetBlock(TestCase):
    """Tests for WarpedImageProvider.get_block()."""

    def test_returns_chw_ndarray(self):
        source = _MockSource(num_bands=3, width=256, height=256, block_w=256, block_h=256)
        builder = _MockGridBuilder(output_w=256, output_h=256, block_w=256, block_h=256)
        provider = WarpedImageProvider(source, builder)
        block = provider.get_block(0, 0)
        self.assertEqual(block.shape, (3, 256, 256))

    def test_zero_filled_when_grid_is_none(self):
        source = _MockSource()
        builder = _MockGridBuilder(return_none_for={(0, 0)})
        provider = WarpedImageProvider(source, builder)
        block = provider.get_block(0, 0)
        self.assertEqual(block.shape, (3, 256, 256))
        np.testing.assert_array_equal(block, 0)

    def test_raises_on_nonzero_resolution_level(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        with self.assertRaises(ValueError):
            provider.get_block(0, 0, resolution_level=1)

    def test_identity_remap_preserves_pixels(self):
        # With identity mapping (map_x = col indices, map_y = row indices),
        # the output should match the source block
        source = _MockSource(num_bands=1, width=256, height=256, block_w=256, block_h=256)
        builder = _MockGridBuilder(output_w=256, output_h=256, block_w=256, block_h=256)
        provider = WarpedImageProvider(source, builder)
        block = provider.get_block(0, 0)
        expected = source.get_block(0, 0)
        np.testing.assert_array_equal(block, expected)

    def test_band_selection(self):
        source = _MockSource(num_bands=4, width=256, height=256, block_w=256, block_h=256)
        builder = _MockGridBuilder(output_w=256, output_h=256, block_w=256, block_h=256)
        provider = WarpedImageProvider(source, builder)
        block = provider.get_block(0, 0, bands=(0, 2))
        self.assertEqual(block.shape[0], 2)

    def test_single_band_source(self):
        source = _MockSource(num_bands=1, width=128, height=128, block_w=128, block_h=128)
        builder = _MockGridBuilder(output_w=128, output_h=128, block_w=128, block_h=128)
        provider = WarpedImageProvider(source, builder)
        block = provider.get_block(0, 0)
        self.assertEqual(block.shape, (1, 128, 128))


class TestWarpedProviderValidMask(TestCase):
    """Tests for WarpedImageProvider.get_valid_mask()."""

    def test_returns_valid_mask(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        mask = provider.get_valid_mask(0, 0)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (256, 256))
        self.assertTrue(mask.all())

    def test_returns_false_when_no_grid(self):
        source = _MockSource()
        builder = _MockGridBuilder(return_none_for={(0, 0)})
        provider = WarpedImageProvider(source, builder)
        mask = provider.get_valid_mask(0, 0)
        self.assertFalse(mask.any())


class TestWarpedProviderHasBlock(TestCase):
    """Tests for WarpedImageProvider.has_block()."""

    def test_true_when_grid_exists(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        self.assertTrue(provider.has_block(0, 0))

    def test_false_when_grid_is_none(self):
        source = _MockSource()
        builder = _MockGridBuilder(return_none_for={(1, 1)})
        provider = WarpedImageProvider(source, builder)
        self.assertFalse(provider.has_block(1, 1))

    def test_raises_on_nonzero_resolution_level(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        with self.assertRaises(ValueError):
            provider.has_block(0, 0, resolution_level=2)


class TestWarpedProviderGridCaching(TestCase):
    """Tests for internal WarpGrid LRU caching behavior."""

    def test_get_block_then_get_valid_mask_no_double_compute(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        provider.get_block(0, 0)
        provider.get_valid_mask(0, 0)
        self.assertEqual(builder.build_call_count, 1)

    def test_different_block_recomputes(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        provider.get_block(0, 0)
        provider.get_block(0, 1)
        self.assertEqual(builder.build_call_count, 2)

    def test_has_block_then_get_block_no_double_compute(self):
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        provider.has_block(0, 0)
        provider.get_block(0, 0)
        self.assertEqual(builder.build_call_count, 1)

    def test_multi_entry_grid_cache_retains_recent(self):
        """The multi-entry LRU retains recently accessed grids."""
        source = _MockSource()
        builder = _MockGridBuilder()
        provider = WarpedImageProvider(source, builder)
        provider.get_block(0, 0)
        provider.get_block(0, 1)
        # Access (0, 0) again — should be a cache hit
        provider.get_block(0, 0)
        self.assertEqual(builder.build_call_count, 2)


class TestWarpedProviderTileCache(TestCase):
    """Tests for output tile caching in the shared TileCache."""

    def test_output_tile_cached(self):
        """Output tiles are cached when a TileCache is provided."""
        source = _MockSource(num_bands=3, width=256, height=256, block_w=256, block_h=256)
        builder = _MockGridBuilder(output_w=256, output_h=256, block_w=256, block_h=256)
        cache = TileCache(max_bytes=1024 * 1024)
        provider = WarpedImageProvider(source, builder, cache=cache)

        result1 = provider.get_block(0, 0)
        self.assertGreater(cache.current_bytes, 0)

        result2 = provider.get_block(0, 0)
        # Both should return same data (cache hit)
        np.testing.assert_array_equal(result1, result2)
        # Grid builder should only be called once
        self.assertEqual(builder.build_call_count, 1)

    def test_no_cache_no_error(self):
        """When cache=None, provider works normally without caching."""
        source = _MockSource(num_bands=1, width=128, height=128, block_w=128, block_h=128)
        builder = _MockGridBuilder(output_w=128, output_h=128, block_w=128, block_h=128)
        provider = WarpedImageProvider(source, builder)
        block = provider.get_block(0, 0)
        self.assertEqual(block.shape, (1, 128, 128))

    def test_none_grid_tile_cached(self):
        """Zero-filled tiles for None grids are also cached."""
        source = _MockSource()
        builder = _MockGridBuilder(return_none_for={(0, 0)})
        cache = TileCache(max_bytes=1024 * 1024)
        provider = WarpedImageProvider(source, builder, cache=cache)

        result1 = provider.get_block(0, 0)
        result2 = provider.get_block(0, 0)
        np.testing.assert_array_equal(result1, 0)
        np.testing.assert_array_equal(result2, 0)
        # Grid builder called once for first miss, cache hit on second
        self.assertEqual(builder.build_call_count, 1)

    def test_cache_key_uses_provider_key(self):
        """Cache entries are keyed by provider.key for isolation."""
        source = _MockSource(num_bands=1, width=128, height=128, block_w=128, block_h=128)
        builder = _MockGridBuilder(output_w=128, output_h=128, block_w=128, block_h=128)
        cache = TileCache(max_bytes=1024 * 1024)
        provider = WarpedImageProvider(source, builder, cache=cache)

        provider.get_block(0, 0)
        expected_key = (provider.key, 0, 0, 0, None)
        self.assertIsNotNone(cache.get(expected_key))


class _AbsoluteCoordGridBuilder(GridBuilder):
    """Grid builder with non-zero tile_limits to test absolute coordinate handling."""

    def __init__(self, min_row=10, min_col=20, max_row=12, max_col=22, block_w=256, block_h=256):
        super().__init__(WarpGridOptions.FAST)
        self._min_row = min_row
        self._min_col = min_col
        self._max_row = max_row
        self._max_col = max_col
        self._block_w = block_w
        self._block_h = block_h
        self.received_coords = []

    @property
    def tile_limits(self) -> Tuple[int, int, int, int]:
        return (self._min_row, self._min_col, self._max_row, self._max_col)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self._block_w, self._block_h)

    def build(self, row: int, col: int) -> Optional[WarpGrid]:
        self.received_coords.append((row, col))
        h = self._block_h
        w = self._block_w
        map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
        map_y = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)
        valid_mask = np.ones((h, w), dtype=np.bool_)
        source_bbox = (0, 0, w, h)
        return WarpGrid(
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid_mask,
            source_bbox=source_bbox,
            source_resolution_level=0,
        )


class TestWarpedProviderAbsoluteCoordinates(TestCase):
    """Tests for absolute tile coordinate handling (non-zero tile_limits)."""

    def test_tile_limits_exposed(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=10, min_col=20, max_row=12, max_col=22)
        provider = WarpedImageProvider(source, builder)
        self.assertEqual(provider.tile_limits, (10, 20, 12, 22))

    def test_num_rows_from_absolute_tile_limits(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=10, min_col=20, max_row=12, max_col=22, block_h=256)
        provider = WarpedImageProvider(source, builder)
        # (12 - 10 + 1) * 256 = 768
        self.assertEqual(provider.num_rows, 768)

    def test_num_columns_from_absolute_tile_limits(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=10, min_col=20, max_row=12, max_col=22, block_w=256)
        provider = WarpedImageProvider(source, builder)
        # (22 - 20 + 1) * 256 = 768
        self.assertEqual(provider.num_columns, 768)

    def test_block_grid_size_from_tile_limits(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=5, min_col=3, max_row=8, max_col=6)
        provider = WarpedImageProvider(source, builder)
        # rows = 8-5+1=4, cols = 6-3+1=4
        self.assertEqual(provider.block_grid_size, (4, 4))

    def test_get_block_passes_absolute_coords_to_builder(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=10, min_col=20, max_row=12, max_col=22)
        provider = WarpedImageProvider(source, builder)
        provider.get_block(11, 21)
        self.assertEqual(builder.received_coords, [(11, 21)])

    def test_has_block_passes_absolute_coords(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=10, min_col=20, max_row=12, max_col=22)
        provider = WarpedImageProvider(source, builder)
        result = provider.has_block(10, 20)
        self.assertTrue(result)
        self.assertEqual(builder.received_coords, [(10, 20)])

    def test_get_valid_mask_passes_absolute_coords(self):
        source = _MockSource()
        builder = _AbsoluteCoordGridBuilder(min_row=10, min_col=20, max_row=12, max_col=22)
        provider = WarpedImageProvider(source, builder)
        mask = provider.get_valid_mask(12, 22)
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(builder.received_coords, [(12, 22)])


class TestWarpedProviderComposition(TestCase):
    """Tests for composition with MappedImageProvider."""

    def test_compose_with_mapped_provider(self):
        source = _MockSource(num_bands=3, width=256, height=256, block_w=256, block_h=256)
        builder = _MockGridBuilder(output_w=256, output_h=256, block_w=256, block_h=256)
        warped = WarpedImageProvider(source, builder)

        def double_fn(block):
            return (block.astype(np.uint16) * 2).astype(np.uint8)

        mapped = MappedImageProvider(warped, double_fn)

        self.assertEqual(mapped.num_rows, 256)
        self.assertEqual(mapped.num_columns, 256)
        self.assertEqual(mapped.num_bands, 3)

        block = mapped.get_block(0, 0)
        self.assertEqual(block.shape, (3, 256, 256))

    def test_mapped_delegates_has_block(self):
        source = _MockSource()
        builder = _MockGridBuilder(return_none_for={(1, 1)})
        warped = WarpedImageProvider(source, builder)
        mapped = MappedImageProvider(warped, lambda b: b)

        self.assertTrue(mapped.has_block(0, 0))
        # has_block delegates to WarpedImageProvider which checks the grid
        self.assertFalse(mapped.has_block(1, 1))
