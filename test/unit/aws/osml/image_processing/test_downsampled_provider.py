#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :class:`DownsampledImageProvider`.

Tests cover the resample path (single resolution level source), the
native path (multi-resolution J2K-like source), chained operations,
TileCache integration, input validation, structural guards, and
interface delegation.

The tests use a minimal ``_MockProvider`` that mimics the osml-imagery-io
``ImageAssetProvider`` contract, consistent with the mock used in the
``test_pyramid_builder.py`` tests.
"""

from unittest import TestCase

import numpy as np

from aws.osml.image_processing.downsampled_provider import DownsampledImageProvider
from aws.osml.image_processing.resample import area_resample
from aws.osml.image_processing.sips_resample import sips_rrds_resample
from aws.osml.image_processing.tile_cache import TileCache
from aws.osml.io import PixelType

# ----------------------------------------------------------------------
# Mocks — minimal duck-typed ImageAssetProvider for unit tests
# ----------------------------------------------------------------------


class _MockProvider:
    """Minimal ImageAssetProvider duck-type backed by a single CHW array.

    Splits the backing array into an aligned tile grid. Supports
    configurable native ``num_resolution_levels``. When multiple
    resolution levels are advertised, the mock stores pre-computed
    half-size arrays for each level so tests can verify the native
    path returns them verbatim.
    """

    def __init__(
        self,
        image,
        tile_height=256,
        tile_width=256,
        num_resolution_levels=1,
        pixel_value_type=PixelType.UInt8,
        pad_pixel_value=0.0,
    ):
        self._base_image = image
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)
        self._num_resolution_levels = int(num_resolution_levels)
        self._pixel_value_type = pixel_value_type
        self._pad_pixel_value = float(pad_pixel_value)
        # Pre-compute reduced-resolution copies for levels 1..N-1.
        self._level_images = [image]
        cur = image
        for _ in range(1, self._num_resolution_levels):
            cur = cur[:, ::2, ::2].copy()
            self._level_images.append(cur)

        self.get_block_calls = []

    @property
    def key(self):
        return "mock:0"

    @property
    def num_rows(self):
        return int(self._base_image.shape[-2])

    @property
    def num_columns(self):
        return int(self._base_image.shape[-1])

    @property
    def num_bands(self):
        return int(self._base_image.shape[0])

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
    def pixel_value_type(self):
        return self._pixel_value_type

    @property
    def pad_pixel_value(self):
        return self._pad_pixel_value

    @property
    def block_grid_size(self):
        rows = (self.num_rows + self._tile_height - 1) // self._tile_height
        cols = (self.num_columns + self._tile_width - 1) // self._tile_width
        return (rows, cols)

    def has_block(self, row, col, resolution_level=0):
        img = self._level_images[resolution_level]
        divisor = 2**resolution_level
        tile_h = max(1, self._tile_height // divisor)
        tile_w = max(1, self._tile_width // divisor)
        grid_rows = (img.shape[-2] + tile_h - 1) // tile_h
        grid_cols = (img.shape[-1] + tile_w - 1) // tile_w
        if row < 0 or col < 0 or row >= grid_rows or col >= grid_cols:
            return False
        return True

    def get_block(self, row, col, resolution_level=0, bands=None):
        self.get_block_calls.append((row, col, resolution_level))
        img = self._level_images[resolution_level]
        divisor = 2**resolution_level
        tile_h = max(1, self._tile_height // divisor)
        tile_w = max(1, self._tile_width // divisor)
        y0 = row * tile_h
        x0 = col * tile_w
        y1 = min(y0 + tile_h, img.shape[-2])
        x1 = min(x0 + tile_w, img.shape[-1])
        tile = img[:, y0:y1, x0:x1].copy()
        if tile.shape[-2] != tile_h or tile.shape[-1] != tile_w:
            padded = np.full(
                (tile.shape[0], tile_h, tile_w),
                self._pad_pixel_value,
                dtype=tile.dtype,
            )
            padded[:, : tile.shape[-2], : tile.shape[-1]] = tile
            tile = padded
        return tile

    @property
    def metadata(self):
        return {}


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestDownsampledImageProviderValidation(TestCase):
    """Input validation tests."""

    def _make_source(self):
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        return _MockProvider(image, tile_height=256, tile_width=256)

    def test_non_power_of_two_scale_factor_raises(self):
        source = self._make_source()
        with self.assertRaises(ValueError):
            DownsampledImageProvider(source, scale_factor=3)

    def test_zero_scale_factor_raises(self):
        source = self._make_source()
        with self.assertRaises(ValueError):
            DownsampledImageProvider(source, scale_factor=0)

    def test_valid_power_of_two_scale_factors(self):
        source = self._make_source()
        for sf in [1, 2, 4, 8]:
            op = DownsampledImageProvider(source, scale_factor=sf)
            self.assertEqual(op.num_rows, (512 + sf - 1) // sf)
            self.assertEqual(op.num_columns, (512 + sf - 1) // sf)


class TestDownsampledImageProviderDimensions(TestCase):
    """Verify output dimensions follow SIPS even/odd rounding."""

    def test_even_dimensions(self):
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=2)
        self.assertEqual(op.num_rows, 256)
        self.assertEqual(op.num_columns, 256)

    def test_odd_dimensions(self):
        image = np.zeros((3, 513, 513), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=2)
        # (513 + 1) // 2 = 257
        self.assertEqual(op.num_rows, 257)
        self.assertEqual(op.num_columns, 257)

    def test_scale_factor_4(self):
        image = np.zeros((3, 1024, 1024), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=4)
        self.assertEqual(op.num_rows, 256)
        self.assertEqual(op.num_columns, 256)


class TestDownsampledImageProviderDefaultTileSize(TestCase):
    """Verify default output tile size is ceil(source_block / scale_factor)."""

    def test_default_tile_size_square(self):
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=2)
        self.assertEqual(op.num_pixels_per_block_horizontal, 128)
        self.assertEqual(op.num_pixels_per_block_vertical, 128)

    def test_default_tile_size_non_square(self):
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=100, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample)
        # ceil(100/2) = 50, ceil(256/2) = 128
        self.assertEqual(op.num_pixels_per_block_vertical, 50)
        self.assertEqual(op.num_pixels_per_block_horizontal, 128)

    def test_default_tile_size_odd_source_block(self):
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=255, tile_width=255)
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample)
        # ceil(255/2) = 128
        self.assertEqual(op.num_pixels_per_block_vertical, 128)
        self.assertEqual(op.num_pixels_per_block_horizontal, 128)

    def test_explicit_tile_size_overrides_default(self):
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=2, tile_width=64, tile_height=64)
        self.assertEqual(op.num_pixels_per_block_horizontal, 64)
        self.assertEqual(op.num_pixels_per_block_vertical, 64)


class TestDownsampledImageProviderInterfaceDelegation(TestCase):
    """Verify ImageAssetProvider interface delegation."""

    def setUp(self):
        self.image = np.zeros((3, 512, 512), dtype=np.uint8)
        self.source = _MockProvider(self.image, tile_height=256, tile_width=256)
        self.op = DownsampledImageProvider(self.source, scale_factor=2, tile_width=256, tile_height=256)

    def test_key_format(self):
        self.assertEqual(self.op.key, "mock:0:downsample:2")

    def test_num_bands_delegated(self):
        self.assertEqual(self.op.num_bands, 3)

    def test_pixel_value_type_delegated(self):
        self.assertEqual(self.op.pixel_value_type, PixelType.UInt8)

    def test_tile_dimensions(self):
        self.assertEqual(self.op.num_pixels_per_block_horizontal, 256)
        self.assertEqual(self.op.num_pixels_per_block_vertical, 256)

    def test_block_grid_size(self):
        # 256 rows / 256 tile = 1 row, 256 cols / 256 tile = 1 col
        self.assertEqual(self.op.block_grid_size, (1, 1))

    def test_has_block_in_bounds(self):
        self.assertTrue(self.op.has_block(0, 0))

    def test_has_block_out_of_bounds(self):
        self.assertFalse(self.op.has_block(1, 0))
        self.assertFalse(self.op.has_block(0, 1))
        self.assertFalse(self.op.has_block(-1, 0))

    def test_metadata_delegated(self):
        self.assertEqual(self.op.metadata, {})

    def test_num_resolution_levels_single(self):
        self.assertEqual(self.op.num_resolution_levels, 1)

    def test_num_resolution_levels_native(self):
        source = _MockProvider(self.image, tile_height=256, tile_width=256, num_resolution_levels=3)
        op = DownsampledImageProvider(source, scale_factor=2)
        # source has 3 levels, we consume level 1, so 3 - 1 = 2 remaining
        self.assertEqual(op.num_resolution_levels, 2)


class TestDownsampledImageProviderResamplePath(TestCase):
    """Tests for the resample path (single resolution level source)."""

    def test_resample_path_output_matches_direct_resample(self):
        """Non-J2K source (single resolution level) → resample path used,
        output matches direct resample of the source region."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (3, 256, 256), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256, pixel_value_type="float64")

        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, tile_width=128, tile_height=128)
        block = op.get_block(0, 0)

        expected = area_resample(image, 128, 128)
        self.assertEqual(block.shape, (3, 128, 128))
        np.testing.assert_array_almost_equal(block, expected)

    def test_resample_path_with_small_tiles(self):
        """Verify resample path works with smaller tiles that require
        stitching multiple source blocks."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 64, 64), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=32, tile_width=32)

        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, tile_height=32, tile_width=32)
        self.assertEqual(op.num_rows, 32)
        self.assertEqual(op.num_columns, 32)
        self.assertEqual(op.block_grid_size, (1, 1))

        block = op.get_block(0, 0)
        self.assertEqual(block.shape, (1, 32, 32))

    def test_resample_path_preserves_dtype(self):
        """Output dtype matches source dtype."""
        image = np.zeros((1, 256, 256), dtype=np.float32)
        source = _MockProvider(image, tile_height=256, tile_width=256, pixel_value_type="float32")
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample)
        block = op.get_block(0, 0)
        self.assertEqual(block.dtype, np.float32)


class TestDownsampledImageProviderNativePath(TestCase):
    """Tests for the native path (multi-resolution J2K-like source)."""

    def test_native_path_used_when_available(self):
        """J2K-like mock (num_resolution_levels=3) → native path used,
        output matches source level-1 image stitched from native blocks."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (3, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256, num_resolution_levels=3)

        op = DownsampledImageProvider(source, scale_factor=2, tile_width=256, tile_height=256)

        block = op.get_block(0, 0)

        # The native path should have read from resolution_level=1.
        native_calls = [c for c in source.get_block_calls if c[2] == 1]
        self.assertTrue(len(native_calls) > 0, "Expected native path to read at resolution_level=1")

        # The output should match the level-1 image (256x256 for a 512x512
        # source with scale_factor=2). Blocks at level 1 are 128x128, so
        # the stitch assembles all 4 native blocks into the output tile.
        expected = source._level_images[1]
        np.testing.assert_array_equal(block, expected)

    def test_native_path_not_used_for_single_level(self):
        """Single resolution level source → resample path used."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 256, 256), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256, num_resolution_levels=1)

        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample)
        op.get_block(0, 0)

        # All reads should be at resolution_level=0.
        for call in source.get_block_calls:
            self.assertEqual(call[2], 0, "Expected resample path to read at resolution_level=0")

    def test_native_path_no_structural_guard(self):
        """Native path does not trigger structural guard even for small blocks."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        # 1x1 block but 3 resolution levels means native path is used — no guard.
        source = _MockProvider(image, tile_height=1, tile_width=1, num_resolution_levels=3)
        op = DownsampledImageProvider(source, scale_factor=2)
        self.assertIsNotNone(op)


class TestDownsampledImageProviderChaining(TestCase):
    """Tests for chaining DownsampledImageProvider instances."""

    def test_chained_operations(self):
        """Chained DownsampledImageProvider(DownsampledImageProvider(source)) produces
        the expected cumulative downsample."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 256, 256), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256, pixel_value_type="float64")

        op1 = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, tile_width=128, tile_height=128)
        op2 = DownsampledImageProvider(op1, scale_factor=2, resample_func=area_resample)

        self.assertEqual(op2.num_rows, 64)
        self.assertEqual(op2.num_columns, 64)

        block = op2.get_block(0, 0)
        self.assertEqual(block.shape[0], 1)

        step1 = area_resample(image, 128, 128)
        step2 = area_resample(step1, 64, 64)
        np.testing.assert_array_almost_equal(block[:, :64, :64], step2)


class TestDownsampledImageProviderTileCache(TestCase):
    """Tests for shared TileCache integration."""

    def test_cache_hit_invokes_source_once(self):
        """Cache hit: repeated get_block with same args invokes source once."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 256, 256), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256)

        cache = TileCache(max_bytes=64 * 1024**2)
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, cache=cache)

        block1 = op.get_block(0, 0)
        calls_after_first = len(source.get_block_calls)

        block2 = op.get_block(0, 0)
        calls_after_second = len(source.get_block_calls)

        self.assertEqual(calls_after_second, calls_after_first)
        np.testing.assert_array_equal(block1, block2)

    def test_no_cache_invokes_source_each_time(self):
        """No cache: repeated get_block invokes source each time."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 256, 256), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256)

        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample)

        op.get_block(0, 0)
        calls_after_first = len(source.get_block_calls)

        op.get_block(0, 0)
        calls_after_second = len(source.get_block_calls)

        self.assertGreater(calls_after_second, calls_after_first)

    def test_cache_returns_equal_data(self):
        """Cached result is element-wise equal to the first call."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 256, 256), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256)

        cache = TileCache(max_bytes=64 * 1024**2)
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, cache=cache)

        block1 = op.get_block(0, 0)
        block2 = op.get_block(0, 0)
        np.testing.assert_array_equal(block1, block2)

    def test_shared_cache_between_operators(self):
        """Two operators with different scale_factors sharing a TileCache don't collide."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 512, 512), dtype=np.uint8).astype(np.float64)
        source = _MockProvider(image, tile_height=256, tile_width=256, pixel_value_type="float64", num_resolution_levels=3)

        cache = TileCache(max_bytes=64 * 1024**2)
        op2 = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, cache=cache)
        op4 = DownsampledImageProvider(source, scale_factor=4, resample_func=area_resample, cache=cache)

        block2 = op2.get_block(0, 0)
        block4 = op4.get_block(0, 0)

        self.assertNotEqual(op2.key, op4.key)
        self.assertGreater(cache.current_bytes, 0)
        self.assertFalse(np.array_equal(block2, block4))

    def test_cache_stores_bytes(self):
        """After a get_block, cache.current_bytes increases."""
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)

        cache = TileCache(max_bytes=64 * 1024**2)
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, cache=cache)

        self.assertEqual(cache.current_bytes, 0)
        op.get_block(0, 0)
        self.assertGreater(cache.current_bytes, 0)


class TestDownsampledImageProviderStructuralGuards(TestCase):
    """Tests for structural guards on source block dimensions."""

    def test_small_height_raises(self):
        """Source with height=1 blocks raises on resample path."""
        image = np.zeros((1, 4, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=1, tile_width=256)
        with self.assertRaises(ValueError) as ctx:
            DownsampledImageProvider(source, scale_factor=2)
        self.assertIn("height >= 2", str(ctx.exception))
        self.assertIn("RetiledImageProvider", str(ctx.exception))

    def test_small_width_raises(self):
        """Source with width=1 blocks raises on resample path."""
        image = np.zeros((1, 256, 4), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=1)
        with self.assertRaises(ValueError) as ctx:
            DownsampledImageProvider(source, scale_factor=2)
        self.assertIn("width >= 2", str(ctx.exception))
        self.assertIn("RetiledImageProvider", str(ctx.exception))

    def test_guard_checks_source_not_output(self):
        """Guard validates source block dims, not output tile params."""
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        # Source blocks are 256x256 (valid), output tiles are 1x1 — should not raise.
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample, tile_width=1, tile_height=1)
        self.assertIsNotNone(op)

    def test_native_path_bypasses_guard(self):
        """Native path (multi-resolution source) bypasses the structural guard."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        # Source has 1x1 blocks but enough resolution levels for native path.
        source = _MockProvider(image, tile_height=1, tile_width=1, num_resolution_levels=3)
        op = DownsampledImageProvider(source, scale_factor=2)
        self.assertIsNotNone(op)


class TestDownsampledImageProviderErrorHandling(TestCase):
    """Tests for error handling in get_block."""

    def _make_op(self):
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        return DownsampledImageProvider(source, scale_factor=2, resample_func=area_resample)

    def test_out_of_range_block_raises_index_error(self):
        op = self._make_op()
        with self.assertRaises(IndexError):
            op.get_block(5, 0)

    def test_negative_block_raises_index_error(self):
        op = self._make_op()
        with self.assertRaises(IndexError):
            op.get_block(-1, 0)

    def test_invalid_resolution_level_raises_value_error(self):
        op = self._make_op()
        with self.assertRaises(ValueError):
            op.get_block(0, 0, resolution_level=5)


class TestDownsampledImageProviderDefaultResampleFunc(TestCase):
    """Verify default resample_func is sips_rrds_resample."""

    def test_default_resample_func(self):
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        op = DownsampledImageProvider(source)
        self.assertIs(op._resample_func, sips_rrds_resample)
