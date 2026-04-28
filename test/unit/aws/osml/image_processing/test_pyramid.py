#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.pyramid`.

Tests cover :class:`TiledImagePyramid` (construction, properties, level
access, ``best_level_for``, factory methods), :func:`iter_blocks`
(raster ordering, sparse block skipping), and :func:`build_pyramid_levels`
(chain structure, stopping condition).

The tests use a minimal ``_MockProvider`` that mimics the osml-imagery-io
``ImageAssetProvider`` contract, consistent with the mocks used in the
``test_downsample.py`` and ``test_pyramid_builder.py`` tests.
"""

from unittest import TestCase

import numpy as np

from aws.osml.image_processing.cached_provider import CachedImageProvider
from aws.osml.image_processing.downsampled_provider import DownsampledImageProvider
from aws.osml.image_processing.pyramid import TiledImagePyramid, build_pyramid_levels, iter_blocks
from aws.osml.image_processing.resample import area_resample
from aws.osml.image_processing.retiled_provider import RetiledImageProvider
from aws.osml.image_processing.sips_resample import sips_rrds_resample
from aws.osml.image_processing.tile_cache import TileCache
from aws.osml.io import PixelType

# ----------------------------------------------------------------------
# Mocks — minimal duck-typed ImageAssetProvider for unit tests
# ----------------------------------------------------------------------


class _MockProvider:
    """Minimal ImageAssetProvider duck-type backed by a single CHW array.

    Splits the backing array into an aligned tile grid. Supports
    configurable sparsity via ``sparse_tiles`` and configurable native
    ``num_resolution_levels``.
    """

    def __init__(
        self,
        image,
        tile_height=256,
        tile_width=256,
        sparse_tiles=None,
        num_resolution_levels=1,
        pixel_value_type=PixelType.UInt8,
        pad_pixel_value=0.0,
    ):
        self._base_image = image
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)
        self._sparse_tiles = set(sparse_tiles or [])
        self._num_resolution_levels = int(num_resolution_levels)
        self._pixel_value_type = pixel_value_type
        self._pad_pixel_value = float(pad_pixel_value)

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
        grid_rows, grid_cols = self.block_grid_size
        if row < 0 or col < 0 or row >= grid_rows or col >= grid_cols:
            return False
        if (row, col) in self._sparse_tiles:
            return False
        return True

    def get_block(self, row, col, resolution_level=0, bands=None):
        self.get_block_calls.append((row, col, resolution_level))
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self._base_image.shape[-2])
        x1 = min(x0 + self._tile_width, self._base_image.shape[-1])
        tile = self._base_image[:, y0:y1, x0:x1].copy()
        if tile.shape[-2] != self._tile_height or tile.shape[-1] != self._tile_width:
            padded = np.full(
                (tile.shape[0], self._tile_height, self._tile_width),
                self._pad_pixel_value,
                dtype=tile.dtype,
            )
            padded[:, : tile.shape[-2], : tile.shape[-1]] = tile
            tile = padded
        return tile

    @property
    def metadata(self):
        return {}


class _MockReader:
    """Minimal DatasetReader duck-type for ``from_dataset`` tests.

    Stores a dict of ``{key: provider}`` and optionally supports
    ``get_assets_by_role``.
    """

    def __init__(self, assets, role_assets=None):
        self._assets = dict(assets)
        self._role_assets = role_assets or {}

    def get_asset(self, key):
        if key not in self._assets:
            raise KeyError(f"Asset not found: {key}")
        return self._assets[key]

    def get_assets_by_role(self, role):
        return self._role_assets.get(role, [])


# ----------------------------------------------------------------------
# TiledImagePyramid — basic operations
# ----------------------------------------------------------------------


class TestTiledImagePyramidConstruction(TestCase):
    """Construction and basic property tests."""

    def _make_providers(self, shapes):
        """Create a list of mock providers with the given (bands, rows, cols) shapes."""
        providers = []
        for bands, rows, cols in shapes:
            image = np.zeros((bands, rows, cols), dtype=np.uint8)
            providers.append(_MockProvider(image, tile_height=min(rows, 256), tile_width=min(cols, 256)))
        return providers

    def test_empty_levels_raises_value_error(self):
        with self.assertRaises(ValueError):
            TiledImagePyramid([])

    def test_num_levels(self):
        providers = self._make_providers([(3, 1024, 1024), (3, 512, 512), (3, 256, 256)])
        pyramid = TiledImagePyramid(providers)
        self.assertEqual(pyramid.num_levels, 3)

    def test_scale_factor_default(self):
        providers = self._make_providers([(3, 512, 512)])
        pyramid = TiledImagePyramid(providers)
        self.assertEqual(pyramid.scale_factor, 2)

    def test_scale_factor_custom(self):
        providers = self._make_providers([(3, 512, 512)])
        pyramid = TiledImagePyramid(providers, scale_factor=4)
        self.assertEqual(pyramid.scale_factor, 4)


class TestTiledImagePyramidLevelAccess(TestCase):
    """Tests for get_level, image_shape_at_level, tile_grid_at_level."""

    def setUp(self):
        self.providers = []
        for rows, cols in [(1024, 1024), (512, 512), (256, 256)]:
            image = np.zeros((3, rows, cols), dtype=np.uint8)
            self.providers.append(_MockProvider(image, tile_height=256, tile_width=256))
        self.pyramid = TiledImagePyramid(self.providers)

    def test_get_level_returns_correct_provider(self):
        for i, provider in enumerate(self.providers):
            self.assertIs(self.pyramid.get_level(i), provider)

    def test_get_level_out_of_range_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.pyramid.get_level(3)

    def test_get_level_negative_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.pyramid.get_level(-1)

    def test_image_shape_at_level(self):
        self.assertEqual(self.pyramid.image_shape_at_level(0), (3, 1024, 1024))
        self.assertEqual(self.pyramid.image_shape_at_level(1), (3, 512, 512))
        self.assertEqual(self.pyramid.image_shape_at_level(2), (3, 256, 256))

    def test_tile_grid_at_level(self):
        # 1024 / 256 = 4x4 grid
        self.assertEqual(self.pyramid.tile_grid_at_level(0), (4, 4))
        # 512 / 256 = 2x2 grid
        self.assertEqual(self.pyramid.tile_grid_at_level(1), (2, 2))
        # 256 / 256 = 1x1 grid
        self.assertEqual(self.pyramid.tile_grid_at_level(2), (1, 1))

    def test_image_shape_out_of_range_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.pyramid.image_shape_at_level(5)

    def test_tile_grid_out_of_range_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.pyramid.tile_grid_at_level(5)


# ----------------------------------------------------------------------
# TiledImagePyramid — best_level_for
# ----------------------------------------------------------------------


class TestBestLevelFor(TestCase):
    """Tests for best_level_for(src_size, output_size)."""

    def setUp(self):
        providers = []
        for rows, cols in [(1024, 1024), (512, 512), (256, 256), (128, 128)]:
            image = np.zeros((3, rows, cols), dtype=np.uint8)
            providers.append(_MockProvider(image, tile_height=min(rows, 256), tile_width=min(cols, 256)))
        self.pyramid = TiledImagePyramid(providers)

    def test_output_larger_than_src_returns_level_0(self):
        """Upsampling case: output > src at level 0, return level 0."""
        level = self.pyramid.best_level_for((128, 128), (256, 256))
        self.assertEqual(level, 0)

    def test_no_scaling_returns_level_0(self):
        """src_size == output_size → level 0 (1024/1=1024 >= 1024)."""
        level = self.pyramid.best_level_for((1024, 1024), (1024, 1024))
        self.assertEqual(level, 0)

    def test_2x_downscale_returns_level_1(self):
        """src 1024, output 512 → level 1 (1024/2=512 >= 512)."""
        level = self.pyramid.best_level_for((1024, 1024), (512, 512))
        self.assertEqual(level, 1)

    def test_4x_downscale_returns_level_2(self):
        """src 1024, output 256 → level 2 (1024/4=256 >= 256)."""
        level = self.pyramid.best_level_for((1024, 1024), (256, 256))
        self.assertEqual(level, 2)

    def test_8x_downscale_returns_level_3(self):
        """src 1024, output 128 → level 3 (1024/8=128 >= 128)."""
        level = self.pyramid.best_level_for((1024, 1024), (128, 128))
        self.assertEqual(level, 3)

    def test_very_small_output_uses_deepest_level(self):
        """src 1024, output 64 → all levels qualify, use deepest."""
        level = self.pyramid.best_level_for((1024, 1024), (64, 64))
        self.assertEqual(level, 3)

    def test_between_levels(self):
        """src 1024, output 600 → level 0 (1024>=600), level 1 (512<600)."""
        level = self.pyramid.best_level_for((1024, 1024), (600, 600))
        self.assertEqual(level, 0)

    def test_zero_src_width_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.pyramid.best_level_for((0, 512), (256, 256))

    def test_negative_output_height_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.pyramid.best_level_for((512, 512), (256, -1))

    def test_asymmetric_src_size(self):
        """src (1024, 512), output (256, 256) →
        level 0: 1024>=256, 512>=256 ✓
        level 1: 512>=256, 256>=256 ✓
        level 2: 256>=256, 128>=256? No ✗ → level 1."""
        level = self.pyramid.best_level_for((1024, 512), (256, 256))
        self.assertEqual(level, 1)

    def test_small_src_no_valid_level(self):
        """Single-level pyramid. src < output returns level 0."""
        image = np.zeros((3, 64, 64), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=64, tile_width=64)
        pyramid = TiledImagePyramid([provider])
        level = pyramid.best_level_for((64, 64), (128, 128))
        self.assertEqual(level, 0)


# ----------------------------------------------------------------------
# TiledImagePyramid — factory methods
# ----------------------------------------------------------------------


class TestFromProviders(TestCase):
    """Tests for TiledImagePyramid.from_providers."""

    def test_preserves_order(self):
        providers = []
        for rows in [1024, 512, 256]:
            image = np.zeros((3, rows, rows), dtype=np.uint8)
            providers.append(_MockProvider(image, tile_height=256, tile_width=256))
        pyramid = TiledImagePyramid.from_providers(providers)
        self.assertEqual(pyramid.num_levels, 3)
        for i, p in enumerate(providers):
            self.assertIs(pyramid.get_level(i), p)

    def test_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            TiledImagePyramid.from_providers([])

    def test_custom_scale_factor(self):
        image = np.zeros((3, 512, 512), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        pyramid = TiledImagePyramid.from_providers([provider], scale_factor=4)
        self.assertEqual(pyramid.scale_factor, 4)


class TestFromDataset(TestCase):
    """Tests for TiledImagePyramid.from_dataset."""

    def _make_provider(self, rows, cols):
        image = np.zeros((3, rows, cols), dtype=np.uint8)
        return _MockProvider(image, tile_height=min(rows, 256), tile_width=min(cols, 256))

    def test_three_level_pyramid_by_key_convention(self):
        """Reader with image:0, image:0:overview:1, image:0:overview:2
        returns a 3-level pyramid ordered by resolution."""
        base = self._make_provider(1024, 1024)
        ov1 = self._make_provider(512, 512)
        ov2 = self._make_provider(256, 256)
        reader = _MockReader(
            {
                "image:0": base,
                "image:0:overview:1": ov1,
                "image:0:overview:2": ov2,
            }
        )
        pyramid = TiledImagePyramid.from_dataset(reader, "image:0")
        self.assertEqual(pyramid.num_levels, 3)
        self.assertIs(pyramid.get_level(0), base)
        self.assertIs(pyramid.get_level(1), ov1)
        self.assertIs(pyramid.get_level(2), ov2)

    def test_base_only_returns_single_level(self):
        """Reader with only the base asset returns a 1-level pyramid."""
        base = self._make_provider(1024, 1024)
        reader = _MockReader({"image:0": base})
        pyramid = TiledImagePyramid.from_dataset(reader, "image:0")
        self.assertEqual(pyramid.num_levels, 1)
        self.assertIs(pyramid.get_level(0), base)

    def test_missing_base_raises_key_error(self):
        """Reader without the base asset raises KeyError."""
        reader = _MockReader({})
        with self.assertRaises(KeyError):
            TiledImagePyramid.from_dataset(reader, "image:0")

    def test_role_based_discovery(self):
        """When key convention finds nothing, fall back to role-based
        discovery."""
        base = self._make_provider(1024, 1024)
        ov1 = self._make_provider(512, 512)
        ov2 = self._make_provider(256, 256)
        reader = _MockReader(
            {"image:0": base},
            role_assets={"overview": [ov2, ov1]},  # intentionally unordered
        )
        pyramid = TiledImagePyramid.from_dataset(reader, "image:0")
        self.assertEqual(pyramid.num_levels, 3)
        self.assertIs(pyramid.get_level(0), base)
        # Role-based assets are sorted by decreasing resolution.
        self.assertEqual(pyramid.image_shape_at_level(1)[1], 512)
        self.assertEqual(pyramid.image_shape_at_level(2)[1], 256)

    def test_custom_base_key(self):
        """from_dataset works with a non-default base_key."""
        base = self._make_provider(512, 512)
        ov1 = self._make_provider(256, 256)
        reader = _MockReader(
            {
                "image:1": base,
                "image:1:overview:1": ov1,
            }
        )
        pyramid = TiledImagePyramid.from_dataset(reader, "image:1")
        self.assertEqual(pyramid.num_levels, 2)
        self.assertIs(pyramid.get_level(0), base)
        self.assertIs(pyramid.get_level(1), ov1)


# ----------------------------------------------------------------------
# iter_blocks
# ----------------------------------------------------------------------


class TestIterBlocks(TestCase):
    """Tests for iter_blocks."""

    def test_visits_all_blocks_in_raster_order(self):
        """All non-sparse blocks are visited in row-major order."""
        image = np.arange(3 * 64 * 64, dtype=np.uint8).reshape(3, 64, 64)
        provider = _MockProvider(image, tile_height=32, tile_width=32)
        # 64/32 = 2x2 grid
        results = list(iter_blocks(provider))
        self.assertEqual(len(results), 4)
        expected_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        actual_coords = [(r, c) for r, c, _ in results]
        self.assertEqual(actual_coords, expected_coords)

    def test_sparse_blocks_skipped(self):
        """Sparse blocks are not yielded."""
        image = np.zeros((1, 64, 64), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=32, tile_width=32, sparse_tiles={(0, 1), (1, 0)})
        results = list(iter_blocks(provider))
        self.assertEqual(len(results), 2)
        actual_coords = [(r, c) for r, c, _ in results]
        self.assertEqual(actual_coords, [(0, 0), (1, 1)])

    def test_all_sparse_yields_nothing(self):
        """When all blocks are sparse, nothing is yielded."""
        image = np.zeros((1, 32, 32), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=32, tile_width=32, sparse_tiles={(0, 0)})
        results = list(iter_blocks(provider))
        self.assertEqual(len(results), 0)

    def test_single_block(self):
        """Single-block provider yields one result."""
        image = np.ones((1, 32, 32), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=32, tile_width=32)
        results = list(iter_blocks(provider))
        self.assertEqual(len(results), 1)
        r, c, block = results[0]
        self.assertEqual(r, 0)
        self.assertEqual(c, 0)
        self.assertEqual(block.shape, (1, 32, 32))

    def test_block_data_is_correct(self):
        """Yielded block data matches the provider's get_block output."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 64, 64), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=32, tile_width=32)
        for r, c, block in iter_blocks(provider):
            expected = provider.get_block(r, c, 0)
            np.testing.assert_array_equal(block, expected)


# ----------------------------------------------------------------------
# build_pyramid_levels
# ----------------------------------------------------------------------


class TestBuildPyramidLevels(TestCase):
    """Tests for build_pyramid_levels."""

    def test_first_element_is_source(self):
        """The first element of the returned list is the source itself."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        self.assertIs(levels[0], source)

    def test_chain_structure(self):
        """Each level after the first is a DownsampledImageProvider whose
        source is the previous level."""
        image = np.zeros((1, 1024, 1024), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        self.assertGreater(len(levels), 1)
        for i in range(1, len(levels)):
            self.assertIsInstance(levels[i], DownsampledImageProvider)
            self.assertIs(levels[i]._source, levels[i - 1])

    def test_stopping_condition(self):
        """Generation stops when either dim falls below min_size;
        that level is included."""
        image = np.zeros((1, 1024, 1024), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        # 1024 -> 512 -> 256 -> 128. Level at 128 has either dim < 256,
        # so it is included as the final level.
        # levels = [source(1024), ds(512), ds(256), ds(128)]
        self.assertEqual(len(levels), 4)
        # Last level should have either dim < min_size.
        last = levels[-1]
        self.assertTrue(
            last.num_rows < 256 or last.num_columns < 256,
        )

    def test_small_source_below_min_size(self):
        """When source already has either axis below min_size, return
        just the source."""
        image = np.zeros((1, 64, 64), dtype=np.uint8)
        source = _MockProvider(image, tile_height=64, tile_width=64)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        self.assertEqual(len(levels), 1)
        self.assertIs(levels[0], source)

    def test_default_resample_func_is_sips(self):
        """When resample_func is None, defaults to sips_rrds_resample."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256)
        # The DownsampledImageProvider at level 1 should use sips_rrds_resample.
        self.assertIs(levels[1]._resample_func, sips_rrds_resample)

    def test_level_dimensions_follow_sips_rounding(self):
        """Level dimensions follow the SIPS even/odd rounding rule."""
        image = np.zeros((1, 513, 513), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=64, resample_func=area_resample)
        # 513 -> 257 -> 129 -> 65 -> 33
        expected_dims = [513, 257, 129, 65, 33]
        for i, expected in enumerate(expected_dims):
            if i == 0:
                self.assertEqual(source.num_rows, expected)
            else:
                self.assertEqual(levels[i].num_rows, expected)
                self.assertEqual(levels[i].num_columns, expected)

    def test_asymmetric_dimensions(self):
        """Asymmetric source dimensions stop when the shorter axis
        drops below min_size."""
        image = np.zeros((1, 1024, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        # rows: 1024 -> 512 -> 256
        # cols: 512 -> 256 -> 128
        # Level 2 (256x128): cols=128 < 256, include and stop.
        self.assertEqual(len(levels), 3)
        self.assertEqual(levels[1].num_rows, 512)
        self.assertEqual(levels[1].num_columns, 256)

    def test_custom_tile_dimensions(self):
        """Custom tile_width and tile_height are passed through."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample, tile_width=128, tile_height=128)
        for i in range(1, len(levels)):
            self.assertEqual(levels[i].num_pixels_per_block_horizontal, 128)
            self.assertEqual(levels[i].num_pixels_per_block_vertical, 128)

    def test_cache_passed_through(self):
        """TileCache is passed through to DownsampledImageProvider."""
        from aws.osml.image_processing.tile_cache import TileCache

        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        cache = TileCache(max_bytes=64 * 1024**2)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample, cache=cache)
        for i in range(1, len(levels)):
            self.assertIs(levels[i]._tile_cache, cache)

    def test_no_cache_default(self):
        """Default cache=None means no caching on DownsampledImageProvider."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        for i in range(1, len(levels)):
            self.assertIsNone(levels[i]._tile_cache)


# ----------------------------------------------------------------------
# build_pyramid_levels — auto-wrapping
# ----------------------------------------------------------------------


class TestBuildPyramidLevelsAutoWrap(TestCase):
    """Tests for convenience auto-wrapping in build_pyramid_levels."""

    def test_auto_wrap_with_cache_when_retiling_needed(self):
        """When cache is provided and source block dims != requested tile dims,
        auto-wraps with CachedImageProvider + RetiledImageProvider."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=512, tile_width=512)
        cache = TileCache(max_bytes=64 * 1024**2)
        levels = build_pyramid_levels(
            source, min_size=128, resample_func=area_resample, tile_width=256, tile_height=256, cache=cache
        )
        # levels[0] should be a RetiledImageProvider
        self.assertIsInstance(levels[0], RetiledImageProvider)
        # Its source should be a CachedImageProvider
        self.assertIsInstance(levels[0]._source, CachedImageProvider)
        # The CachedImageProvider's source should be the original source
        self.assertIs(levels[0]._source._source, source)

    def test_auto_wrap_without_cache_when_retiling_needed(self):
        """When no cache and source block dims != requested tile dims,
        auto-wraps with RetiledImageProvider only (no CachedImageProvider)."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=512, tile_width=512)
        levels = build_pyramid_levels(source, min_size=128, resample_func=area_resample, tile_width=256, tile_height=256)
        # levels[0] should be a RetiledImageProvider
        self.assertIsInstance(levels[0], RetiledImageProvider)
        # Its source should be the original source directly (no CachedImageProvider)
        self.assertIs(levels[0]._source, source)

    def test_no_wrap_when_dims_already_match(self):
        """When source block dims match requested tile dims, no wrapping occurs."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        cache = TileCache(max_bytes=64 * 1024**2)
        levels = build_pyramid_levels(
            source, min_size=128, resample_func=area_resample, tile_width=256, tile_height=256, cache=cache
        )
        # levels[0] should be the source itself (no wrapping)
        self.assertIs(levels[0], source)

    def test_no_double_wrap_with_retiled_source(self):
        """When source is already a RetiledImageProvider with matching dims,
        no second wrapper is inserted."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        raw_source = _MockProvider(image, tile_height=512, tile_width=512)
        retiled = RetiledImageProvider(raw_source, tile_width=256, tile_height=256)
        cache = TileCache(max_bytes=64 * 1024**2)
        levels = build_pyramid_levels(
            retiled, min_size=128, resample_func=area_resample, tile_width=256, tile_height=256, cache=cache
        )
        # levels[0] should be the retiled source itself (already matches)
        self.assertIs(levels[0], retiled)

    def test_auto_wrap_defaults_to_source_block_dims(self):
        """When tile_width/tile_height are None, defaults to source block dims (no wrap needed)."""
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        source = _MockProvider(image, tile_height=256, tile_width=256)
        cache = TileCache(max_bytes=64 * 1024**2)
        levels = build_pyramid_levels(source, min_size=128, resample_func=area_resample, cache=cache)
        # tile_width/tile_height default to source dims, so no retiling needed
        self.assertIs(levels[0], source)

    def test_oversized_source_block_clamped_to_max(self):
        """Source with block dims > 8192 (untiled NITF) should be clamped."""
        from aws.osml.image_processing.pyramid import _MAX_TILE_SIZE

        image = np.zeros((1, 4096, 19278), dtype=np.uint8)
        source = _MockProvider(image, tile_height=4096, tile_width=19278)
        levels = build_pyramid_levels(source, min_size=256, resample_func=area_resample)
        # Level 0 should be retiled since inherited dims were clamped
        self.assertIsInstance(levels[0], RetiledImageProvider)
        self.assertLessEqual(levels[0].num_pixels_per_block_horizontal, _MAX_TILE_SIZE)
        self.assertLessEqual(levels[0].num_pixels_per_block_vertical, _MAX_TILE_SIZE)

    def test_auto_wrap_produces_correct_output(self):
        """End-to-end: auto-wrapping a single-block source produces correct pyramid output."""
        rng = np.random.default_rng(99)
        image = rng.integers(0, 255, (1, 512, 512), dtype=np.uint8)
        # Source reports as a single 512x512 block (simulating J2K single-codestream)
        source = _MockProvider(image, tile_height=512, tile_width=512)
        cache = TileCache(max_bytes=64 * 1024**2)
        levels = build_pyramid_levels(
            source, min_size=128, resample_func=area_resample, tile_width=256, tile_height=256, cache=cache
        )
        # Should produce multiple levels
        self.assertGreater(len(levels), 1)
        # All DownsampledImageProvider levels should have 256x256 output tiles
        for i in range(1, len(levels)):
            self.assertIsInstance(levels[i], DownsampledImageProvider)
            self.assertEqual(levels[i].num_pixels_per_block_horizontal, 256)
            self.assertEqual(levels[i].num_pixels_per_block_vertical, 256)


# ----------------------------------------------------------------------
# TiledImagePyramid — compute_statistics
# ----------------------------------------------------------------------


class TestPyramidComputeStatistics(TestCase):
    """Tests for TiledImagePyramid.compute_statistics and _best_level_for_statistics."""

    def _build_pyramid(self, base_image, min_size=64):
        """Build a multi-level pyramid from a base image using area resampling."""
        source = _MockProvider(base_image, tile_height=64, tile_width=64)
        levels = build_pyramid_levels(source, min_size=min_size, resample_func=area_resample)
        return TiledImagePyramid(levels)

    def test_best_level_for_statistics_selects_coarsest_meeting_threshold(self):
        """_best_level_for_statistics picks the coarsest level with >= max_pixels."""
        # 3-band 256x256 => 65536 pixels at level 0
        # level 1: 128x128 => 16384 pixels
        # level 2: 64x64 => 4096 pixels
        # level 3: 32x32 => 1024 pixels
        image = np.random.default_rng(42).integers(0, 256, (3, 256, 256), dtype=np.uint8)
        pyramid = self._build_pyramid(image, min_size=32)

        # Threshold 4096 should select the deepest level with >= 4096 pixels
        level = pyramid._best_level_for_statistics(4096)
        _, rows, cols = pyramid.image_shape_at_level(level)
        self.assertGreaterEqual(rows * cols, 4096)
        # The next deeper level should have < 4096 pixels
        if level < pyramid.num_levels - 1:
            _, next_rows, next_cols = pyramid.image_shape_at_level(level + 1)
            self.assertLess(next_rows * next_cols, 4096)

    def test_best_level_for_statistics_returns_0_when_no_level_meets_threshold(self):
        """When no level has enough pixels, returns level 0."""
        image = np.zeros((1, 64, 64), dtype=np.uint8)
        source = _MockProvider(image, tile_height=64, tile_width=64)
        pyramid = TiledImagePyramid([source])
        level = pyramid._best_level_for_statistics(100_000_000)
        self.assertEqual(level, 0)

    def test_compute_statistics_at_explicit_level(self):
        """With explicit level, computes at exactly that level."""
        rng = np.random.default_rng(7)
        image = rng.integers(50, 200, (1, 128, 128), dtype=np.uint8)
        pyramid = self._build_pyramid(image, min_size=32)

        stats = pyramid.compute_statistics(level=0)
        self.assertEqual(len(stats.bands), 1)
        self.assertEqual(stats.bands[0].count, 128 * 128)

    def test_compute_statistics_auto_selects_level(self):
        """Without explicit level, delegates to _best_level_for_statistics."""
        rng = np.random.default_rng(99)
        image = rng.integers(0, 256, (1, 256, 256), dtype=np.uint8)
        pyramid = self._build_pyramid(image, min_size=32)

        # max_pixels=1000 should allow a coarser level than 0
        stats = pyramid.compute_statistics(max_pixels=1000)
        self.assertEqual(len(stats.bands), 1)
        # Count should be less than level-0's pixel count
        self.assertLess(stats.bands[0].count, 256 * 256)

    def test_compute_statistics_accuracy_mean(self):
        """Overview-level mean is within 1% of level-0 mean."""
        # Use a spatially structured image (smooth gradient + low noise) to
        # simulate real imagery where nearby pixels are correlated.
        rng = np.random.default_rng(123)
        base = np.linspace(20, 230, 256 * 256).reshape(256, 256)
        image = np.stack(
            [
                (base + rng.normal(0, 5, (256, 256))).clip(0, 255).astype(np.uint8),
                (base * 0.8 + 30 + rng.normal(0, 5, (256, 256))).clip(0, 255).astype(np.uint8),
                (255 - base + rng.normal(0, 5, (256, 256))).clip(0, 255).astype(np.uint8),
            ]
        )
        pyramid = self._build_pyramid(image, min_size=32)

        stats_r0 = pyramid.compute_statistics(level=0)
        stats_overview = pyramid.compute_statistics(level=1)

        for b in range(3):
            r0_mean = stats_r0.bands[b].mean
            ov_mean = stats_overview.bands[b].mean
            relative_error = abs(ov_mean - r0_mean) / max(abs(r0_mean), 1e-10)
            self.assertLess(relative_error, 0.01, f"Band {b} mean error {relative_error:.4f} exceeds 1%")  # noqa: E231

    def test_compute_statistics_accuracy_stddev(self):
        """Overview-level stddev is within 5% of level-0 stddev."""
        # Spatially correlated image: smooth gradient dominates variance,
        # 2x2 averaging preserves large-scale structure. This matches real
        # imagery behavior where SIPS RRDS resampling maintains stddev accuracy.
        rng = np.random.default_rng(456)
        base = np.linspace(20, 230, 256 * 256).reshape(256, 256)
        image = np.stack(
            [
                (base + rng.normal(0, 3, (256, 256))).clip(0, 255).astype(np.uint8),
                (base * 0.8 + 30 + rng.normal(0, 3, (256, 256))).clip(0, 255).astype(np.uint8),
                (255 - base + rng.normal(0, 3, (256, 256))).clip(0, 255).astype(np.uint8),
            ]
        )
        pyramid = self._build_pyramid(image, min_size=32)

        stats_r0 = pyramid.compute_statistics(level=0)
        stats_overview = pyramid.compute_statistics(level=1)

        for b in range(3):
            r0_std = stats_r0.bands[b].stddev
            ov_std = stats_overview.bands[b].stddev
            relative_error = abs(ov_std - r0_std) / max(abs(r0_std), 1e-10)
            self.assertLess(relative_error, 0.05, f"Band {b} stddev error {relative_error:.4f} exceeds 5%")  # noqa: E231

    def test_compute_statistics_passes_num_bins(self):
        """num_bins parameter is forwarded to compute_image_statistics."""
        image = np.random.default_rng(10).integers(0, 256, (1, 128, 128), dtype=np.uint8)
        pyramid = self._build_pyramid(image, min_size=32)

        stats = pyramid.compute_statistics(level=0, num_bins=32)
        self.assertEqual(len(stats.bands[0].histogram), 32)

    def test_compute_statistics_passes_bin_edges(self):
        """bin_edges parameter is forwarded to compute_image_statistics."""
        image = np.random.default_rng(11).integers(0, 256, (1, 128, 128), dtype=np.uint8)
        pyramid = self._build_pyramid(image, min_size=32)

        edges = np.linspace(0, 255, 17)
        stats = pyramid.compute_statistics(level=0, bin_edges=edges)
        self.assertEqual(len(stats.bands[0].histogram), 16)

    def test_compute_statistics_force_recompute(self):
        """compute_statistics always force-recomputes (ignores metadata)."""
        rng = np.random.default_rng(77)
        image = rng.integers(0, 256, (1, 64, 64), dtype=np.uint8)
        source = _MockProvider(image, tile_height=64, tile_width=64)
        pyramid = TiledImagePyramid([source])

        stats = pyramid.compute_statistics(level=0)
        # Should produce valid statistics even though provider has no GDAL metadata
        self.assertIsNotNone(stats)
        self.assertEqual(stats.bands[0].count, 64 * 64)
