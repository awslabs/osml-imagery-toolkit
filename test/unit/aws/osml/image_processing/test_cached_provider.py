#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import collections.abc
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.cached_provider import CachedImageProvider
from aws.osml.image_processing.tile_cache import TileCache


class _MockMetadata(collections.abc.Mapping):
    def __init__(self, d=None):
        self._d = d or {}

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


class _MockProvider:
    """Minimal duck-typed ImageAssetProvider mock."""

    def __init__(self, image, key="mock-key"):
        self._image = image
        self._key = key
        self.read_count = 0

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
        return self._image.shape[2]

    @property
    def num_pixels_per_block_vertical(self):
        return self._image.shape[1]

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def block_grid_size(self):
        return (1, 1)

    @property
    def num_bands(self):
        return self._image.shape[0]

    @property
    def pixel_value_type(self):
        return "uint8"

    def has_block(self, row, col, resolution_level=0):
        return row == 0 and col == 0

    def get_block(self, row, col, resolution_level=0, bands=None):
        self.read_count += 1
        if bands is not None:
            return self._image[list(bands), :, :].copy()
        return self._image.copy()

    @property
    def metadata(self):
        return _MockMetadata({"test": "value"})


class TestCachedImageProviderCacheHit(TestCase):
    """Tests that cache hits avoid source reads."""

    def test_second_read_is_cache_hit(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        source = _MockProvider(image)
        cache = TileCache(max_bytes=1024 * 1024)
        cached = CachedImageProvider(source, cache=cache)

        result1 = cached.get_block(0, 0)
        result2 = cached.get_block(0, 0)

        np.testing.assert_array_equal(result1, result2)
        self.assertEqual(source.read_count, 1)

    def test_different_positions_are_separate(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        source = _MockProvider(image)
        cache = TileCache(max_bytes=1024 * 1024)
        cached = CachedImageProvider(source, cache=cache)

        cached.get_block(0, 0)
        cached.get_block(0, 0)
        cached.get_block(0, 1)  # different col → cache miss (even if source lacks it)

        self.assertEqual(source.read_count, 2)

    def test_different_bands_are_separate(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        source = _MockProvider(image)
        cache = TileCache(max_bytes=1024 * 1024)
        cached = CachedImageProvider(source, cache=cache)

        cached.get_block(0, 0, bands=(0,))
        cached.get_block(0, 0, bands=(0,))  # hit
        cached.get_block(0, 0, bands=(0, 1))  # miss

        self.assertEqual(source.read_count, 2)


class TestCachedImageProviderNoCache(TestCase):
    """Tests that cache=None is a no-op pass-through."""

    def test_no_cache_reads_every_time(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        source = _MockProvider(image)
        cached = CachedImageProvider(source, cache=None)

        cached.get_block(0, 0)
        cached.get_block(0, 0)
        cached.get_block(0, 0)

        self.assertEqual(source.read_count, 3)

    def test_no_cache_returns_source_data(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        source = _MockProvider(image)
        cached = CachedImageProvider(source, cache=None)

        result = cached.get_block(0, 0)
        np.testing.assert_array_equal(result, image)


class TestCachedImageProviderTransparency(TestCase):
    """Tests that all properties delegate to source unchanged."""

    def setUp(self):
        self.image = np.zeros((3, 64, 128), dtype=np.uint8)
        self.source = _MockProvider(self.image, key="test-asset")
        self.cached = CachedImageProvider(self.source, cache=TileCache())

    def test_key_delegates(self):
        self.assertEqual(self.cached.key, "test-asset")

    def test_num_rows_delegates(self):
        self.assertEqual(self.cached.num_rows, 64)

    def test_num_columns_delegates(self):
        self.assertEqual(self.cached.num_columns, 128)

    def test_num_pixels_per_block_horizontal_delegates(self):
        self.assertEqual(self.cached.num_pixels_per_block_horizontal, 128)

    def test_num_pixels_per_block_vertical_delegates(self):
        self.assertEqual(self.cached.num_pixels_per_block_vertical, 64)

    def test_num_resolution_levels_delegates(self):
        self.assertEqual(self.cached.num_resolution_levels, 1)

    def test_block_grid_size_delegates(self):
        self.assertEqual(self.cached.block_grid_size, (1, 1))

    def test_num_bands_delegates(self):
        self.assertEqual(self.cached.num_bands, 3)

    def test_pixel_value_type_delegates(self):
        self.assertEqual(self.cached.pixel_value_type, "uint8")

    def test_has_block_delegates(self):
        self.assertTrue(self.cached.has_block(0, 0))
        self.assertFalse(self.cached.has_block(1, 1))

    def test_metadata_delegates(self):
        self.assertEqual(dict(self.cached.metadata), {"test": "value"})


class TestCachedImageProviderKeyTransparency(TestCase):
    """Tests that CachedImageProvider.key delegates to source.key unchanged."""

    def test_key_is_source_key(self):
        source = _MockProvider(np.zeros((1, 4, 4), dtype=np.uint8), key="unique-asset-id")
        cached = CachedImageProvider(source, cache=TileCache())
        self.assertEqual(cached.key, "unique-asset-id")
        self.assertEqual(cached.key, source.key)
