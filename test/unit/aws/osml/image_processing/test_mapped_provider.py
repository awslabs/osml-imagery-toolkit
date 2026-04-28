#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import collections.abc
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.mapped_provider import MappedImageProvider
from aws.osml.image_processing.tile_cache import TileCache


class _MockMetadata(collections.abc.Mapping):
    """Minimal mock for the metadata object returned by ImageAssetProvider.metadata."""

    def __init__(self, metadata_dict):
        self._dict = metadata_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)


class _MockProvider:
    """Minimal mock for a duck-typed ImageAssetProvider.

    Stores a CHW array and exposes it as a single-block grid with all
    the properties that MappedImageProvider delegates to.
    """

    def __init__(self, image, metadata_dict=None):
        self._image = image
        self._metadata_dict = metadata_dict or {}
        self._read_count = 0

    @property
    def key(self):
        return "mock-key"

    @property
    def num_rows(self):
        return self._image.shape[-2]

    @property
    def num_columns(self):
        return self._image.shape[-1]

    @property
    def num_pixels_per_block_horizontal(self):
        return self._image.shape[-1]

    @property
    def num_pixels_per_block_vertical(self):
        return self._image.shape[-2]

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def block_grid_size(self):
        return (1, 1)

    @property
    def num_bands(self):
        return self._image.shape[0] if self._image.ndim == 3 else 1

    @property
    def pixel_value_type(self):
        return "uint8"

    def has_block(self, row, col, resolution_level=0):
        return row == 0 and col == 0

    def get_block(self, row, col, resolution_level=0, bands=None):
        self._read_count += 1
        return self._image.copy()

    @property
    def metadata(self):
        return _MockMetadata(self._metadata_dict)


class TestMappedImageProviderFunctionApplication(TestCase):
    """Tests that get_block applies the function to the source block."""

    def test_function_is_applied(self):
        """get_block returns func(source.get_block(...))."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        source = _MockProvider(image)

        def double(block):
            return block * 2

        mapped = MappedImageProvider(source, double)
        result = mapped.get_block(0, 0)

        expected = image * 2
        np.testing.assert_array_equal(result, expected)

    def test_function_receives_source_block(self):
        """The function receives exactly what the source returns."""
        image = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)
        source = _MockProvider(image)
        received_blocks = []

        def capture(block):
            received_blocks.append(block.copy())
            return block

        mapped = MappedImageProvider(source, capture)
        mapped.get_block(0, 0)

        self.assertEqual(len(received_blocks), 1)
        np.testing.assert_array_equal(received_blocks[0], image)


class TestMappedImageProviderCache(TestCase):
    """Tests for the TileCache caching behavior."""

    def test_cache_hit_function_called_once(self):
        """With cache provided, repeated get_block calls with same args only call func once."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        source = _MockProvider(image)
        call_count = [0]

        def counting_func(block):
            call_count[0] += 1
            return block + 1

        cache = TileCache(max_bytes=1024 * 1024)
        mapped = MappedImageProvider(source, counting_func, cache=cache, name="test")

        result1 = mapped.get_block(0, 0)
        result2 = mapped.get_block(0, 0)

        # Function should have been called only once
        self.assertEqual(call_count[0], 1)
        # Source should have been read only once
        self.assertEqual(source._read_count, 1)
        # Both results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_no_cache_function_called_every_time(self):
        """With cache=None, repeated get_block calls invoke func each time."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        source = _MockProvider(image)
        call_count = [0]

        def counting_func(block):
            call_count[0] += 1
            return block + 1

        mapped = MappedImageProvider(source, counting_func)

        mapped.get_block(0, 0)
        mapped.get_block(0, 0)
        mapped.get_block(0, 0)

        # Function should have been called every time
        self.assertEqual(call_count[0], 3)
        # Source should have been read every time
        self.assertEqual(source._read_count, 3)

    def test_different_args_produce_separate_cache_entries(self):
        """Different (row, col) arguments are cached independently."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        source = _MockProvider(image)
        call_count = [0]

        def counting_func(block):
            call_count[0] += 1
            return block

        cache = TileCache(max_bytes=1024 * 1024)
        mapped = MappedImageProvider(source, counting_func, cache=cache, name="test")

        mapped.get_block(0, 0)
        mapped.get_block(0, 0)  # cache hit
        mapped.get_block(0, 1)  # different col → cache miss

        self.assertEqual(call_count[0], 2)

    def test_bands_parameter_in_cache_key(self):
        """Different bands arguments produce separate cache entries."""
        image = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)
        source = _MockProvider(image)
        call_count = [0]

        def counting_func(block):
            call_count[0] += 1
            return block

        cache = TileCache(max_bytes=1024 * 1024)
        mapped = MappedImageProvider(source, counting_func, cache=cache, name="test")

        mapped.get_block(0, 0, bands=(0,))
        mapped.get_block(0, 0, bands=(0,))  # cache hit
        mapped.get_block(0, 0, bands=(0, 1))  # different bands → cache miss

        self.assertEqual(call_count[0], 2)


class TestMappedImageProviderDelegation(TestCase):
    """Tests that properties and methods delegate to the source provider."""

    def setUp(self):
        self.image = np.zeros((3, 64, 128), dtype=np.uint8)
        self.source = _MockProvider(self.image, metadata_dict={"some_key": "some_value"})
        self.mapped = MappedImageProvider(self.source, lambda b: b, name="identity")

    def test_key_includes_name(self):
        """key property includes source key and name."""
        self.assertEqual(self.mapped.key, "mock-key:mapped:identity")

    def test_num_rows_delegates(self):
        """num_rows property delegates to source."""
        self.assertEqual(self.mapped.num_rows, 64)

    def test_num_columns_delegates(self):
        """num_columns property delegates to source."""
        self.assertEqual(self.mapped.num_columns, 128)

    def test_num_pixels_per_block_horizontal_delegates(self):
        """num_pixels_per_block_horizontal property delegates to source."""
        self.assertEqual(self.mapped.num_pixels_per_block_horizontal, 128)

    def test_num_pixels_per_block_vertical_delegates(self):
        """num_pixels_per_block_vertical property delegates to source."""
        self.assertEqual(self.mapped.num_pixels_per_block_vertical, 64)

    def test_num_resolution_levels_delegates(self):
        """num_resolution_levels property delegates to source."""
        self.assertEqual(self.mapped.num_resolution_levels, 1)

    def test_block_grid_size_delegates(self):
        """block_grid_size property delegates to source."""
        self.assertEqual(self.mapped.block_grid_size, (1, 1))

    def test_has_block_delegates(self):
        """has_block method delegates to source."""
        self.assertTrue(self.mapped.has_block(0, 0))
        self.assertFalse(self.mapped.has_block(1, 1))

    def test_metadata_delegates(self):
        """metadata property delegates to source."""
        metadata = self.mapped.metadata
        self.assertEqual(dict(metadata), {"some_key": "some_value"})


class TestMappedImageProviderOverrides(TestCase):
    """Tests for num_bands and pixel_value_type overrides."""

    def test_num_bands_override(self):
        """When num_bands override is provided, it is returned instead of source value."""
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)

        mapped = MappedImageProvider(source, lambda b: b, num_bands=1)
        self.assertEqual(mapped.num_bands, 1)

    def test_num_bands_delegates_when_no_override(self):
        """When no num_bands override, delegates to source."""
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)

        mapped = MappedImageProvider(source, lambda b: b)
        self.assertEqual(mapped.num_bands, 3)

    def test_pixel_value_type_override(self):
        """When pixel_value_type override is provided, it is returned instead of source value."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)

        mapped = MappedImageProvider(source, lambda b: b, pixel_value_type="float32")
        self.assertEqual(mapped.pixel_value_type, "float32")

    def test_pixel_value_type_delegates_when_no_override(self):
        """When no pixel_value_type override, delegates to source."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)

        mapped = MappedImageProvider(source, lambda b: b)
        self.assertEqual(mapped.pixel_value_type, "uint8")


class TestMappedImageProviderKey(TestCase):
    """Tests for the .key property with name parameter."""

    def test_key_with_name(self):
        """When name is provided, key uses it."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)
        mapped = MappedImageProvider(source, lambda b: b, name="my_op")
        self.assertEqual(mapped.key, "mock-key:mapped:my_op")

    def test_key_without_name_uses_id(self):
        """When name is None, key uses id(self) for uniqueness."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)
        mapped = MappedImageProvider(source, lambda b: b)
        expected = f"mock-key:mapped:{id(mapped)}"
        self.assertEqual(mapped.key, expected)

    def test_two_unnamed_providers_have_different_keys(self):
        """Two unnamed providers have distinct keys."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        source = _MockProvider(image)
        mapped1 = MappedImageProvider(source, lambda b: b)
        mapped2 = MappedImageProvider(source, lambda b: b)
        self.assertNotEqual(mapped1.key, mapped2.key)

    def test_shared_cache_isolation_with_name(self):
        """Two named providers sharing a cache don't collide."""
        image = np.ones((1, 2, 2), dtype=np.uint8)
        source = _MockProvider(image)
        cache = TileCache(max_bytes=1024 * 1024)

        mapped_a = MappedImageProvider(source, lambda b: b * 2, cache=cache, name="double")
        mapped_b = MappedImageProvider(source, lambda b: b * 3, cache=cache, name="triple")

        result_a = mapped_a.get_block(0, 0)
        result_b = mapped_b.get_block(0, 0)

        np.testing.assert_array_equal(result_a, image * 2)
        np.testing.assert_array_equal(result_b, image * 3)
