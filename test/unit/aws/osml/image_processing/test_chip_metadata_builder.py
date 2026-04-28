#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.chip_metadata_builder`.

Tests cover the :class:`ChipMetadataBuilder` protocol,
:class:`GeoTiffChipMetadataBuilder` (GeoTransform derivation from source
metadata and from sensor model), and integration with
:class:`ChipFactory` (auto-selection, metadata_overrides merging).
"""

import collections.abc
from math import radians
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.chip_factory import ChipFactory, ImageSize, PixelWindow
from aws.osml.image_processing.chip_metadata_builder import (
    GeoTiffChipMetadataBuilder,
    NitfChipMetadataBuilder,
    _build_ichipb_fields,
    _compute_corner_coords,
    _geotransform_for_chip,
    _geotransform_from_sensor_model,
    _parse_geotransform,
)
from aws.osml.io import BufferedMetadataProvider, PixelType

# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------


class _MockProvider:
    """Minimal ImageAssetProvider duck-type backed by a CHW array."""

    def __init__(self, image, tile_height=256, tile_width=256):
        self._base_image = image
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)

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
    def pixel_value_type(self):
        return PixelType.UInt8

    @property
    def pad_pixel_value(self):
        return 0.0

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def block_grid_size(self):
        rows = (self.num_rows + self._tile_height - 1) // self._tile_height
        cols = (self.num_columns + self._tile_width - 1) // self._tile_width
        return (rows, cols)

    def has_block(self, row, col, resolution_level=0):
        grid_rows, grid_cols = self.block_grid_size
        return 0 <= row < grid_rows and 0 <= col < grid_cols

    def get_block(self, row, col, resolution_level=0, bands=None):
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self._base_image.shape[-2])
        x1 = min(x0 + self._tile_width, self._base_image.shape[-1])
        tile = self._base_image[:, y0:y1, x0:x1].copy()
        if tile.shape[-2] != self._tile_height or tile.shape[-1] != self._tile_width:
            padded = np.full(
                (tile.shape[0], self._tile_height, self._tile_width),
                0,
                dtype=tile.dtype,
            )
            padded[:, : tile.shape[-2], : tile.shape[-1]] = tile
            tile = padded
        return tile


class _MockMetadata(collections.abc.Mapping):
    """Minimal metadata accessor supporting dict() conversion."""

    def __init__(self, meta_dict):
        self._dict = meta_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def entries(self, prefix=None):
        if prefix:
            return {k: v for k, v in self._dict.items() if k.startswith(prefix)}
        return dict(self._dict)


class _MockReader:
    """Minimal DatasetReader duck-type with metadata."""

    def __init__(self, assets, metadata_dict=None):
        self._assets = dict(assets)
        self._metadata = _MockMetadata(metadata_dict or {})

    def get_asset(self, key):
        if key not in self._assets:
            raise KeyError(f"Asset not found: {key}")
        return self._assets[key]

    @property
    def metadata(self):
        return self._metadata

    def get_data_assets(self):
        return []


class _MockSensorModel:
    """Sensor model using a simple affine transform for testing.

    Maps pixel (x, y) to geographic (lon_deg, lat_deg) using:
        lon = x_origin + x * pixel_width
        lat = y_origin + y * pixel_height (negative for north-up)
    """

    def __init__(self, x_origin, y_origin, pixel_width, pixel_height):
        self._x_origin = x_origin
        self._y_origin = y_origin
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height

    def image_to_world(self, image_coordinate, elevation_model=None, options=None):
        from aws.osml.photogrammetry import GeodeticWorldCoordinate

        x = image_coordinate.x
        y = image_coordinate.y
        lon_deg = self._x_origin + x * self._pixel_width
        lat_deg = self._y_origin + y * self._pixel_height
        return GeodeticWorldCoordinate([radians(lon_deg), radians(lat_deg), 0.0])

    def world_to_image(self, world_coordinate):
        from math import degrees

        from aws.osml.photogrammetry import ImageCoordinate

        lon_deg = degrees(world_coordinate.longitude)
        lat_deg = degrees(world_coordinate.latitude)
        x = (lon_deg - self._x_origin) / self._pixel_width
        y = (lat_deg - self._y_origin) / self._pixel_height
        return ImageCoordinate([x, y])


# ----------------------------------------------------------------------
# _parse_geotransform
# ----------------------------------------------------------------------


class TestParseGeoTransform(TestCase):
    """Tests for _parse_geotransform helper."""

    def test_string_values(self):
        metadata = {
            "33550": "10.0,10.0,0.0",
            "33922": "0.0,0.0,0.0,100.0,200.0,0.0",
        }
        result = _parse_geotransform(metadata)
        self.assertIsNotNone(result)
        x_origin, pixel_width, x_rot, y_origin, y_rot, pixel_height = result
        self.assertAlmostEqual(x_origin, 100.0)
        self.assertAlmostEqual(pixel_width, 10.0)
        self.assertAlmostEqual(x_rot, 0.0)
        self.assertAlmostEqual(y_origin, 200.0)
        self.assertAlmostEqual(y_rot, 0.0)
        self.assertAlmostEqual(pixel_height, -10.0)

    def test_list_values(self):
        metadata = {
            "33550": [5.0, 5.0, 0.0],
            "33922": [0.0, 0.0, 0.0, -77.0, 39.0, 0.0],
        }
        result = _parse_geotransform(metadata)
        self.assertIsNotNone(result)
        x_origin, pixel_width, x_rot, y_origin, y_rot, pixel_height = result
        self.assertAlmostEqual(x_origin, -77.0)
        self.assertAlmostEqual(pixel_width, 5.0)
        self.assertAlmostEqual(y_origin, 39.0)
        self.assertAlmostEqual(pixel_height, -5.0)

    def test_with_tiepoint_offset(self):
        # Tiepoint at pixel (10, 20) maps to geo (500.0, 1000.0)
        # with pixel scale (2.0, 3.0)
        metadata = {
            "33550": "2.0,3.0,0.0",
            "33922": "10.0,20.0,0.0,500.0,1000.0,0.0",
        }
        result = _parse_geotransform(metadata)
        self.assertIsNotNone(result)
        x_origin, pixel_width, _, y_origin, _, pixel_height = result
        # x_origin = 500.0 - 10*2.0 = 480.0
        self.assertAlmostEqual(x_origin, 480.0)
        # y_origin = 1000.0 - 20*(-3.0) = 1060.0
        self.assertAlmostEqual(y_origin, 1060.0)

    def test_missing_scale_returns_none(self):
        metadata = {"33922": "0.0,0.0,0.0,100.0,200.0,0.0"}
        self.assertIsNone(_parse_geotransform(metadata))

    def test_missing_tiepoint_returns_none(self):
        metadata = {"33550": "10.0,10.0,0.0"}
        self.assertIsNone(_parse_geotransform(metadata))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(_parse_geotransform({}))

    def test_insufficient_values_returns_none(self):
        metadata = {
            "33550": "10.0",
            "33922": "0.0,0.0",
        }
        self.assertIsNone(_parse_geotransform(metadata))


# ----------------------------------------------------------------------
# _geotransform_for_chip
# ----------------------------------------------------------------------


class TestGeoTransformForChip(TestCase):
    """Tests for _geotransform_for_chip helper."""

    def test_identity_chip(self):
        # Full image, no scaling
        gt = (100.0, 0.001, 0.0, 50.0, 0.0, -0.001)
        window = PixelWindow(0, 0, 1000, 1000)
        output_size = ImageSize(1000, 1000)
        result = _geotransform_for_chip(gt, window, output_size)
        self.assertAlmostEqual(result[0], 100.0)
        self.assertAlmostEqual(result[1], 0.001)
        self.assertAlmostEqual(result[3], 50.0)
        self.assertAlmostEqual(result[5], -0.001)

    def test_chip_offset(self):
        # Chip at pixel (100, 200) from a north-up image
        gt = (10.0, 0.01, 0.0, 50.0, 0.0, -0.01)
        window = PixelWindow(100, 200, 256, 256)
        output_size = ImageSize(256, 256)
        result = _geotransform_for_chip(gt, window, output_size)
        # new_x_origin = 10.0 + 100*0.01 = 11.0
        self.assertAlmostEqual(result[0], 11.0)
        # new_y_origin = 50.0 + 200*(-0.01) = 48.0
        self.assertAlmostEqual(result[3], 48.0)
        # pixel size unchanged
        self.assertAlmostEqual(result[1], 0.01)
        self.assertAlmostEqual(result[5], -0.01)

    def test_chip_with_scaling(self):
        # 512x512 src window rendered at 256x256 output (2x downsample)
        gt = (0.0, 0.001, 0.0, 0.0, 0.0, -0.001)
        window = PixelWindow(0, 0, 512, 512)
        output_size = ImageSize(256, 256)
        result = _geotransform_for_chip(gt, window, output_size)
        # pixel size doubles
        self.assertAlmostEqual(result[1], 0.002)
        self.assertAlmostEqual(result[5], -0.002)

    def test_chip_offset_plus_scaling(self):
        gt = (100.0, 0.0001, 0.0, 40.0, 0.0, -0.0001)
        window = PixelWindow(500, 300, 1000, 1000)
        output_size = ImageSize(500, 500)
        result = _geotransform_for_chip(gt, window, output_size)
        # new_x_origin = 100.0 + 500*0.0001 = 100.05
        self.assertAlmostEqual(result[0], 100.05)
        # new_y_origin = 40.0 + 300*(-0.0001) = 39.97
        self.assertAlmostEqual(result[3], 39.97)
        # pixel_width = 0.0001 * (1000/500) = 0.0002
        self.assertAlmostEqual(result[1], 0.0002)
        self.assertAlmostEqual(result[5], -0.0002)


# ----------------------------------------------------------------------
# _geotransform_from_sensor_model
# ----------------------------------------------------------------------


class TestGeoTransformFromSensorModel(TestCase):
    """Tests for sensor-model-derived GeoTransform."""

    def test_affine_sensor_model(self):
        # Simple affine: pixel (x,y) -> lon=x*0.01-77.0, lat=39.0-y*0.01
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        window = PixelWindow(0, 0, 100, 100)
        output_size = ImageSize(100, 100)
        result = _geotransform_from_sensor_model(model, window, output_size)
        x_origin, pixel_width, x_rot, y_origin, y_rot, pixel_height = result
        self.assertAlmostEqual(x_origin, -77.0, places=5)
        self.assertAlmostEqual(y_origin, 39.0, places=5)
        self.assertAlmostEqual(pixel_width, 0.01, places=5)
        self.assertAlmostEqual(pixel_height, -0.01, places=5)
        self.assertAlmostEqual(x_rot, 0.0, places=5)
        self.assertAlmostEqual(y_rot, 0.0, places=5)

    def test_with_chip_offset(self):
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        window = PixelWindow(50, 25, 100, 100)
        output_size = ImageSize(100, 100)
        result = _geotransform_from_sensor_model(model, window, output_size)
        x_origin, pixel_width, x_rot, y_origin, y_rot, pixel_height = result
        # UL at pixel (50, 25): lon=-77+50*0.01=-76.5, lat=39+25*(-0.01)=38.75
        self.assertAlmostEqual(x_origin, -76.5, places=5)
        self.assertAlmostEqual(y_origin, 38.75, places=5)

    def test_with_scaling(self):
        model = _MockSensorModel(x_origin=0.0, y_origin=0.0, pixel_width=0.001, pixel_height=-0.001)
        window = PixelWindow(0, 0, 1000, 1000)
        output_size = ImageSize(500, 500)
        result = _geotransform_from_sensor_model(model, window, output_size)
        _, pixel_width, _, _, _, pixel_height = result
        # 1000 pixels source / 500 output = 2x scale
        self.assertAlmostEqual(pixel_width, 0.002, places=5)
        self.assertAlmostEqual(pixel_height, -0.002, places=5)


# ----------------------------------------------------------------------
# GeoTiffChipMetadataBuilder
# ----------------------------------------------------------------------


class TestGeoTiffChipMetadataBuilder(TestCase):
    """Tests for GeoTiffChipMetadataBuilder."""

    def _make_reader_with_geotransform(self, x_origin, y_origin, pixel_width, pixel_height):
        """Create a mock reader with GeoTIFF metadata."""
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        metadata_dict = {
            "33550": f"{pixel_width},{abs(pixel_height)},0.0",  # noqa: E231
            "33922": f"0.0,0.0,0.0,{x_origin},{y_origin},0.0",  # noqa: E231
            "34735": "1,1,0,7,1024,0,1,1,1025,0,1,1,2048,0,1,4326",  # noqa: E231
        }
        return _MockReader({"image:0": provider}, metadata_dict=metadata_dict)

    def test_build_from_source_metadata(self):
        reader = self._make_reader_with_geotransform(-77.0, 39.0, 0.001, 0.001)
        builder = GeoTiffChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(0, 0, 256, 256),
            output_size=ImageSize(256, 256),
        )
        self.assertIsNotNone(result)
        meta_dict = result.entries()
        self.assertIn("33550", meta_dict)
        self.assertIn("33922", meta_dict)

    def test_geotransform_correct_for_chip_offset(self):
        reader = self._make_reader_with_geotransform(100.0, 50.0, 0.01, 0.01)
        builder = GeoTiffChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(100, 200, 256, 256),
            output_size=ImageSize(256, 256),
        )
        meta_dict = result.entries()
        tiepoint = meta_dict["33922"]
        new_x_origin = tiepoint[3]
        new_y_origin = tiepoint[4]
        # new_x = 100.0 + 100*0.01 = 101.0
        self.assertAlmostEqual(new_x_origin, 101.0, places=5)
        # new_y = 50.0 + 200*(-0.01) = 48.0
        self.assertAlmostEqual(new_y_origin, 48.0, places=5)

    def test_geotransform_with_scaling(self):
        reader = self._make_reader_with_geotransform(0.0, 0.0, 0.001, 0.001)
        builder = GeoTiffChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(0, 0, 512, 512),
            output_size=ImageSize(256, 256),
        )
        meta_dict = result.entries()
        scale = meta_dict["33550"]
        # Pixel size doubles (512 / 256 = 2x)
        self.assertAlmostEqual(scale[0], 0.002, places=6)
        self.assertAlmostEqual(scale[1], 0.002, places=6)

    def test_crs_propagated(self):
        reader = self._make_reader_with_geotransform(-77.0, 39.0, 0.001, 0.001)
        builder = GeoTiffChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta_dict = result.entries()
        self.assertIn("34735", meta_dict)

    def test_sensor_model_derived_geotransform(self):
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        builder = GeoTiffChipMetadataBuilder(reader=None, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(0, 0, 100, 100),
            output_size=ImageSize(100, 100),
        )
        meta_dict = result.entries()
        self.assertIn("33550", meta_dict)
        self.assertIn("33922", meta_dict)
        tiepoint = meta_dict["33922"]
        self.assertAlmostEqual(tiepoint[3], -77.0, places=4)
        self.assertAlmostEqual(tiepoint[4], 39.0, places=4)

    def test_no_reader_no_sensor_model_returns_empty(self):
        builder = GeoTiffChipMetadataBuilder(reader=None, sensor_model=None)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta_dict = result.entries()
        self.assertNotIn("33550", meta_dict)
        self.assertNotIn("33922", meta_dict)

    def test_build_is_stateless(self):
        reader = self._make_reader_with_geotransform(0.0, 0.0, 0.01, 0.01)
        builder = GeoTiffChipMetadataBuilder(reader=reader)
        result1 = builder.build(
            src_window=PixelWindow(0, 0, 100, 100),
            output_size=ImageSize(100, 100),
        )
        result2 = builder.build(
            src_window=PixelWindow(100, 100, 200, 200),
            output_size=ImageSize(200, 200),
        )
        # Results should differ — proves no shared mutable state
        tp1 = result1.entries()["33922"]
        tp2 = result2.entries()["33922"]
        self.assertNotAlmostEqual(tp1[3], tp2[3])


# ----------------------------------------------------------------------
# ChipFactory integration
# ----------------------------------------------------------------------


class TestChipFactoryMetadataBuilderIntegration(TestCase):
    """Tests for ChipFactory's metadata builder integration."""

    def _make_pyramid(self, image, tile_size=256, metadata_dict=None):
        from aws.osml.image_processing.pyramid import TiledImagePyramid

        provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
        if metadata_dict is not None:
            reader = _MockReader({"image:0": provider}, metadata_dict=metadata_dict)
            return TiledImagePyramid.from_dataset(reader)
        return TiledImagePyramid([provider])

    def test_auto_selects_geotiff_builder(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {
            "33550": "0.001,0.001,0.0",
            "33922": "0.0,0.0,0.0,-77.0,39.0,0.0",
        }
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        factory = ChipFactory(source=pyramid, output_format="geotiff")
        self.assertIsNotNone(factory._metadata_builder)
        self.assertIsInstance(factory._metadata_builder, GeoTiffChipMetadataBuilder)

    def test_no_auto_select_for_png(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=128)
        factory = ChipFactory(source=pyramid, output_format="png")
        self.assertIsNone(factory._metadata_builder)

    def test_no_auto_select_for_nitf(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=128)
        factory = ChipFactory(source=pyramid, output_format="nitf")
        self.assertIsNone(factory._metadata_builder)

    def test_explicit_builder_overrides_auto(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=128)
        custom_builder = GeoTiffChipMetadataBuilder(reader=None, sensor_model=None)
        factory = ChipFactory(source=pyramid, output_format="geotiff", metadata_builder=custom_builder)
        self.assertIs(factory._metadata_builder, custom_builder)

    def test_geotiff_tile_has_metadata(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {
            "33550": "0.001,0.001,0.0",
            "33922": "0.0,0.0,0.0,-77.0,39.0,0.0",
        }
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        factory = ChipFactory(source=pyramid, output_format="geotiff")
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        # Verify TIFF magic
        self.assertTrue(result[:2] in (b"II", b"MM"))
        # Verify the metadata builder produces correct output
        meta = factory._metadata_builder.build(PixelWindow(0, 0, 128, 128), ImageSize(128, 128))
        meta_dict = meta.entries()
        self.assertIn("33550", meta_dict)
        self.assertIn("33922", meta_dict)

    def test_metadata_overrides_merged(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {
            "33550": "0.001,0.001,0.0",
            "33922": "0.0,0.0,0.0,-77.0,39.0,0.0",
        }
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        overrides = BufferedMetadataProvider()
        overrides["259"] = 5  # Compression = LZW
        factory = ChipFactory(source=pyramid, output_format="geotiff", metadata_overrides=overrides)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

    def test_sensor_model_derived_metadata_in_tile(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=128)
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        factory = ChipFactory(source=pyramid, output_format="geotiff", sensor_model=model)
        self.assertIsNotNone(factory._metadata_builder)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        # Verify the metadata builder produces correct geospatial metadata
        meta = factory._metadata_builder.build(PixelWindow(0, 0, 128, 128), ImageSize(128, 128))
        meta_dict = meta.entries()
        self.assertIn("33550", meta_dict)
        self.assertIn("33922", meta_dict)
        tiepoint = meta_dict["33922"]
        self.assertAlmostEqual(tiepoint[3], -77.0, places=4)
        self.assertAlmostEqual(tiepoint[4], 39.0, places=4)

    def test_geotiff_metadata_roundtrip(self):
        """Verify GeoTIFF extension tags survive a full write/read cycle."""
        import io

        from aws.osml.io import IO

        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {
            "33550": "0.001,0.001,0.0",
            "33922": "0.0,0.0,0.0,-77.0,39.0,0.0",
        }
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        factory = ChipFactory(source=pyramid, output_format="geotiff")
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

        with IO.open(io.BytesIO(result), "r", "tiff") as reader:
            asset = reader.get_asset("image:0")
            meta = asset.metadata.entries()
            self.assertIn("33550", meta)
            self.assertIn("33922", meta)
            scale = meta["33550"]
            tiepoint = meta["33922"]
            self.assertAlmostEqual(scale[0], 0.001, places=6)
            self.assertAlmostEqual(scale[1], 0.001, places=6)
            self.assertAlmostEqual(tiepoint[3], -77.0, places=4)
            self.assertAlmostEqual(tiepoint[4], 39.0, places=4)


# ----------------------------------------------------------------------
# _compute_corner_coords
# ----------------------------------------------------------------------


class TestComputeCornerCoords(TestCase):
    """Tests for _compute_corner_coords helper."""

    def test_corner_order(self):
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        corners = _compute_corner_coords(model, PixelWindow(0, 0, 100, 100))
        # UL, UR, LR, LL
        self.assertEqual(len(corners), 4)
        ul_lat, ul_lon = corners[0]
        ur_lat, ur_lon = corners[1]
        lr_lat, lr_lon = corners[2]
        ll_lat, ll_lon = corners[3]
        # UL: (0,0) -> lon=-77, lat=39
        self.assertAlmostEqual(ul_lon, -77.0, places=5)
        self.assertAlmostEqual(ul_lat, 39.0, places=5)
        # UR: (100,0) -> lon=-76, lat=39
        self.assertAlmostEqual(ur_lon, -76.0, places=5)
        self.assertAlmostEqual(ur_lat, 39.0, places=5)
        # LR: (100,100) -> lon=-76, lat=38
        self.assertAlmostEqual(lr_lon, -76.0, places=5)
        self.assertAlmostEqual(lr_lat, 38.0, places=5)
        # LL: (0,100) -> lon=-77, lat=38
        self.assertAlmostEqual(ll_lon, -77.0, places=5)
        self.assertAlmostEqual(ll_lat, 38.0, places=5)

    def test_with_offset(self):
        model = _MockSensorModel(x_origin=0.0, y_origin=0.0, pixel_width=0.001, pixel_height=-0.001)
        corners = _compute_corner_coords(model, PixelWindow(50, 25, 100, 100))
        ul_lat, ul_lon = corners[0]
        self.assertAlmostEqual(ul_lon, 0.05, places=5)
        self.assertAlmostEqual(ul_lat, -0.025, places=5)


# ----------------------------------------------------------------------
# _build_ichipb_fields
# ----------------------------------------------------------------------


class TestBuildIchipbFields(TestCase):
    """Tests for _build_ichipb_fields helper."""

    def test_first_gen_chip(self):
        """First-generation chip (no source ICHIPB)."""
        fields = _build_ichipb_fields(
            src_window=PixelWindow(100, 200, 256, 256),
            output_size=ImageSize(256, 256),
            source_ichipb=None,
        )
        # OP corners: UL=(0,0), UR=(255,0), LL=(0,255), LR=(255,255)
        self.assertAlmostEqual(float(fields["OP_COL_11"]), 0.0)
        self.assertAlmostEqual(float(fields["OP_ROW_11"]), 0.0)
        self.assertAlmostEqual(float(fields["OP_COL_12"]), 255.0)
        self.assertAlmostEqual(float(fields["OP_ROW_12"]), 0.0)
        self.assertAlmostEqual(float(fields["OP_COL_21"]), 0.0)
        self.assertAlmostEqual(float(fields["OP_ROW_21"]), 255.0)
        self.assertAlmostEqual(float(fields["OP_COL_22"]), 255.0)
        self.assertAlmostEqual(float(fields["OP_ROW_22"]), 255.0)
        # FI corners: offset by src_window origin, 1:1 scale
        self.assertAlmostEqual(float(fields["FI_COL_11"]), 100.0)
        self.assertAlmostEqual(float(fields["FI_ROW_11"]), 200.0)
        self.assertAlmostEqual(float(fields["FI_COL_12"]), 355.0)
        self.assertAlmostEqual(float(fields["FI_ROW_12"]), 200.0)
        self.assertAlmostEqual(float(fields["FI_COL_21"]), 100.0)
        self.assertAlmostEqual(float(fields["FI_ROW_21"]), 455.0)
        self.assertAlmostEqual(float(fields["FI_COL_22"]), 355.0)
        self.assertAlmostEqual(float(fields["FI_ROW_22"]), 455.0)

    def test_first_gen_chip_with_scaling(self):
        """First-gen chip with output smaller than source (downsampled)."""
        fields = _build_ichipb_fields(
            src_window=PixelWindow(0, 0, 512, 512),
            output_size=ImageSize(256, 256),
            source_ichipb=None,
        )
        # OP corners: 256x256 output
        self.assertAlmostEqual(float(fields["OP_COL_22"]), 255.0)
        self.assertAlmostEqual(float(fields["OP_ROW_22"]), 255.0)
        # FI corners: scale_x = 512/256 = 2.0
        # FI_COL_12 = 0 + 255 * 2.0 = 510.0
        self.assertAlmostEqual(float(fields["FI_COL_12"]), 510.0)
        self.assertAlmostEqual(float(fields["FI_ROW_21"]), 510.0)

    def test_chained_ichipb(self):
        """Source is itself a chip — coordinates chain to original full image."""
        source_ichipb = {
            "OP_COL_11": "0.0",
            "OP_ROW_11": "0.0",
            "OP_COL_12": "511.0",
            "OP_ROW_12": "0.0",
            "OP_COL_21": "0.0",
            "OP_ROW_21": "511.0",
            "OP_COL_22": "511.0",
            "OP_ROW_22": "511.0",
            "FI_COL_11": "1000.0",
            "FI_ROW_11": "2000.0",
            "FI_COL_12": "1511.0",
            "FI_ROW_12": "2000.0",
            "FI_COL_21": "1000.0",
            "FI_ROW_21": "2511.0",
            "FI_COL_22": "1511.0",
            "FI_ROW_22": "2511.0",
            "FI_COL": "4096",
            "FI_ROW": "4096",
        }
        # Take a 128x128 chip from pixel (100, 50) of the source
        fields = _build_ichipb_fields(
            src_window=PixelWindow(100, 50, 128, 128),
            output_size=ImageSize(128, 128),
            source_ichipb=source_ichipb,
        )
        # Source ICHIPB maps [0..511] → [1000..1511] (1:1 scale)
        # So our chip at (100, 50) maps to FI (1100, 2050)
        self.assertAlmostEqual(float(fields["FI_COL_11"]), 1100.0, places=3)
        self.assertAlmostEqual(float(fields["FI_ROW_11"]), 2050.0, places=3)
        # FI_COL at (100+127, 50+127) = FI (1227, 2177)
        self.assertAlmostEqual(float(fields["FI_COL_22"]), 1227.0, places=3)
        self.assertAlmostEqual(float(fields["FI_ROW_22"]), 2177.0, places=3)
        # Full image dimensions chained from source
        self.assertEqual(int(fields["FI_COL"]), 4096)
        self.assertEqual(int(fields["FI_ROW"]), 4096)

    def test_fi_dimensions_first_gen(self):
        """FI_COL/FI_ROW defaults to src extent when no source ICHIPB."""
        fields = _build_ichipb_fields(
            src_window=PixelWindow(0, 0, 1024, 768),
            output_size=ImageSize(1024, 768),
            source_ichipb=None,
        )
        self.assertEqual(int(fields["FI_COL"]), 1024)
        self.assertEqual(int(fields["FI_ROW"]), 768)


# ----------------------------------------------------------------------
# NitfChipMetadataBuilder
# ----------------------------------------------------------------------


class TestNitfChipMetadataBuilder(TestCase):
    """Tests for NitfChipMetadataBuilder."""

    def _make_nitf_reader(self, metadata_dict=None):
        """Create a mock reader with NITF-style metadata."""
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        return _MockReader({"image:0": provider}, metadata_dict=metadata_dict or {})

    def test_igeolo_computed_from_sensor_model(self):
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(0, 0, 100, 100),
            output_size=ImageSize(100, 100),
        )
        meta_dict = result.entries()
        self.assertIn("IGEOLO", meta_dict)
        self.assertIn("ICORDS", meta_dict)
        self.assertEqual(meta_dict["ICORDS"], "G")
        igeolo = meta_dict["IGEOLO"]
        self.assertEqual(len(igeolo), 60)

    def test_igeolo_dms_format(self):
        # Known point: lat=39, lon=-77 → "390000N0770000W"
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(0, 0, 100, 100),
            output_size=ImageSize(100, 100),
        )
        igeolo = result.entries()["IGEOLO"]
        # UL corner: lat=39N, lon=77W → "390000N0770000W"
        self.assertEqual(igeolo[:15], "390000N0770000W")

    def test_ichipb_present(self):
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(50, 25, 100, 100),
            output_size=ImageSize(100, 100),
        )
        meta_dict = result.entries()
        self.assertIn("ICHIPB", meta_dict)
        ichipb = meta_dict["ICHIPB"]
        self.assertIn("OP_COL_11", ichipb)
        self.assertIn("FI_COL_11", ichipb)
        self.assertAlmostEqual(float(ichipb["FI_COL_11"]), 50.0)
        self.assertAlmostEqual(float(ichipb["FI_ROW_11"]), 25.0)

    def test_source_subheader_fields_propagated(self):
        metadata_dict = {
            "IREP": "MULTI",
            "ICAT": "VIS",
            "IC": "C8",
            "IMODE": "B",
        }
        reader = self._make_nitf_reader(metadata_dict=metadata_dict)
        model = _MockSensorModel(x_origin=0.0, y_origin=0.0, pixel_width=0.001, pixel_height=-0.001)
        builder = NitfChipMetadataBuilder(reader=reader, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta_dict = result.entries()
        self.assertEqual(meta_dict["IREP"], "MULTI")
        self.assertEqual(meta_dict["ICAT"], "VIS")
        self.assertEqual(meta_dict["IC"], "C8")
        self.assertEqual(meta_dict["IMODE"], "B")

    def test_chained_ichipb_from_source(self):
        """When source has ICHIPB, chip coordinates chain through it."""
        metadata_dict = {
            "ICHIPB": {
                "OP_COL_11": "0.0",
                "OP_ROW_11": "0.0",
                "OP_COL_12": "255.0",
                "OP_ROW_12": "0.0",
                "OP_COL_21": "0.0",
                "OP_ROW_21": "255.0",
                "OP_COL_22": "255.0",
                "OP_ROW_22": "255.0",
                "FI_COL_11": "500.0",
                "FI_ROW_11": "600.0",
                "FI_COL_12": "755.0",
                "FI_ROW_12": "600.0",
                "FI_COL_21": "500.0",
                "FI_ROW_21": "855.0",
                "FI_COL_22": "755.0",
                "FI_ROW_22": "855.0",
                "FI_COL": "2048",
                "FI_ROW": "2048",
            }
        }
        reader = self._make_nitf_reader(metadata_dict=metadata_dict)
        model = _MockSensorModel(x_origin=0.0, y_origin=0.0, pixel_width=0.001, pixel_height=-0.001)
        builder = NitfChipMetadataBuilder(reader=reader, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(50, 50, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta_dict = result.entries()
        ichipb = meta_dict["ICHIPB"]
        # Source maps [0..255] -> [500..755] (1:1)
        # Chip at (50, 50) → FI (550, 650)
        self.assertAlmostEqual(float(ichipb["FI_COL_11"]), 550.0, places=3)
        self.assertAlmostEqual(float(ichipb["FI_ROW_11"]), 650.0, places=3)
        self.assertEqual(int(ichipb["FI_COL"]), 2048)
        self.assertEqual(int(ichipb["FI_ROW"]), 2048)

    def test_cross_format_derives_from_sensor_model(self):
        """Non-NITF source → NITF output uses sensor model for IGEOLO."""
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=model)
        result = builder.build(
            src_window=PixelWindow(0, 0, 50, 50),
            output_size=ImageSize(50, 50),
        )
        meta_dict = result.entries()
        self.assertIn("IGEOLO", meta_dict)
        self.assertEqual(len(meta_dict["IGEOLO"]), 60)
        self.assertIn("ICHIPB", meta_dict)

    def test_no_sensor_model_no_igeolo(self):
        """Without sensor model, no IGEOLO/ICORDS but ICHIPB still present."""
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=None)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta_dict = result.entries()
        self.assertNotIn("IGEOLO", meta_dict)
        self.assertNotIn("ICORDS", meta_dict)
        self.assertIn("ICHIPB", meta_dict)

    def test_build_is_stateless(self):
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=model)
        result1 = builder.build(
            src_window=PixelWindow(0, 0, 100, 100),
            output_size=ImageSize(100, 100),
        )
        result2 = builder.build(
            src_window=PixelWindow(200, 300, 100, 100),
            output_size=ImageSize(100, 100),
        )
        # Different windows produce different IGEOLO
        self.assertNotEqual(result1.entries()["IGEOLO"], result2.entries()["IGEOLO"])
        # Different ICHIPB FI coordinates
        ichipb1 = result1.entries()["ICHIPB"]
        ichipb2 = result2.entries()["ICHIPB"]
        self.assertNotEqual(ichipb1["FI_COL_11"], ichipb2["FI_COL_11"])


# ----------------------------------------------------------------------
# ChipFactory NITF metadata builder integration
# ----------------------------------------------------------------------


class TestChipFactoryNitfMetadataBuilderIntegration(TestCase):
    """Tests for ChipFactory's NITF metadata builder auto-selection."""

    def _make_pyramid(self, image, tile_size=256, metadata_dict=None):
        from aws.osml.image_processing.pyramid import TiledImagePyramid

        provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
        if metadata_dict is not None:
            reader = _MockReader({"image:0": provider}, metadata_dict=metadata_dict)
            return TiledImagePyramid.from_dataset(reader)
        return TiledImagePyramid([provider])

    def test_auto_selects_nitf_builder_with_sensor_model(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=128)
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        factory = ChipFactory(source=pyramid, output_format="nitf", sensor_model=model)
        self.assertIsNotNone(factory._metadata_builder)
        self.assertIsInstance(factory._metadata_builder, NitfChipMetadataBuilder)

    def test_auto_selects_nitf_builder_with_reader(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {"IREP": "MULTI", "ICAT": "VIS"}
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        factory = ChipFactory(source=pyramid, output_format="nitf")
        self.assertIsNotNone(factory._metadata_builder)
        self.assertIsInstance(factory._metadata_builder, NitfChipMetadataBuilder)

    def test_no_nitf_builder_without_reader_or_model(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=128)
        factory = ChipFactory(source=pyramid, output_format="nitf")
        self.assertIsNone(factory._metadata_builder)

    def test_nitf_tile_has_igeolo(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {"IREP": "MULTI"}
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        factory = ChipFactory(source=pyramid, output_format="nitf", sensor_model=model)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

    def test_nitf_metadata_roundtrip(self):
        """Verify NITF tile can be read back and has expected metadata."""
        import io

        from aws.osml.io import IO

        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        metadata_dict = {"IREP": "MULTI", "ICAT": "VIS"}
        pyramid = self._make_pyramid(image, tile_size=128, metadata_dict=metadata_dict)
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        factory = ChipFactory(source=pyramid, output_format="nitf", sensor_model=model)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

        with IO.open(io.BytesIO(result), "r", "nitf") as reader:
            asset = reader.get_asset("image:0")
            meta = asset.metadata.entries()
            self.assertIn("IGEOLO", meta)
            self.assertIn("ICORDS", meta)
            self.assertEqual(meta["ICORDS"], "G")
            self.assertEqual(len(meta["IGEOLO"]), 60)
            # Verify ICHIPB TRE roundtrips
            self.assertIn("ICHIPB", meta)
            ichipb = meta["ICHIPB"]
            self.assertAlmostEqual(float(ichipb["FI_COL_11"]), 0.0, places=1)
            self.assertAlmostEqual(float(ichipb["FI_ROW_11"]), 0.0, places=1)
            self.assertAlmostEqual(float(ichipb["OP_COL_22"]), 127.0, places=1)
            self.assertAlmostEqual(float(ichipb["OP_ROW_22"]), 127.0, places=1)

    def test_nitf_ichipb_roundtrip_with_offset(self):
        """Verify ICHIPB records correct FI coordinates for offset chip."""
        import io

        from aws.osml.io import IO

        image = np.random.randint(0, 256, (3, 256, 256), dtype=np.uint8)
        pyramid = self._make_pyramid(image, tile_size=256)
        model = _MockSensorModel(x_origin=-77.0, y_origin=39.0, pixel_width=0.01, pixel_height=-0.01)
        factory = ChipFactory(source=pyramid, output_format="nitf", sensor_model=model)
        result = factory.create_chip(PixelWindow(50, 75, 128, 128))
        self.assertIsNotNone(result)

        with IO.open(io.BytesIO(result), "r", "nitf") as reader:
            asset = reader.get_asset("image:0")
            meta = asset.metadata.entries()
            self.assertIn("ICHIPB", meta)
            ichipb = meta["ICHIPB"]
            # FI origin should reflect src_window offset
            self.assertAlmostEqual(float(ichipb["FI_COL_11"]), 50.0, places=1)
            self.assertAlmostEqual(float(ichipb["FI_ROW_11"]), 75.0, places=1)


# ----------------------------------------------------------------------
# SICD/SIDD Metadata Handling (Phase 5)
# ----------------------------------------------------------------------


class _MockDesAsset:
    """Minimal DES asset duck-type for testing."""

    def __init__(self, xml_content, metadata_dict=None):
        self._xml = xml_content
        self._metadata = _MockMetadata(metadata_dict or {"DESID": "XML_DATA_CONTENT"})

    @property
    def raw_asset(self):
        import io

        return io.BytesIO(self._xml.encode("utf-8"))

    @property
    def metadata(self):
        return self._metadata


class _MockReaderWithDes:
    """Mock reader that exposes DES assets alongside image assets."""

    def __init__(self, provider, metadata_dict=None, des_xml=None, des_metadata=None):
        self._provider = provider
        self._metadata = _MockMetadata(metadata_dict or {})
        self._des_xml = des_xml
        self._des_metadata = des_metadata

    def get_asset(self, key):
        if key == "image:0":
            return self._provider
        if key == "des:0" and self._des_xml is not None:
            return _MockDesAsset(self._des_xml, self._des_metadata)
        raise KeyError(f"Asset not found: {key}")

    def get_asset_keys(self):
        keys = ["image:0"]
        if self._des_xml is not None:
            keys.append("des:0")
        return keys

    @property
    def metadata(self):
        return self._metadata


def _load_sicd_xml():
    """Load SICD XML from test data."""
    with open("test/data/sicd/example.sicd121.capella.xml") as f:
        return f.read()


def _load_sidd_xml():
    """Load SIDD XML from test data."""
    with open("test/data/sidd/example.sidd.xml") as f:
        return f.read()


class TestSicdUpdaterStateless(TestCase):
    """Tests for the stateless update_sicd_for_chip function."""

    def test_updates_image_data(self):
        from aws.osml.formats.model_utils import sicd_parser
        from aws.osml.image_processing.sicd_updater import update_sicd_for_chip

        xml = _load_sicd_xml()
        result = update_sicd_for_chip(xml, [100, 200, 256, 256])

        sicd = sicd_parser.from_string(result)
        self.assertEqual(sicd.image_data.first_row, 200)
        self.assertEqual(sicd.image_data.first_col, 100)
        self.assertEqual(sicd.image_data.num_rows, 256)
        self.assertEqual(sicd.image_data.num_cols, 256)

    def test_raises_on_scaling(self):
        from aws.osml.image_processing.sicd_updater import update_sicd_for_chip

        xml = _load_sicd_xml()
        with self.assertRaises(ValueError):
            update_sicd_for_chip(xml, [0, 0, 512, 512], output_size=(256, 256))

    def test_no_error_for_1_to_1(self):
        from aws.osml.image_processing.sicd_updater import update_sicd_for_chip

        xml = _load_sicd_xml()
        result = update_sicd_for_chip(xml, [0, 0, 512, 512], output_size=(512, 512))
        self.assertIn("SICD", result)

    def test_stateless_multiple_calls(self):
        """Multiple calls on same XML produce independent results."""
        from aws.osml.formats.model_utils import sicd_parser
        from aws.osml.image_processing.sicd_updater import update_sicd_for_chip

        xml = _load_sicd_xml()
        result1 = update_sicd_for_chip(xml, [0, 0, 128, 128])
        result2 = update_sicd_for_chip(xml, [200, 300, 64, 64])

        sicd1 = sicd_parser.from_string(result1)
        sicd2 = sicd_parser.from_string(result2)

        self.assertEqual(sicd1.image_data.first_row, 0)
        self.assertEqual(sicd1.image_data.first_col, 0)
        self.assertEqual(sicd2.image_data.first_row, 300)
        self.assertEqual(sicd2.image_data.first_col, 200)

    def test_chained_chip(self):
        """Chipping from an already-chipped SICD accumulates offsets."""
        from aws.osml.formats.model_utils import sicd_parser
        from aws.osml.image_processing.sicd_updater import update_sicd_for_chip

        xml = _load_sicd_xml()
        # First chip at (100, 200)
        first_chip = update_sicd_for_chip(xml, [100, 200, 256, 256])
        # Second chip from first_chip at (50, 50)
        second_chip = update_sicd_for_chip(first_chip, [50, 50, 64, 64])

        sicd = sicd_parser.from_string(second_chip)
        # FirstRow = 200 + 50 = 250, FirstCol = 100 + 50 = 150
        self.assertEqual(sicd.image_data.first_row, 250)
        self.assertEqual(sicd.image_data.first_col, 150)


class TestSiddUpdaterStateless(TestCase):
    """Tests for the stateless update_sidd_for_chip function."""

    def test_adds_geometric_chip(self):
        from aws.osml.image_processing.sidd_updater import update_sidd_for_chip

        xml = _load_sidd_xml()
        result = update_sidd_for_chip(xml, [100, 200, 256, 256])
        self.assertIn("GeometricChip", result)
        self.assertIn("ChipSize", result)

    def test_chip_size_matches_output(self):
        from aws.osml.formats.model_utils import sidd_parser
        from aws.osml.image_processing.sidd_updater import update_sidd_for_chip

        xml = _load_sidd_xml()
        result = update_sidd_for_chip(xml, [0, 0, 512, 512], output_size=(256, 256))
        sidd = sidd_parser.from_string(result)
        chip = sidd.downstream_reprocessing.geometric_chip
        self.assertEqual(chip.chip_size.col, 256)
        self.assertEqual(chip.chip_size.row, 256)

    def test_original_corners_recorded(self):
        from aws.osml.formats.model_utils import sidd_parser
        from aws.osml.image_processing.sidd_updater import update_sidd_for_chip

        xml = _load_sidd_xml()
        result = update_sidd_for_chip(xml, [100, 200, 256, 256])
        sidd = sidd_parser.from_string(result)
        chip = sidd.downstream_reprocessing.geometric_chip
        # UL corner
        self.assertAlmostEqual(chip.original_upper_left_coordinate.col, 100.0)
        self.assertAlmostEqual(chip.original_upper_left_coordinate.row, 200.0)
        # LR corner (col+width-1, row+height-1)
        self.assertAlmostEqual(chip.original_lower_right_coordinate.col, 355.0)
        self.assertAlmostEqual(chip.original_lower_right_coordinate.row, 455.0)

    def test_stateless_multiple_calls(self):
        """Multiple calls produce independent results."""
        from aws.osml.formats.model_utils import sidd_parser
        from aws.osml.image_processing.sidd_updater import update_sidd_for_chip

        xml = _load_sidd_xml()
        result1 = update_sidd_for_chip(xml, [0, 0, 128, 128])
        result2 = update_sidd_for_chip(xml, [500, 600, 64, 64])

        sidd1 = sidd_parser.from_string(result1)
        sidd2 = sidd_parser.from_string(result2)

        chip1 = sidd1.downstream_reprocessing.geometric_chip
        chip2 = sidd2.downstream_reprocessing.geometric_chip

        self.assertAlmostEqual(chip1.original_upper_left_coordinate.col, 0.0)
        self.assertAlmostEqual(chip2.original_upper_left_coordinate.col, 500.0)


class TestNitfChipMetadataBuilderSicdSidd(TestCase):
    """Tests for NitfChipMetadataBuilder SICD/SIDD DES handling."""

    def _make_sicd_reader(self):
        image = np.zeros((2, 512, 512), dtype=np.int16)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        return _MockReaderWithDes(
            provider,
            metadata_dict={"IREP": "NODISPLY", "ICAT": "SAR"},
            des_xml=_load_sicd_xml(),
            des_metadata={"DESID": "XML_DATA_CONTENT", "DESVER": "01", "DESCLAS": "U"},
        )

    def _make_sidd_reader(self):
        image = np.zeros((1, 512, 512), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        return _MockReaderWithDes(
            provider,
            metadata_dict={"IREP": "MONO", "ICAT": "SAR"},
            des_xml=_load_sidd_xml(),
            des_metadata={"DESID": "XML_DATA_CONTENT", "DESVER": "01", "DESCLAS": "U"},
        )

    def test_detects_sicd(self):
        reader = self._make_sicd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        self.assertTrue(builder.has_sicd)
        self.assertFalse(builder.has_sidd)

    def test_detects_sidd(self):
        reader = self._make_sidd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        self.assertFalse(builder.has_sicd)
        self.assertTrue(builder.has_sidd)

    def test_sicd_des_included_in_build(self):
        reader = self._make_sicd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta = result.entries()
        self.assertIn("_DES_XML", meta)
        self.assertIn("SICD", meta["_DES_XML"])

    def test_sicd_des_updated_for_chip_bounds(self):
        from aws.osml.formats.model_utils import sicd_parser

        reader = self._make_sicd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(100, 200, 64, 64),
            output_size=ImageSize(64, 64),
        )
        des_xml = result.entries()["_DES_XML"]
        sicd = sicd_parser.from_string(des_xml)
        self.assertEqual(sicd.image_data.first_col, 100)
        self.assertEqual(sicd.image_data.first_row, 200)
        self.assertEqual(sicd.image_data.num_cols, 64)
        self.assertEqual(sicd.image_data.num_rows, 64)

    def test_sidd_des_included_in_build(self):
        reader = self._make_sidd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(50, 50, 200, 200),
            output_size=ImageSize(200, 200),
        )
        meta = result.entries()
        self.assertIn("_DES_XML", meta)
        self.assertIn("SIDD", meta["_DES_XML"])
        self.assertIn("GeometricChip", meta["_DES_XML"])

    def test_skip_des_omits_xml(self):
        reader = self._make_sicd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
            skip_des=True,
        )
        meta = result.entries()
        self.assertNotIn("_DES_XML", meta)

    def test_des_metadata_included(self):
        reader = self._make_sicd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        result = builder.build(
            src_window=PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        meta = result.entries()
        self.assertIn("_DES_METADATA", meta)
        self.assertEqual(meta["_DES_METADATA"]["DESID"], "XML_DATA_CONTENT")

    def test_build_stateless_sicd(self):
        """Multiple builds from same builder produce independent SICD results."""
        from aws.osml.formats.model_utils import sicd_parser

        reader = self._make_sicd_reader()
        builder = NitfChipMetadataBuilder(reader=reader)
        r1 = builder.build(PixelWindow(0, 0, 64, 64), ImageSize(64, 64))
        r2 = builder.build(PixelWindow(100, 100, 64, 64), ImageSize(64, 64))

        sicd1 = sicd_parser.from_string(r1.entries()["_DES_XML"])
        sicd2 = sicd_parser.from_string(r2.entries()["_DES_XML"])

        self.assertEqual(sicd1.image_data.first_col, 0)
        self.assertEqual(sicd2.image_data.first_col, 100)

    def test_no_des_without_reader(self):
        builder = NitfChipMetadataBuilder(reader=None, sensor_model=None)
        self.assertFalse(builder.has_sicd)
        self.assertFalse(builder.has_sidd)
        result = builder.build(PixelWindow(0, 0, 128, 128), ImageSize(128, 128))
        self.assertNotIn("_DES_XML", result.entries())


class TestChipFactorySicdValidation(TestCase):
    """Tests for ChipFactory SICD scaling validation."""

    def _make_sicd_pyramid(self):
        from aws.osml.image_processing.pyramid import TiledImagePyramid

        image = np.zeros((2, 512, 512), dtype=np.int16)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        reader = _MockReaderWithDes(
            provider,
            metadata_dict={"IREP": "NODISPLY", "ICAT": "SAR"},
            des_xml=_load_sicd_xml(),
        )
        return TiledImagePyramid.from_dataset(reader)

    def test_sicd_1_to_1_nitf_succeeds(self):
        pyramid = self._make_sicd_pyramid()
        factory = ChipFactory(source=pyramid, output_format="nitf")
        result = factory.create_chip(PixelWindow(0, 0, 128, 128), ImageSize(128, 128))
        self.assertIsNotNone(result)

    def test_sicd_scaled_nitf_raises(self):
        pyramid = self._make_sicd_pyramid()
        factory = ChipFactory(source=pyramid, output_format="nitf")
        with self.assertRaises(ValueError) as ctx:
            factory.create_chip(PixelWindow(0, 0, 256, 256), ImageSize(128, 128))
        self.assertIn("SICD", str(ctx.exception))
        self.assertIn("decimation", str(ctx.exception))

    def test_sicd_scaled_non_nitf_allowed(self):
        """Scaling to non-NITF formats is allowed (DES not carried)."""
        from aws.osml.image_processing.processing_chain import ProcessingChain

        pyramid = self._make_sicd_pyramid()

        def to_uint8(pixels):
            return np.clip(np.abs(pixels[:1]).astype(np.float32), 0, 255).astype(np.uint8)

        chain = ProcessingChain(steps=[to_uint8], output_bands=1, output_dtype=np.dtype(np.uint8))
        factory = ChipFactory(source=pyramid, output_format="png", processing_chain=chain)
        result = factory.create_chip(PixelWindow(0, 0, 256, 256), ImageSize(128, 128))
        self.assertIsNotNone(result)

    def test_processing_chain_omits_des(self):
        """When processing chain is active, DES XML is omitted from NITF output."""
        import io as sio

        from aws.osml.image_processing.processing_chain import ProcessingChain
        from aws.osml.io import IO

        pyramid = self._make_sicd_pyramid()

        def to_uint8(pixels):
            return np.clip(np.abs(pixels[:1]).astype(np.float32), 0, 255).astype(np.uint8)

        chain = ProcessingChain(steps=[to_uint8], output_bands=1, output_dtype=np.dtype(np.uint8))
        factory = ChipFactory(source=pyramid, output_format="nitf", processing_chain=chain)
        # 1:1 with chain - should produce output without DES
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

        with IO.open(sio.BytesIO(result), "r", "nitf") as reader:
            keys = reader.get_asset_keys()
            has_des = any(k.startswith("des:") or k.startswith("text:") for k in keys)
            self.assertFalse(has_des)

    def test_no_chain_includes_des(self):
        """Without processing chain, DES XML is included in NITF output."""
        import io as sio

        from aws.osml.io import IO

        pyramid = self._make_sicd_pyramid()
        factory = ChipFactory(source=pyramid, output_format="nitf")
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

        with IO.open(sio.BytesIO(result), "r", "nitf") as reader:
            keys = reader.get_asset_keys()
            has_des = any(k.startswith("des:") for k in keys)
            self.assertTrue(has_des)

    def test_concurrent_create_chip(self):
        """Thread-safe: concurrent create_chip calls produce correct independent results."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pyramid = self._make_sicd_pyramid()
        factory = ChipFactory(source=pyramid, output_format="nitf")

        windows = [
            PixelWindow(0, 0, 64, 64),
            PixelWindow(64, 0, 64, 64),
            PixelWindow(0, 64, 64, 64),
            PixelWindow(64, 64, 64, 64),
        ]

        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(factory.create_chip, w): w for w in windows}
            for future in as_completed(futures):
                w = futures[future]
                results[w] = future.result()

        # All tiles should be produced and non-None
        for w, result in results.items():
            self.assertIsNotNone(result, f"Tile for {w} was None")
            self.assertGreater(len(result), 0)
