#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest
from math import degrees, radians
from unittest.mock import MagicMock, patch

import numpy as np

from aws.osml.elevation import StoredDEMTileFactory
from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate


class TestStoredDEMTileFactoryWithChips(unittest.TestCase):
    """Tests using small GeoTIFF chips stored in data/unit/."""

    def setUp(self):
        self.factory = StoredDEMTileFactory("data/unit")

    def test_geotiff_returns_valid_tuple(self):
        arr, sm, summary = self.factory.get_tile("dem_chip_10x10.tif")
        self.assertIsNotNone(arr)
        self.assertIsNotNone(sm)
        self.assertIsNotNone(summary)

    def test_array_shape(self):
        arr, _, _ = self.factory.get_tile("dem_chip_10x10.tif")
        self.assertEqual(arr.shape, (10, 10))
        self.assertEqual(arr.dtype, np.float64)

    def test_elevation_values_preserved(self):
        arr, _, _ = self.factory.get_tile("dem_chip_10x10.tif")
        self.assertAlmostEqual(arr[0, 0], 100.0)
        self.assertAlmostEqual(arr[0, 9], 190.0)
        self.assertAlmostEqual(arr[9, 9], 235.0)

    def test_sensor_model_round_trip(self):
        _, sm, _ = self.factory.get_tile("dem_chip_10x10.tif")
        center = ImageCoordinate([5, 5])
        world = sm.image_to_world(center)
        back = sm.world_to_image(world)
        self.assertAlmostEqual(back.coordinate[0], 5.0, places=6)
        self.assertAlmostEqual(back.coordinate[1], 5.0, places=6)

    def test_sensor_model_known_coordinates(self):
        """UL pixel maps to NW corner of chip at (-78, 39)."""
        _, sm, _ = self.factory.get_tile("dem_chip_10x10.tif")
        ul_world = sm.image_to_world(ImageCoordinate([0, 0]))
        self.assertAlmostEqual(degrees(ul_world.longitude), -78.0, places=4)
        self.assertAlmostEqual(degrees(ul_world.latitude), 39.0, places=4)

    def test_summary_min_max(self):
        _, _, summary = self.factory.get_tile("dem_chip_10x10.tif")
        self.assertAlmostEqual(summary.min_elevation, 100.0)
        self.assertAlmostEqual(summary.max_elevation, 235.0)

    def test_summary_post_spacing_positive(self):
        _, _, summary = self.factory.get_tile("dem_chip_10x10.tif")
        self.assertGreater(summary.post_spacing, 0)

    def test_no_nodata_tag_uses_all_values(self):
        """Without a no-data indicator, all pixel values contribute to min/max."""
        _, _, summary = self.factory.get_tile("dem_chip_nodata.tif")
        self.assertAlmostEqual(summary.min_elevation, -32767.0)
        self.assertAlmostEqual(summary.max_elevation, 260.0)

    def test_world_to_image_lookup_elevation(self):
        arr, sm, _ = self.factory.get_tile("dem_chip_10x10.tif")
        coord = GeodeticWorldCoordinate([radians(-78.0), radians(39.0), 0.0])
        pixel = sm.world_to_image(coord)
        row, col = int(round(pixel.coordinate[1])), int(round(pixel.coordinate[0]))
        self.assertAlmostEqual(arr[row, col], 100.0)

    def test_missing_tile_returns_none_tuple(self):
        result = self.factory.get_tile("nonexistent_tile.tif")
        self.assertEqual(result, (None, None, None))


class TestStoredDEMTileFactoryDTED(unittest.TestCase):
    """Tests for DTED path using mocks (DTED uses 'elevation' asset key)."""

    def _make_mock_reader(self, metadata, data, pad_pixel_value=0.0):
        mock_asset = MagicMock()
        mock_asset.metadata = metadata
        mock_asset.block_grid_size = (1, 1)
        mock_asset.num_pixels_per_block_vertical = data.shape[0]
        mock_asset.num_pixels_per_block_horizontal = data.shape[1]
        mock_asset.num_rows = data.shape[0]
        mock_asset.num_columns = data.shape[1]
        mock_asset.get_block.return_value = data.reshape(1, *data.shape)
        mock_asset.pad_pixel_value = pad_pixel_value

        mock_reader = MagicMock()
        mock_reader.get_asset_keys.return_value = ["elevation"]
        mock_reader.get_asset.return_value = mock_asset
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        return mock_reader

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_first_image_asset_used(self, mock_io):
        metadata = {
            "dted:origin_longitude": -78.0,
            "dted:origin_latitude": 38.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 5,
        }
        data = np.array([[100, 200, 300, 400, 500]] * 5, dtype=np.int16)
        mock_reader = self._make_mock_reader(metadata, data)
        mock_io.open.return_value = mock_reader

        factory = StoredDEMTileFactory("/tiles")
        arr, sm, summary = factory.get_tile("test.dt2")

        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (5, 5))
        mock_reader.get_asset.assert_called_with("elevation")

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_dted_sensor_model_coordinates(self, mock_io):
        metadata = {
            "dted:origin_longitude": 120.0,
            "dted:origin_latitude": 22.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 5,
        }
        data = np.ones((5, 5), dtype=np.int16) * 50
        mock_reader = self._make_mock_reader(metadata, data)
        mock_io.open.return_value = mock_reader

        factory = StoredDEMTileFactory("/tiles")
        _, sm, _ = factory.get_tile("test.dt2")

        # NW corner lat = origin_lat + (num_lat_points - 1) * interval_deg
        # = 22.0 + 4 * (10/10/3600) = 22.0 + 4/3600
        expected_lat = 22.0 + 4.0 / 3600.0
        ul_world = sm.image_to_world(ImageCoordinate([0, 0]))
        self.assertAlmostEqual(degrees(ul_world.longitude), 120.0, places=4)
        self.assertAlmostEqual(degrees(ul_world.latitude), expected_lat, places=6)

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_no_geo_transform_returns_none(self, mock_io):
        mock_asset = MagicMock()
        mock_asset.metadata = {}
        mock_reader = MagicMock()
        mock_reader.get_asset_keys.return_value = ["image:0"]
        mock_reader.get_asset.return_value = mock_asset
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_io.open.return_value = mock_reader

        factory = StoredDEMTileFactory("/tiles")
        result = factory.get_tile("bad_tile.tif")
        self.assertEqual(result, (None, None, None))

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_os_error_returns_none(self, mock_io):
        mock_io.open.side_effect = OSError("IO error: No such file or directory (os error 2)")
        factory = StoredDEMTileFactory("/tiles")
        result = factory.get_tile("missing.tif")
        self.assertEqual(result, (None, None, None))

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_unexpected_error_returns_none(self, mock_io):
        mock_io.open.side_effect = RuntimeError("unexpected")
        factory = StoredDEMTileFactory("/tiles")
        result = factory.get_tile("broken.tif")
        self.assertEqual(result, (None, None, None))

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_nodata_from_geotiff_tag(self, mock_io):
        """GeoTIFF tag 42113 (GDAL_NODATA) is used to filter void pixels."""
        metadata = {
            "33550": [1.0 / 3600.0, 1.0 / 3600.0, 0],
            "33922": [0, 0, 0, 10.0, 20.0, 0],
            "42113": "-32767",
        }
        data = np.array([[100, 200, -32767], [300, -32767, 400], [500, 600, 700]], dtype=np.int16)
        mock_reader = self._make_mock_reader(metadata, data, pad_pixel_value=0.0)
        mock_io.open.return_value = mock_reader

        factory = StoredDEMTileFactory("/tiles")
        _, _, summary = factory.get_tile("tile.tif")

        self.assertAlmostEqual(summary.min_elevation, 100.0)
        self.assertAlmostEqual(summary.max_elevation, 700.0)
        self.assertEqual(summary.no_data_value, -32767)

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_nodata_from_pad_pixel_value(self, mock_io):
        """DTED pad_pixel_value (-32767) is used when tag 42113 is absent."""
        metadata = {
            "dted:origin_longitude": -78.0,
            "dted:origin_latitude": 38.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3,
        }
        data = np.array([[100, 200, -32767], [300, -32767, 400], [500, 600, 700]], dtype=np.int16)
        mock_reader = self._make_mock_reader(metadata, data, pad_pixel_value=-32767.0)
        mock_io.open.return_value = mock_reader

        factory = StoredDEMTileFactory("/tiles")
        _, _, summary = factory.get_tile("tile.dt2")

        self.assertAlmostEqual(summary.min_elevation, 100.0)
        self.assertAlmostEqual(summary.max_elevation, 700.0)
        self.assertEqual(summary.no_data_value, -32767)

    @patch("aws.osml.elevation.dem_tile_factory.IO")
    def test_no_nodata_when_pad_is_zero_and_no_tag(self, mock_io):
        """When pad_pixel_value is 0 and no tag 42113, no filtering occurs."""
        metadata = {
            "33550": [1.0 / 3600.0, 1.0 / 3600.0, 0],
            "33922": [0, 0, 0, 10.0, 20.0, 0],
        }
        data = np.array([[0, 100, 200], [300, 400, 500], [0, 0, 0]], dtype=np.int16)
        mock_reader = self._make_mock_reader(metadata, data, pad_pixel_value=0.0)
        mock_io.open.return_value = mock_reader

        factory = StoredDEMTileFactory("/tiles")
        _, _, summary = factory.get_tile("tile.tif")

        self.assertAlmostEqual(summary.min_elevation, 0.0)
        self.assertAlmostEqual(summary.max_elevation, 500.0)
        self.assertEqual(summary.no_data_value, 0)


if __name__ == "__main__":
    unittest.main()
