#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest
from math import degrees, radians
from unittest.mock import MagicMock, patch

import numpy as np

from aws.osml.elevation import StoredDEMTileFactory
from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, geodetic_to_geocentric


def _ecf_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Straight-line ECF distance in meters between two points given in decimal degrees."""
    a = geodetic_to_geocentric(GeodeticWorldCoordinate([radians(lon1), radians(lat1), 0.0])).coordinate
    b = geodetic_to_geocentric(GeodeticWorldCoordinate([radians(lon2), radians(lat2), 0.0])).coordinate
    return float(np.linalg.norm(a - b))


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

    def test_summary_post_spacing_matches_raster_extent(self):
        """
        post_spacing is the raster's UL-to-LR extent divided by its diagonal in posts.

        Computed here from the fixture's documented georeferencing rather than from the
        sensor model, so the assertion is independent of the transform derivation. The
        chip is PixelIsArea, so its (0,0) tiepoint at (-78, 39) is already the UL corner
        of the raster and the LR corner is 10 pixels SE of it.
        """
        arr, _, summary = self.factory.get_tile("dem_chip_10x10.tif")
        height, width = arr.shape
        pixel_size = 0.00024999999999977265

        extent = _ecf_distance(-78.0, 39.0, -78.0 + width * pixel_size, 39.0 - height * pixel_size)
        expected = extent / np.sqrt(width * width + height * height)

        self.assertAlmostEqual(summary.post_spacing, expected, places=9)
        # Sanity check on magnitude: ~0.00025 deg of latitude is ~28 m
        self.assertAlmostEqual(summary.post_spacing, 24.89, places=2)

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

        # The transform is corner-referenced, so pixel (0, 0) is half a post NW of the NW post
        # NW post lat = origin_lat + (num_lat_points - 1) * interval_deg
        # = 22.0 + 4 * (10/10/3600) = 22.0 + 4/3600
        post_spacing = 1.0 / 3600.0
        expected_lat = 22.0 + 4.0 / 3600.0 + post_spacing / 2
        ul_world = sm.image_to_world(ImageCoordinate([0, 0]))
        self.assertAlmostEqual(degrees(ul_world.longitude), 120.0 - post_spacing / 2, places=6)
        self.assertAlmostEqual(degrees(ul_world.latitude), expected_lat, places=6)

        # The NW post itself is at the centre of pixel (0, 0)
        nw_post = sm.image_to_world(ImageCoordinate([0.5, 0.5]))
        self.assertAlmostEqual(degrees(nw_post.longitude), 120.0, places=6)
        self.assertAlmostEqual(degrees(nw_post.latitude), 22.0 + 4.0 / 3600.0, places=6)

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
    def test_post_spacing_spans_exact_extent_for_pixel_is_point(self, mock_io):
        """
        For a post-referenced source, [0,0]-to-[width,height] is the exact raster extent.

        Before the area/point normalization the same span ran from the centre of the NW
        post to one post beyond the SE post — the right length but the wrong footprint.
        Now both endpoints are true raster corners, half a post outside the post grid.
        """
        nw_post_lon, nw_post_lat, res = 10.0, 50.0, 0.01
        metadata = {
            "33550": [res, res, 0],
            "33922": [0, 0, 0, nw_post_lon, nw_post_lat, 0],
            # GTRasterTypeGeoKey (1025) = 2, RasterPixelIsPoint
            "34735": [1, 1, 0, 2, 1024, 0, 1, 2, 1025, 0, 1, 2],
        }
        data = np.arange(16, dtype=np.int16).reshape(4, 4)
        mock_io.open.return_value = self._make_mock_reader(metadata, data)

        _, _, summary = StoredDEMTileFactory("/tiles").get_tile("point.tif")

        rows, cols = data.shape
        ul_lon, ul_lat = nw_post_lon - res / 2, nw_post_lat + res / 2
        extent = _ecf_distance(ul_lon, ul_lat, ul_lon + cols * res, ul_lat - rows * res)
        expected = extent / np.sqrt(cols * cols + rows * rows)

        self.assertAlmostEqual(summary.post_spacing, expected, places=9)

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
