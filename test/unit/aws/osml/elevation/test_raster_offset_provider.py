#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest
from math import radians
from unittest.mock import MagicMock, patch

import numpy as np

from aws.osml.elevation import RasterOffsetProvider
from aws.osml.photogrammetry import GeodeticWorldCoordinate


class TestRasterOffsetProviderLazyInit(unittest.TestCase):
    """Verify lazy initialization behavior."""

    def test_grid_not_loaded_on_construction(self):
        provider = RasterOffsetProvider("/some/grid.tif")
        self.assertIsNone(provider.offset_grid)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_grid_loaded_on_first_get_offset(self, mock_io):
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), _make_uniform_grid(5, 10, value=15.0))
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        self.assertIsNone(provider.offset_grid)

        coord = GeodeticWorldCoordinate([radians(5.0), radians(2.5), 0.0])
        provider.get_offset(coord)
        self.assertIsNotNone(provider.offset_grid)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_grid_loaded_only_once(self, mock_io):
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), _make_uniform_grid(5, 10, value=10.0))
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(5.0), radians(2.5), 0.0])
        provider.get_offset(coord)
        provider.get_offset(coord)
        mock_io.open.assert_called_once()


class TestRasterOffsetProviderInterpolation(unittest.TestCase):
    """Verify interpolation accuracy."""

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_interpolation_at_grid_center(self, mock_io):
        """A uniform grid should return the same value everywhere."""
        data = _make_uniform_grid(5, 10, value=42.0)
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(5.0), radians(2.5), 0.0])
        offset = provider.get_offset(coord)
        self.assertAlmostEqual(offset, 42.0, places=4)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_interpolation_at_grid_points(self, mock_io):
        """Values at grid cell centers should match the raw data."""
        rows, cols = 4, 4
        data = np.arange(16, dtype=np.float64).reshape(rows, cols)
        metadata = _make_metadata(origin_lon=0.0, origin_lat=0.0, x_res=1.0, y_res=1.0, rows=rows, cols=cols)
        mock_reader = _make_mock_reader(metadata, data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")

        # Query the center of pixel (0,0) after flipping: data is flipped so
        # row 0 becomes the southernmost row. After flip, ascending lat axis:
        # lat_center[0] = origin_lat + y_res/2 = 0.5
        # lon_center[0] = origin_lon + x_res/2 = 0.5
        # After north-up flip: row 0 of flipped data = row 3 of original = [12,13,14,15]
        # So pixel (row=0, col=0) in ascending order = original data[3,0] = 12
        coord = GeodeticWorldCoordinate([radians(0.5), radians(0.5), 0.0])
        offset = provider.get_offset(coord)
        self.assertAlmostEqual(offset, 12.0, places=4)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_bilinear_interpolation_between_points(self, mock_io):
        """Midpoint between two known values should be the average."""
        rows, cols = 2, 2
        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        metadata = _make_metadata(origin_lon=0.0, origin_lat=0.0, x_res=1.0, y_res=1.0, rows=rows, cols=cols)
        mock_reader = _make_mock_reader(metadata, data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")

        # After flip (north-up → ascending lat): row 0 = original row 1 = [30, 40]
        # lat axis: [0.5, 1.5], lon axis: [0.5, 1.5]
        # Midpoint lat=1.0, lon=1.0 → average of all 4 corners = 25.0
        coord = GeodeticWorldCoordinate([radians(1.0), radians(1.0), 0.0])
        offset = provider.get_offset(coord)
        self.assertAlmostEqual(offset, 25.0, places=4)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_scale_factor_applied(self, mock_io):
        """Scale factor multiplies all values."""
        data = _make_uniform_grid(5, 10, value=100.0)
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif", scale_factor=0.5)
        coord = GeodeticWorldCoordinate([radians(5.0), radians(2.5), 0.0])
        offset = provider.get_offset(coord)
        self.assertAlmostEqual(offset, 50.0, places=4)


class TestRasterOffsetProviderAxisFlipping(unittest.TestCase):
    """Verify correct handling of north-up (negative y_res) rasters."""

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_north_up_raster(self, mock_io):
        """Standard north-up GeoTIFF (negative y_res) is handled correctly."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        metadata = {
            "33550": [1.0, 1.0, 0],
            "33922": [0, 0, 0, 10.0, 50.0, 0],
        }
        mock_reader = _make_mock_reader(metadata, data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        # With this tiepoint: origin is (10, 50), pixel scale (1, 1)
        # GeoTransform: [10, 1, 0, 50, 0, -1]
        # After flip: lat axis ascending from 47.5 to 49.5
        # Center of grid: lat=48.5, lon=11.5 → original pixel (1,1) = 5.0
        coord = GeodeticWorldCoordinate([radians(11.5), radians(48.5), 0.0])
        offset = provider.get_offset(coord)
        self.assertAlmostEqual(offset, 5.0, places=4)


class TestRasterOffsetProviderBoundsChecking(unittest.TestCase):
    """Verify bounds checking raises ValueError."""

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_latitude_out_of_bounds(self, mock_io):
        data = _make_uniform_grid(5, 10, value=1.0)
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(0.0), radians(91.0), 0.0])
        with self.assertRaises(ValueError):
            provider.get_offset(coord)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_longitude_out_of_bounds(self, mock_io):
        data = _make_uniform_grid(5, 10, value=1.0)
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(181.0), radians(0.0), 0.0])
        with self.assertRaises(ValueError):
            provider.get_offset(coord)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_negative_latitude_out_of_bounds(self, mock_io):
        data = _make_uniform_grid(5, 10, value=1.0)
        mock_reader = _make_mock_reader(_uniform_grid_metadata(), data)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(0.0), radians(-91.0), 0.0])
        with self.assertRaises(ValueError):
            provider.get_offset(coord)


class TestRasterOffsetProviderErrors(unittest.TestCase):
    """Verify error conditions."""

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_non_uniform_grid_raises(self, mock_io):
        """GeoTransform with rotation (non-zero gt[2] or gt[4]) raises ValueError."""
        mock_asset = MagicMock()
        # Rotated transform: gt[2] != 0
        mock_asset.metadata = {"34264": [1.0, 0.5, 0, 10.0, 0.5, -1.0, 0, 50.0, 0, 0, 0, 0, 0, 0, 0, 1.0]}
        mock_reader = MagicMock()
        mock_reader.get_asset_keys.return_value = ["image:0"]
        mock_reader.get_asset.return_value = mock_asset
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(10.5), radians(49.5), 0.0])
        with self.assertRaises(ValueError):
            provider.get_offset(coord)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_no_geo_transform_raises(self, mock_io):
        """Missing geo transform metadata raises ValueError."""
        mock_asset = MagicMock()
        mock_asset.metadata = {}
        mock_reader = MagicMock()
        mock_reader.get_asset_keys.return_value = ["image:0"]
        mock_reader.get_asset.return_value = mock_asset
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(5.0), radians(2.5), 0.0])
        with self.assertRaises(ValueError):
            provider.get_offset(coord)

    @patch("aws.osml.elevation.raster_offset_provider.IO")
    def test_no_image_assets_raises(self, mock_io):
        """Raster with no image assets raises ValueError."""
        mock_reader = MagicMock()
        mock_reader.get_asset_keys.return_value = []
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_io.open.return_value = mock_reader

        provider = RasterOffsetProvider("/grid.tif")
        coord = GeodeticWorldCoordinate([radians(5.0), radians(2.5), 0.0])
        with self.assertRaises(ValueError):
            provider.get_offset(coord)


# --- Test helpers ---


def _make_uniform_grid(rows: int, cols: int, value: float) -> np.ndarray:
    return np.full((rows, cols), value, dtype=np.float64)


def _uniform_grid_metadata() -> dict:
    """Metadata for a 5x10 grid covering lon=[0,10], lat=[0,5], north-up."""
    return _make_metadata(origin_lon=0.0, origin_lat=0.0, x_res=1.0, y_res=1.0, rows=5, cols=10)


def _make_metadata(origin_lon: float, origin_lat: float, x_res: float, y_res: float, rows: int, cols: int) -> dict:
    """Create GeoTIFF metadata for a north-up raster.

    The tiepoint ties pixel (0,0) to the NW corner, and ModelPixelScale gives
    the positive pixel sizes. The derived geo transform will be:
    [origin_lon, x_res, 0, nw_lat, 0, -y_res]
    """
    nw_lat = origin_lat + y_res * rows
    return {
        "33550": [x_res, y_res, 0],
        "33922": [0, 0, 0, origin_lon, nw_lat, 0],
    }


def _make_mock_reader(metadata: dict, data: np.ndarray):
    """Create a mock IO reader with a single image asset."""
    mock_asset = MagicMock()
    mock_asset.metadata = metadata
    mock_asset.block_grid_size = (1, 1)
    mock_asset.num_pixels_per_block_vertical = data.shape[0]
    mock_asset.num_pixels_per_block_horizontal = data.shape[1]
    mock_asset.num_rows = data.shape[0]
    mock_asset.num_columns = data.shape[1]
    mock_asset.get_block.return_value = data.reshape(1, *data.shape)

    mock_reader = MagicMock()
    mock_reader.get_asset_keys.return_value = ["image:0"]
    mock_reader.get_asset.return_value = mock_asset
    mock_reader.__enter__ = MagicMock(return_value=mock_reader)
    mock_reader.__exit__ = MagicMock(return_value=False)
    return mock_reader


if __name__ == "__main__":
    unittest.main()
