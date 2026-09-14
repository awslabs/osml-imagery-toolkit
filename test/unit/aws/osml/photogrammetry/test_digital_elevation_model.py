#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest

import mock
import numpy as np
import pytest


class TestDigitalElevationModel(unittest.TestCase):
    def test_dem_interpolation(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileFactory,
            DigitalElevationModelTileSet,
        )
        from aws.osml.photogrammetry.elevation_model import ElevationRegionSummary
        from aws.osml.photogrammetry.sensor_model import SensorModel

        mock_tile_set = mock.Mock(DigitalElevationModelTileSet)
        mock_tile_set.find_tile_id.return_value = "MockN00E000V0.tif"

        # This is a sample 3x3 grid of elevation data
        test_elevation_data = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        test_elevation_summary = ElevationRegionSummary(0.0, 4.0, -1, 30.0)

        # These are the points we will test for interpolation. The grid is queried in
        # continuous image coordinates, so post [row][col] sits at (col + 0.5, row + 0.5).
        test_grid_coordinates = [
            ImageCoordinate([-0.5, -0.5]),
            ImageCoordinate([1.0, 1.0]),
            ImageCoordinate([1.5, 1.0]),
            ImageCoordinate([1.5, 2.0]),
            ImageCoordinate([2.0, 0.5]),
            ImageCoordinate([2.0, 2.0]),
            ImageCoordinate([3.0, 3.0]),
            ImageCoordinate([0.5, 0.5]),
            ImageCoordinate([2.5, 2.5]),
        ]

        # These are the expected interpolated values
        expected_values = [0.0, 1.0, 1.5, 2.5, 2.5, 3.0, 4.0, 0.0, 4.0]

        # This mock sensor model will return the sequence of test_grid_coordinates each time a
        # world_to_image call is made
        mock_sensor_model = mock.Mock(SensorModel)
        mock_sensor_model.world_to_image.side_effect = iter(test_grid_coordinates)

        # This mock tile factory will always return the 3x3 elevation grid and the sensor model
        mock_tile_factory = mock.Mock(DigitalElevationModelTileFactory)
        mock_tile_factory.get_tile.return_value = test_elevation_data, mock_sensor_model, test_elevation_summary

        dem = DigitalElevationModel(mock_tile_set, mock_tile_factory)

        # Loop over all the expected values and verify that the world coordinate elevation is updated
        # while the latitude, longitude are unchanged
        for grid_coordinate, expected_value in zip(test_grid_coordinates, expected_values):
            world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
            assert dem.set_elevation(world_coordinate)

            assert world_coordinate.longitude == 1.0
            assert world_coordinate.latitude == 2.0
            assert world_coordinate.elevation == pytest.approx(expected_value)

        # Verify that find_tile_id was called for each test but that get_tile was only called
        # once because the grid and sensor model were cached
        assert mock_tile_set.find_tile_id.call_count == len(test_grid_coordinates)
        assert mock_tile_factory.get_tile.call_count == 1

    def test_unknown_tile(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileFactory,
            DigitalElevationModelTileSet,
        )

        # This is the case when the tile set does not know about a tile for the requested coordinate
        mock_tile_set = mock.Mock(DigitalElevationModelTileSet)
        mock_tile_set.find_tile_id.return_value = None
        mock_tile_factory = mock.Mock(DigitalElevationModelTileFactory)

        dem = DigitalElevationModel(mock_tile_set, mock_tile_factory)

        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert not dem.set_elevation(world_coordinate)
        assert world_coordinate.elevation == 0.0
        assert mock_tile_set.find_tile_id.call_count == 1
        assert mock_tile_factory.get_tile.call_count == 0

    def test_missing_tile(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileFactory,
            DigitalElevationModelTileSet,
        )

        # This is the case when the area doesn't have an elevation tile associated with the region
        mock_tile_set = mock.Mock(DigitalElevationModelTileSet)
        mock_tile_set.find_tile_id.return_value = "MockN00E000V0.tif"
        mock_tile_factory = mock.Mock(DigitalElevationModelTileFactory)
        mock_tile_factory.get_tile.return_value = None, None, None

        dem = DigitalElevationModel(mock_tile_set, mock_tile_factory)

        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert not dem.set_elevation(world_coordinate)
        assert world_coordinate.elevation == 0.0
        assert mock_tile_set.find_tile_id.call_count == 1
        assert mock_tile_factory.get_tile.call_count == 1

    def test_tile_exception(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileFactory,
            DigitalElevationModelTileSet,
        )

        # This is the case when the tile factory has an exception
        mock_tile_set = mock.Mock(DigitalElevationModelTileSet)
        mock_tile_set.find_tile_id.return_value = "MockN00E000V0.tif"
        mock_tile_factory = mock.Mock(DigitalElevationModelTileFactory)
        mock_tile_factory.get_tile.side_effect = Exception

        dem = DigitalElevationModel(mock_tile_set, mock_tile_factory)

        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert not dem.set_elevation(world_coordinate)
        assert not dem.set_elevation(world_coordinate)
        assert world_coordinate.elevation == 0.0
        assert mock_tile_set.find_tile_id.call_count == 2
        assert mock_tile_factory.get_tile.call_count == 1

    def test_missing_elevation(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileFactory,
            DigitalElevationModelTileSet,
        )
        from aws.osml.photogrammetry.elevation_model import ElevationRegionSummary
        from aws.osml.photogrammetry.sensor_model import SensorModel

        mock_tile_set = mock.Mock(DigitalElevationModelTileSet)
        mock_tile_set.find_tile_id.return_value = "MockN00E000V0.tif"

        # This is a sample 3x3 grid of elevation data with a no data value at 2,2
        test_elevation_data = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 3.0], [2.0, 3.0, -9999]])
        test_elevation_summary = ElevationRegionSummary(0.0, 4.0, -9999, 30.0)

        # These are the points we will test for interpolation. The grid is queried in
        # continuous image coordinates, so post [row][col] sits at (col + 0.5, row + 0.5).
        test_grid_coordinates = [
            ImageCoordinate([-0.5, -0.5]),
            ImageCoordinate([1.0, 1.0]),
            ImageCoordinate([1.5, 1.0]),
            ImageCoordinate([1.5, 2.0]),
            ImageCoordinate([2.0, 0.5]),
            ImageCoordinate([2.0, 2.0]),
            ImageCoordinate([3.0, 3.0]),
            ImageCoordinate([0.5, 0.5]),
            ImageCoordinate([2.5, 2.5]),
        ]

        # This is the default elevation value. If there is missing data in the elevation
        # array, the interpolator should fall back to this elevation to avoid feeding large
        # negative numbers to the interpolation grid
        default_elevation = 2.0

        # These are the expected interpolation values AND the expected return value of the elevation update
        expected_values = [
            (0.0, True),
            (1.0, True),
            (1.5, True),
            (default_elevation, False),
            (2.5, True),
            (default_elevation, False),
            (default_elevation, False),
            (0.0, True),
            (default_elevation, False),
        ]

        # This mock sensor model will return the sequence of test_grid_coordinates each time a
        # world_to_image call is made
        mock_sensor_model = mock.Mock(SensorModel)
        mock_sensor_model.world_to_image.side_effect = iter(test_grid_coordinates)

        # This mock tile factory will always return the 3x3 elevation grid and the sensor model
        mock_tile_factory = mock.Mock(DigitalElevationModelTileFactory)
        mock_tile_factory.get_tile.return_value = test_elevation_data, mock_sensor_model, test_elevation_summary

        dem = DigitalElevationModel(mock_tile_set, mock_tile_factory, propagate_nans=True)

        # Loop over all the expected values and verify that the interpolated value matched the
        # expected value. If no data is present in the elevation array, the interpolator should
        # match the default elevation and not update the elevation
        for grid_coordinate, (expected_value, updated_elevation) in zip(test_grid_coordinates, expected_values):
            world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, default_elevation])
            assert dem.set_elevation(world_coordinate) == updated_elevation

            assert world_coordinate.longitude == 1.0
            assert world_coordinate.latitude == 2.0
            assert world_coordinate.elevation == pytest.approx(expected_value)

        # Verify that find_tile_id was called for each test but that get_tile was only called
        # once because the grid and sensor model were cached
        assert mock_tile_set.find_tile_id.call_count == len(test_grid_coordinates)
        assert mock_tile_factory.get_tile.call_count == 1


class TestInterpolationGridConvention(unittest.TestCase):
    """
    Pin the coordinate convention of get_interpolation_grid() for external callers.

    The grid is queried in the same continuous image coordinates the sensor models
    produce, so post [row][col] is at (col + 0.5, row + 0.5) and querying a post
    returns its stored value rather than an interpolation of its neighbors.
    """

    # Deliberately asymmetric so a transposed or half-pixel-shifted lookup cannot pass
    elevations = np.array([[10.0, 20.0, 40.0, 80.0], [15.0, 25.0, 45.0, 85.0], [17.0, 27.0, 47.0, 87.0]])

    def _build_dem(self, no_data_value, propagate_nans):
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileFactory,
            DigitalElevationModelTileSet,
        )
        from aws.osml.photogrammetry.elevation_model import ElevationRegionSummary
        from aws.osml.photogrammetry.sensor_model import SensorModel

        mock_tile_set = mock.Mock(DigitalElevationModelTileSet)
        mock_tile_set.find_tile_id.return_value = "MockN00E000V0.tif"
        mock_tile_factory = mock.Mock(DigitalElevationModelTileFactory)
        mock_tile_factory.get_tile.return_value = (
            self.elevations,
            mock.Mock(SensorModel),
            ElevationRegionSummary(10.0, 87.0, no_data_value, 30.0),
        )
        return DigitalElevationModel(mock_tile_set, mock_tile_factory, propagate_nans=propagate_nans)

    def _grids(self):
        """The non-propagating spline path and the NaN-propagating interpolator path."""
        # no_data_value=-9999 is absent from the array, so propagate_nans still yields a spline
        yield "spline", self._build_dem(-9999, propagate_nans=False).get_interpolation_grid("t")[0]
        # A no_data_value that matches a real post forces the RegularGridInterpolator path
        yield "interpolator", self._build_dem(10, propagate_nans=True).get_interpolation_grid("t")[0]

    def test_posts_are_at_half_pixel_offsets(self):
        height, width = self.elevations.shape
        for name, grid in self._grids():
            with self.subTest(path=name):
                for row in range(height):
                    for col in range(width):
                        expected = self.elevations[row][col]
                        if name == "interpolator" and expected == 10.0:
                            continue  # masked to NaN on this path
                        assert grid(col + 0.5, row + 0.5)[0][0] == pytest.approx(expected)

    def test_corner_posts(self):
        height, width = self.elevations.shape
        for name, grid in self._grids():
            with self.subTest(path=name):
                assert grid(width - 0.5, height - 0.5)[0][0] == pytest.approx(self.elevations[-1][-1])

    def test_midpoint_between_posts_interpolates(self):
        for name, grid in self._grids():
            with self.subTest(path=name):
                # Halfway between posts [0][1]=20 and [0][2]=40 along x
                assert grid(2.0, 0.5)[0][0] == pytest.approx(30.0)

    def test_out_of_bounds_queries_clamp(self):
        height, width = self.elevations.shape
        for name, grid in self._grids():
            with self.subTest(path=name):
                # Clamped to the LR post rather than extrapolated beyond it
                assert grid(width + 5.0, height + 5.0)[0][0] == pytest.approx(self.elevations[-1][-1])
                # Clamped to the UR post
                assert grid(width + 5.0, -5.0)[0][0] == pytest.approx(self.elevations[0][-1])


class TestDigitalElevationModelRealTiles(unittest.TestCase):
    """
    Unmocked end-to-end tests over real georeferenced tiles.

    These exercise the seam the mocked tests cannot see: the contract between the
    image coordinates a real AffineSensorModel produces and the coordinates the
    interpolation grid expects. Both GeoTIFF raster type conventions are covered, so
    a half-pixel error of either sign in either layer fails at least one case.
    """

    def _query_post(self, tile_directory, tile_name, row, col):
        """Return (stored post value, elevation the DEM reports at that post's centre)."""
        from aws.osml.elevation import StoredDEMTileFactory
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.digital_elevation_model import (
            DigitalElevationModel,
            DigitalElevationModelTileSet,
        )

        class SingleTileSet(DigitalElevationModelTileSet):
            def find_tile_id(self, geodetic_world_coordinate):
                return tile_name

        tile_factory = StoredDEMTileFactory(tile_directory)
        elevations, sensor_model, _ = tile_factory.get_tile(tile_name)
        assert elevations is not None, f"could not load {tile_name}"

        # The geodetic location of the centre of post [row][col]
        post_centre = sensor_model.image_to_world(ImageCoordinate([col + 0.5, row + 0.5]))

        dem = DigitalElevationModel(SingleTileSet(), tile_factory)
        world_coordinate = GeodeticWorldCoordinate([post_centre.longitude, post_centre.latitude, 0.0])
        assert dem.set_elevation(world_coordinate)
        return elevations[row][col], world_coordinate.elevation

    def test_pixel_is_area_geotiff(self):
        """
        data/unit/dem_chip_10x10.tif is PixelIsArea (GTRasterTypeGeoKey = 1).

        Post [0][0] is 100.0 with neighbors 110.0 east and 105.0 south. Before the
        area/point normalization this returned 107.5 — the average of the four
        surrounding posts — because the corner-referenced transform was queried
        against an index-based grid.
        """
        truth, actual = self._query_post("data/unit", "dem_chip_10x10.tif", 0, 0)
        assert truth == pytest.approx(100.0)
        assert actual == pytest.approx(truth)

    def test_pixel_is_point_geotiff(self):
        """
        test/data/n47_e034_3arc_v2.tif is PixelIsPoint (GTRasterTypeGeoKey = 2).

        Post [456][52] is -29.0 m with row neighbors [-62, -29, 19] — a steep enough
        local gradient that a half-post shift lands at -42.5 m. This case is correct
        both before and after the fix, so it proves the normalization did not trade
        one convention's correctness for the other's.
        """
        truth, actual = self._query_post("test/data", "n47_e034_3arc_v2.tif", 456, 52)
        assert truth == pytest.approx(-29.0)
        assert actual == pytest.approx(truth)


if __name__ == "__main__":
    unittest.main()
