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

        # These are the points we will test for interpolation
        test_grid_coordinates = [
            ImageCoordinate([-1.0, -1.0]),
            ImageCoordinate([0.5, 0.5]),
            ImageCoordinate([1.0, 0.5]),
            ImageCoordinate([1.0, 1.5]),
            ImageCoordinate([1.5, 0.0]),
            ImageCoordinate([1.5, 1.5]),
            ImageCoordinate([2.5, 2.5]),
            ImageCoordinate([0.0, 0.0]),
            ImageCoordinate([2.0, 2.0]),
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


if __name__ == "__main__":
    unittest.main()
