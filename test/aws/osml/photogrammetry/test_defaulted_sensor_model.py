#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest
from math import degrees

import numpy as np


class TestDefaultedSensorModel(unittest.TestCase):
    def test_no_defaults(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.defaulted_sensor_model import DefaultedSensorModel
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.gdal_sensor_model import GDALAffineSensorModel

        sensor_model = DefaultedSensorModel(
            inner_sensor_model=GDALAffineSensorModel([0, degrees(2), 0, 0, 0, degrees(3)]),
        )

        image_coordinate = ImageCoordinate([10, 20])
        world_coordinate = sensor_model.image_to_world(image_coordinate)
        assert np.allclose(world_coordinate.coordinate, [20, 60, 0], atol=1e-6)
        new_image_coordinate = sensor_model.world_to_image(world_coordinate)
        assert np.allclose(new_image_coordinate.coordinate, image_coordinate.coordinate)
        world_coordinate = sensor_model.image_to_world(
            image_coordinate,
            elevation_model=ConstantElevationModel(4),
        )
        assert np.allclose(world_coordinate.coordinate, [20, 60, 4], atol=1e-6)

    def test_defaults(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.defaulted_sensor_model import DefaultedSensorModel
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.gdal_sensor_model import GDALAffineSensorModel
        from aws.osml.photogrammetry.sensor_model import SensorModelOptions

        sensor_model = DefaultedSensorModel(
            inner_sensor_model=GDALAffineSensorModel([0, degrees(2), 0, 0, 0, degrees(3)]),
            elevation_model=ConstantElevationModel(3),
            options={SensorModelOptions.IGNORE_DEFAULT_ELEVATION_MODEL: False},
        )

        image_coordinate = ImageCoordinate([10, 20])
        world_coordinate = sensor_model.image_to_world(image_coordinate)
        assert np.allclose(world_coordinate.coordinate, [20, 60, 3], atol=1e-6)
        world_coordinate = sensor_model.image_to_world(
            image_coordinate,
            elevation_model=ConstantElevationModel(4),
        )
        assert np.allclose(world_coordinate.coordinate, [20, 60, 4], atol=1e-6)
        world_coordinate = sensor_model.image_to_world(
            image_coordinate,
            options={SensorModelOptions.IGNORE_DEFAULT_ELEVATION_MODEL: True},
        )
        assert np.allclose(world_coordinate.coordinate, [20, 60, 0], atol=1e-6)

    def test_nans(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.defaulted_sensor_model import DefaultedSensorModel
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.gdal_sensor_model import GDALAffineSensorModel

        # Add elevation adjustment to allow testing nan replacement.
        class FakeHeightSensorModel(GDALAffineSensorModel):
            def world_to_image(self, world_coordinate):
                new_world_coordinate = GeodeticWorldCoordinate(world_coordinate.coordinate)
                new_world_coordinate.longitude += new_world_coordinate.z
                return super().world_to_image(new_world_coordinate)

        sensor_model = DefaultedSensorModel(
            inner_sensor_model=DefaultedSensorModel(
                inner_sensor_model=FakeHeightSensorModel([0, degrees(2), 0, 0, 0, degrees(3)]),
                elevation_model=ConstantElevationModel(10),
            )
        )

        world_coordinate = GeodeticWorldCoordinate([20, 60, np.nan])
        image_coordinate = sensor_model.world_to_image(world_coordinate)
        assert np.allclose(image_coordinate.coordinate, [15, 20], atol=1e-6)
