#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from typing import Any, Dict, Optional

import numpy as np

from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
from aws.osml.photogrammetry.elevation_model import ElevationModel
from aws.osml.photogrammetry.sensor_model import SensorModel


class FakeSensorModel(SensorModel):
    def __init__(self):
        super().__init__()

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        world_coordinate = GeodeticWorldCoordinate([image_coordinate.x, image_coordinate.y, 0.0])
        if elevation_model:
            elevation_model.set_elevation(world_coordinate)
        return world_coordinate

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        return ImageCoordinate([world_coordinate.x, world_coordinate.y])


class TestChippedImageSensorModel(unittest.TestCase):
    def test_chipped_image_sensor_model(self):
        from aws.osml.photogrammetry.chipped_image_sensor_model import ChippedImageSensorModel
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel

        original_image_coordinates = [
            ImageCoordinate([10.0, 10.0]),
            ImageCoordinate([10.0, 20.0]),
            ImageCoordinate([20.0, 20.0]),
            ImageCoordinate([20.0, 10.0]),
        ]

        chipped_image_coordinates = [
            ImageCoordinate([0.0, 0.0]),
            ImageCoordinate([0.0, 5.0]),
            ImageCoordinate([5.0, 5.0]),
            ImageCoordinate([5.0, 0.0]),
        ]
        sensor_model = ChippedImageSensorModel(original_image_coordinates, chipped_image_coordinates, FakeSensorModel())
        elevation_model = ConstantElevationModel(42.0)
        image_coordinate = ImageCoordinate([2, 2])

        # Test with an external elevation model
        world_coordinate = sensor_model.image_to_world(image_coordinate, elevation_model=elevation_model)
        assert np.allclose(world_coordinate.coordinate, np.array([14.0, 14.0, 42.0]))

        # Test without an external elevation model
        world_coordinate = sensor_model.image_to_world(image_coordinate)
        assert np.allclose(world_coordinate.coordinate, np.array([14.0, 14.0, 0.0]))
        new_image_coordinate = sensor_model.world_to_image(world_coordinate)
        assert np.allclose(image_coordinate.coordinate, new_image_coordinate.coordinate)


if __name__ == "__main__":
    unittest.main()
