#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import itertools
import unittest

import geojson
import numpy as np
from mock import Mock


class TestCompositeSensorModel(unittest.TestCase):
    def setUp(self) -> None:
        from aws.osml.photogrammetry.composite_sensor_model import CompositeSensorModel
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.sensor_model import SensorModel

        self.approximate_sensor_model = Mock(SensorModel)
        self.approximate_sensor_model.image_to_world.side_effect = (
            GeodeticWorldCoordinate([0.1, 0.2, 0.3]) for x in itertools.cycle([True])
        )
        self.approximate_sensor_model.world_to_image.side_effect = (
            ImageCoordinate([10, 11]) for x in itertools.cycle([True])
        )
        self.precision_sensor_model = Mock(SensorModel)
        self.precision_sensor_model.image_to_world.side_effect = (
            GeodeticWorldCoordinate([0.11, 0.21, 0.31]) for x in itertools.cycle([True])
        )
        self.precision_sensor_model.world_to_image.side_effect = (
            ImageCoordinate([10.1, 11.1]) for x in itertools.cycle([True])
        )
        self.composite_sensor_model = CompositeSensorModel(self.approximate_sensor_model, self.precision_sensor_model)
        self.elevation_model = ConstantElevationModel(0.31)
        self.sample_geojson_detections = self.build_geojson_detections()

    def test_image_to_world(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate

        assert np.array_equal(
            self.composite_sensor_model.image_to_world(ImageCoordinate([1, 2])).coordinate,
            np.array([0.11, 0.21, 0.31]),
        )
        assert self.approximate_sensor_model.image_to_world.call_count == 1
        assert self.precision_sensor_model.image_to_world.call_count == 1

    def test_world_to_image(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        assert np.array_equal(
            self.composite_sensor_model.world_to_image(GeodeticWorldCoordinate([0.1, 0.2, 0.3])).coordinate,
            np.array([10.1, 11.1]),
        )
        assert self.approximate_sensor_model.world_to_image.call_count == 0
        assert self.precision_sensor_model.world_to_image.call_count == 1

    @staticmethod
    def build_geojson_detections():
        with open("./test/data/detections.geojson", "r") as geojson_file:
            return geojson.load(geojson_file)
