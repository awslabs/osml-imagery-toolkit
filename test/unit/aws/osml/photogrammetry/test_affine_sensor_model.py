#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from math import radians

import numpy as np
import pytest


class TestAffineSensorModel(unittest.TestCase):
    def test_affine_sensor_model(self):
        from aws.osml.photogrammetry.affine_sensor_model import AffineSensorModel
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel

        sensor_model = AffineSensorModel([0.0, 0.002, 0.0, 0.0, 0.0, 0.003])
        elevation_model = ConstantElevationModel(42.0)
        image_coordinate = ImageCoordinate([200, 300])
        world_coordinate = sensor_model.image_to_world(image_coordinate, elevation_model=elevation_model)
        assert np.array_equal(world_coordinate.coordinate, np.array([radians(0.4), radians(0.9), 42.0]))
        new_image_coordinate = sensor_model.world_to_image(world_coordinate)
        assert np.array_equal(image_coordinate.coordinate, new_image_coordinate.coordinate)

    def test_affine_sensor_model_real_example(self):
        from aws.osml.photogrammetry.affine_sensor_model import AffineSensorModel
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate

        transform = (
            -43.681640625,
            4.487879136029412e-06,
            0.0,
            -22.939453125,
            0.0,
            -4.487879136029412e-06,
        )
        sensor_model = AffineSensorModel(transform)
        sample_image_bounds = [
            ImageCoordinate((0, 0)),
            ImageCoordinate((19584, 0)),
            ImageCoordinate((19584, 19584)),
            ImageCoordinate((0, 19584)),
        ]
        sample_geo_bounds = [
            GeodeticWorldCoordinate((radians(-43.681640625), radians(-22.939453125), 0.0)),
            GeodeticWorldCoordinate((radians(-43.59375), radians(-22.939453125), 0.0)),
            GeodeticWorldCoordinate((radians(-43.59375), radians(-23.02734375), 0.0)),
            GeodeticWorldCoordinate((radians(-43.681640625), radians(-23.02734375), 0.0)),
        ]
        assert (
            pytest.approx(sample_geo_bounds[0].coordinate, rel=1e-6, abs=1e-6)
            == sensor_model.image_to_world(sample_image_bounds[0]).coordinate
        )
        assert (
            pytest.approx(sample_geo_bounds[1].coordinate, rel=1e-6, abs=1e-6)
            == sensor_model.image_to_world(sample_image_bounds[1]).coordinate
        )
        assert (
            pytest.approx(sample_image_bounds[0].coordinate, rel=1e-6, abs=1e-6)
            == sensor_model.world_to_image(sample_geo_bounds[0]).coordinate
        )
        assert (
            pytest.approx(sample_image_bounds[1].coordinate, rel=1e-6, abs=1e-6)
            == sensor_model.world_to_image(sample_geo_bounds[1]).coordinate
        )

    def test_non_invertable_transform(self):
        from aws.osml.photogrammetry.affine_sensor_model import AffineSensorModel

        transform = [
            -43.681640625,
            0,
            0.0,
            -22.939453125,
            0.0,
            0,
        ]
        with pytest.raises(ValueError):
            AffineSensorModel(transform)

    def test_sample_tiff_geotransform(self):
        from aws.osml.photogrammetry.affine_sensor_model import AffineSensorModel
        from aws.osml.photogrammetry.coordinates import ImageCoordinate

        geo_transform = (8.98125, 0.0375, 0.0, 52.01875, 0.0, -0.0375)
        proj_wkt = (
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
            'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
            'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
        )

        sensor_model = AffineSensorModel(geo_transform, proj_wkt)
        world_coord = sensor_model.image_to_world(ImageCoordinate([50, 50]))
        assert pytest.approx(world_coord.coordinate, abs=0.1) == [radians(9.0), radians(52.0), 0.0]
        image_coord = sensor_model.world_to_image(world_coord)
        assert pytest.approx(image_coord.coordinate, abs=0.1) == [50, 50]


if __name__ == "__main__":
    unittest.main()
