#  Copyright 2023-2025 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2026 General Atomics Integrated Intelligence, Inc.

import unittest
from math import radians
from pathlib import Path

import numpy as np

import aws.osml.formats.sidd.models.sidd_v2_0_0 as sidd200
from aws.osml.formats.model_utils import sidd_parser
from aws.osml.metadata.sidd_sensor_model_builder import SIDDSensorModelBuilder
from aws.osml.photogrammetry import (
    ChippedImageSensorModel,
    GeodeticWorldCoordinate,
    ImageCoordinate,
    SICDSensorModel,
    geocentric_to_geodetic,
)


class TestSIDDSensorModel(unittest.TestCase):
    def test_planar_projection(self):
        sidd: sidd200.SIDD = sidd_parser.from_path(Path("./test/data/sidd/example.sidd.xml"))

        sm = SIDDSensorModelBuilder.from_dataclass(sidd)

        assert sm is not None
        assert isinstance(sm, SICDSensorModel)

        scp_image_coord = ImageCoordinate(
            [
                sm.coord_converter.scp_pixel.x - sm.coord_converter.first_pixel.x,
                sm.coord_converter.scp_pixel.y - sm.coord_converter.first_pixel.y,
            ]
        )
        scp_world_coord = geocentric_to_geodetic(sm.coord_converter.scp_ecf)

        assert np.allclose(scp_image_coord.coordinate, sm.world_to_image(scp_world_coord).coordinate, atol=1.0)
        assert np.allclose(scp_world_coord.coordinate, sm.image_to_world(scp_image_coord).coordinate)

        num_cols = sidd.measurement.pixel_footprint.col
        num_rows = sidd.measurement.pixel_footprint.row
        for icp in sidd.geo_data.image_corners.icp:
            world_location = GeodeticWorldCoordinate([radians(icp.lon), radians(icp.lat), scp_world_coord.elevation])
            if icp.index.value == "1:FRFC":
                image_location = ImageCoordinate([0, 0])
            elif icp.index.value == "2:FRLC":
                image_location = ImageCoordinate([num_cols, 0])
            elif icp.index.value == "3:LRLC":
                image_location = ImageCoordinate([num_cols, num_rows])
            elif icp.index.value == "4:LRFC":
                image_location = ImageCoordinate([0, num_rows])
            else:
                raise ValueError(f"Unexpected ICP in test data {icp.index.value}")

            computed_world_location = sm.image_to_world(image_location)
            computed_image_location = sm.world_to_image(computed_world_location)

            assert np.allclose(computed_world_location.coordinate[0:2], world_location.coordinate[0:2], atol=0.000001)
            assert np.allclose(computed_image_location.coordinate, image_location.coordinate, atol=0.5)

    def test_chipped_sidd(self):
        sidd: sidd200.SIDD = sidd_parser.from_path(Path("./test/data/sidd/example.sidd-chip.xml"))

        sm = SIDDSensorModelBuilder.from_dataclass(sidd)
        assert sm is not None
        assert isinstance(sm, ChippedImageSensorModel)

        coord_calculated_from_chip = sm.image_to_world(ImageCoordinate([0, 0]))
        coord_calculated_from_full = sm.full_image_sensor_model.image_to_world(ImageCoordinate([512, 512]))

        assert np.allclose(coord_calculated_from_chip.coordinate, coord_calculated_from_full.coordinate)
