#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from math import radians

import numpy as np
import pytest


class TestMathUtils(unittest.TestCase):
    def setUp(self):
        self.sample_image_domain = self.build_image_domain()
        self.sample_geodetic_ground_domain = self.build_geodetic_ground_domain()
        self.sample_rectangular_ground_domain = self.build_rectangular_ground_domain()
        self.sample_polynomial_sensor_model = self.build_polynomial_sensor_model(
            self.sample_geodetic_ground_domain, self.sample_image_domain
        )
        self.sample_sectioned_polynomial_sensor_model = self.build_sectioned_polynomial_sensor_model(
            self.sample_geodetic_ground_domain, self.sample_image_domain
        )

    def test_image_domain(self):
        assert self.sample_image_domain.min_row == 0
        assert self.sample_image_domain.max_row == 2048
        assert self.sample_image_domain.min_column == 10
        assert self.sample_image_domain.max_column == 2038

    def test_geodetic_ground_domain(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        assert len(self.sample_geodetic_ground_domain.ground_domain_vertices) == 8
        assert np.array_equal(
            self.sample_geodetic_ground_domain.ground_domain_vertices[0].coordinate,
            np.array([radians(0.0), radians(10.0), -100.0]),
        )
        assert np.array_equal(
            self.sample_geodetic_ground_domain.ground_domain_vertices[7].coordinate,
            np.array([radians(10.0), radians(0.0), 100.0]),
        )

        world_coordinate = GeodeticWorldCoordinate([radians(5.0), radians(5.0), 50.0])
        domain_coordinate = self.sample_geodetic_ground_domain.geodetic_to_ground_domain_coordinate(world_coordinate)
        assert np.array_equal(domain_coordinate.coordinate, np.array([radians(5.0), radians(5.0), 50.0]))
        assert np.array_equal(
            self.sample_geodetic_ground_domain.ground_reference_point.coordinate, np.array([radians(5.0), radians(5.0), 0.0])
        )
        assert np.allclose(
            np.array(self.sample_geodetic_ground_domain.geodetic_lonlat_bbox),
            np.array([radians(0), radians(0), radians(10.0), radians(10.0)]),
            atol=0.000001,
        )

    def test_rectangular_ground_domain(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        world_coordinate = GeodeticWorldCoordinate([radians(5.0), radians(10.0), 0.0])
        domain_coordinate = self.sample_rectangular_ground_domain.geodetic_to_ground_domain_coordinate(world_coordinate)
        assert np.array_equal(domain_coordinate.coordinate, np.array([0.0, 0.0, 0.0]))

        new_world_coordinate = self.sample_rectangular_ground_domain.ground_domain_coordinate_to_geodetic(domain_coordinate)
        assert world_coordinate.longitude == pytest.approx(new_world_coordinate.longitude, abs=0.000001)
        assert world_coordinate.latitude == pytest.approx(new_world_coordinate.latitude, abs=0.000001)
        assert world_coordinate.elevation == pytest.approx(new_world_coordinate.elevation, abs=0.1)

        # TODO: More Testing!!!

    def test_rsmpolynomial_eval(self):
        from aws.osml.photogrammetry.coordinates import WorldCoordinate
        from aws.osml.photogrammetry.replacement_sensor_model import RSMPolynomial

        polynomial = RSMPolynomial(
            2,
            1,
            1,
            [
                1.0,  # constant
                1.0,  # X
                0.0,  # XX
                2.0,  # Y
                0.0,  # XY
                0.0,  # XXY
                3.0,  # Z
                0.0,  # XZ
                0.0,  # XXZ
                0.0,  # YZ
                0.0,  # XYZ
                1.0,  # XXYZ
            ],
        )

        world_coordinate = WorldCoordinate([10, 20, 30])
        assert polynomial(world_coordinate) == 1.0 + 10.0 + 40.0 + 90.0 + (100.0 * 20.0 * 30.0)

    def test_rsmloworderpolynomial_eval(self):
        from aws.osml.photogrammetry.coordinates import WorldCoordinate
        from aws.osml.photogrammetry.replacement_sensor_model import RSMLowOrderPolynomial

        low_order_polynomial = RSMLowOrderPolynomial(
            [
                42.0,  # constant
                1.0,  # X
                1.0,  # Y
                1.0,  # Z
                0.0,  # XX
                0.0,  # XY
                2.0,  # XZ
                0.0,  # YY
                0.0,  # YZ
                3.0,  # ZZ
            ]
        )

        world_coordinate = WorldCoordinate([1.0, 2.0, 3.0])
        assert low_order_polynomial(world_coordinate) == 42.0 + 1.0 + 2.0 + 3.0 + 6.0 + 27.0

    def test_polynomial_sensor_model(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.sensor_model import SensorModelOptions

        world_coordinate = GeodeticWorldCoordinate([radians(5.0), radians(5.0), 0.0])
        image_coordinate = self.sample_polynomial_sensor_model.world_to_image(world_coordinate)

        assert np.array_equal(image_coordinate.coordinate, np.array([100.0 * radians(5.0), 100.0 * radians(5.0)]))
        new_world_coordinate = self.sample_polynomial_sensor_model.image_to_world(
            image_coordinate,
            options={
                SensorModelOptions.INITIAL_GUESS: [radians(5.1), radians(5.1)],
                SensorModelOptions.INITIAL_SEARCH_DISTANCE: radians(0.1),
            },
        )

        assert np.allclose(world_coordinate.coordinate, new_world_coordinate.coordinate)

    def test_segmented_polynomial_sensor_model(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel

        elevation_model = ConstantElevationModel(42.0)
        world_coordinate = GeodeticWorldCoordinate([radians(5.0), radians(5.0), 42.0])
        image_coordinate = self.sample_sectioned_polynomial_sensor_model.world_to_image(world_coordinate)
        assert np.allclose(image_coordinate.coordinate, np.array([radians(5.0), radians(5.0)]))
        new_world_coordinate = self.sample_sectioned_polynomial_sensor_model.image_to_world(
            image_coordinate, elevation_model=elevation_model
        )
        assert np.allclose(world_coordinate.coordinate, new_world_coordinate.coordinate)

    def test_build_rsm_ground_domain_invalid_count_exception(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.replacement_sensor_model import RSMGroundDomain, RSMGroundDomainForm

        sample_geodetic_ground_vertices = [
            GeodeticWorldCoordinate([radians(0.0), radians(10.0), -100.0]),
            GeodeticWorldCoordinate([radians(0.0), radians(0.0), -100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(10.0), -100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(0.0), -100.0]),
            GeodeticWorldCoordinate([radians(0.0), radians(10.0), 100.0]),
            GeodeticWorldCoordinate([radians(0.0), radians(0.0), 100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(10.0), 100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(0.0), 100.0]),
        ]
        del sample_geodetic_ground_vertices[0]  # remove an element, throw exception if it has only 7 items

        with pytest.raises(ValueError):
            RSMGroundDomain(RSMGroundDomainForm.GEODETIC, sample_geodetic_ground_vertices)

    def test_build_rsm_ground_domain_empty_coordinate_exception(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, WorldCoordinate
        from aws.osml.photogrammetry.replacement_sensor_model import RSMGroundDomain, RSMGroundDomainForm

        rectangular_coordinate_origin = GeodeticWorldCoordinate([radians(5.0), radians(10.0), 0.0])

        ground_domain_vertices = [
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 10.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 0.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 10.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 0.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 10.0, 100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 0.0, 100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 10.0, 100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 0.0, 100.0])),
        ]

        rectangular_coordinate_unit_vectors = None
        with pytest.raises(ValueError):
            RSMGroundDomain(
                RSMGroundDomainForm.RECTANGULAR,
                ground_domain_vertices,
                rectangular_coordinate_origin,
                rectangular_coordinate_unit_vectors,
            )

    def test_invalid_rsmloworderpolynomial_exception(self):
        from aws.osml.photogrammetry.replacement_sensor_model import RSMLowOrderPolynomial

        # Missing 1 index (ZZ), will throw exception if it doesn't meet the requirements
        with pytest.raises(ValueError):
            RSMLowOrderPolynomial(
                [
                    42.0,  # constant
                    1.0,  # X
                    1.0,  # Y
                    1.0,  # Z
                    0.0,  # XX
                    0.0,  # XY
                    2.0,  # XZ
                    0.0,  # YY
                    0.0,  # YZ
                ]
            )

    def test_invalid_rsmpolynomial_exception(self):
        from aws.osml.photogrammetry.replacement_sensor_model import RSMPolynomial

        # Missing 1 index (XYZ), will throw exception if it doesn't meet the requirements
        with pytest.raises(ValueError):
            RSMPolynomial(
                2,
                1,
                1,
                [
                    1.0,  # constant
                    1.0,  # X
                    0.0,  # XX
                    2.0,  # Y
                    0.0,  # XY
                    0.0,  # XXY
                    3.0,  # Z
                    0.0,  # XZ
                    0.0,  # XXZ
                    0.0,  # YZ
                    0.0,  # XYZ
                ],
            )

    @staticmethod
    def build_image_domain():
        from aws.osml.photogrammetry.replacement_sensor_model import RSMImageDomain

        return RSMImageDomain(0, 2048, 10, 2038)

    @staticmethod
    def build_geodetic_ground_domain():
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.replacement_sensor_model import RSMGroundDomain, RSMGroundDomainForm

        sample_geodetic_ground_vertices = [
            GeodeticWorldCoordinate([radians(0.0), radians(10.0), -100.0]),
            GeodeticWorldCoordinate([radians(0.0), radians(0.0), -100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(10.0), -100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(0.0), -100.0]),
            GeodeticWorldCoordinate([radians(0.0), radians(10.0), 100.0]),
            GeodeticWorldCoordinate([radians(0.0), radians(0.0), 100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(10.0), 100.0]),
            GeodeticWorldCoordinate([radians(10.0), radians(0.0), 100.0]),
        ]
        return RSMGroundDomain(
            RSMGroundDomainForm.GEODETIC,
            sample_geodetic_ground_vertices,
            ground_reference_point=GeodeticWorldCoordinate([radians(5.0), radians(5.0), 0.0]),
        )

    @staticmethod
    def build_rectangular_ground_domain():
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, WorldCoordinate, geodetic_to_geocentric
        from aws.osml.photogrammetry.replacement_sensor_model import RSMGroundDomain, RSMGroundDomainForm

        geodetic_coordinate_origin = GeodeticWorldCoordinate([radians(5.0), radians(10.0), 0.0])
        rectangular_coordinate_origin = geodetic_to_geocentric(geodetic_coordinate_origin)

        ground_domain_vertices = [
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 10.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 0.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 10.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 0.0, -100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 10.0, 100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [0.0, 0.0, 100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 10.0, 100.0])),
            WorldCoordinate(np.add(rectangular_coordinate_origin.coordinate, [10.0, 0.0, 100.0])),
        ]

        rectangular_coordinate_unit_vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        return RSMGroundDomain(
            RSMGroundDomainForm.RECTANGULAR,
            ground_domain_vertices,
            rectangular_coordinate_origin,
            rectangular_coordinate_unit_vectors,
        )

    @staticmethod
    def build_polynomial_sensor_model(sample_geodetic_ground_domain, sample_image_domain):
        from aws.osml.photogrammetry.replacement_sensor_model import RSMContext, RSMPolynomial, RSMPolynomialSensorModel

        context = RSMContext(sample_geodetic_ground_domain, sample_image_domain)
        coln = RSMPolynomial(1, 1, 1, [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cold = RSMPolynomial(0, 0, 0, [1.0])
        rown = RSMPolynomial(1, 1, 1, [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rowd = RSMPolynomial(0, 0, 0, [1.0])

        return RSMPolynomialSensorModel(
            context, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, rown, rowd, coln, cold
        )

    @staticmethod
    def build_sectioned_polynomial_sensor_model(sample_geodetic_ground_domain, sample_image_domain):
        from aws.osml.photogrammetry.replacement_sensor_model import (
            RSMContext,
            RSMLowOrderPolynomial,
            RSMPolynomial,
            RSMPolynomialSensorModel,
            RSMSectionedPolynomialSensorModel,
        )

        context = RSMContext(sample_geodetic_ground_domain, sample_image_domain)

        col_poly = RSMLowOrderPolynomial([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        row_poly = RSMLowOrderPolynomial([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        coln = RSMPolynomial(1, 1, 1, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cold = RSMPolynomial(0, 0, 0, [1.0])
        rown = RSMPolynomial(1, 1, 1, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rowd = RSMPolynomial(0, 0, 0, [1.0])
        identity_polynomial_sensor_model = RSMPolynomialSensorModel(
            context, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, rown, rowd, coln, cold
        )
        identity_polynomial_sensor_model2 = RSMPolynomialSensorModel(
            context, 2, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, rown, rowd, coln, cold
        )

        return RSMSectionedPolynomialSensorModel(
            context,
            2,
            1,
            1024,
            1024,
            row_poly,
            col_poly,
            [[identity_polynomial_sensor_model], [identity_polynomial_sensor_model2]],
        )


if __name__ == "__main__":
    unittest.main()
