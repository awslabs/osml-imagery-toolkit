#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from math import radians

import numpy as np


class TestRPCSensorModel(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        self.sample_rpc_sensor_model = self.build_rpc_sensor_model()
        self.realworld_rpc_sensor_model = self.build_realworld_rpc_sensor_model()

    def test_rpc_polynomial_eval(self):
        from aws.osml.photogrammetry.coordinates import WorldCoordinate
        from aws.osml.photogrammetry.rpc_sensor_model import RPCPolynomial

        polynomial = RPCPolynomial(
            [
                42.0,  # constant
                1.0,  # L
                2.0,  # P
                3.0,  # H
                0.0,  # LP
                0.0,  # LH
                0.0,  # PH
                0.0,  # LL
                0.0,  # PP
                0.0,  # HH
                0.0,  # LPH
                0.0,  # LLL
                100.0,  # LPP
                0.0,  # LHH
                0.0,  # LLP
                0.0,  # PPP
                0.0,  # PHH
                0.0,  # LLH
                0.0,  # PPH
                0.0,  # HHH
            ]
        )

        world_coordinate = WorldCoordinate([10, 20, 30])
        assert polynomial(world_coordinate) == 42.0 + 10.0 + 40.0 + 90.0 + (100.0 * 10.0 * 20.0 * 20.0)

    def test_rpc_sensor_model(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.sensor_model import SensorModelOptions

        elevation_model = ConstantElevationModel(42.0)
        world_coordinate = GeodeticWorldCoordinate([radians(5.0), radians(3.0), 42.0])
        image_coordinate = self.sample_rpc_sensor_model.world_to_image(world_coordinate)

        assert np.allclose(image_coordinate.coordinate, np.array([5.0, 3.0]))
        new_world_coordinate = self.sample_rpc_sensor_model.image_to_world(
            image_coordinate,
            elevation_model=elevation_model,
            options={
                SensorModelOptions.INITIAL_GUESS: [radians(5.1), radians(3.1)],
                SensorModelOptions.INITIAL_SEARCH_DISTANCE: radians(0.1),
            },
        )

        assert np.allclose(world_coordinate.coordinate, new_world_coordinate.coordinate)

    def test_rpc_sensor_model_realworld(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        world_coordinate = GeodeticWorldCoordinate([radians(56.305048278781925), radians(27.15809908265984), -9.0])
        image_coordinate = self.realworld_rpc_sensor_model.world_to_image(world_coordinate)

        assert np.allclose(image_coordinate.coordinate, np.array([0.0, 13854.0]), atol=1.0)
        new_world_coordinate = self.realworld_rpc_sensor_model.image_to_world(image_coordinate)

        assert np.allclose(world_coordinate.coordinate, new_world_coordinate.coordinate)

    @staticmethod
    def build_rpc_sensor_model():
        from aws.osml.photogrammetry.rpc_sensor_model import RPCPolynomial, RPCSensorModel

        samp_num_poly = RPCPolynomial(
            [
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        samp_den_poly = RPCPolynomial(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        line_num_poly = RPCPolynomial(
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        line_den_poly = RPCPolynomial(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        return RPCSensorModel(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            line_num_poly,
            line_den_poly,
            samp_num_poly,
            samp_den_poly,
        )

    @staticmethod
    def build_realworld_rpc_sensor_model():
        from aws.osml.photogrammetry.rpc_sensor_model import RPCPolynomial, RPCSensorModel

        samp_num_poly = RPCPolynomial(
            [
                -2.204284e7,
                +9.999999e8,
                +6.250255e6,
                -1.342506e8,
                -3.107754e7,
                -1.101236e7,
                +5.563589e6,
                -1.101821e7,
                -1.698152e5,
                +1.657516e6,
                +1.773720e5,
                -5.277249e4,
                +3.404038e5,
                +1.492974e4,
                +2.384937e5,
                +1.576174e3,
                -3.092346e4,
                +8.307716e4,
                -8.854074e4,
                -4.077669e3,
            ]
        )
        samp_den_poly = RPCPolynomial(
            [
                +8.560069e8,
                -9.431402e6,
                -2.600458e7,
                -1.520253e7,
                +1.983045e5,
                +1.213286e5,
                +3.161597e5,
                -4.535808e4,
                +2.744588e5,
                +7.995991e4,
                -1.081926e3,
                -2.324262e1,
                +6.363693e1,
                -4.086394e2,
                -3.527186e1,
                +8.591534e1,
                -9.908944e2,
                +1.995737e2,
                -1.545197e3,
                -1.442957e2,
            ]
        )

        line_num_poly = RPCPolynomial(
            [
                +1.705526e7,
                +4.409292e6,
                -9.999999e8,
                +6.029603e7,
                +1.088467e7,
                -4.977649e5,
                +4.250149e6,
                -3.148845e5,
                +3.036354e7,
                -4.262737e5,
                -4.712155e4,
                +2.661822e3,
                -2.302774e5,
                -2.449691e3,
                +6.146575e4,
                -3.203005e5,
                -5.104847e3,
                +2.006485e4,
                -1.298168e4,
                +1.209481e3,
            ]
        )
        line_den_poly = RPCPolynomial(
            [
                +8.729689e8,
                -9.618288e6,
                -2.651987e7,
                -1.550378e7,
                +2.022340e5,
                +1.237328e5,
                +3.224245e5,
                -4.625686e4,
                +2.798973e5,
                +8.154434e4,
                -1.103364e3,
                -2.370318e1,
                +6.489792e1,
                -4.167367e2,
                -3.597079e1,
                +8.761778e1,
                -1.010529e3,
                +2.035283e2,
                -1.575816e3,
                -1.471550e2,
            ]
        )

        return RPCSensorModel(
            0005.14,
            0000.50,
            006927.0,
            06163.0,
            27.2197,
            056.3653,
            -0009.0,
            006927.0,
            06164.0,
            00.0716,
            000.0727,
            3234.0,
            line_num_poly,
            line_den_poly,
            samp_num_poly,
            samp_den_poly,
        )


if __name__ == "__main__":
    unittest.main()
