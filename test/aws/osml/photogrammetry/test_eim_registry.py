#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import unittest
from unittest.mock import patch

import numpy as np


class TestEimRegistry(unittest.TestCase):
    @patch.dict("aws.osml.photogrammetry.eim_registry._REGISTRY", clear=True)
    def test_custom_registration(self):
        from aws.osml.photogrammetry import eim_registry
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.earth_intersection_minimizer import EarthIntersectionMinimizer
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel

        class FakeEimSolver(EarthIntersectionMinimizer):
            def solve(
                self,
                minimization_function,
                elevation_model,
                initial_guess,
                search_distance,
                lon_bounds=None,
                lat_bounds=None,
                height_bounds=None,
            ):
                world_coordinate = GeodeticWorldCoordinate((*initial_guess, 0.0))
                elevation_model.set_elevation(world_coordinate)
                return (world_coordinate, True)

        eim_registry.register("fakesolve", FakeEimSolver())
        fakesolve = eim_registry.get("fakesolve")
        assert np.allclose(
            fakesolve.solve(
                lambda lonlat, em: 0.0,
                ConstantElevationModel(3.0),
                [1.0, 2.0],
                0.5,
            )[0].coordinate,
            [1.0, 2.0, 3.0],
        )
