#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import unittest

import numpy as np
import pytest


@pytest.mark.usefixtures("fake_minimization")
class TestEIMNelderMead(unittest.TestCase):
    def test_solve(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.eim_neldermead import EIMNelderMead
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel

        solver = EIMNelderMead()
        coordinate, success = solver.solve(
            self.fake_minimization(ImageCoordinate((0.0, 0.0))),
            elevation_model=ConstantElevationModel(1.0),
            initial_guess=np.array([0.1, 0.2]),
            search_distance=0.5,
        )

        assert success
        assert np.allclose(coordinate.coordinate, [0.0, 0.0, 1.0], atol=0.1)
