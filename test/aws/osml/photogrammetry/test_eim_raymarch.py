#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import unittest
from typing import Optional

import numpy as np
import pytest


@pytest.mark.usefixtures("fake_minimization")
class TestEIMRayMarch(unittest.TestCase):
    def test_solve(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.eim_raymarch import EIMRayMarch
        from aws.osml.photogrammetry.elevation_model import ElevationModel, ElevationRegionSummary

        # This elevation model will create a wall to intersect.
        class FakeElevationModel(ElevationModel):
            def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
                if world_coordinate.x >= 0.1 and world_coordinate.x <= 0.3:
                    world_coordinate.z = 20
                else:
                    world_coordinate.z = 0
                return True

            def describe_region(self, world_coordinate: GeodeticWorldCoordinate) -> Optional[ElevationRegionSummary]:
                return None

        solver = EIMRayMarch()
        coordinate, success = solver.solve(
            self.fake_minimization(ImageCoordinate((0.0, 0.0))),
            elevation_model=FakeElevationModel(),
            initial_guess=np.array([0.0, 0.0]),
            search_distance=0.5,
        )

        assert success
        assert np.allclose(coordinate.coordinate, [0.2, 0.0, 20.0], atol=0.1)
