#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest
from math import pi


class TestNormalizedElevationModel(unittest.TestCase):
    def test_restricted_range(self):
        from aws.osml.photogrammetry.conditional_elevation_model import ConditionalElevationModel
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.em_condition import ElevationModelCondition
        from aws.osml.photogrammetry.normalized_elevation_model import NormalizedElevationModel

        class BoundedCondition(ElevationModelCondition):
            def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
                return (
                    world_coordinate.latitude >= -pi / 2
                    and world_coordinate.latitude <= pi / 2
                    and world_coordinate.longitude >= -pi
                    and world_coordinate.longitude <= pi
                )

        conditional_model = ConditionalElevationModel(
            inner_elevation_model=ConstantElevationModel(3.0),
            em_condition=BoundedCondition(),
        )
        normalized_model = NormalizedElevationModel(
            inner_elevation_model=conditional_model,
        )

        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert world_coordinate.elevation == 0.0
        assert not conditional_model.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 0.0
        assert normalized_model.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 3.0


if __name__ == "__main__":
    unittest.main()
