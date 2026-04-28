#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest


class TestConditionalElevationModel(unittest.TestCase):
    def test_true_false(self):
        from aws.osml.photogrammetry.conditional_elevation_model import ConditionalElevationModel
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.em_condition import EMConditionFalse, EMConditionTrue

        inner_elevation_model = ConstantElevationModel(10.0)
        elevation_model_false = ConditionalElevationModel(
            inner_elevation_model=inner_elevation_model,
            em_condition=EMConditionFalse(),
        )
        elevation_model_true = ConditionalElevationModel(
            inner_elevation_model=inner_elevation_model,
            em_condition=EMConditionTrue(),
        )
        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert world_coordinate.elevation == 0.0
        assert not elevation_model_false.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 0.0
        assert elevation_model_true.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 10.0


if __name__ == "__main__":
    unittest.main()
