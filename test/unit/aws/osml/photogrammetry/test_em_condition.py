#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest


class TestEMCondition(unittest.TestCase):
    def test_fixed_conditions(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.em_condition import EMConditionFalse, EMConditionTrue

        em_condition_true = EMConditionTrue()
        em_condition_false = EMConditionFalse()
        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert em_condition_true.is_true(world_coordinate)
        assert not em_condition_false.is_true(world_coordinate)


if __name__ == "__main__":
    unittest.main()
