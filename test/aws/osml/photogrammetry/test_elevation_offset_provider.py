#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest


class TestElevationOffsetProvider(unittest.TestCase):
    def test_constant_offset_provider(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_offset_provider import ConstantOffsetProvider

        offset_provider = ConstantOffsetProvider(10.0)
        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert offset_provider.get_offset(world_coordinate) == 10.0


if __name__ == "__main__":
    unittest.main()
