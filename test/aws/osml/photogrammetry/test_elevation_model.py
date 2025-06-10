#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest


class TestElevationModel(unittest.TestCase):
    def test_constant_elevation_model(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel

        elevation_model = ConstantElevationModel(10.0)
        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert world_coordinate.elevation == 0.0
        assert elevation_model.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 10.0


if __name__ == "__main__":
    unittest.main()
