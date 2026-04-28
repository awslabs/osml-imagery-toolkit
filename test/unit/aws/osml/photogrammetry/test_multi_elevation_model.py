#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest


class TestMultiElevationModel(unittest.TestCase):
    def test_empty_list(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.multi_elevation_model import MultiElevationModel

        elevation_model = MultiElevationModel([])
        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert not elevation_model.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 0.0

    def test_order(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.multi_elevation_model import MultiElevationModel

        elevation_model = MultiElevationModel(
            [
                MultiElevationModel([]),
                ConstantElevationModel(1.0),
                ConstantElevationModel(2.0),
            ],
        )

        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert elevation_model.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 1.0


if __name__ == "__main__":
    unittest.main()
