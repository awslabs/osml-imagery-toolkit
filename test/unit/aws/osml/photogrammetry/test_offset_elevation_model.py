#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest


class TestOffsetElevationModel(unittest.TestCase):
    def test_constant_offset(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.elevation_offset_provider import ConstantOffsetProvider
        from aws.osml.photogrammetry.offset_elevation_model import OffsetElevationModel

        elevation_model = OffsetElevationModel(
            inner_elevation_model=ConstantElevationModel(10.0),
            offset_provider=ConstantOffsetProvider(5.0),
        )
        world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert world_coordinate.elevation == 0.0
        assert elevation_model.set_elevation(world_coordinate)
        assert world_coordinate.longitude == 1
        assert world_coordinate.latitude == 2
        assert world_coordinate.elevation == 15.0


if __name__ == "__main__":
    unittest.main()
