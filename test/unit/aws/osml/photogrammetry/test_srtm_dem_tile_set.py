#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from math import radians


class TestSRTMDEMTileSet(unittest.TestCase):
    def test_ne_location(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.srtm_dem_tile_set import SRTMTileSet

        tile_set = SRTMTileSet()
        tile_path = tile_set.find_tile_id(GeodeticWorldCoordinate([radians(142), radians(3), 0.0]))
        assert "n03_e142_1arc_v3.tif" == tile_path

    def test_sw_location(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.srtm_dem_tile_set import SRTMTileSet

        tile_set = SRTMTileSet()
        tile_path = tile_set.find_tile_id(GeodeticWorldCoordinate([radians(-2), radians(-11), 0.0]))
        assert "s11_w002_1arc_v3.tif" == tile_path

    def test_zeros_and_overrides(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.srtm_dem_tile_set import SRTMTileSet

        tile_set = SRTMTileSet(prefix="CustomPrefix_", version="?", format_extension=".foo")
        tile_path = tile_set.find_tile_id(GeodeticWorldCoordinate([radians(0.0), radians(0.0), 0.0]))
        assert "CustomPrefix_n00_e000_?.foo" == tile_path


if __name__ == "__main__":
    unittest.main()
