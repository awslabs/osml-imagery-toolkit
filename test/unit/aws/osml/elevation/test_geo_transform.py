#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest

from aws.osml.elevation._geo_transform import _derive_dted_geo_transform, derive_geo_transform


class TestDeriveGeoTransform(unittest.TestCase):
    def test_geotiff_single_tiepoint_with_scale(self):
        metadata = {
            "33550": [0.0002777777777777778, 0.0002777777777777778, 0],
            "33922": [0, 0, 0, -78.0, 39.0, 0],
        }
        gt = derive_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], -78.0, places=10)
        self.assertAlmostEqual(gt[1], 0.0002777777777777778, places=15)
        self.assertAlmostEqual(gt[2], 0.0)
        self.assertAlmostEqual(gt[3], 39.0, places=10)
        self.assertAlmostEqual(gt[4], 0.0)
        self.assertAlmostEqual(gt[5], -0.0002777777777777778, places=15)

    def test_geotiff_model_transformation(self):
        metadata = {
            "34264": [
                0.5,
                0.1,
                0,
                100.0,
                0.2,
                -0.5,
                0,
                200.0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        }
        gt = derive_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], 100.0)
        self.assertAlmostEqual(gt[1], 0.5)
        self.assertAlmostEqual(gt[2], 0.1)
        self.assertAlmostEqual(gt[3], 200.0)
        self.assertAlmostEqual(gt[4], 0.2)
        self.assertAlmostEqual(gt[5], -0.5)

    def test_dted_metadata(self):
        metadata = {
            "dted:origin_longitude": -78.0,
            "dted:origin_latitude": 38.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }
        gt = derive_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], -78.0, places=10)
        self.assertAlmostEqual(gt[1], 1.0 / 3600.0, places=15)
        self.assertAlmostEqual(gt[2], 0.0)
        self.assertAlmostEqual(gt[3], 39.0, places=10)
        self.assertAlmostEqual(gt[4], 0.0)
        self.assertAlmostEqual(gt[5], -1.0 / 3600.0, places=15)

    def test_dted_missing_keys_returns_none(self):
        metadata = {
            "dted:origin_longitude": -78.0,
            "dted:origin_latitude": 38.0,
        }
        gt = derive_geo_transform(metadata)
        self.assertIsNone(gt)

    def test_empty_metadata_returns_none(self):
        gt = derive_geo_transform({})
        self.assertIsNone(gt)

    def test_geotiff_takes_priority_over_dted(self):
        metadata = {
            "33550": [0.001, 0.001, 0],
            "33922": [0, 0, 0, 10.0, 20.0, 0],
            "dted:origin_longitude": -78.0,
            "dted:origin_latitude": 38.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }
        gt = derive_geo_transform(metadata)
        self.assertAlmostEqual(gt[0], 10.0)
        self.assertAlmostEqual(gt[3], 20.0)


class TestDeriveDtedGeoTransform(unittest.TestCase):
    def test_southern_hemisphere(self):
        metadata = {
            "dted:origin_longitude": -43.0,
            "dted:origin_latitude": -23.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }
        gt = _derive_dted_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], -43.0)
        self.assertAlmostEqual(gt[3], -23.0 + (3600 / 3600.0))
        self.assertAlmostEqual(gt[3], -22.0, places=10)

    def test_eastern_hemisphere(self):
        metadata = {
            "dted:origin_longitude": 120.0,
            "dted:origin_latitude": 22.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }
        gt = _derive_dted_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], 120.0)
        self.assertAlmostEqual(gt[3], 23.0, places=10)

    def test_dted_geotiff_consistency(self):
        """Verify DTED geo transform matches the GeoTIFF geo transform for the same tile."""
        pixel_size = 1.0 / 3600.0

        tif_metadata = {
            "33550": [pixel_size, pixel_size, 0],
            "33922": [0, 0, 0, -78.0, 39.0, 0],
        }

        dted_metadata = {
            "dted:origin_longitude": -78.0,
            "dted:origin_latitude": 38.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }

        tif_gt = derive_geo_transform(tif_metadata)
        dted_gt = derive_geo_transform(dted_metadata)

        self.assertIsNotNone(tif_gt)
        self.assertIsNotNone(dted_gt)
        for i in range(6):
            self.assertAlmostEqual(tif_gt[i], dted_gt[i], places=10, msg=f"Mismatch at index {i}")


if __name__ == "__main__":
    unittest.main()
