#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest

from aws.osml.elevation._geo_transform import _derive_dted_geo_transform, derive_geo_transform


def _geokey_directory(raster_type: int) -> list:
    """Build a minimal GeoKey directory (tag 34735) carrying GTRasterTypeGeoKey."""
    return [1, 1, 0, 2, 1024, 0, 1, 2, 1025, 0, 1, raster_type]


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
        post_spacing = 1.0 / 3600.0
        gt = derive_geo_transform(metadata)
        self.assertIsNotNone(gt)
        # Origin is the NW pixel corner, half a post NW of the (-78, 39) NW post
        self.assertAlmostEqual(gt[0], -78.0 - post_spacing / 2, places=10)
        self.assertAlmostEqual(gt[1], post_spacing, places=15)
        self.assertAlmostEqual(gt[2], 0.0)
        self.assertAlmostEqual(gt[3], 39.0 + post_spacing / 2, places=10)
        self.assertAlmostEqual(gt[4], 0.0)
        self.assertAlmostEqual(gt[5], -post_spacing, places=15)

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


class TestRasterTypeNormalization(unittest.TestCase):
    """GTRasterTypeGeoKey (1025) must be normalized so every transform is corner-referenced."""

    pixel_size = 0.001
    tiepoint_metadata = {
        "33550": [pixel_size, pixel_size, 0],
        "33922": [0, 0, 0, -78.0, 39.0, 0],
    }

    def test_pixel_is_area_is_unshifted(self):
        metadata = {**self.tiepoint_metadata, "34735": _geokey_directory(1)}
        gt = derive_geo_transform(metadata)
        self.assertAlmostEqual(gt[0], -78.0, places=12)
        self.assertAlmostEqual(gt[3], 39.0, places=12)

    def test_absent_raster_type_behaves_as_pixel_is_area(self):
        with_geokeys = derive_geo_transform({**self.tiepoint_metadata, "34735": _geokey_directory(1)})
        without_geokeys = derive_geo_transform(dict(self.tiepoint_metadata))
        self.assertEqual(without_geokeys, with_geokeys)

    def test_pixel_is_point_shifts_origin_half_a_pixel(self):
        area = derive_geo_transform({**self.tiepoint_metadata, "34735": _geokey_directory(1)})
        point = derive_geo_transform({**self.tiepoint_metadata, "34735": _geokey_directory(2)})

        # PixelIsPoint tiepoints reference sample centres, so the corner is half a pixel NW
        self.assertAlmostEqual(point[0] - area[0], -self.pixel_size / 2, places=12)
        self.assertAlmostEqual(point[3] - area[3], self.pixel_size / 2, places=12)
        # Resolution and rotation terms are untouched
        for i in (1, 2, 4, 5):
            self.assertEqual(point[i], area[i])

    def test_pixel_is_point_shifts_model_transformation(self):
        """GDAL shifts regardless of which tag supplied the transform, including rotated ones."""
        model_transform = [0.5, 0.1, 0, 100.0, 0.2, -0.5, 0, 200.0, 0, 0, 0, 0, 0, 0, 0, 1]
        area = derive_geo_transform({"34264": model_transform, "34735": _geokey_directory(1)})
        point = derive_geo_transform({"34264": model_transform, "34735": _geokey_directory(2)})

        self.assertAlmostEqual(point[0], area[0] - (0.5 + 0.1) / 2, places=12)
        self.assertAlmostEqual(point[3], area[3] - (0.2 - 0.5) / 2, places=12)

    def test_real_srtm_tile_matches_gdal_origin(self):
        """test/data/n47_e034_3arc_v2.tif is PixelIsPoint; gdalinfo reports the shifted origin."""
        from aws.osml.io import IO

        with IO.open("./test/data/n47_e034_3arc_v2.tif", "r") as reader:
            keys = [k for k in reader.get_asset_keys() if k.startswith("image:")]
            metadata = dict(reader.get_asset(keys[0]).metadata)

        gt = derive_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], 33.999583333333334, places=12)
        self.assertAlmostEqual(gt[3], 48.000416666666666, places=12)


class TestDeriveDtedGeoTransform(unittest.TestCase):
    def test_southern_hemisphere(self):
        metadata = {
            "dted:origin_longitude": -43.0,
            "dted:origin_latitude": -23.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }
        post_spacing = 1.0 / 3600.0
        gt = _derive_dted_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], -43.0 - post_spacing / 2)
        self.assertAlmostEqual(gt[3], -22.0 + post_spacing / 2, places=10)

    def test_eastern_hemisphere(self):
        metadata = {
            "dted:origin_longitude": 120.0,
            "dted:origin_latitude": 22.0,
            "dted:longitude_interval": 10,
            "dted:latitude_interval": 10,
            "dted:num_latitude_points": 3601,
        }
        post_spacing = 1.0 / 3600.0
        gt = _derive_dted_geo_transform(metadata)
        self.assertIsNotNone(gt)
        self.assertAlmostEqual(gt[0], 120.0 - post_spacing / 2)
        self.assertAlmostEqual(gt[3], 23.0 + post_spacing / 2, places=10)

    def test_dted_geotiff_consistency(self):
        """
        Verify DTED and an equivalent PixelIsPoint GeoTIFF produce the same transform.

        Both are asserted against the absolute NW corner as well as each other, so a
        convention flip applied to both sources cannot slip through.
        """
        pixel_size = 1.0 / 3600.0

        # An SRTM-style tile: post-referenced tiepoint at the NW post, so 1025 = 2
        tif_metadata = {
            "33550": [pixel_size, pixel_size, 0],
            "33922": [0, 0, 0, -78.0, 39.0, 0],
            "34735": _geokey_directory(2),
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

        # Absolute check: pixel (0, 0) is the corner half a post NW of the (-78, 39) post
        expected_corner = (-78.0 - pixel_size / 2, 39.0 + pixel_size / 2)
        for gt in (tif_gt, dted_gt):
            self.assertAlmostEqual(gt[0], expected_corner[0], places=10)
            self.assertAlmostEqual(gt[3], expected_corner[1], places=10)


if __name__ == "__main__":
    unittest.main()
