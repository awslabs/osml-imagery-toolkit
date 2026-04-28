#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest
from math import radians

import pytest


class TestCoordinates(unittest.TestCase):
    def test_worldcoordinate_list_constructor(self):
        from aws.osml.photogrammetry.coordinates import WorldCoordinate

        world_coordinate = WorldCoordinate([1.0, 2.0, 3.0])
        assert world_coordinate.x == 1.0
        assert world_coordinate.y == 2.0
        assert world_coordinate.z == 3.0
        assert world_coordinate.coordinate.shape == (3,)  # 1D numpy array

    def test_worldcoordinate_repr(self):
        from aws.osml.photogrammetry.coordinates import WorldCoordinate

        world_coordinate = WorldCoordinate([1.0, 2.0, 3.0])
        assert f"{world_coordinate!r}" == "WorldCoordinate(coordinate=array([1., 2., 3.]))"

    def test_imagecoordinate_list_constructor(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate

        image_coordinate = ImageCoordinate([1.0, 2.0])
        assert image_coordinate.x == 1.0
        assert image_coordinate.y == 2.0
        assert image_coordinate.c == 1.0
        assert image_coordinate.r == 2.0
        assert image_coordinate.coordinate.shape == (2,)  # 1D numpy array

    def test_imagecoordinate_default_constructor(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate

        image_coordinate = ImageCoordinate()
        assert image_coordinate.x == 0.0
        assert image_coordinate.y == 0.0

    def test_imagecoordinate_bad_values(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate

        with pytest.raises(ValueError) as value_error:
            image_coordinate = ImageCoordinate([1.0, 2.0, 3.0])  # noqa: F841
        assert "must have 2 components" in str(value_error.value)

    def test_imagecoordinate_repr(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate

        image_coordinate = ImageCoordinate([-10.2, 5.0])
        assert f"{image_coordinate!r}" == "ImageCoordinate(coordinate=array([-10.2,   5. ]))"

    def test_geodeticworldcoordinate_list_constructor(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate([-1.2, 1.5, 10.0])
        assert geodetic_coordinate.longitude == -1.2
        assert geodetic_coordinate.latitude == 1.5
        assert geodetic_coordinate.elevation == 10.0
        assert geodetic_coordinate.x == -1.2
        assert geodetic_coordinate.y == 1.5
        assert geodetic_coordinate.z == 10.0
        assert geodetic_coordinate.coordinate.shape == (3,)  # 1D numpy array

    def test_geodeticworldcoordinate_default_constructor(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate()
        assert geodetic_coordinate.longitude == 0.0
        assert geodetic_coordinate.latitude == 0.0
        assert geodetic_coordinate.elevation == 0.0

    def test_geodeticworldcoordinate_bad_values(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        with pytest.raises(ValueError) as value_error:
            geodetic_coordinate = GeodeticWorldCoordinate([1.0, 2.0])  # noqa: F841
        assert "must have 3 components" in str(value_error.value)

    def test_setters(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate, WorldCoordinate

        wc = WorldCoordinate()
        assert wc.x == 0.0
        with pytest.raises(ValueError):
            wc.x = "a"

        assert wc.y == 0.0
        with pytest.raises(ValueError):
            wc.y = "a"

        gwc = GeodeticWorldCoordinate()
        assert gwc.longitude == 0.0
        with pytest.raises(ValueError):
            gwc.longitude = "a"

        assert gwc.latitude == 0.0
        with pytest.raises(ValueError):
            gwc.latitude = "a"

        assert gwc.elevation == 0.0
        with pytest.raises(ValueError):
            gwc.elevation = "a"

        ic = ImageCoordinate()
        assert ic.c == 0.0
        with pytest.raises(ValueError):
            ic.c = "a"

        assert ic.r == 0.0
        with pytest.raises(ValueError):
            ic.r = "a"

        assert ic.x == 0.0
        with pytest.raises(ValueError):
            ic.x = "a"

        assert ic.y == 0.0
        with pytest.raises(ValueError):
            ic.y = "a"

    def test_geodeticworldcoordinate_to_dms_string(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate([radians(-121.5125), radians(-10.5125), 10.0])
        assert geodetic_coordinate.to_dms_string() == "103045S1213045W"
        geodetic_coordinate = GeodeticWorldCoordinate([radians(1.5125), radians(1.5125), 10.0])
        assert geodetic_coordinate.to_dms_string() == "013045N0013045E"

    def test_geodeticworldcoordinate_format(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate([radians(115.25), radians(-45.5), 3.0])
        assert f"{geodetic_coordinate:%ld%lm%ls%lH %od%om%os%oH %E}" == "453000S 1151500E 3.0"
        assert f"{geodetic_coordinate:%ld %lm %ls %lh %od %om %os %oh %E}" == "45 30 00 s 115 15 00 e 3.0"
        assert f"{geodetic_coordinate:%l %o %E}" == "45.5 115.25 3.0"
        assert f"{geodetic_coordinate:%L %O %E}" == "-0.7941248096574199 2.011491962923465 3.0"
        assert f"{geodetic_coordinate}" == "453000S 1151500E 3.0"
        assert f"{geodetic_coordinate:100%% unexpected usage: %X}" == "100% unexpected usage: "

    def test_geodeticworldcoordinate_repr(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate([1.1, 2.2, 3.3])
        assert f"{geodetic_coordinate!r}" == "GeodeticWorldCoordinate(coordinate=array([1.1, 2.2, 3.3]))"

    def test_geodeticworldcoordinate_normalized(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate((radians(0.0), radians(0.0), 1.0)).normalized()
        assert geodetic_coordinate.coordinate == pytest.approx((radians(0.0), radians(0.0), 1.0), abs=1e-6)
        geodetic_coordinate = GeodeticWorldCoordinate((radians(360.0), radians(0.0), 1.0)).normalized()
        assert geodetic_coordinate.coordinate == pytest.approx((radians(0.0), radians(0.0), 1.0), abs=1e-6)
        geodetic_coordinate = GeodeticWorldCoordinate((radians(1.0), radians(181.0), 1.0)).normalized()
        assert geodetic_coordinate.coordinate == pytest.approx((radians(-179.0), radians(-1.0), 1.0), abs=1e-6)
        geodetic_coordinate = GeodeticWorldCoordinate((radians(-700.0), radians(-91.0), 1.0)).normalized()
        assert geodetic_coordinate.coordinate == pytest.approx((radians(-160.0), radians(-89.0), 1.0), abs=1e-6)

    def test_geodeticworldcoordinate_range_adjusted(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        geodetic_coordinate = GeodeticWorldCoordinate((radians(-700.0), radians(-91.0), 1.0)).range_adjusted(
            radians(-180.0),
            radians(180.0),
            radians(-90.0),
            radians(90.0),
        )
        assert geodetic_coordinate.coordinate == pytest.approx((radians(-160.0), radians(-89.0), 1.0), abs=1e-6)
        geodetic_coordinate = GeodeticWorldCoordinate((radians(-700.0), radians(-91.0), 1.0)).range_adjusted(
            radians(0.0),
            radians(360.0),
            radians(-90.0),
            radians(90.0),
        )
        assert geodetic_coordinate.coordinate == pytest.approx((radians(200.0), radians(-89.0), 1.0), abs=1e-6)
        geodetic_coordinate = GeodeticWorldCoordinate((radians(181.0), radians(179.0), 1.0)).range_adjusted(
            radians(360.0),
            radians(720.0),
            radians(-270.0),
            radians(-180.0),
        )
        assert geodetic_coordinate.coordinate == pytest.approx((radians(541.0), radians(-181.0), 1.0), abs=1e-6)
        geodetic_coordinate = GeodeticWorldCoordinate((radians(181.0), radians(1.0), 1.0)).range_adjusted(
            radians(0.0),
            radians(355.0),
            radians(-185.0),
            radians(0.0),
        )
        assert geodetic_coordinate.coordinate == pytest.approx((radians(1.0), radians(-181.0), 1.0), abs=1e-6)
        with pytest.raises(ValueError):
            geodetic_coordinate = GeodeticWorldCoordinate((radians(0.0), radians(0.0), 0.0)).range_adjusted(
                radians(1.0),
                radians(2.0),
                radians(1.0),
                radians(2.0),
            )

    def test_ecef_to_geodetic(self):
        from aws.osml.photogrammetry.coordinates import WorldCoordinate, geocentric_to_geodetic

        ecef_world_coordinate = WorldCoordinate([6257968.0, 547501.0, 1100249.0])
        geodetic_world_coordinate = geocentric_to_geodetic(ecef_world_coordinate)
        assert geodetic_world_coordinate.longitude == pytest.approx(radians(5.0), abs=0.000001)
        assert geodetic_world_coordinate.latitude == pytest.approx(radians(10.0), abs=0.000001)
        assert geodetic_world_coordinate.elevation == pytest.approx(0.0, abs=1.0)

    def test_geodetic_to_ecef(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, geodetic_to_geocentric

        geodetic_world_coordinate = GeodeticWorldCoordinate([radians(5.0), radians(10.0), 0.0])
        ecef_world_coordinate = geodetic_to_geocentric(geodetic_world_coordinate)
        assert ecef_world_coordinate.x == pytest.approx(6257968.0, abs=1.0)
        assert ecef_world_coordinate.y == pytest.approx(547501.0, abs=1.0)
        assert ecef_world_coordinate.z == pytest.approx(1100249.0, abs=1.0)


if __name__ == "__main__":
    unittest.main()
