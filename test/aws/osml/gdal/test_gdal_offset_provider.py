#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import unittest
from typing import Callable

import mock
import numpy as np
import numpy.typing as npt
import pytest
from osgeo import gdal, gdal_array, osr

from configuration import TEST_ENV_CONFIG


@mock.patch.dict("os.environ", TEST_ENV_CONFIG, clear=True)
class TestGDALOffsetProvider(unittest.TestCase):
    def test_orientation_np(self):
        from aws.osml.gdal.gdal_offset_provider import GDALOffsetProvider
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        with mock.patch(
            "osgeo.gdal.Open",
            TestGDALOffsetProvider.create_gdal_fake_open(
                np.array(
                    [
                        [0, 1],
                        [2, 3],
                    ],
                    dtype=np.uint8,
                ),
                (-360.0, 360.0, 0.0,
                 180.0, 0.0, -180.0),  # fmt: skip
            ),
        ):
            offset_provider = GDALOffsetProvider("fakepath", 1.0)
            offset = offset_provider.get_offset(
                GeodeticWorldCoordinate(
                    (
                        np.radians(-90.0),
                        np.radians(90.0),
                        np.nan,
                    ),
                ),
            )
            assert offset == pytest.approx(0.25, abs=1e-3)

    def test_orientation_pp(self):
        from aws.osml.gdal.gdal_offset_provider import GDALOffsetProvider
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        with mock.patch(
            "osgeo.gdal.Open",
            TestGDALOffsetProvider.create_gdal_fake_open(
                np.array(
                    [
                        [0, 1],
                        [2, 3],
                    ],
                    dtype=np.uint8,
                ),
                (360.0, -360.0, 0.0,
                 180.0, 0.0, -180.0),  # fmt: skip
            ),
        ):
            offset_provider = GDALOffsetProvider("fakepath", 1.0)
            offset = offset_provider.get_offset(
                GeodeticWorldCoordinate(
                    (
                        np.radians(-90.0),
                        np.radians(90.0),
                        np.nan,
                    ),
                ),
            )
            assert offset == pytest.approx(0.75, abs=1e-3)

    def test_orientation_pn(self):
        from aws.osml.gdal.gdal_offset_provider import GDALOffsetProvider
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        with mock.patch(
            "osgeo.gdal.Open",
            TestGDALOffsetProvider.create_gdal_fake_open(
                np.array(
                    [
                        [0, 1],
                        [2, 3],
                    ],
                    dtype=np.uint8,
                ),
                (360.0, -360.0, 0.0,
                 -180.0, 0.0, 180.0),  # fmt: skip
            ),
        ):
            offset_provider = GDALOffsetProvider("fakepath", 1.0)
            offset = offset_provider.get_offset(
                GeodeticWorldCoordinate(
                    (
                        np.radians(-90.0),
                        np.radians(90.0),
                        np.nan,
                    ),
                ),
            )
            assert offset == pytest.approx(2.75, abs=1e-3)

    def test_orientation_nn(self):
        from aws.osml.gdal.gdal_offset_provider import GDALOffsetProvider
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        with mock.patch(
            "osgeo.gdal.Open",
            TestGDALOffsetProvider.create_gdal_fake_open(
                np.array(
                    [
                        [0, 1],
                        [2, 3],
                    ],
                    dtype=np.uint8,
                ),
                (-360.0, 360.0, 0.0,
                 -180.0, 0.0, 180.0),  # fmt: skip
            ),
        ):
            offset_provider = GDALOffsetProvider("fakepath", 1.0)
            offset = offset_provider.get_offset(
                GeodeticWorldCoordinate(
                    (
                        np.radians(-90.0),
                        np.radians(90.0),
                        np.nan,
                    ),
                ),
            )
            assert offset == pytest.approx(2.25, abs=1e-3)

    @staticmethod
    def create_gdal_fake_open(
        data: npt.NDArray,
        geotransform: npt.ArrayLike,
    ) -> Callable[[str], gdal.Dataset]:
        """Create in-memory dataset from numpy array and transform."""
        # Get dimensions.
        nrows, ncols = data.shape

        # Create dataset in memory.
        driver = gdal.GetDriverByName("MEM")
        f = driver.Create(
            "FAKEFILE",
            ncols,
            nrows,
            1,
            gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype),
        )

        # Set WGS84.
        t = osr.SpatialReference()
        t.SetWellKnownGeogCS("WGS84")
        projection = t.ExportToWkt()
        f.SetProjection(projection)

        # Set data.
        f.SetGeoTransform(geotransform)
        f.GetRasterBand(1).WriteArray(data)

        def gdal_fake_open(filepath):
            return f

        return gdal_fake_open


if __name__ == "__main__":
    unittest.main()
