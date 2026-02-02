#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import numpy as np
from osgeo import gdal
from scipy.interpolate import RectBivariateSpline

from aws.osml.photogrammetry import ElevationOffsetProvider, GeodeticWorldCoordinate


class GDALOffsetProvider(ElevationOffsetProvider):
    """
    Provide WGS84 offsets from any raster format supported by GDAL.

    Notes
    -----
    Reads the datatype as signed even if unsigned. Assumes a GDAL GeoTransform is provided, which must be independent in
    x and y (only scale / offset). Also assumes single raster.
    """

    def __init__(
        self,
        offset_path: str,
        scale_factor: float = 1.0,
    ) -> None:
        """
        Store information for creating the offset grid using any path supported by GDAL.

        :param offset_path: the file path
        :param scale_factor: the amount to multiply by to get meters

        :return: None
        """
        self.offset_path = offset_path
        self.scale_factor = scale_factor
        self.offset_grid = None

    def _initialize_grid(self):
        """
        Initialize the offset grid.
        """
        f = gdal.Open(self.offset_path)
        raster = f.GetRasterBand(1)
        gt = list(f.GetGeoTransform())
        if not gt or gt[2] != 0.0 or gt[4] != 0.0:
            raise ValueError(f"GeoTransform {gt} not uniform grid.")
        data = raster.ReadAsArray() * self.scale_factor
        # The grid needs ascending order.
        flips = []
        if gt[5] < 0:
            flips.append(0)
            gt[3] += gt[5] * raster.YSize
            gt[5] *= -1
        if gt[1] < 0:
            flips.append(1)
            gt[0] += gt[1] * raster.XSize
            gt[1] *= -1
        data = np.flip(data, flips)
        # The grid needs pixel centers and the transform is for corner.
        self.offset_grid = RectBivariateSpline(
            np.radians(gt[3] + gt[5] / 2 + gt[5] * np.arange(raster.YSize)),
            np.radians(gt[0] + gt[1] / 2 + gt[1] * np.arange(raster.XSize)),
            data,
            kx=1,
            ky=1,
        )
        f.Close()

    def get_offset(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> float:
        """
        Interpolate a WGS84 offset from the grid. Raise a ValueError if the coordinate is outside the grid.

        :param geodetic_world_coordinate: a normalized world coordinate of interest

        :return: meters above WGS84 ellipsoid
        """
        if self.offset_grid is None:
            self._initialize_grid()
        return self.offset_grid(
            geodetic_world_coordinate.latitude,
            geodetic_world_coordinate.longitude,
        )[0][0]  # fmt: skip
