#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from typing import Optional

import numpy as np
from scipy.interpolate import RectBivariateSpline

from aws.osml.io import IO, AssetType
from aws.osml.photogrammetry import ElevationOffsetProvider, GeodeticWorldCoordinate

from ._geo_transform import derive_geo_transform
from ._raster_utils import read_single_band


class RasterOffsetProvider(ElevationOffsetProvider):
    """
    Provide WGS84 elevation offsets from any raster supported by osml-imagery-io.
    Builds a bilinear spline interpolator for fast queries at arbitrary coordinates.
    """

    def __init__(
        self,
        offset_path: str,
        scale_factor: float = 1.0,
        tol: float = 1e-6,
    ) -> None:
        """
        :param offset_path: path to the offset raster file (GeoTIFF or DTED)
        :param scale_factor: multiplier to convert raster values to meters
        :param tol: tolerance in radians for bounds checking
        """
        super().__init__()
        self.offset_path = offset_path
        self.scale_factor = scale_factor
        self.tol = tol
        self.offset_grid: Optional[RectBivariateSpline] = None

    def _initialize_grid(self) -> None:
        """Load the offset raster and build the bilinear interpolation spline."""
        with IO.open(self.offset_path, "r") as reader:
            keys = reader.get_asset_keys(asset_type=AssetType.Image)
            if not keys:
                raise ValueError(f"No image asset found in {self.offset_path}")
            image = reader.get_asset(keys[0])
            metadata = dict(image.metadata) if image.metadata else {}

            gt = derive_geo_transform(metadata)
            if not gt or gt[2] != 0.0 or gt[4] != 0.0:
                raise ValueError(f"GeoTransform {gt} is not a uniform grid.")

            gt = list(gt)
            data = read_single_band(image) * self.scale_factor
            rows, cols = data.shape

            flips = []
            if gt[5] < 0:
                flips.append(0)
                gt[3] += gt[5] * rows
                gt[5] *= -1
            if gt[1] < 0:
                flips.append(1)
                gt[0] += gt[1] * cols
                gt[1] *= -1
            if flips:
                data = np.flip(data, flips)

            self.offset_grid = RectBivariateSpline(
                np.radians(gt[3] + gt[5] / 2 + gt[5] * np.arange(rows)),
                np.radians(gt[0] + gt[1] / 2 + gt[1] * np.arange(cols)),
                data,
                kx=1,
                ky=1,
            )

    def get_offset(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> float:
        """
        Interpolate a WGS84 offset from the grid.

        :param geodetic_world_coordinate: a normalized world coordinate (radians)
        :return: offset in meters
        :raises ValueError: if coordinate is outside valid bounds
        """
        if self.offset_grid is None:
            self._initialize_grid()
        if abs(geodetic_world_coordinate.latitude) > (np.pi / 2 + self.tol) or abs(geodetic_world_coordinate.longitude) > (
            np.pi + self.tol
        ):
            raise ValueError(f"Coordinate {geodetic_world_coordinate} out of bounds.")
        return self.offset_grid(
            geodetic_world_coordinate.latitude,
            geodetic_world_coordinate.longitude,
        )[0][0]
