#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import math
from typing import Optional, Tuple

import numpy as np
import pyproj
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import distance_transform_edt

from ..photogrammetry import ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, SensorModel
from .map_tileset import MapTileId, MapTileSet
from .warp_grid import GridBuilder, WarpGrid, WarpGridOptions

logger = logging.getLogger(__name__)


class OrthoGridBuilder(GridBuilder):
    """Maps output tiles from a MapTileSet to source pixels via a sensor model.

    The output geometry is defined entirely by the tile set and tile matrix level.
    This builder is responsible only for computing the backward-projection mapping
    from output tile pixels to source image coordinates.

    Each call to build(tile_row, tile_col) produces a WarpGrid that maps one
    tile of the output raster back to source image coordinates via the sensor model.
    Tiles that extend past the source footprint produce partial output with
    invalid pixels marked in valid_mask.
    """

    def __init__(
        self,
        tile_set: MapTileSet,
        tile_matrix: int,
        sensor_model: SensorModel,
        source_width: int,
        source_height: int,
        elevation_model: Optional[ElevationModel] = None,
        options: WarpGridOptions = WarpGridOptions.TERRAIN_CORRECTED,
        num_source_levels: int = 1,
    ) -> None:
        """
        :param tile_set: The MapTileSet defining output tile geometry.
        :param tile_matrix: The tile matrix (zoom level) to operate at.
        :param sensor_model: Converts between image pixels and world coordinates.
        :param source_width: Width of the source image in pixels (full resolution).
        :param source_height: Height of the source image in pixels (full resolution).
        :param elevation_model: Optional terrain model for height-corrected projection.
        :param options: Control point density and interpolation quality settings.
        :param num_source_levels: Number of resolution levels available in the source
            image pyramid.
        """
        super().__init__(options)
        self._tile_set = tile_set
        self._tile_matrix = tile_matrix
        self._sensor_model = sensor_model
        self._source_width = source_width
        self._source_height = source_height
        self._elevation_model = elevation_model
        self._num_source_levels = num_source_levels

        # Resolve tile set CRS
        crs = pyproj.CRS.from_user_input(tile_set.crs_id)
        self._crs = crs
        crs_4326 = pyproj.CRS.from_epsg(4326)

        # Create transformers for CRS conversions
        if crs == crs_4326 or (crs.is_geographic and crs.to_epsg() == 4326):
            self._to_4326 = None
            self._from_4326 = None
        else:
            self._to_4326 = pyproj.Transformer.from_crs(crs, crs_4326, always_xy=True)
            self._from_4326 = pyproj.Transformer.from_crs(crs_4326, crs, always_xy=True)

        # Project image corners → WGS84 → compute tile_limits
        corners_wgs84 = self._project_image_corners()
        self._tile_limits = tile_set.get_tile_matrix_limits_for_area(corners_wgs84, tile_matrix)

        # Compute per-matrix output GSD in native CRS units
        self._output_gsd = self._compute_output_gsd()

        # Estimate source GSD in native CRS units (one-time, at image center)
        self._source_gsd = self._estimate_source_gsd()

    @property
    def tile_limits(self) -> Tuple[int, int, int, int]:
        return self._tile_limits

    @property
    def tile_size(self) -> Tuple[int, int]:
        sample_tile = self._tile_set.get_tile(MapTileId(self._tile_matrix, self._tile_limits[0], self._tile_limits[1]))
        return (sample_tile.size[0], sample_tile.size[1])

    def build(self, tile_row: int, tile_col: int) -> Optional[WarpGrid]:
        """Compute the warp grid for output tile at (tile_row, tile_col).

        Returns None if the tile has no coverage in the source image.
        Handles partial control-point failures gracefully — edge tiles that
        extend past the source footprint produce valid output with correct
        valid_mask.
        """
        n = self._options.control_points_per_side

        # --- Step 1: Get tile geometry from tile set ---
        tile = self._tile_set.get_tile(MapTileId(self._tile_matrix, tile_row, tile_col))
        tile_w, tile_h = tile.size[0], tile.size[1]
        native_bounds = tile.native_bounds  # (xmin, ymin, xmax, ymax)

        # --- Step 2: Build control-point grid in native CRS ---
        x0, y0, x1, y1 = native_bounds
        # x increases left-to-right, y decreases top-to-bottom (y1 is top, y0 is bottom)
        ctrl_x = np.linspace(x0, x1, n)
        ctrl_y = np.linspace(y1, y0, n)  # top to bottom
        grid_x, grid_y = np.meshgrid(ctrl_x, ctrl_y)

        # --- Step 3: Transform to WGS84 radians ---
        if self._to_4326 is not None:
            lon_deg, lat_deg = self._to_4326.transform(grid_x.ravel(), grid_y.ravel())
            if np.any(np.isinf(lon_deg)) or np.any(np.isinf(lat_deg)):
                return None
        else:
            # CRS is already 4326 — native coords are degrees
            lon_deg = grid_x.ravel().copy()
            lat_deg = grid_y.ravel().copy()

        lon_rad = np.radians(lon_deg)
        lat_rad = np.radians(lat_deg)

        # --- Step 4: Project each control point to source image pixels ---
        src_x = np.full(n * n, np.nan, dtype=np.float64)
        src_y = np.full(n * n, np.nan, dtype=np.float64)

        for i in range(n * n):
            if np.isnan(lon_rad[i]) or np.isnan(lat_rad[i]):
                continue
            world_coord = GeodeticWorldCoordinate([lon_rad[i], lat_rad[i], 0.0])
            if self._elevation_model is not None:
                self._elevation_model.set_elevation(world_coord)
            try:
                img_coord = self._sensor_model.world_to_image(world_coord)
                src_x[i] = img_coord.x
                src_y[i] = img_coord.y
            except Exception:
                pass  # Leave as NaN

        # Check how many control points succeeded
        valid_points = ~np.isnan(src_x)
        num_valid = valid_points.sum()
        if num_valid == 0:
            return None

        # --- Step 5: Overlap check ---
        valid_src_x = src_x[valid_points]
        valid_src_y = src_y[valid_points]
        has_overlap = np.any(
            (valid_src_x >= -self._source_width)
            & (valid_src_x < 2 * self._source_width)
            & (valid_src_y >= -self._source_height)
            & (valid_src_y < 2 * self._source_height)
        )
        if not has_overlap:
            return None

        src_x_grid = src_x.reshape(n, n)
        src_y_grid = src_y.reshape(n, n)

        # --- Step 6: Fill NaN control points via nearest-neighbor ---
        nan_mask = np.isnan(src_x_grid)
        if nan_mask.any():
            src_x_grid = self._fill_nans_nearest(src_x_grid, nan_mask)
            src_y_grid = self._fill_nans_nearest(src_y_grid, nan_mask)

        # --- Step 7: Pyramid level selection ---
        source_resolution_level = self._select_resolution_level()

        # Scale source coordinates to the selected pyramid level
        scale = 2**source_resolution_level
        scaled_x = src_x_grid / scale
        scaled_y = src_y_grid / scale

        # Compute source bounding box at selected level, clamped to source dimensions
        level_w = int(math.ceil(self._source_width / scale))
        level_h = int(math.ceil(self._source_height / scale))
        min_sx = max(0, int(math.floor(np.nanmin(scaled_x))))
        min_sy = max(0, int(math.floor(np.nanmin(scaled_y))))
        max_sx = min(level_w, int(math.ceil(np.nanmax(scaled_x))) + 1)
        max_sy = min(level_h, int(math.ceil(np.nanmax(scaled_y))) + 1)
        bbox_w = max(1, max_sx - min_sx)
        bbox_h = max(1, max_sy - min_sy)
        source_bbox = (min_sx, min_sy, bbox_w, bbox_h)

        # Convert to coordinates local to the source bbox
        local_x = scaled_x - min_sx
        local_y = scaled_y - min_sy

        # --- Step 8: Interpolate sparse control grid → dense pixel map ---
        ctrl_rows = np.linspace(0, tile_h - 1, n)
        ctrl_cols = np.linspace(0, tile_w - 1, n)

        spline_x = RectBivariateSpline(ctrl_rows, ctrl_cols, local_x, kx=1, ky=1)
        spline_y = RectBivariateSpline(ctrl_rows, ctrl_cols, local_y, kx=1, ky=1)

        dense_rows = np.arange(tile_h)
        dense_cols = np.arange(tile_w)
        map_x = spline_x(dense_rows, dense_cols).astype(np.float32)
        map_y = spline_y(dense_rows, dense_cols).astype(np.float32)

        # Mark pixels that map outside the source bbox as invalid
        valid_mask = (map_x >= 0) & (map_x < bbox_w) & (map_y >= 0) & (map_y < bbox_h)

        # Also mark pixels that correspond to NaN control-point regions as invalid
        if nan_mask.any():
            nan_mask_dense = self._expand_nan_mask(nan_mask, tile_h, tile_w)
            valid_mask = valid_mask & ~nan_mask_dense

        return WarpGrid(
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid_mask,
            source_bbox=source_bbox,
            source_resolution_level=source_resolution_level,
        )

    def _project_image_corners(self) -> list:
        """Project the four image corners through the sensor model to WGS84."""
        corners = [
            ImageCoordinate([0.0, 0.0]),
            ImageCoordinate([self._source_width - 1.0, 0.0]),
            ImageCoordinate([self._source_width - 1.0, self._source_height - 1.0]),
            ImageCoordinate([0.0, self._source_height - 1.0]),
        ]
        world_corners = []
        for corner in corners:
            world = self._sensor_model.image_to_world(corner, self._elevation_model)
            world_corners.append(world)
        return world_corners

    def _compute_output_gsd(self) -> float:
        """Compute the output GSD in native CRS units from the tile geometry."""
        min_row, min_col, _, _ = self._tile_limits
        tile = self._tile_set.get_tile(MapTileId(self._tile_matrix, min_row, min_col))
        x0, y0, x1, y1 = tile.native_bounds
        return (x1 - x0) / tile.size[0]

    def _estimate_source_gsd(self) -> float:
        """Estimate source image GSD in tile set's native CRS units."""
        center = ImageCoordinate([self._source_width / 2.0, self._source_height / 2.0])
        right = ImageCoordinate([self._source_width / 2.0 + 1.0, self._source_height / 2.0])

        world_center = self._sensor_model.image_to_world(center, self._elevation_model)
        world_right = self._sensor_model.image_to_world(right, self._elevation_model)

        cx_deg = math.degrees(world_center.longitude)
        cy_deg = math.degrees(world_center.latitude)
        rx_deg = math.degrees(world_right.longitude)
        ry_deg = math.degrees(world_right.latitude)

        if self._from_4326 is not None:
            tx_c, ty_c = self._from_4326.transform(cx_deg, cy_deg)
            tx_r, ty_r = self._from_4326.transform(rx_deg, ry_deg)
        else:
            tx_c, ty_c = cx_deg, cy_deg
            tx_r, ty_r = rx_deg, ry_deg

        dx = tx_r - tx_c
        dy = ty_r - ty_c
        return math.sqrt(dx * dx + dy * dy)

    def _select_resolution_level(self) -> int:
        """Choose the coarsest source pyramid level with sufficient resolution."""
        if self._num_source_levels <= 1:
            return 0
        if self._source_gsd <= 0:
            return 0
        ratio = self._output_gsd / self._source_gsd
        if ratio <= 1.0:
            return 0
        level = int(math.floor(math.log2(ratio)))
        return min(level, self._num_source_levels - 1)

    @staticmethod
    def _fill_nans_nearest(grid: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
        """Fill NaN entries by nearest-neighbor from valid points."""
        filled = grid.copy()
        if not nan_mask.any():
            return filled
        # distance_transform_edt gives indices of nearest valid cell
        _, indices = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        filled[nan_mask] = grid[indices[0][nan_mask], indices[1][nan_mask]]
        return filled

    @staticmethod
    def _expand_nan_mask(nan_mask: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
        """Expand the n×n NaN mask to dense tile dimensions via nearest-neighbor."""
        n = nan_mask.shape[0]
        # Map each dense pixel to its nearest control point
        row_indices = np.round(np.linspace(0, n - 1, tile_h)).astype(int)
        col_indices = np.round(np.linspace(0, n - 1, tile_w)).astype(int)
        return nan_mask[np.ix_(row_indices, col_indices)]
