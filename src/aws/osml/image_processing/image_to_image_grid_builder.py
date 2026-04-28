#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import math
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..photogrammetry import ElevationModel, ImageCoordinate, SensorModel
from .warp_grid import GridBuilder, WarpGrid, WarpGridOptions

logger = logging.getLogger(__name__)


class ImageToImageGridBuilder(GridBuilder):
    """Maps output blocks in a target image's pixel space to source pixels.

    For each output block in the target image, evaluates the target sensor
    model at control points to get world coordinates, then projects those
    world coordinates back into the source image via the source sensor model.
    """

    def __init__(
        self,
        source_sensor_model: SensorModel,
        target_sensor_model: SensorModel,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        elevation_model: Optional[ElevationModel] = None,
        block_width: int = 1024,
        block_height: int = 1024,
        options: WarpGridOptions = WarpGridOptions.TERRAIN_CORRECTED,
    ) -> None:
        super().__init__(options)
        self._source_sensor_model = source_sensor_model
        self._target_sensor_model = target_sensor_model
        self._source_width = source_width
        self._source_height = source_height
        self._target_width = target_width
        self._target_height = target_height
        self._elevation_model = elevation_model
        self._block_width = block_width
        self._block_height = block_height

    @property
    def tile_limits(self) -> Tuple[int, int, int, int]:
        max_row = math.ceil(self._target_height / self._block_height) - 1
        max_col = math.ceil(self._target_width / self._block_width) - 1
        return (0, 0, max_row, max_col)

    @property
    def tile_size(self) -> Tuple[int, int]:
        return (self._block_width, self._block_height)

    def build(self, row: int, col: int) -> Optional[WarpGrid]:
        n = self._options.control_points_per_side

        px0 = col * self._block_width
        py0 = row * self._block_height

        actual_w = min(self._block_width, self._target_width - px0)
        actual_h = min(self._block_height, self._target_height - py0)
        if actual_w <= 0 or actual_h <= 0:
            return None

        ctrl_x = np.linspace(px0, px0 + actual_w - 1, n)
        ctrl_y = np.linspace(py0, py0 + actual_h - 1, n)

        src_x = np.empty(n * n, dtype=np.float64)
        src_y = np.empty(n * n, dtype=np.float64)

        for i in range(n):
            for j in range(n):
                target_coord = ImageCoordinate([ctrl_x[j], ctrl_y[i]])
                try:
                    world_coord = self._target_sensor_model.image_to_world(target_coord, self._elevation_model)
                    source_coord = self._source_sensor_model.world_to_image(world_coord)
                except Exception:
                    return None
                idx = i * n + j
                src_x[idx] = source_coord.x
                src_y[idx] = source_coord.y

        src_x_grid = src_x.reshape(n, n)
        src_y_grid = src_y.reshape(n, n)

        check_indices = [0, n - 1, n * (n - 1), n * n - 1, n * n // 2]
        has_overlap = False
        for idx in check_indices:
            px, py = src_x[idx], src_y[idx]
            if 0 <= px < self._source_width and 0 <= py < self._source_height:
                has_overlap = True
                break
        if not has_overlap:
            return None

        source_resolution_level = 0

        scale = 2**source_resolution_level
        scaled_x = src_x_grid / scale
        scaled_y = src_y_grid / scale

        min_sx = max(0, int(math.floor(scaled_x.min())))
        min_sy = max(0, int(math.floor(scaled_y.min())))
        max_sx = int(math.ceil(scaled_x.max())) + 1
        max_sy = int(math.ceil(scaled_y.max())) + 1
        bbox_w = max_sx - min_sx
        bbox_h = max_sy - min_sy
        source_bbox = (min_sx, min_sy, bbox_w, bbox_h)

        local_x = scaled_x - min_sx
        local_y = scaled_y - min_sy

        ctrl_rows = np.linspace(0, actual_h - 1, n)
        ctrl_cols = np.linspace(0, actual_w - 1, n)

        spline_x = RectBivariateSpline(ctrl_rows, ctrl_cols, local_x, kx=1, ky=1)
        spline_y = RectBivariateSpline(ctrl_rows, ctrl_cols, local_y, kx=1, ky=1)

        dense_rows = np.arange(actual_h)
        dense_cols = np.arange(actual_w)
        map_x = spline_x(dense_rows, dense_cols).astype(np.float32)
        map_y = spline_y(dense_rows, dense_cols).astype(np.float32)

        valid_mask = (map_x >= 0) & (map_x < bbox_w) & (map_y >= 0) & (map_y < bbox_h)

        if actual_h < self._block_height or actual_w < self._block_width:
            full_map_x = np.zeros((self._block_height, self._block_width), dtype=np.float32)
            full_map_y = np.zeros((self._block_height, self._block_width), dtype=np.float32)
            full_valid = np.zeros((self._block_height, self._block_width), dtype=np.bool_)
            full_map_x[:actual_h, :actual_w] = map_x
            full_map_y[:actual_h, :actual_w] = map_y
            full_valid[:actual_h, :actual_w] = valid_mask
            map_x = full_map_x
            map_y = full_map_y
            valid_mask = full_valid

        return WarpGrid(
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid_mask,
            source_bbox=source_bbox,
            source_resolution_level=source_resolution_level,
        )
