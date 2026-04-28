#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class OcclusionMode(Enum):
    """Occlusion detection mode for warp grid construction."""

    NONE = "none"
    Z_BUFFER = "z_buffer"


@dataclass(frozen=True)
class WarpGridOptions:
    """Quality/fidelity configuration for warp grid construction."""

    control_points_per_side: int = 16
    remap_interpolation: int = cv2.INTER_LINEAR
    occlusion_mode: OcclusionMode = OcclusionMode.NONE

    FAST: ClassVar["WarpGridOptions"]
    TERRAIN_CORRECTED: ClassVar["WarpGridOptions"]
    VISIBILITY_AWARE: ClassVar["WarpGridOptions"]


WarpGridOptions.FAST = WarpGridOptions(control_points_per_side=4)
WarpGridOptions.TERRAIN_CORRECTED = WarpGridOptions(control_points_per_side=16)
WarpGridOptions.VISIBILITY_AWARE = WarpGridOptions(control_points_per_side=32, occlusion_mode=OcclusionMode.Z_BUFFER)


@dataclass(frozen=True)
class WarpGrid:
    """Dense pixel-correspondence mapping from output tile to source image."""

    map_x: NDArray[np.float32]
    map_y: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    source_bbox: Tuple[int, int, int, int]
    source_resolution_level: int


class GridBuilder(ABC):
    """Produces WarpGrids for tiles in an output grid."""

    def __init__(self, options: WarpGridOptions = WarpGridOptions.TERRAIN_CORRECTED) -> None:
        if options.occlusion_mode == OcclusionMode.Z_BUFFER:
            raise NotImplementedError("Z_BUFFER occlusion mode is not yet implemented")
        self._options = options

    @abstractmethod
    def build(self, tile_row: int, tile_col: int) -> Optional[WarpGrid]:
        """Compute the warp grid for the given tile. Returns None if no coverage."""

    @property
    @abstractmethod
    def tile_limits(self) -> Tuple[int, int, int, int]:
        """(min_row, min_col, max_row, max_col) of tiles with potential coverage."""

    @property
    @abstractmethod
    def tile_size(self) -> Tuple[int, int]:
        """(width, height) in pixels of each output tile."""
