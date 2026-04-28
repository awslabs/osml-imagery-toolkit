#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from typing import Any, Optional, Tuple

import cachetools
import cv2
import numpy as np
from numpy.typing import NDArray

from .block_utils import _source_dtype, read_window
from .tile_cache import TileCache
from .warp_grid import GridBuilder, WarpGrid

_WARP_GRID_CACHE_SIZE = 16


class WarpedImageProvider:
    """Presents a source image warped into an arbitrary output coordinate system.

    Satisfies the duck-typed ImageAssetProvider contract. Composable
    with MappedImageProvider, PyramidBuilder, ChipFactory, etc.

    The source may be a single ImageAssetProvider or a TiledImagePyramid.
    When a pyramid is provided, the provider reads from the level
    indicated by ``WarpGrid.source_resolution_level``, enabling
    resolution-aware warping that reads from overviews instead of
    full-resolution data when appropriate.
    """

    def __init__(
        self,
        source: Any,
        grid_builder: GridBuilder,
        num_bands: Optional[int] = None,
        cache: Optional[TileCache] = None,
    ) -> None:
        self._source = source
        self._pyramid = source if hasattr(source, "get_level") else None
        self._grid_builder = grid_builder
        self._num_bands = num_bands if num_bands is not None else int(self._base_source.num_bands)
        self._cache = cache
        self._grid_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=_WARP_GRID_CACHE_SIZE)

    @property
    def _base_source(self) -> Any:
        if self._pyramid is not None:
            return self._pyramid.get_level(0)
        return self._source

    @property
    def key(self) -> str:
        return f"{self._base_source.key}:warped"  # noqa: E231

    @property
    def num_rows(self) -> int:
        min_row, min_col, max_row, max_col = self._grid_builder.tile_limits
        _, tile_h = self._grid_builder.tile_size
        return (max_row - min_row + 1) * tile_h

    @property
    def num_columns(self) -> int:
        min_row, min_col, max_row, max_col = self._grid_builder.tile_limits
        tile_w, _ = self._grid_builder.tile_size
        return (max_col - min_col + 1) * tile_w

    @property
    def num_pixels_per_block_horizontal(self) -> int:
        tile_w, _ = self._grid_builder.tile_size
        return tile_w

    @property
    def num_pixels_per_block_vertical(self) -> int:
        _, tile_h = self._grid_builder.tile_size
        return tile_h

    @property
    def num_bands(self) -> int:
        return self._num_bands

    @property
    def pixel_value_type(self) -> Any:
        return self._base_source.pixel_value_type

    @property
    def num_resolution_levels(self) -> int:
        return 1

    @property
    def tile_limits(self) -> Tuple[int, int, int, int]:
        return self._grid_builder.tile_limits

    @property
    def block_grid_size(self) -> Tuple[int, int]:
        min_row, min_col, max_row, max_col = self._grid_builder.tile_limits
        return (max_row - min_row + 1, max_col - min_col + 1)

    @property
    def metadata(self) -> Any:
        return None

    def get_block(
        self,
        row: int,
        col: int,
        resolution_level: int = 0,
        bands: Optional[Tuple[int, ...]] = None,
    ) -> NDArray:
        if resolution_level != 0:
            raise ValueError(f"WarpedImageProvider only supports resolution_level=0, got {resolution_level}")

        hashable_bands = tuple(bands) if bands is not None else None
        if self._cache is not None:
            cache_key = (self.key, row, col, resolution_level, hashable_bands)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        grid = self._get_grid(row, col)
        out_bands = len(bands) if bands is not None else self._num_bands
        block_w, block_h = self._grid_builder.tile_size
        dtype = _source_dtype(self._base_source)

        if grid is None:
            result = np.zeros((out_bands, block_h, block_w), dtype=dtype)
            if self._cache is not None:
                self._cache.put(cache_key, result)
            return result

        if self._pyramid is not None:
            level = min(grid.source_resolution_level, self._pyramid.num_levels - 1)
            source_for_read = self._pyramid.get_level(level)
        else:
            source_for_read = self._source

        source_pixels = read_window(
            source_for_read,
            window=grid.source_bbox,
            resolution_level=0,
            bands=bands,
        )

        # source_pixels is CHW — transpose to HWC for cv2.remap
        if source_pixels.ndim == 3:
            hwc = np.moveaxis(source_pixels, 0, -1)
        else:
            hwc = source_pixels

        remapped = cv2.remap(
            hwc,
            grid.map_x,
            grid.map_y,
            interpolation=self._grid_builder._options.remap_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Transpose back to CHW
        if remapped.ndim == 3:
            chw = np.moveaxis(remapped, -1, 0)
        elif remapped.ndim == 2:
            chw = remapped[np.newaxis, :, :]
        else:
            chw = remapped

        # Zero-fill invalid pixels
        chw[:, ~grid.valid_mask] = 0

        if self._cache is not None:
            self._cache.put(cache_key, chw)

        return chw

    def get_valid_mask(self, row: int, col: int) -> NDArray[np.bool_]:
        grid = self._get_grid(row, col)
        if grid is None:
            block_w, block_h = self._grid_builder.tile_size
            return np.zeros((block_h, block_w), dtype=np.bool_)
        return grid.valid_mask

    def has_block(self, row: int, col: int, resolution_level: int = 0) -> bool:
        if resolution_level != 0:
            raise ValueError(f"WarpedImageProvider only supports resolution_level=0, got {resolution_level}")
        grid = self._get_grid(row, col)
        return grid is not None

    def _get_grid(self, row: int, col: int) -> Optional[WarpGrid]:
        grid_key = (row, col)
        cached = self._grid_cache.get(grid_key)
        if cached is not None:
            return cached
        grid = self._grid_builder.build(row, col)
        if grid is not None:
            self._grid_cache[grid_key] = grid
        return grid
