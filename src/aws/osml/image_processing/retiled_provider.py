#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""RetiledImageProvider — virtual tile grid adapter.

This module provides :class:`RetiledImageProvider`, an adapter that wraps
any duck-typed ``ImageAssetProvider`` and presents a virtual tile grid of
configurable dimensions.  Virtual tiles are sliced from source blocks
(when source blocks are larger) or stitched from multiple source blocks
(when source blocks are smaller).

This cleanly separates the physical I/O grid (how the codec stores blocks
on disk) from the logical processing grid (how consumers want to iterate
the image), eliminating pathological tile-size inheritance that previously
caused downstream operators to break.
"""

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .block_utils import _source_dtype, _source_pad_value, stitch_source_blocks
from .tile_cache import TileCache


class RetiledImageProvider:
    """Adapter that presents a virtual tile grid over any source provider.

    Implements the ``ImageAssetProvider`` duck-typed interface.  Virtual
    tiles are sliced from source blocks (when source blocks are larger)
    or stitched from multiple source blocks (when source blocks are
    smaller).  Optionally caches its own output tiles in a shared
    :class:`TileCache`.

    :param source: A duck-typed ``ImageAssetProvider`` instance.
    :param tile_width: Virtual tile width in pixels. Default 1024.
    :param tile_height: Virtual tile height in pixels. Default 1024.
    :param pad_edges: When ``True``, edge tiles are padded to full tile
        dimensions.  When ``False`` (default), edge tiles are partial.
    :param cache: Shared :class:`TileCache` instance, or ``None`` to
        disable output caching.
    """

    def __init__(
        self,
        source: Any,
        tile_width: int = 1024,
        tile_height: int = 1024,
        pad_edges: bool = False,
        cache: Optional[TileCache] = None,
    ) -> None:
        self._source = source
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._pad_edges = pad_edges
        self._cache = cache

        self._src_tile_width = int(source.num_pixels_per_block_horizontal)
        self._src_tile_height = int(source.num_pixels_per_block_vertical)
        self._src_num_rows = int(source.num_rows)
        self._src_num_cols = int(source.num_columns)
        self._src_num_bands = int(source.num_bands)
        self._dtype = _source_dtype(source)
        self._pad_value = _source_pad_value(source)

        self._grid_rows = math.ceil(self._src_num_rows / tile_height)
        self._grid_cols = math.ceil(self._src_num_cols / tile_width)

    @property
    def key(self) -> str:
        """Chain-unique key including retiling parameters."""
        return f"{self._source.key}:retiled:{self._tile_width}x{self._tile_height}"

    @property
    def num_rows(self) -> int:
        """Image height in pixels."""
        if self._pad_edges:
            return self._grid_rows * self._tile_height
        return self._src_num_rows

    @property
    def num_columns(self) -> int:
        """Image width in pixels."""
        if self._pad_edges:
            return self._grid_cols * self._tile_width
        return self._src_num_cols

    @property
    def num_pixels_per_block_horizontal(self) -> int:
        """Virtual tile width."""
        return self._tile_width

    @property
    def num_pixels_per_block_vertical(self) -> int:
        """Virtual tile height."""
        return self._tile_height

    @property
    def num_resolution_levels(self) -> int:
        """Number of usable resolution levels.

        Capped to ``min(source.num_resolution_levels, floor(log2(min(tw, th))) + 1)``
        since higher levels would produce sub-1-pixel tiles.
        """
        max_from_tile = int(math.log2(min(self._tile_width, self._tile_height))) + 1
        return min(int(self._source.num_resolution_levels), max_from_tile)

    @property
    def block_grid_size(self) -> Tuple[int, int]:
        """Grid dimensions ``(rows, cols)`` of the virtual tile layout."""
        return (self._grid_rows, self._grid_cols)

    @property
    def num_bands(self) -> int:
        """Number of image bands."""
        return self._src_num_bands

    @property
    def pixel_value_type(self) -> Any:
        """Pixel value type of the source image."""
        return self._source.pixel_value_type

    @property
    def metadata(self) -> Any:
        """Delegate to source metadata."""
        return self._source.metadata

    def has_block(self, row: int, col: int, resolution_level: int = 0) -> bool:
        """Always ``True`` for valid grid positions."""
        return 0 <= row < self._grid_rows and 0 <= col < self._grid_cols

    def get_block(
        self,
        row: int,
        col: int,
        resolution_level: int = 0,
        bands: Union[Tuple[int, ...], None] = None,
    ) -> NDArray:
        """Return the virtual tile at ``(row, col)``.

        Determines which source block(s) overlap this virtual tile,
        retrieves them via the source's ``get_block()``, and slices/stitches
        the requested region.

        At resolution levels > 0, the spatial position remains the same
        but the returned array shrinks by ``2^level`` on each axis.
        """
        if self._cache is not None:
            hashable_bands = tuple(bands) if bands is not None else None
            cache_key = (self.key, row, col, resolution_level, hashable_bands)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        tile = self._read_virtual_tile(row, col, resolution_level, bands)

        if self._cache is not None:
            self._cache.put(cache_key, tile)

        return tile

    def _read_virtual_tile(
        self,
        row: int,
        col: int,
        resolution_level: int,
        bands: Union[Tuple[int, ...], None],
    ) -> NDArray:
        """Read a virtual tile by stitching from source blocks."""
        scale = 2**resolution_level

        # Pixel coordinates in the full-resolution source image
        y0 = row * self._tile_height
        x0 = col * self._tile_width

        if self._pad_edges:
            out_h = self._tile_height // scale
            out_w = self._tile_width // scale
        else:
            y1 = min(y0 + self._tile_height, self._src_num_rows)
            x1 = min(x0 + self._tile_width, self._src_num_cols)
            actual_h = y1 - y0
            actual_w = x1 - x0
            out_h = math.ceil(actual_h / scale)
            out_w = math.ceil(actual_w / scale)

        if resolution_level == 0:
            return self._read_level0(row, col, bands)

        # For resolution levels > 0, delegate to source's resolution level support
        return self._read_higher_level(row, col, resolution_level, bands, out_h, out_w)

    def _read_level0(
        self,
        row: int,
        col: int,
        bands: Union[Tuple[int, ...], None],
    ) -> NDArray:
        """Read a virtual tile at full resolution (level 0)."""
        y0 = row * self._tile_height
        x0 = col * self._tile_width

        if self._pad_edges:
            out_h = self._tile_height
            out_w = self._tile_width
        else:
            y1 = min(y0 + self._tile_height, self._src_num_rows)
            x1 = min(x0 + self._tile_width, self._src_num_cols)
            out_h = y1 - y0
            out_w = x1 - x0

        num_bands = len(bands) if bands is not None else self._src_num_bands

        # Fast path: if virtual tile aligns exactly with a single source block
        if self._tile_width == self._src_tile_width and self._tile_height == self._src_tile_height and not self._pad_edges:
            block = self._source.get_block(row, col, resolution_level=0, bands=bands)
            return block

        # General path: stitch from source blocks
        tile = stitch_source_blocks(
            provider=self._source,
            row_range=(y0, y0 + out_h),
            col_range=(x0, x0 + out_w),
            tile_height=self._src_tile_height,
            tile_width=self._src_tile_width,
            num_bands=self._src_num_bands,
            pad_value=self._pad_value,
            dtype=self._dtype,
            resolution_level=0,
            bands=bands,
        )

        # If pad_edges=True and we got a partial tile, pad it
        if self._pad_edges and (tile.shape[1] < self._tile_height or tile.shape[2] < self._tile_width):
            padded = np.full((num_bands, self._tile_height, self._tile_width), self._pad_value, dtype=self._dtype)
            padded[:, : tile.shape[1], : tile.shape[2]] = tile
            return padded

        return tile

    def _read_higher_level(
        self,
        row: int,
        col: int,
        resolution_level: int,
        bands: Union[Tuple[int, ...], None],
        out_h: int,
        out_w: int,
    ) -> NDArray:
        """Read a virtual tile at a reduced resolution level.

        The spatial position (row, col) is the same as level 0.  The
        source is queried at the given resolution level, and we slice
        the corresponding reduced-resolution region.
        """
        scale = 2**resolution_level
        num_bands = len(bands) if bands is not None else self._src_num_bands

        # Source block dimensions at this resolution level
        src_tile_h_at_level = max(1, self._src_tile_height // scale)
        src_tile_w_at_level = max(1, self._src_tile_width // scale)

        # Pixel coordinates at this resolution level
        y0 = (row * self._tile_height) // scale
        x0 = (col * self._tile_width) // scale

        tile = stitch_source_blocks(
            provider=self._source,
            row_range=(y0, y0 + out_h),
            col_range=(x0, x0 + out_w),
            tile_height=src_tile_h_at_level,
            tile_width=src_tile_w_at_level,
            num_bands=self._src_num_bands,
            pad_value=self._pad_value,
            dtype=self._dtype,
            resolution_level=resolution_level,
            bands=bands,
        )

        # If pad_edges=True and tile is smaller than expected, pad it
        expected_h = self._tile_height // scale
        expected_w = self._tile_width // scale
        if self._pad_edges and (tile.shape[1] < expected_h or tile.shape[2] < expected_w):
            padded = np.full((num_bands, expected_h, expected_w), self._pad_value, dtype=self._dtype)
            padded[:, : tile.shape[1], : tile.shape[2]] = tile
            return padded

        return tile
