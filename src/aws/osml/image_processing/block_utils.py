#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Block-stitching utilities for assembling pixel windows from block-based providers.

This module provides public functions for reading arbitrary rectangular
pixel regions from an :class:`ImageAssetProvider` (or any duck-typed
equivalent) that exposes a block grid. The core routines —
:func:`stitch_source_blocks` and :func:`read_block_or_pad` — were
originally private helpers in :mod:`pyramid_builder` and
:mod:`downsample`; they are promoted here so that multiple v2.0
components (tile factory, DEM tile factory, elevation offset provider)
can import them directly.

:func:`read_window` is a convenience wrapper that infers block
parameters from the provider, reducing boilerplate at call sites.
"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

_PIXEL_TYPE_STR_MAP = {
    "UINT8": np.uint8,
    "BYTE": np.uint8,
    "INT8": np.int8,
    "UINT16": np.uint16,
    "INT16": np.int16,
    "UINT32": np.uint32,
    "INT32": np.int32,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
    "COMPLEX64": np.complex64,
    "COMPLEX128": np.complex128,
}


def _source_dtype(provider: Any) -> np.dtype:
    """Return the numpy dtype associated with ``provider``'s pixel type.

    Supports osml-imagery-io's ``PixelType`` (exposes
    ``to_numpy_dtype()`` returning a string like ``"uint8"``) and plain
    string / dtype descriptors used by duck-typed mocks.
    """
    pvt = provider.pixel_value_type
    to_numpy = getattr(pvt, "to_numpy_dtype", None)
    if callable(to_numpy):
        return np.dtype(to_numpy())
    if hasattr(pvt, "numpy_dtype"):
        return np.dtype(pvt.numpy_dtype)
    try:
        return np.dtype(pvt)
    except TypeError:
        if isinstance(pvt, str):
            np_type = _PIXEL_TYPE_STR_MAP.get(pvt.upper())
            if np_type is not None:
                return np.dtype(np_type)
        return np.dtype(np.uint8)


def _source_pad_value(provider: Any) -> float:
    """Return ``provider.pad_pixel_value`` or ``0.0`` if unavailable."""
    return float(getattr(provider, "pad_pixel_value", 0.0))


def read_block_or_pad(
    provider: Any,
    tile_row: int,
    tile_col: int,
    resolution_level: int,
    tile_height: int,
    tile_width: int,
    num_bands: int,
    pad_value: float,
    dtype: np.dtype,
) -> NDArray:
    """Return ``provider.get_block()`` when the block exists; else a pad-filled tile.

    Parameters
    ----------
    provider : Any
        Duck-typed ImageAssetProvider.
    tile_row : int
        Block row index.
    tile_col : int
        Block column index.
    resolution_level : int
        Resolution level for the read.
    tile_height : int
        Expected block height.
    tile_width : int
        Expected block width.
    num_bands : int
        Number of image bands.
    pad_value : float
        Fill value for sparse blocks.
    dtype : np.dtype
        Output array dtype.

    Returns
    -------
    NDArray
        CHW array of shape (num_bands, tile_height, tile_width).
    """
    if provider.has_block(tile_row, tile_col, resolution_level):
        return provider.get_block(tile_row, tile_col, resolution_level)
    return np.full((num_bands, tile_height, tile_width), pad_value, dtype=dtype)


def stitch_source_blocks(
    provider: Any,
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
    tile_height: int,
    tile_width: int,
    num_bands: int,
    pad_value: float,
    dtype: np.dtype,
    resolution_level: int,
    bands: Optional[Tuple[int, ...]] = None,
) -> NDArray:
    """Assemble an arbitrary pixel window from a block-based provider.

    Reads every block overlapping the requested window and copies the
    overlapping region into the output. Sparse blocks (has_block == False)
    and out-of-bounds regions contribute pad pixels. The output shape
    matches the requested window exactly.

    Out-of-bounds windows (partially or fully beyond image dimensions,
    or with negative coordinates) are handled gracefully: the valid
    portion is read normally, and the remainder is filled with
    ``pad_value``.

    Parameters
    ----------
    provider : Any
        Duck-typed ImageAssetProvider with get_block() and has_block().
    row_range : tuple[int, int]
        (y_start, y_end) pixel rows — half-open interval.
    col_range : tuple[int, int]
        (x_start, x_end) pixel columns — half-open interval.
    tile_height : int
        Provider's block height in pixels.
    tile_width : int
        Provider's block width in pixels.
    num_bands : int
        Number of image bands.
    pad_value : float
        Fill value for sparse blocks and out-of-bounds regions.
    dtype : np.dtype
        Output array dtype.
    resolution_level : int
        Resolution level passed to get_block / has_block.
    bands : tuple[int, ...], optional
        Band indices to select from each block. When provided, only
        these bands are included in the output (output channel dim
        equals ``len(bands)``). When None, all bands are returned.

    Returns
    -------
    NDArray
        CHW array of shape (out_bands, y_end - y_start, x_end - x_start)
        where out_bands is ``len(bands)`` if bands is provided, else
        num_bands.
    """
    y0, y1 = row_range
    x0, x1 = col_range
    out_h = y1 - y0
    out_w = x1 - x0
    out_bands = len(bands) if bands is not None else num_bands
    out = np.full((out_bands, out_h, out_w), pad_value, dtype=dtype)
    if out_h <= 0 or out_w <= 0:
        return out

    # Determine valid block grid bounds from provider dimensions.
    num_rows = getattr(provider, "num_rows", None)
    num_cols = getattr(provider, "num_columns", None)

    # Clamp block iteration to the valid grid to prevent overflow in the
    # provider's has_block/get_block when coordinates exceed image bounds.
    first_block_row = y0 // tile_height if y0 >= 0 else -((-y0 - 1) // tile_height + 1)
    last_block_row = (y1 - 1) // tile_height if y1 > 0 else -1
    first_block_col = x0 // tile_width if x0 >= 0 else -((-x0 - 1) // tile_width + 1)
    last_block_col = (x1 - 1) // tile_width if x1 > 0 else -1

    if num_rows is not None:
        max_block_row = (num_rows - 1) // tile_height if num_rows > 0 else -1
        first_block_row = max(first_block_row, 0)
        last_block_row = min(last_block_row, max_block_row)

    if num_cols is not None:
        max_block_col = (num_cols - 1) // tile_width if num_cols > 0 else -1
        first_block_col = max(first_block_col, 0)
        last_block_col = min(last_block_col, max_block_col)

    band_indices = list(bands) if bands is not None else None

    for br in range(first_block_row, last_block_row + 1):
        for bc in range(first_block_col, last_block_col + 1):
            if provider.has_block(br, bc, resolution_level):
                block = provider.get_block(br, bc, resolution_level)
            else:
                block = np.full((num_bands, tile_height, tile_width), pad_value, dtype=dtype)

            if band_indices is not None:
                block = block[band_indices, :, :]

            # Pixel-space bounds of this block.
            block_y0 = br * tile_height
            block_x0 = bc * tile_width
            block_y1 = block_y0 + block.shape[1]
            block_x1 = block_x0 + block.shape[2]

            # Overlap between block and requested window.
            overlap_y0 = max(y0, block_y0)
            overlap_x0 = max(x0, block_x0)
            overlap_y1 = min(y1, block_y1)
            overlap_x1 = min(x1, block_x1)

            # Source slice inside the block.
            src_y0 = overlap_y0 - block_y0
            src_x0 = overlap_x0 - block_x0
            src_y1 = overlap_y1 - block_y0
            src_x1 = overlap_x1 - block_x0

            # Destination slice inside the output window.
            dst_y0 = overlap_y0 - y0
            dst_x0 = overlap_x0 - x0
            dst_y1 = overlap_y1 - y0
            dst_x1 = overlap_x1 - x0

            out[:, dst_y0:dst_y1, dst_x0:dst_x1] = block[:, src_y0:src_y1, src_x0:src_x1]
    return out


def read_window(
    provider: Any,
    window: Tuple[int, int, int, int],
    resolution_level: int = 0,
    fill_value: float = 0,
    bands: Optional[Tuple[int, ...]] = None,
) -> NDArray:
    """Read an arbitrary rectangular window from a block-based source.

    Convenience wrapper around :func:`stitch_source_blocks` that infers
    block parameters from the provider.

    Parameters
    ----------
    provider : Any
        Duck-typed ImageAssetProvider exposing num_pixels_per_block_vertical,
        num_pixels_per_block_horizontal, num_bands, and pixel_value_type.
    window : tuple[int, int, int, int]
        Pixel region as (x, y, width, height).
    resolution_level : int
        Resolution level for block decoding. Default 0.
    fill_value : float
        Value for sparse block regions. Default 0.
    bands : tuple[int, ...], optional
        Band indices to select. When provided, only these bands are
        returned (output channel dim equals ``len(bands)``). When None,
        all bands are returned.

    Returns
    -------
    NDArray
        CHW array of shape (out_bands, height, width) where out_bands
        is ``len(bands)`` if bands is provided, else num_bands.
    """
    x, y, w, h = window
    tile_height = int(provider.num_pixels_per_block_vertical)
    tile_width = int(provider.num_pixels_per_block_horizontal)
    num_bands = int(provider.num_bands)
    dtype = _source_dtype(provider)

    return stitch_source_blocks(
        provider=provider,
        row_range=(y, y + h),
        col_range=(x, x + w),
        tile_height=tile_height,
        tile_width=tile_width,
        num_bands=num_bands,
        pad_value=fill_value,
        dtype=dtype,
        resolution_level=resolution_level,
        bands=bands,
    )
