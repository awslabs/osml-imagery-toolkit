#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Lazy reduced-resolution view of an ImageAssetProvider.

This module provides :class:`DownsampledImageProvider`, a virtual provider
that wraps a source ``ImageAssetProvider`` and presents a
reduced-resolution tile grid. Blocks are computed on demand — either
by requesting native reduced-resolution blocks from the source (when
available, e.g. J2K wavelet levels) or by reading full-resolution
source blocks and applying a pluggable ``ResampleFunc``.

An optional shared :class:`~aws.osml.image_processing.tile_cache.TileCache`
avoids redundant reads and resampling for repeated requests to the same
block coordinates.

``DownsampledImageProvider`` implements the ``ImageAssetProvider`` interface
(duck-typed) so it can be passed to ``DatasetWriter``,
``MappedImageProvider``, or another ``DownsampledImageProvider`` for
chaining.

See :mod:`aws.osml.image_processing.resample` for the ``ResampleFunc``
protocol and OpenCV-based resamplers, and
:mod:`aws.osml.image_processing.sips_resample` for the default
``sips_rrds_resample``.
"""

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .block_utils import stitch_source_blocks
from .resample import ResampleFunc
from .sips_resample import sips_rrds_resample
from .tile_cache import TileCache


def _halo_pixels(resample_func: ResampleFunc) -> int:
    """Halo width required by ``resample_func`` (0 if not required)."""
    if resample_func is sips_rrds_resample:
        return 5
    return int(getattr(resample_func, "halo_pixels", 0))


def _source_dtype(provider: Any) -> np.dtype:
    """Return the numpy dtype associated with ``provider``'s pixel type.

    Supports osml-imagery-io's ``PixelType`` (exposes
    ``to_numpy_dtype()`` returning a string like ``"uint8"``) and plain
    string / dtype descriptors used by duck-typed mocks.
    """
    from .block_utils import _source_dtype as _shared_source_dtype

    return _shared_source_dtype(provider)


def _source_pad_value(provider: Any) -> float:
    """Return the pad pixel value from a provider, defaulting to 0."""
    return float(getattr(provider, "pad_pixel_value", 0.0))


def _is_power_of_two(n: int) -> bool:
    """Return True if *n* is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0


class DownsampledImageProvider:
    """Lazy reduced-resolution view of an ImageAssetProvider.

    Wraps a source provider and presents a virtual tile grid at
    reduced resolution. Blocks are computed on demand via one of two
    paths:

    - **Native path**: when the source exposes multiple resolution
      levels (e.g. J2K wavelet decomposition), blocks are requested
      at the appropriate ``resolution_level`` and stitched into the
      output tile. No resampling function is invoked.
    - **Resample path**: when the source has only one resolution
      level, full-resolution blocks covering the output tile's
      footprint (plus halo if ``resample_func is sips_rrds_resample``)
      are read, stitched, and passed through ``resample_func``.

    Implements the ``ImageAssetProvider`` interface (duck-typed) so it
    can be passed to ``DatasetWriter``, ``MappedImageProvider``, or
    another ``DownsampledImageProvider`` for chaining.

    Args:
        source: A duck-typed ``ImageAssetProvider`` instance.
        scale_factor: Reduction factor. Must be a positive power of 2.
            Defaults to ``2``.
        resample_func: A ``ResampleFunc`` callable used when the native
            path is not available. Defaults to
            :func:`~aws.osml.image_processing.sips_resample.sips_rrds_resample`
            when ``None``.
        tile_width: Tile width for the output grid. When ``None``,
            defaults to ``ceil(source_block_width / scale_factor)``.
        tile_height: Tile height for the output grid. When ``None``,
            defaults to ``ceil(source_block_height / scale_factor)``.
        cache: Optional shared :class:`TileCache` for caching computed
            output blocks.  When ``None``, no caching is performed.

    Raises:
        ValueError: If ``scale_factor`` is not a power of 2, or if the
            source block dimensions are too small for the resample path.
    """

    def __init__(
        self,
        source: Any,
        scale_factor: int = 2,
        resample_func: Optional[ResampleFunc] = None,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        cache: Optional[TileCache] = None,
    ) -> None:
        if not _is_power_of_two(scale_factor):
            raise ValueError(f"scale_factor must be a positive power of 2, got {scale_factor}")

        self._source = source
        self._scale_factor = int(scale_factor)
        self._resample_func: ResampleFunc = resample_func or sips_rrds_resample
        self._tile_cache = cache

        src_bw = int(source.num_pixels_per_block_horizontal)
        src_bh = int(source.num_pixels_per_block_vertical)

        # Structural guard: resample path requires source blocks >= 2 on each axis.
        log2_scale = int(math.log2(scale_factor))
        is_chained = isinstance(source, DownsampledImageProvider)
        uses_resample_path = is_chained or source.num_resolution_levels <= log2_scale
        if uses_resample_path:
            if src_bh < 2:
                raise ValueError(
                    f"DownsampledImageProvider requires source tile height >= 2, got {src_bh}. "
                    "Wrap the source with RetiledImageProvider to establish a suitable tile grid."
                )
            if src_bw < 2:
                raise ValueError(
                    f"DownsampledImageProvider requires source tile width >= 2, got {src_bw}. "
                    "Wrap the source with RetiledImageProvider to establish a suitable tile grid."
                )

        # Default output tile size: ceil(source_block / scale_factor).
        sf = int(scale_factor)
        self._tile_width = int(tile_width) if tile_width is not None else math.ceil(src_bw / sf)
        self._tile_height = int(tile_height) if tile_height is not None else math.ceil(src_bh / sf)

        # Compute output image dimensions using SIPS even/odd rounding.
        self._num_rows = (source.num_rows + scale_factor - 1) // scale_factor
        self._num_columns = (source.num_columns + scale_factor - 1) // scale_factor

        # Determine whether the native path is available.
        if not is_chained and source.num_resolution_levels > log2_scale:
            self._source_resolution_level: Optional[int] = log2_scale
        else:
            self._source_resolution_level = None

        # Pre-compute the output block grid.
        self._block_grid_rows = (self._num_rows + self._tile_height - 1) // self._tile_height
        self._block_grid_cols = (self._num_columns + self._tile_width - 1) // self._tile_width

        # Cache source metadata for fast access.
        self._num_bands = int(source.num_bands)
        self._dtype = _source_dtype(source)
        self._pad_value = _source_pad_value(source)
        self._halo = _halo_pixels(self._resample_func)

    # ------------------------------------------------------------------
    # ImageAssetProvider interface — properties
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        """Unique key identifying this operator in the processing chain."""
        return f"{self._source.key}:downsample:{self._scale_factor}"

    @property
    def num_rows(self) -> int:
        """Number of pixel rows in the downsampled image."""
        return self._num_rows

    @property
    def num_columns(self) -> int:
        """Number of pixel columns in the downsampled image."""
        return self._num_columns

    @property
    def num_bands(self) -> int:
        """Number of image bands (delegated to source)."""
        return self._num_bands

    @property
    def pixel_value_type(self) -> Any:
        """Pixel value type (delegated to source)."""
        return self._source.pixel_value_type

    @property
    def num_pixels_per_block_horizontal(self) -> int:
        """Tile width in pixels for the output grid."""
        return self._tile_width

    @property
    def num_pixels_per_block_vertical(self) -> int:
        """Tile height in pixels for the output grid."""
        return self._tile_height

    @property
    def num_resolution_levels(self) -> int:
        """Number of resolution levels available.

        When the native path is active, the number of remaining
        resolution levels in the source beyond the one consumed by
        this operation. Otherwise ``1`` (only the resampled level).
        """
        if self._source_resolution_level is not None:
            remaining = self._source.num_resolution_levels - self._source_resolution_level
            return max(remaining, 1)
        return 1

    @property
    def block_grid_size(self) -> Tuple[int, int]:
        """Grid dimensions ``(rows, cols)`` of the output block layout."""
        return (self._block_grid_rows, self._block_grid_cols)

    # ------------------------------------------------------------------
    # ImageAssetProvider interface — methods
    # ------------------------------------------------------------------

    def has_block(self, row: int, col: int, resolution_level: int = 0) -> bool:
        """Check whether a block exists at the given grid position.

        Always returns ``True`` for in-bounds coordinates (the
        downsampled view is never sparse — missing source blocks are
        padded).

        Args:
            row: Block row index.
            col: Block column index.
            resolution_level: Resolution level (default ``0``).

        Returns:
            ``True`` if the coordinates are within the output grid,
            ``False`` otherwise.
        """
        if row < 0 or col < 0 or row >= self._block_grid_rows or col >= self._block_grid_cols:
            return False
        return True

    def get_block(
        self,
        row: int,
        col: int,
        resolution_level: int = 0,
        bands: Union[Tuple[int, ...], None] = None,
    ) -> NDArray:
        """Read a downsampled block at the given grid position.

        Dispatches to the native path or resample path depending on
        whether the source exposes enough resolution levels. Results
        are cached when a :class:`TileCache` is provided.

        Args:
            row: Block row index.
            col: Block column index.
            resolution_level: Resolution level (default ``0``).
            bands: Optional tuple of band indices to read.

        Returns:
            A CHW NDArray of shape
            ``(num_bands, tile_height, tile_width)`` with the source's
            dtype.

        Raises:
            IndexError: If ``(row, col)`` is out of the output grid
                bounds.
            ValueError: If ``resolution_level`` exceeds the available
                levels.
        """
        if row < 0 or col < 0 or row >= self._block_grid_rows or col >= self._block_grid_cols:
            raise IndexError(
                f"Block ({row}, {col}) is out of range for grid ({self._block_grid_rows}, {self._block_grid_cols})"
            )
        if resolution_level < 0 or resolution_level >= self.num_resolution_levels:
            raise ValueError(f"resolution_level {resolution_level} exceeds available levels ({self.num_resolution_levels})")

        hashable_bands = tuple(bands) if bands is not None else None
        cache_key = (self.key, row, col, resolution_level, hashable_bands)
        if self._tile_cache is not None:
            cached = self._tile_cache.get(cache_key)
            if cached is not None:
                return cached

        if self._source_resolution_level is not None:
            result = self._aggregate_native_blocks(row, col, resolution_level, bands)
        else:
            result = self._resample_blocks(row, col, resolution_level, bands)

        if self._tile_cache is not None:
            self._tile_cache.put(cache_key, result)

        return result

    @property
    def metadata(self) -> Any:
        """Return the metadata object from the source provider."""
        return self._source.metadata

    # ------------------------------------------------------------------
    # Internal — native path
    # ------------------------------------------------------------------

    def _aggregate_native_blocks(
        self,
        block_row: int,
        block_col: int,
        resolution_level: int,
        bands: Union[Tuple[int, ...], None],
    ) -> NDArray:
        """Produce an output tile by reading native reduced-resolution
        blocks from the source.

        The source is asked for blocks at
        ``self._source_resolution_level + resolution_level``, which
        maps to the correct wavelet decomposition level for J2K
        sources. Multiple source blocks may need to be stitched when
        the output tile size differs from the source's native block
        size at that resolution level.
        """
        effective_level = self._source_resolution_level + resolution_level

        # Compute the pixel window in the native-level coordinate space.
        # At the native level, the source image dimensions are roughly
        # source.num_rows / scale_factor by source.num_columns / scale_factor.
        y0 = block_row * self._tile_height
        x0 = block_col * self._tile_width
        y1 = min(y0 + self._tile_height, self._num_rows)
        x1 = min(x0 + self._tile_width, self._num_columns)

        # Source tile dimensions at the native level. At reduced
        # resolution levels, block dimensions shrink by the level's
        # scale factor (e.g., level 1 blocks are half-size).
        native_divisor = 2**effective_level
        src_tile_h = max(1, self._source.num_pixels_per_block_vertical // native_divisor)
        src_tile_w = max(1, self._source.num_pixels_per_block_horizontal // native_divisor)

        patch = stitch_source_blocks(
            self._source,
            row_range=(y0, y1),
            col_range=(x0, x1),
            tile_height=src_tile_h,
            tile_width=src_tile_w,
            num_bands=self._num_bands,
            pad_value=self._pad_value,
            dtype=self._dtype,
            resolution_level=effective_level,
        )

        # Pad to full tile size if this is an edge tile.
        if patch.shape[1] != self._tile_height or patch.shape[2] != self._tile_width:
            padded = np.full(
                (self._num_bands, self._tile_height, self._tile_width),
                self._pad_value,
                dtype=self._dtype,
            )
            padded[:, : patch.shape[1], : patch.shape[2]] = patch
            return padded

        return patch

    # ------------------------------------------------------------------
    # Internal — resample path
    # ------------------------------------------------------------------

    def _resample_blocks(
        self,
        block_row: int,
        block_col: int,
        resolution_level: int,
        bands: Union[Tuple[int, ...], None],
    ) -> NDArray:
        """Produce an output tile by reading full-resolution source
        blocks, stitching them, and applying ``resample_func``.

        When ``resample_func is sips_rrds_resample``, a 5-pixel halo
        is requested around the output tile's footprint to support the
        7x7 anti-alias kernel and 4x4 LaGrange interpolation kernel.
        Out-of-image pixels are filled via reflection
        (``numpy.pad(mode='reflect')``), matching SIPS Mirror Edge -
        Odd semantics.
        """
        # Compute the output tile's footprint in source (R0) pixel
        # coordinates. Each output pixel maps to ``scale_factor``
        # source pixels.
        sf = self._scale_factor
        out_y0 = block_row * self._tile_height
        out_x0 = block_col * self._tile_width
        # Actual output tile dimensions (may be smaller at edges).
        out_h = min(self._tile_height, self._num_rows - out_y0)
        out_w = min(self._tile_width, self._num_columns - out_x0)

        # Source footprint for this output tile.
        src_y0 = out_y0 * sf
        src_x0 = out_x0 * sf
        src_y1 = min(src_y0 + out_h * sf, self._source.num_rows)
        src_x1 = min(src_x0 + out_w * sf, self._source.num_columns)

        # Extend by halo if needed.
        halo = self._halo
        ext_y0 = src_y0 - halo
        ext_x0 = src_x0 - halo
        ext_y1 = src_y1 + halo
        ext_x1 = src_x1 + halo

        # Clamp to image bounds for the actual read.
        read_y0 = max(ext_y0, 0)
        read_x0 = max(ext_x0, 0)
        read_y1 = min(ext_y1, self._source.num_rows)
        read_x1 = min(ext_x1, self._source.num_columns)

        src_tile_h = self._source.num_pixels_per_block_vertical
        src_tile_w = self._source.num_pixels_per_block_horizontal

        patch = stitch_source_blocks(
            self._source,
            row_range=(read_y0, read_y1),
            col_range=(read_x0, read_x1),
            tile_height=src_tile_h,
            tile_width=src_tile_w,
            num_bands=self._num_bands,
            pad_value=self._pad_value,
            dtype=self._dtype,
            resolution_level=0,
        )

        # Reflect-pad to fill halo regions that extend beyond the image
        # boundary (Mirror Edge - Odd).
        if halo > 0:
            pad_top = read_y0 - ext_y0
            pad_bottom = ext_y1 - read_y1
            pad_left = read_x0 - ext_x0
            pad_right = ext_x1 - read_x1
            if pad_top or pad_bottom or pad_left or pad_right:
                patch = np.pad(
                    patch,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="reflect",
                )

        # Apply the resample function. When a halo is present the patch
        # is larger than 2*out — resample the full patch at strict 2x then
        # crop the halo contribution from the output.
        src_h = patch.shape[1]
        src_w = patch.shape[2]
        target_rows = (src_h + 1) // 2
        target_cols = (src_w + 1) // 2
        resampled = self._resample_func(patch, target_rows, target_cols)

        if halo > 0:
            halo_out = (halo + sf - 1) // sf
            resampled = resampled[:, halo_out : halo_out + out_h, halo_out : halo_out + out_w]

        # Pad to full tile size if this is an edge tile.
        if resampled.shape[1] != self._tile_height or resampled.shape[2] != self._tile_width:
            padded = np.full(
                (self._num_bands, self._tile_height, self._tile_width),
                self._pad_value,
                dtype=self._dtype,
            )
            padded[:, : resampled.shape[1], : resampled.shape[2]] = resampled
            return padded

        return resampled
