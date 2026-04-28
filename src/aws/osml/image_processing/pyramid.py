#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Read-side pyramid container, lazy pyramid builder, and block iterator.

This module provides:

- :class:`TiledImagePyramid` — a thin container that groups related
  ``ImageAssetProvider`` instances into a multi-resolution pyramid and
  lets callers pick the appropriate level for a given target resolution.
  No computation is performed; the providers are assumed to already
  contain the pixel data (e.g. COG overviews, NITF R-Set levels, or
  the output of :class:`~aws.osml.image_processing.pyramid_builder.PyramidBuilder`).

- :func:`iter_blocks` — a convenience iterator that yields
  ``(block_row, block_col, NDArray)`` tuples from a provider in
  raster (row-major) order, skipping sparse blocks.

- :func:`build_pyramid_levels` — constructs a lazy pyramid as a list
  of chained :class:`~aws.osml.image_processing.downsampled_provider.DownsampledImageProvider`
  instances. No pixels are decoded until ``get_block()`` is called on
  one of the returned providers.

All provider-aware components duck-type on ``ImageAssetProvider``
without requiring a concrete class import.

See :mod:`aws.osml.image_processing.downsampled_provider` for the on-demand
downsampling provider, and
:mod:`aws.osml.image_processing.pyramid_builder` for the write-path
single-pass pyramid generator.
"""

from typing import Any, Iterator, List, Optional, Tuple, Union

from numpy.typing import NDArray

from .cached_provider import CachedImageProvider
from .downsampled_provider import DownsampledImageProvider
from .resample import ResampleFunc
from .retiled_provider import RetiledImageProvider
from .sips_resample import sips_rrds_resample
from .statistics import ImageStatistics, compute_image_statistics
from .tile_cache import TileCache

# Maximum tile dimension inherited from a source provider. Matches the NITF
# NPPBH/NPPBV upper bound; untiled imagery reports full image width as block
# size which can exceed this.
_MAX_TILE_SIZE = 8192


class TiledImagePyramid:
    """Read-side container grouping ImageAssetProviders into a pyramid.

    Level 0 is the highest resolution. This class does no computation —
    it wraps providers that already exist (from COG overviews, NITF
    R-Sets, :class:`PyramidBuilder` output, or
    :func:`build_pyramid_levels`).

    Args:
        levels: A non-empty list of ``ImageAssetProvider`` instances
            ordered from highest resolution (level 0) to lowest.
        scale_factor: The nominal per-level reduction factor.
            Defaults to ``2``.
        reader: Optional ``DatasetReader`` that provides metadata and
            DES segment access. Stored by :meth:`from_dataset`.

    Raises:
        ValueError: If ``levels`` is empty.
    """

    def __init__(self, levels: List[Any], scale_factor: int = 2, reader: Any = None) -> None:
        if not levels:
            raise ValueError("levels must be a non-empty list of providers")
        self._levels = list(levels)
        self._scale_factor = int(scale_factor)
        self._reader = reader

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_levels(self) -> int:
        """Number of pyramid levels."""
        return len(self._levels)

    @property
    def scale_factor(self) -> int:
        """Nominal per-level reduction factor."""
        return self._scale_factor

    @property
    def reader(self) -> Any:
        """The ``DatasetReader`` providing metadata and DES access.

        Returns ``None`` when the pyramid was built via
        :meth:`from_providers` or constructed without a reader.
        """
        return self._reader

    # ------------------------------------------------------------------
    # Level access
    # ------------------------------------------------------------------

    def get_level(self, level: int) -> Any:
        """Return the provider at the given pyramid level.

        Args:
            level: Level index in ``[0, num_levels)``.

        Returns:
            The ``ImageAssetProvider`` at the requested level.

        Raises:
            IndexError: If ``level`` is out of ``[0, num_levels)``.
        """
        if level < 0 or level >= len(self._levels):
            raise IndexError(f"level {level} is out of range [0, {len(self._levels)})")
        return self._levels[level]

    def image_shape_at_level(self, level: int) -> Tuple[int, int, int]:
        """Return ``(num_bands, num_rows, num_columns)`` at the given level.

        Args:
            level: Level index in ``[0, num_levels)``.

        Returns:
            A 3-tuple of ``(num_bands, num_rows, num_columns)``.

        Raises:
            IndexError: If ``level`` is out of ``[0, num_levels)``.
        """
        provider = self.get_level(level)
        return (int(provider.num_bands), int(provider.num_rows), int(provider.num_columns))

    def tile_grid_at_level(self, level: int) -> Tuple[int, int]:
        """Return the block grid size ``(rows, cols)`` at the given level.

        Args:
            level: Level index in ``[0, num_levels)``.

        Returns:
            A 2-tuple of ``(grid_rows, grid_cols)``.

        Raises:
            IndexError: If ``level`` is out of ``[0, num_levels)``.
        """
        provider = self.get_level(level)
        return provider.block_grid_size

    def best_level_for(self, src_size: Tuple[int, int], output_size: Tuple[int, int]) -> int:
        """Pick the deepest pyramid level that avoids upsampling.

        Given a source window of size ``src_size`` (width, height) at R0
        and a desired ``output_size`` (width, height), returns the
        deepest level N where ``src_size / scale_factor^N >= output_size``
        on both axes.

        When no level meets the target (i.e. even level 0's scaled size
        is smaller than output_size), returns ``0``.

        Args:
            src_size: Source window dimensions as ``(width, height)``.
                Both must be > 0.
            output_size: Desired output dimensions as ``(width, height)``.
                Both must be > 0.

        Returns:
            The level index in ``[0, num_levels)``.

        Raises:
            ValueError: If any dimension is <= 0.
        """
        src_w, src_h = src_size
        out_w, out_h = output_size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"src_size dimensions must be > 0, got {src_size}")
        if out_w <= 0 or out_h <= 0:
            raise ValueError(f"output_size dimensions must be > 0, got {output_size}")

        best = 0
        for level in range(self.num_levels):
            divisor = self._scale_factor**level
            scaled_w = src_w // divisor
            scaled_h = src_h // divisor
            if scaled_w >= out_w and scaled_h >= out_h:
                best = level
            else:
                break
        return best

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        level: Optional[int] = None,
        max_pixels: int = 10_000_000,
        num_bins: int = 0,
        bin_edges: Optional[Union[NDArray, List[NDArray]]] = None,
        num_workers: int = 0,
    ) -> ImageStatistics:
        """Compute statistics from the most efficient pyramid level.

        When ``level`` is None, selects the deepest level with at least
        ``max_pixels`` total spatial pixels (~R2-R3 for typical large
        imagery). Gives sub-percent mean/stddev accuracy with minimal I/O.

        Min/max from overview levels are approximate — isolated extrema
        get averaged away by the downsampling kernel. Callers needing
        exact min/max should compute at level 0.

        Args:
            level: Explicit level to compute at. When None, auto-selects
                via :meth:`_best_level_for_statistics`.
            max_pixels: Minimum spatial pixel count for auto-selection.
                Defaults to ``10_000_000``.
            num_bins: Number of histogram bins. See
                :func:`~aws.osml.image_processing.statistics.compute_image_statistics`.
            bin_edges: Explicit bin edges. See
                :func:`~aws.osml.image_processing.statistics.compute_image_statistics`.
            num_workers: Thread count for concurrent block processing.
                See
                :func:`~aws.osml.image_processing.statistics.compute_image_statistics`.

        Returns:
            An :class:`~aws.osml.image_processing.statistics.ImageStatistics`
            instance.
        """
        selected = level if level is not None else self._best_level_for_statistics(max_pixels)
        provider = self.get_level(selected)
        return compute_image_statistics(
            provider,
            num_bins=num_bins,
            bin_edges=bin_edges,
            num_workers=num_workers,
            force_recompute=True,
        )

    def _best_level_for_statistics(self, max_pixels: int) -> int:
        """Pick the coarsest level with at least ``max_pixels`` spatial pixels.

        ``max_pixels`` is a statistical sample-size threshold — the
        minimum number of spatial pixels needed for a reliable estimate
        of per-band mean and stddev. Band count is irrelevant since
        statistics are computed per-band.

        Args:
            max_pixels: Minimum spatial pixel count.

        Returns:
            Level index in ``[0, num_levels)``.
        """
        for level in range(self.num_levels - 1, -1, -1):
            _, rows, cols = self.image_shape_at_level(level)
            if rows * cols >= max_pixels:
                return level
        return 0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_providers(providers: List[Any], scale_factor: int = 2) -> "TiledImagePyramid":
        """Construct a pyramid from an explicit list of providers.

        Args:
            providers: A non-empty list of ``ImageAssetProvider``
                instances ordered from highest to lowest resolution.
            scale_factor: Nominal per-level reduction factor.
                Defaults to ``2``.

        Returns:
            A :class:`TiledImagePyramid` wrapping the providers.

        Raises:
            ValueError: If ``providers`` is empty.
        """
        return TiledImagePyramid(providers, scale_factor)

    @staticmethod
    def from_dataset(reader: Any, base_key: str = "image:0") -> "TiledImagePyramid":
        """Discover overview assets in a dataset reader and build a pyramid.

        Looks for overview assets using the key convention
        ``"{base_key}:overview:{N}"`` for ``N = 1, 2, ...``, and also
        checks for assets with role ``"overview"`` referencing
        ``base_key``. Levels are ordered by decreasing resolution
        (ascending overview index).

        When no overviews are found, returns a single-level pyramid
        containing only the base asset.

        Args:
            reader: A duck-typed ``DatasetReader`` that exposes
                ``get_asset(key)`` and optionally
                ``get_assets_by_role(role)`` or ``list_asset_keys()``.
            base_key: The key identifying the base (full-resolution)
                image asset. Defaults to ``"image:0"``.

        Returns:
            A :class:`TiledImagePyramid` with levels ordered from
            highest to lowest resolution.

        Raises:
            KeyError: If the base asset is not found in the reader.
        """
        base = reader.get_asset(base_key)
        levels = [base]

        # Strategy 1: discover by key convention.
        n = 1
        while True:
            overview_key = f"{base_key}:overview:{n}"  # noqa: E231
            try:
                overview = reader.get_asset(overview_key)
                levels.append(overview)
                n += 1
            except (KeyError, AttributeError):
                break

        # Strategy 2: discover by role (if the reader supports it and
        # we haven't found any overviews yet via key convention).
        if len(levels) == 1:
            get_by_role = getattr(reader, "get_assets_by_role", None)
            if callable(get_by_role):
                try:
                    role_assets = get_by_role("overview")
                    if role_assets:
                        # Sort by decreasing resolution (largest num_rows first).
                        role_assets = sorted(
                            role_assets,
                            key=lambda a: (int(a.num_rows), int(a.num_columns)),
                            reverse=True,
                        )
                        levels.extend(role_assets)
                except (KeyError, AttributeError, TypeError):
                    pass

        return TiledImagePyramid(levels, reader=reader)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def iter_blocks(
    provider: Any,
    order: str = "raster",
    resolution_level: int = 0,
) -> Iterator[Tuple[int, int, NDArray]]:
    """Yield ``(block_row, block_col, block)`` from a provider.

    Iterates the provider's block grid in the specified order, skipping
    sparse blocks (where ``provider.has_block(r, c, resolution_level)``
    returns ``False``).

    Args:
        provider: A duck-typed ``ImageAssetProvider``.
        order: Iteration order. Currently only ``"raster"`` (row-major)
            is supported. Defaults to ``"raster"``.
        resolution_level: Resolution level to read. Defaults to ``0``.

    Yields:
        Tuples of ``(block_row, block_col, NDArray)`` for each
        non-sparse block.
    """
    grid_rows, grid_cols = provider.block_grid_size
    for r in range(grid_rows):
        for c in range(grid_cols):
            if provider.has_block(r, c, resolution_level):
                block = provider.get_block(r, c, resolution_level)
                yield (r, c, block)


def build_pyramid_levels(
    source: Any,
    min_size: int = 256,
    scale_factor: int = 2,
    tile_width: Optional[int] = None,
    tile_height: Optional[int] = None,
    resample_func: Optional[ResampleFunc] = None,
    cache: Optional[TileCache] = None,
) -> List[Any]:
    """Build a lazy pyramid as a chain of DownsampledImageProvider providers.

    The first element of the returned list is the ``source`` itself.
    Each subsequent element is a
    :class:`~aws.osml.image_processing.downsampled_provider.DownsampledImageProvider`
    wrapping the previous element, forming a chain of progressively
    lower-resolution virtual providers. No pixels are decoded until
    ``get_block()`` is called on one of the returned providers.

    Generation stops once either dimension of the current level falls
    strictly below ``min_size``; that level is included as the final
    entry.

    Args:
        source: A duck-typed ``ImageAssetProvider`` for the
            full-resolution image.
        min_size: Minimum dimension threshold. Generation stops when
            either axis falls below this value. Defaults to ``256``.
        scale_factor: Per-level reduction factor. Must be a positive
            power of 2. Defaults to ``2``.
        tile_width: Tile width for the output grid of each
            ``DownsampledImageProvider``. When ``None``, defaults to
            ``ceil(source_block / scale_factor)``.
        tile_height: Tile height for the output grid of each
            ``DownsampledImageProvider``. When ``None``, defaults to
            ``ceil(source_block / scale_factor)``.
        resample_func: A ``ResampleFunc`` callable. Defaults to
            :func:`~aws.osml.image_processing.sips_resample.sips_rrds_resample`
            when ``None``.
        cache: Optional shared :class:`TileCache` for caching computed
            output blocks in each ``DownsampledImageProvider``.

    Returns:
        A list ``[source, level_1, level_2, ...]`` where each
        ``level_i`` (for ``i > 0``) is a ``DownsampledImageProvider``
        whose internal source is ``level_{i-1}``.
    """
    if resample_func is None:
        resample_func = sips_rrds_resample

    tw = tile_width or min(source.num_pixels_per_block_horizontal, _MAX_TILE_SIZE)
    th = tile_height or min(source.num_pixels_per_block_vertical, _MAX_TILE_SIZE)

    if cache is not None and _needs_retiling(source, tw, th):
        source = CachedImageProvider(source, cache=cache)
        source = RetiledImageProvider(source, tile_width=tw, tile_height=th, cache=cache)
    elif _needs_retiling(source, tw, th):
        source = RetiledImageProvider(source, tile_width=tw, tile_height=th)

    levels: List[Any] = [source]
    current = source

    # If the source already has either dimension below min_size, no
    # overview levels are needed.
    if current.num_rows < min_size or current.num_columns < min_size:
        return levels

    while True:
        ds = DownsampledImageProvider(
            current,
            scale_factor=scale_factor,
            resample_func=resample_func,
            tile_width=tile_width,
            tile_height=tile_height,
            cache=cache,
        )
        levels.append(ds)
        # Stop once either dimension drops below min_size; this level
        # is included as the final entry.
        if ds.num_rows < min_size or ds.num_columns < min_size:
            break
        current = ds

    return levels


def _needs_retiling(source: Any, tile_width: int, tile_height: int) -> bool:
    """Return True when the source's block grid doesn't match the requested tile dims."""
    return source.num_pixels_per_block_horizontal != tile_width or source.num_pixels_per_block_vertical != tile_height
