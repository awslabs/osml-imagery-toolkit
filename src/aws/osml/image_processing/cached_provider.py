#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""CachedImageProvider — transparent decode-caching wrapper.

This module provides :class:`CachedImageProvider`, a pass-through wrapper
that caches :meth:`get_block` results from any source in a shared
:class:`~aws.osml.image_processing.tile_cache.TileCache`.  All properties
and methods delegate to the source unchanged — the wrapper is fully
transparent to the processing chain.

When no cache is provided (``cache=None``), this wrapper is a no-op
pass-through with negligible overhead.
"""

from typing import Any, Optional, Tuple, Union

from numpy.typing import NDArray

from .tile_cache import TileCache


class CachedImageProvider:
    """Transparent decode-caching wrapper for any ImageAssetProvider.

    Wraps an uncached source and caches ``get_block()`` results in a
    shared :class:`TileCache`.  All properties and methods delegate to
    the source unchanged.  The ``.key`` property also delegates
    unchanged — this provider is fully transparent to the chain.

    When ``cache`` is ``None``, this wrapper is a no-op pass-through.

    :param source: A duck-typed ``ImageAssetProvider`` instance.
    :param cache: Shared :class:`TileCache` instance, or ``None`` to
        disable caching.
    """

    def __init__(self, source: Any, cache: Optional[TileCache] = None) -> None:
        self._source = source
        self._cache = cache

    def get_block(
        self,
        row: int,
        col: int,
        resolution_level: int = 0,
        bands: Union[Tuple[int, ...], None] = None,
    ) -> NDArray:
        """Return the block from cache if available, else read from source and cache."""
        if self._cache is None:
            return self._source.get_block(row, col, resolution_level=resolution_level, bands=bands)

        hashable_bands = tuple(bands) if bands is not None else None
        cache_key = (self.key, row, col, resolution_level, hashable_bands)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        block = self._source.get_block(row, col, resolution_level=resolution_level, bands=bands)
        self._cache.put(cache_key, block)
        return block

    @property
    def key(self) -> str:
        """Delegates to source — fully transparent."""
        return self._source.key

    @property
    def num_rows(self) -> int:
        return self._source.num_rows

    @property
    def num_columns(self) -> int:
        return self._source.num_columns

    @property
    def num_pixels_per_block_horizontal(self) -> int:
        return self._source.num_pixels_per_block_horizontal

    @property
    def num_pixels_per_block_vertical(self) -> int:
        return self._source.num_pixels_per_block_vertical

    @property
    def num_resolution_levels(self) -> int:
        return self._source.num_resolution_levels

    @property
    def block_grid_size(self) -> Tuple[int, int]:
        return self._source.block_grid_size

    @property
    def num_bands(self) -> int:
        return self._source.num_bands

    @property
    def pixel_value_type(self) -> Any:
        return self._source.pixel_value_type

    def has_block(self, row: int, col: int, resolution_level: int = 0) -> bool:
        return self._source.has_block(row, col, resolution_level=resolution_level)

    @property
    def metadata(self) -> Any:
        return self._source.metadata
