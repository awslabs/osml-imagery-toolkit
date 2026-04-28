#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""MappedImageProvider — block-level function application adapter.

This module provides :class:`MappedImageProvider`, a generic adapter that
wraps a duck-typed ``ImageAssetProvider`` and applies an arbitrary
``ndarray → ndarray`` function to every block returned by
:meth:`get_block`.  An optional shared :class:`TileCache` avoids
redundant reads and function applications for repeated requests.

Typical usage::

    from aws.osml.image_processing.mapped_provider import MappedImageProvider
    from aws.osml.image_processing.tile_cache import TileCache


    def my_transform(block):
        return block * 2


    cache = TileCache(max_bytes=256 * 1024**2)
    mapped = MappedImageProvider(source_provider, my_transform, cache=cache, name="double")
    processed_block = mapped.get_block(0, 0)
"""

from typing import Any, Callable, Optional, Tuple, Union

from numpy.typing import NDArray

from .tile_cache import TileCache


class MappedImageProvider:
    """Adapter that applies a function to each block read from a source provider.

    Delegates all ``ImageAssetProvider`` properties and methods to the
    wrapped *source*, except :meth:`get_block` (which applies *func*)
    and optionally :attr:`num_bands` / :attr:`pixel_value_type` (which
    can be overridden when the function changes band count or dtype).

    Args:
        source: A duck-typed ``ImageAssetProvider`` instance.
        func: A callable that accepts an NDArray and returns an NDArray.
        cache: Optional shared :class:`TileCache` for caching processed
            blocks.  When ``None`` (the default), no caching occurs.
        name: Optional human-readable name used in the cache key.
            When ``None``, ``id(self)`` is used for uniqueness.
        num_bands: If provided, overrides the ``num_bands`` property
            instead of delegating to *source*.
        pixel_value_type: If provided, overrides the ``pixel_value_type``
            property instead of delegating to *source*.
        source_bands: If provided, specifies which bands to read from
            the wrapped source on every ``get_block()`` call, regardless
            of the caller's ``bands`` argument.  When ``None`` (the
            default), ``None`` is passed to the source (reading all bands).
    """

    def __init__(
        self,
        source: Any,
        func: Callable[[NDArray], NDArray],
        cache: Optional[TileCache] = None,
        name: Optional[str] = None,
        num_bands: Optional[int] = None,
        pixel_value_type: Any = None,
        source_bands: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self._source = source
        self._func = func
        self._cache = cache
        self._name = name
        self._num_bands_override = num_bands
        self._pixel_value_type_override = pixel_value_type
        self._source_bands = source_bands

    # ------------------------------------------------------------------
    # Block access
    # ------------------------------------------------------------------

    def get_block(
        self,
        row: int,
        col: int,
        resolution_level: int = 0,
        bands: Union[Tuple[int, ...], None] = None,
    ) -> NDArray:
        """Read a block from the source, apply *func*, and return the result.

        The source is always read using the instance's ``source_bands``
        parameter (set at construction time).  The caller's *bands*
        argument is applied as a band-dimension slice on the function's
        output array.

        When a :class:`TileCache` is provided, the processed result is
        stored and returned directly on subsequent calls with the same
        arguments.

        Args:
            row: Block row index.
            col: Block column index.
            resolution_level: Resolution level (default ``0``).
            bands: Optional tuple of band indices to select from the
                function's output.  When ``None``, the full output is
                returned.

        Returns:
            The processed block as an NDArray.

        Raises:
            IndexError: If any index in *bands* exceeds the function
                output's band count.
        """
        hashable_bands = tuple(bands) if bands is not None else None
        if self._cache is not None:
            cache_key = (self.key, row, col, resolution_level, hashable_bands)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Always read from source using the instance's source_bands
        block = self._source.get_block(row, col, resolution_level=resolution_level, bands=self._source_bands)
        result = self._func(block)

        # Apply caller's bands as an output slice
        if bands is not None:
            if any(b >= result.shape[0] for b in bands):
                raise IndexError(f"Requested band indices {bands} exceed output band count {result.shape[0]}")
            result = result[list(bands), :, :]

        if self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        """Unique key identifying this mapped provider in the chain."""
        return f"{self._source.key}:mapped:{self._name or id(self)}"

    @property
    def num_rows(self) -> int:
        """Number of pixel rows in the image."""
        return self._source.num_rows

    @property
    def num_columns(self) -> int:
        """Number of pixel columns in the image."""
        return self._source.num_columns

    @property
    def num_pixels_per_block_horizontal(self) -> int:
        """Number of pixels per block in the horizontal direction."""
        return self._source.num_pixels_per_block_horizontal

    @property
    def num_pixels_per_block_vertical(self) -> int:
        """Number of pixels per block in the vertical direction."""
        return self._source.num_pixels_per_block_vertical

    @property
    def num_resolution_levels(self) -> int:
        """Number of resolution levels available."""
        return self._source.num_resolution_levels

    @property
    def block_grid_size(self) -> Tuple[int, int]:
        """Grid dimensions ``(rows, cols)`` of the block layout."""
        return self._source.block_grid_size

    @property
    def num_bands(self) -> int:
        """Number of image bands.

        Returns the override value when one was provided at construction
        time; otherwise delegates to the source provider.
        """
        if self._num_bands_override is not None:
            return self._num_bands_override
        return self._source.num_bands

    @property
    def pixel_value_type(self) -> Any:
        """Pixel value type of the image.

        Returns the override value when one was provided at construction
        time; otherwise delegates to the source provider.
        """
        if self._pixel_value_type_override is not None:
            return self._pixel_value_type_override
        return self._source.pixel_value_type

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def has_block(self, row: int, col: int, resolution_level: int = 0) -> bool:
        """Check whether a block exists at the given grid position.

        Args:
            row: Block row index.
            col: Block column index.
            resolution_level: Resolution level (default ``0``).

        Returns:
            ``True`` if the block is available, ``False`` otherwise.
        """
        return self._source.has_block(row, col, resolution_level=resolution_level)

    @property
    def metadata(self) -> Any:
        """Return the metadata object from the source provider."""
        return self._source.metadata
