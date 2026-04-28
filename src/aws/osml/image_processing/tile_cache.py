#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Byte-budget LRU cache for ndarray tiles.

This module provides :class:`TileCache`, a thread-safe byte-budget LRU cache
backed by :class:`cachetools.LRUCache`.  It is designed to be shared across
all operators in an image processing chain, giving users a single knob
(``max_bytes``) to control total memory consumption.

Arrays are frozen (marked read-only) on insertion.  Cache hits return the
same array object (zero-copy).  Consumers that need to mutate must copy
explicitly.
"""

import threading
from typing import Optional, Tuple, Union

import cachetools
from numpy.typing import NDArray

CacheKey = Tuple[str, int, int, int, Union[Tuple[int, ...], None]]


class TileCache:
    """Byte-budget LRU cache for ndarray tiles.

    Thread-safe. Backed by :class:`cachetools.LRUCache` with
    ``getsizeof=arr.nbytes``.  Arrays are marked read-only on insertion;
    cache hits return the same array object (zero-copy).  Consumers that
    need to mutate must copy explicitly.

    Tiles larger than ``max_bytes`` are silently not cached — the cache
    is an optimization and missing is correct behavior.

    :param max_bytes: Maximum total bytes of cached arrays. Default 5 GiB.
    """

    def __init__(self, max_bytes: int = 5 * 1024**3) -> None:
        self._cache: cachetools.LRUCache = cachetools.LRUCache(
            maxsize=max_bytes,
            getsizeof=lambda arr: arr.nbytes,
        )
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> Optional[NDArray]:
        """Retrieve a cached tile by key, or ``None`` if not present."""
        with self._lock:
            return self._cache.get(key)

    def put(self, key: CacheKey, tile: NDArray) -> None:
        """Store a tile in the cache under *key*.

        The tile is frozen (``writeable=False``) before insertion.
        Tiles whose ``nbytes`` exceeds ``max_bytes`` are silently
        discarded — the cache cannot hold them regardless of eviction.
        """
        if tile.nbytes > self._cache.maxsize:
            return
        tile.flags.writeable = False
        with self._lock:
            self._cache[key] = tile

    @property
    def current_bytes(self) -> int:
        """Total bytes currently stored in the cache."""
        with self._lock:
            return self._cache.currsize

    @property
    def max_bytes(self) -> int:
        """Maximum byte budget for the cache."""
        return self._cache.maxsize

    def clear(self) -> None:
        """Evict all entries from the cache."""
        with self._lock:
            self._cache.clear()
