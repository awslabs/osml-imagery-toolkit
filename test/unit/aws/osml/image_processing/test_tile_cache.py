#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import threading
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.tile_cache import TileCache


class TestTileCacheBasics(TestCase):
    """Tests for basic TileCache get/put semantics."""

    def test_put_and_get(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile = np.zeros((3, 64, 64), dtype=np.uint8)
        key = ("src", 0, 0, 0, None)

        cache.put(key, tile)
        result = cache.get(key)

        self.assertIs(result, tile)

    def test_get_missing_key_returns_none(self):
        cache = TileCache(max_bytes=1024 * 1024)
        result = cache.get(("src", 99, 99, 0, None))
        self.assertIsNone(result)

    def test_clear_empties_cache(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile = np.zeros((3, 32, 32), dtype=np.uint8)
        cache.put(("src", 0, 0, 0, None), tile)
        self.assertGreater(cache.current_bytes, 0)

        cache.clear()

        self.assertEqual(cache.current_bytes, 0)
        self.assertIsNone(cache.get(("src", 0, 0, 0, None)))


class TestTileCacheByteBudget(TestCase):
    """Tests for byte-budget eviction."""

    def test_current_bytes_tracks_inserts(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile = np.zeros((1, 10, 10), dtype=np.uint8)
        cache.put(("src", 0, 0, 0, None), tile)

        self.assertEqual(cache.current_bytes, 100)

    def test_max_bytes_reports_configured_limit(self):
        cache = TileCache(max_bytes=256)
        self.assertEqual(cache.max_bytes, 256)

    def test_eviction_when_budget_exceeded(self):
        cache = TileCache(max_bytes=200)
        tile_a = np.zeros((1, 10, 10), dtype=np.uint8)  # 100 bytes
        tile_b = np.zeros((1, 10, 10), dtype=np.uint8)  # 100 bytes
        tile_c = np.zeros((1, 10, 10), dtype=np.uint8)  # 100 bytes

        cache.put(("src", 0, 0, 0, None), tile_a)
        cache.put(("src", 0, 1, 0, None), tile_b)
        # At this point cache is at 200 bytes (full)
        cache.put(("src", 0, 2, 0, None), tile_c)

        # tile_a should have been evicted (LRU)
        self.assertIsNone(cache.get(("src", 0, 0, 0, None)))
        # tile_c should be present
        self.assertIsNotNone(cache.get(("src", 0, 2, 0, None)))
        self.assertLessEqual(cache.current_bytes, 200)

    def test_oversized_tile_silently_not_cached(self):
        cache = TileCache(max_bytes=50)
        tile = np.zeros((1, 10, 10), dtype=np.uint8)  # 100 bytes > 50

        cache.put(("src", 0, 0, 0, None), tile)

        self.assertIsNone(cache.get(("src", 0, 0, 0, None)))
        self.assertEqual(cache.current_bytes, 0)


class TestTileCacheReadOnly(TestCase):
    """Tests that cached tiles are marked read-only."""

    def test_stored_array_is_read_only(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile = np.ones((1, 4, 4), dtype=np.uint8)
        cache.put(("src", 0, 0, 0, None), tile)

        result = cache.get(("src", 0, 0, 0, None))
        self.assertFalse(result.flags.writeable)

    def test_mutation_raises_valueerror(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile = np.ones((1, 4, 4), dtype=np.uint8)
        cache.put(("src", 0, 0, 0, None), tile)

        result = cache.get(("src", 0, 0, 0, None))
        with self.assertRaises(ValueError):
            result[0, 0, 0] = 99

    def test_original_array_is_frozen_after_put(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile = np.ones((1, 4, 4), dtype=np.uint8)
        cache.put(("src", 0, 0, 0, None), tile)

        self.assertFalse(tile.flags.writeable)


class TestTileCacheKeyIsolation(TestCase):
    """Tests that different keys are stored independently."""

    def test_different_provider_keys(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile_a = np.ones((1, 4, 4), dtype=np.uint8)
        tile_b = np.full((1, 4, 4), 2, dtype=np.uint8)

        cache.put(("provider_a", 0, 0, 0, None), tile_a)
        cache.put(("provider_b", 0, 0, 0, None), tile_b)

        np.testing.assert_array_equal(cache.get(("provider_a", 0, 0, 0, None)), tile_a)
        np.testing.assert_array_equal(cache.get(("provider_b", 0, 0, 0, None)), tile_b)

    def test_different_bands_keys(self):
        cache = TileCache(max_bytes=1024 * 1024)
        tile_all = np.ones((3, 4, 4), dtype=np.uint8)
        tile_rgb = np.full((3, 4, 4), 2, dtype=np.uint8)

        cache.put(("src", 0, 0, 0, None), tile_all)
        cache.put(("src", 0, 0, 0, (0, 1, 2)), tile_rgb)

        np.testing.assert_array_equal(cache.get(("src", 0, 0, 0, None)), tile_all)
        np.testing.assert_array_equal(cache.get(("src", 0, 0, 0, (0, 1, 2))), tile_rgb)


class TestTileCacheThreadSafety(TestCase):
    """Tests that TileCache is safe under concurrent access."""

    def test_concurrent_puts_and_gets(self):
        cache = TileCache(max_bytes=10 * 1024 * 1024)
        num_threads = 8
        num_ops = 100
        errors = []

        def worker(thread_id):
            try:
                for i in range(num_ops):
                    key = (f"thread-{thread_id}", i, 0, 0, None)
                    fill_val = (thread_id * 13 + i) % 256
                    tile = np.full((1, 8, 8), fill_val, dtype=np.uint8)
                    cache.put(key, tile)

                    result = cache.get(key)
                    if result is not None:
                        if not np.all(result == fill_val):
                            errors.append(f"Thread {thread_id}, iter {i}: data corruption")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_concurrent_puts_with_eviction(self):
        # Small budget forces eviction under contention
        cache = TileCache(max_bytes=500)
        num_threads = 4
        num_ops = 50
        errors = []

        def worker(thread_id):
            try:
                for i in range(num_ops):
                    key = (f"t{thread_id}", i, 0, 0, None)
                    tile = np.full((1, 4, 4), i, dtype=np.uint8)  # 16 bytes each
                    cache.put(key, tile)
                    cache.get(key)  # may or may not hit (eviction possible)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(cache.current_bytes, 500)
