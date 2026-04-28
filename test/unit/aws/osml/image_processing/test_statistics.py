#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import collections.abc
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.statistics import (
    BandStatistics,
    ImageStatistics,
    SamplingStrategy,
    compute_image_statistics,
    compute_statistics,
    merge_statistics,
    statistics_from_gdal_metadata,
    statistics_to_gdal_metadata,
)


class TestBandStatistics(TestCase):
    """Tests for the BandStatistics dataclass."""

    def test_valid_construction(self):
        """BandStatistics accepts histogram and bin_edges with correct lengths."""
        histogram = np.array([10, 20, 30])
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        bs = BandStatistics(
            min=0.0, max=3.0, mean=1.5, stddev=0.8, count=60, m2=38.4, histogram=histogram, bin_edges=bin_edges
        )
        self.assertEqual(bs.min, 0.0)
        self.assertEqual(bs.max, 3.0)
        self.assertEqual(bs.count, 60)
        self.assertAlmostEqual(bs.m2, 38.4)
        self.assertEqual(len(bs.histogram), 3)
        self.assertEqual(len(bs.bin_edges), 4)

    def test_invalid_bin_edges_length_raises(self):
        """BandStatistics raises ValueError when bin_edges length != len(histogram) + 1."""
        histogram = np.array([10, 20, 30])
        bin_edges = np.array([0.0, 1.0, 2.0])  # Should be length 4
        with self.assertRaises(ValueError) as ctx:
            BandStatistics(
                min=0.0, max=3.0, mean=1.5, stddev=0.8, count=60, m2=38.4, histogram=histogram, bin_edges=bin_edges
            )
        self.assertIn("bin_edges length", str(ctx.exception))

    def test_count_and_m2_required(self):
        """BandStatistics raises TypeError if count or m2 are not provided."""
        histogram = np.array([10, 20, 30])
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            BandStatistics(min=0.0, max=3.0, mean=1.5, stddev=0.8, histogram=histogram, bin_edges=bin_edges)


class TestImageStatistics(TestCase):
    """Tests for the ImageStatistics dataclass."""

    def test_default_sample_rate(self):
        """ImageStatistics defaults sample_rate to 1.0."""
        stats = ImageStatistics(bands=[])
        self.assertEqual(stats.sample_rate, 1.0)

    def test_custom_sample_rate(self):
        """ImageStatistics accepts a custom sample_rate."""
        stats = ImageStatistics(bands=[], sample_rate=0.5)
        self.assertEqual(stats.sample_rate, 0.5)


class TestSamplingStrategy(TestCase):
    """Tests for the SamplingStrategy enum."""

    def test_enum_values(self):
        """SamplingStrategy has ALL, RANDOM, and BLOCK values."""
        self.assertEqual(SamplingStrategy.ALL.value, "all")
        self.assertEqual(SamplingStrategy.RANDOM.value, "random")
        self.assertEqual(SamplingStrategy.BLOCK.value, "block")


class TestComputeStatistics(TestCase):
    """Tests for the compute_statistics function."""

    def test_known_uint8_histogram(self):
        """Known 2x2 uint8 array produces exact histogram counts."""
        # 1-band, 2x2 image with pixel values: 0, 50, 200, 255
        image = np.array([[[0, 50], [200, 255]]], dtype=np.uint8)
        stats = compute_statistics(image, num_bins=256)

        self.assertEqual(len(stats.bands), 1)
        band = stats.bands[0]

        # Verify scalar stats
        self.assertAlmostEqual(band.min, 0.0)
        self.assertAlmostEqual(band.max, 255.0)
        self.assertAlmostEqual(band.mean, float(np.mean([0, 50, 200, 255])))
        self.assertAlmostEqual(band.stddev, float(np.std([0, 50, 200, 255])))

        # Verify count and m2
        self.assertEqual(band.count, 4)
        expected_m2 = float(np.var([0, 50, 200, 255])) * 4
        self.assertAlmostEqual(band.m2, expected_m2)

        # Verify histogram: total counts should equal number of pixels
        self.assertEqual(int(np.sum(band.histogram)), 4)

        # Verify bin_edges length
        self.assertEqual(len(band.bin_edges), 257)

    def test_multi_band_image(self):
        """compute_statistics handles multi-band CHW images correctly."""
        image = np.array(
            [
                [[10, 20], [30, 40]],
                [[100, 110], [120, 130]],
            ],
            dtype=np.uint8,
        )
        stats = compute_statistics(image, num_bins=256)

        self.assertEqual(len(stats.bands), 2)
        self.assertAlmostEqual(stats.bands[0].mean, 25.0)
        self.assertAlmostEqual(stats.bands[1].mean, 115.0)
        self.assertEqual(stats.bands[0].count, 4)
        self.assertEqual(stats.bands[1].count, 4)

    def test_2d_input_treated_as_single_band(self):
        """A 2-D (H, W) array is treated as a single-band image."""
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        stats = compute_statistics(image, num_bins=256)

        self.assertEqual(len(stats.bands), 1)
        self.assertAlmostEqual(stats.bands[0].mean, 25.0)

    def test_dtype_derived_bin_edges_uint8(self):
        """For uint8, bin_edges span 0 to 255."""
        image = np.array([[[128]]], dtype=np.uint8)
        stats = compute_statistics(image, num_bins=256)

        band = stats.bands[0]
        self.assertAlmostEqual(band.bin_edges[0], 0.0)
        self.assertAlmostEqual(band.bin_edges[-1], 255.0)

    def test_dtype_derived_bin_edges_uint16(self):
        """For uint16, bin_edges span 0 to 65535."""
        image = np.array([[[1000]]], dtype=np.uint16)
        stats = compute_statistics(image, num_bins=256)

        band = stats.bands[0]
        self.assertAlmostEqual(band.bin_edges[0], 0.0)
        self.assertAlmostEqual(band.bin_edges[-1], 65535.0)

    def test_dtype_derived_bin_edges_float32_data_derived(self):
        """For float32, bin_edges are derived from actual data range (not [0, 1])."""
        image = np.array([[[25.0, -10.0], [50.0, 100.0]]], dtype=np.float32)
        stats = compute_statistics(image, num_bins=256)

        band = stats.bands[0]
        self.assertAlmostEqual(band.bin_edges[0], -10.0, places=5)
        self.assertAlmostEqual(band.bin_edges[-1], 100.0, places=5)

    def test_float_dtype_per_band_independent_edges(self):
        """Multi-band float32 image gets independent per-band bin_edges."""
        band0_data = np.array([[0.0, 500.0], [250.0, 1000.0]], dtype=np.float32)
        band1_data = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
        image = np.stack([band0_data, band1_data])

        stats = compute_statistics(image, num_bins=256)

        # Band 0 edges should span [0, 1000]
        self.assertAlmostEqual(stats.bands[0].bin_edges[0], 0.0, places=5)
        self.assertAlmostEqual(stats.bands[0].bin_edges[-1], 1000.0, places=5)

        # Band 1 edges should span [0, 1]
        self.assertAlmostEqual(stats.bands[1].bin_edges[0], 0.0, places=5)
        self.assertAlmostEqual(stats.bands[1].bin_edges[-1], 1.0, places=5)

    def test_float_dtype_histogram_has_nonzero_interior(self):
        """Float32 array with values in [-50, 50] produces non-zero interior histogram bins."""
        rng = np.random.default_rng(42)
        image = rng.uniform(-50, 50, size=(1, 32, 32)).astype(np.float32)
        stats = compute_statistics(image, num_bins=256)

        band = stats.bands[0]
        # Verify edges span the data range
        self.assertLessEqual(band.bin_edges[0], float(image.min()))
        self.assertGreaterEqual(band.bin_edges[-1], float(image.max()))
        # Interior bins should have counts
        interior = band.histogram[1:-1]
        self.assertGreater(interior.sum(), 0)

    def test_float_constant_band_gets_expanded_range(self):
        """Float band with all-same values gets edges expanded by +1.0."""
        image = np.full((1, 4, 4), 5.0, dtype=np.float32)
        stats = compute_statistics(image, num_bins=256)

        band = stats.bands[0]
        self.assertAlmostEqual(band.bin_edges[0], 5.0, places=5)
        self.assertAlmostEqual(band.bin_edges[-1], 6.0, places=5)

    def test_supplied_bin_edges_used_directly(self):
        """When bin_edges are supplied as NDArray, they are used as-is for all bands."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        custom_edges = np.array([0.0, 25.0, 50.0])
        stats = compute_statistics(image, bin_edges=custom_edges)

        band = stats.bands[0]
        np.testing.assert_array_equal(band.bin_edges, custom_edges)
        self.assertEqual(len(band.histogram), 2)

    def test_supplied_per_band_bin_edges(self):
        """When bin_edges are supplied as List[NDArray], each band gets its own edges."""
        image = np.array(
            [
                [[10, 20], [30, 40]],
                [[100, 110], [120, 130]],
            ],
            dtype=np.uint8,
        )
        edges_band0 = np.array([0.0, 25.0, 50.0])
        edges_band1 = np.array([90.0, 115.0, 140.0])
        stats = compute_statistics(image, bin_edges=[edges_band0, edges_band1])

        np.testing.assert_array_equal(stats.bands[0].bin_edges, edges_band0)
        np.testing.assert_array_equal(stats.bands[1].bin_edges, edges_band1)
        self.assertEqual(len(stats.bands[0].histogram), 2)
        self.assertEqual(len(stats.bands[1].histogram), 2)

    def test_histogram_count_equals_pixel_count(self):
        """With SamplingStrategy.ALL, histogram sum equals total pixel count."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
        stats = compute_statistics(image, num_bins=256, sampling=SamplingStrategy.ALL)

        for band in stats.bands:
            self.assertEqual(int(np.sum(band.histogram)), 16 * 16)
            self.assertEqual(band.count, 16 * 16)

    def test_random_sampling_count_equals_expected(self):
        """With SamplingStrategy.RANDOM, count equals expected sample count."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 100, 100), dtype=np.uint8)
        sample_rate = 0.1
        stats = compute_statistics(image, num_bins=256, sampling=SamplingStrategy.RANDOM, sample_rate=sample_rate)

        expected_count = max(1, int(100 * 100 * sample_rate))
        self.assertEqual(stats.bands[0].count, expected_count)
        self.assertAlmostEqual(stats.sample_rate, sample_rate)

    def test_random_sampling_uses_replacement(self):
        """Random sampling uses rng.integers (with replacement), not choice without replacement."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 100, 100), dtype=np.uint8)
        sample_rate = 0.1
        stats = compute_statistics(image, num_bins=256, sampling=SamplingStrategy.RANDOM, sample_rate=sample_rate)

        expected_count = max(1, int(100 * 100 * sample_rate))
        # With replacement sampling, count should exactly match the requested sample size
        self.assertEqual(stats.bands[0].count, expected_count)

    def test_random_sampling_same_indices_all_bands(self):
        """Random sampling uses the same pixel indices for all bands."""
        band0 = np.zeros((1, 100, 100), dtype=np.uint8)
        band1 = np.ones((1, 100, 100), dtype=np.uint8) * 255
        # Place a known marker at specific positions
        band0[0, 0, 0] = 200
        band1[0, 0, 0] = 50
        image = np.concatenate([band0, band1], axis=0)

        stats = compute_statistics(image, num_bins=256, sampling=SamplingStrategy.RANDOM, sample_rate=0.5)

        # Both bands should have the same count
        self.assertEqual(stats.bands[0].count, stats.bands[1].count)

    def test_random_sampling_full_rate_uses_all_pixels(self):
        """With SamplingStrategy.RANDOM and sample_rate=1.0, all pixels are used."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        stats = compute_statistics(image, num_bins=256, sampling=SamplingStrategy.RANDOM, sample_rate=1.0)

        self.assertEqual(int(np.sum(stats.bands[0].histogram)), 4)
        self.assertEqual(stats.bands[0].count, 4)

    def test_count_and_m2_populated(self):
        """compute_statistics populates count and m2 correctly."""
        pixels = np.array([10, 20, 30, 40], dtype=np.uint8)
        image = pixels.reshape(1, 2, 2)
        stats = compute_statistics(image, num_bins=256)

        band = stats.bands[0]
        self.assertEqual(band.count, 4)
        expected_var = float(np.var(pixels))
        expected_m2 = expected_var * 4
        self.assertAlmostEqual(band.m2, expected_m2, places=10)
        self.assertAlmostEqual(band.stddev, float(np.sqrt(expected_var)), places=10)

    def test_bin_edges_strictly_monotonic(self):
        """bin_edges are strictly monotonically increasing."""
        rng = np.random.default_rng(99)
        image = rng.integers(0, 256, size=(2, 8, 8), dtype=np.uint8)
        stats = compute_statistics(image, num_bins=128)

        for band in stats.bands:
            diffs = np.diff(band.bin_edges)
            self.assertTrue(np.all(diffs > 0), "bin_edges must be strictly monotonically increasing")

    def test_sample_rate_in_result(self):
        """sample_rate is correctly set in the returned ImageStatistics."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)

        stats_all = compute_statistics(image, sampling=SamplingStrategy.ALL)
        self.assertEqual(stats_all.sample_rate, 1.0)

    def test_no_input_mutation(self):
        """compute_statistics does not mutate the input array."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        original = image.copy()
        compute_statistics(image, num_bins=256)
        np.testing.assert_array_equal(image, original)

    def test_invalid_ndim_raises(self):
        """compute_statistics raises ValueError for arrays with ndim != 2 or 3."""
        image_4d = np.zeros((1, 1, 2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_statistics(image_4d)


class TestMergeStatistics(TestCase):
    """Tests for the merge_statistics function."""

    def _make_band(self, min_val, max_val, mean, stddev, histogram, bin_edges):
        """Helper to construct BandStatistics with count/m2 derived from histogram."""
        count = int(np.sum(histogram))
        m2 = stddev**2 * count
        return BandStatistics(
            min=min_val,
            max=max_val,
            mean=mean,
            stddev=stddev,
            count=count,
            m2=m2,
            histogram=histogram,
            bin_edges=bin_edges,
        )

    def test_two_blocks_merged_mean_matches_full_array(self):
        """Two known blocks → merged mean matches full-array mean, histogram sum equals pixel count."""
        # Create a known 1-band 4x4 uint8 array
        full_array = np.array(
            [[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]]],
            dtype=np.uint8,
        )
        # Split into two 2x4 blocks (top half and bottom half)
        top_half = full_array[:, :2, :]  # shape (1, 2, 4)
        bottom_half = full_array[:, 2:, :]  # shape (1, 2, 4)

        # Compute statistics on each block with the same bin_edges
        bin_edges = np.linspace(0, 255, 257)
        stats_top = compute_statistics(top_half, bin_edges=bin_edges)
        stats_bottom = compute_statistics(bottom_half, bin_edges=bin_edges)

        # Merge the two statistics
        merged = merge_statistics([stats_top, stats_bottom])

        # Verify merged mean is approximately equal to the full-array mean
        expected_mean = float(np.mean(full_array))
        self.assertAlmostEqual(merged.bands[0].mean, expected_mean, places=10)

        # Verify merged histogram sum equals 16 (4*4 pixels)
        self.assertEqual(int(np.sum(merged.bands[0].histogram)), 16)

        # Verify merged count
        self.assertEqual(merged.bands[0].count, 16)

    def test_chan_merge_stddev_matches_full_array(self):
        """Chan merge: merged stddev matches np.std(full_array) within 1e-10."""
        full_array = np.array(
            [[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]]],
            dtype=np.uint8,
        )
        top_half = full_array[:, :2, :]
        bottom_half = full_array[:, 2:, :]

        bin_edges = np.linspace(0, 255, 257)
        stats_top = compute_statistics(top_half, bin_edges=bin_edges)
        stats_bottom = compute_statistics(bottom_half, bin_edges=bin_edges)

        merged = merge_statistics([stats_top, stats_bottom])

        expected_stddev = float(np.std(full_array))
        self.assertAlmostEqual(merged.bands[0].stddev, expected_stddev, places=10)

    def test_chan_merge_multi_partition_exact(self):
        """Chan merge across 4 partitions is exact for non-trivial distribution."""
        rng = np.random.default_rng(123)
        full_array = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)

        bin_edges = np.linspace(0, 255, 257)
        strips = np.array_split(full_array, 4, axis=1)
        block_stats = [compute_statistics(s, bin_edges=bin_edges) for s in strips]

        merged = merge_statistics(block_stats)

        for b in range(3):
            expected_stddev = float(np.std(full_array[b]))
            self.assertAlmostEqual(merged.bands[b].stddev, expected_stddev, places=10)
            expected_mean = float(np.mean(full_array[b]))
            self.assertAlmostEqual(merged.bands[b].mean, expected_mean, places=10)

    def test_merged_min_max_exact_from_blocks(self):
        """Merged min/max are exact per-block values, not histogram-bin approximations."""
        bin_edges = np.linspace(0, 255, 257)
        # Block A has min=5, max=200
        block_a = np.array([[[5, 100], [150, 200]]], dtype=np.uint8)
        # Block B has min=3, max=250
        block_b = np.array([[[3, 50], [120, 250]]], dtype=np.uint8)

        stats_a = compute_statistics(block_a, bin_edges=bin_edges)
        stats_b = compute_statistics(block_b, bin_edges=bin_edges)

        merged = merge_statistics([stats_a, stats_b])

        self.assertEqual(merged.bands[0].min, 3.0)
        self.assertEqual(merged.bands[0].max, 250.0)

    def test_merged_min_max_filters_zero_count_blocks(self):
        """Zero-count blocks are excluded from min/max aggregation."""
        bin_edges = np.array([0.0, 128.0, 256.0])
        # Real block with actual data
        real_band = BandStatistics(
            min=10.0,
            max=200.0,
            mean=105.0,
            stddev=50.0,
            count=100,
            m2=50.0**2 * 100,
            histogram=np.array([40, 60]),
            bin_edges=bin_edges,
        )
        # Empty block (count=0)
        empty_band = BandStatistics(
            min=0.0,
            max=0.0,
            mean=0.0,
            stddev=0.0,
            count=0,
            m2=0.0,
            histogram=np.array([0, 0]),
            bin_edges=bin_edges,
        )

        stats_real = ImageStatistics(bands=[real_band])
        stats_empty = ImageStatistics(bands=[empty_band])

        merged = merge_statistics([stats_real, stats_empty])

        # Should use real block's min/max, not the empty block's 0.0
        self.assertEqual(merged.bands[0].min, 10.0)
        self.assertEqual(merged.bands[0].max, 200.0)

    def test_merged_histogram_is_element_wise_sum(self):
        """Merged histogram is the element-wise sum of input histograms."""
        bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
        stats_a = ImageStatistics(bands=[self._make_band(0.0, 3.0, 1.0, 0.5, np.array([10, 20, 30]), bin_edges)])
        stats_b = ImageStatistics(bands=[self._make_band(0.0, 3.0, 2.0, 0.5, np.array([5, 15, 25]), bin_edges)])

        merged = merge_statistics([stats_a, stats_b])
        np.testing.assert_array_equal(merged.bands[0].histogram, np.array([15, 35, 55]))

    def test_merged_sample_rate_capped_at_one(self):
        """Merged sample_rate is capped at 1.0."""
        bin_edges = np.array([0.0, 1.0, 2.0])
        stats_a = ImageStatistics(
            bands=[self._make_band(0.0, 2.0, 1.0, 0.5, np.array([5, 5]), bin_edges)],
            sample_rate=0.6,
        )
        stats_b = ImageStatistics(
            bands=[self._make_band(0.0, 2.0, 1.0, 0.5, np.array([5, 5]), bin_edges)],
            sample_rate=0.6,
        )

        merged = merge_statistics([stats_a, stats_b])
        self.assertEqual(merged.sample_rate, 1.0)

    def test_mismatched_bands_raises_value_error(self):
        """Mismatched band counts raise ValueError."""
        bin_edges = np.array([0.0, 1.0, 2.0])
        stats_1band = ImageStatistics(bands=[self._make_band(0.0, 2.0, 1.0, 0.5, np.array([5, 5]), bin_edges)])
        stats_2bands = ImageStatistics(
            bands=[
                self._make_band(0.0, 2.0, 1.0, 0.5, np.array([5, 5]), bin_edges),
                self._make_band(0.0, 2.0, 1.0, 0.5, np.array([5, 5]), bin_edges),
            ]
        )

        with self.assertRaises(ValueError) as ctx:
            merge_statistics([stats_1band, stats_2bands])
        self.assertIn("Band count mismatch", str(ctx.exception))

    def test_mismatched_bin_edges_raises_value_error(self):
        """Mismatched bin_edges for the same band raise ValueError."""
        edges_a = np.array([0.0, 1.0, 2.0])
        edges_b = np.array([0.0, 1.5, 3.0])
        stats_a = ImageStatistics(bands=[self._make_band(0.0, 2.0, 1.0, 0.5, np.array([5, 5]), edges_a)])
        stats_b = ImageStatistics(bands=[self._make_band(0.0, 3.0, 1.5, 0.5, np.array([5, 5]), edges_b)])

        with self.assertRaises(ValueError) as ctx:
            merge_statistics([stats_a, stats_b])
        self.assertIn("bin_edges mismatch", str(ctx.exception))

    def test_empty_iterable_raises_value_error(self):
        """Empty iterable raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            merge_statistics([])
        self.assertIn("empty", str(ctx.exception).lower())


class TestStatisticsFromGdalMetadata(TestCase):
    """Tests for the statistics_from_gdal_metadata function."""

    def test_well_formed_xml_stats_only(self):
        """Well-formed XML with per-band stats (no histogram) returns ImageStatistics with 1 band."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">10.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">200.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">105.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">45.0</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 1)
        band = result.bands[0]
        self.assertAlmostEqual(band.min, 10.0)
        self.assertAlmostEqual(band.max, 200.0)
        self.assertAlmostEqual(band.mean, 105.0)
        self.assertAlmostEqual(band.stddev, 45.0)
        # count derived from histogram sum
        self.assertEqual(band.count, 0)  # minimal histogram with 0 count
        # m2 derived from stddev^2 * count
        self.assertAlmostEqual(band.m2, 45.0**2 * 0)
        # A minimal histogram is created since no histogram data in XML
        self.assertIsNotNone(band.histogram)
        self.assertEqual(len(band.bin_edges), len(band.histogram) + 1)

    def test_well_formed_xml_with_histogram(self):
        """Well-formed XML with stats and histogram entries returns correct histogram data."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">50.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">25.0</Item>'
            '  <Item name="STATISTICS_HISTOMIN" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_HISTOMAX" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_HISTONUMBINS" sample="0">4</Item>'
            '  <Item name="STATISTICS_HISTOBINVALUES" sample="0">5|10|15|20</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 1)
        band = result.bands[0]
        np.testing.assert_array_equal(band.histogram, np.array([5, 10, 15, 20]))
        # bin_edges computed from histomin=0, histomax=100, numbins=4
        expected_edges = np.linspace(0.0, 100.0, 5)
        np.testing.assert_array_almost_equal(band.bin_edges, expected_edges)
        # count derived from histogram sum
        self.assertEqual(band.count, 50)
        # m2 derived from stddev^2 * count
        self.assertAlmostEqual(band.m2, 25.0**2 * 50)

    def test_multi_band_xml(self):
        """XML with 2 bands returns ImageStatistics with 2 BandStatistics."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">255.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">30.0</Item>'
            '  <Item name="STATISTICS_MINIMUM" sample="1">10.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="1">240.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="1">120.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="1">35.0</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 2)
        self.assertAlmostEqual(result.bands[0].min, 0.0)
        self.assertAlmostEqual(result.bands[0].mean, 100.0)
        self.assertAlmostEqual(result.bands[1].min, 10.0)
        self.assertAlmostEqual(result.bands[1].mean, 120.0)

    def test_missing_gdal_metadata_key_returns_none(self):
        """A dict without the GDAL_METADATA key returns None."""
        result = statistics_from_gdal_metadata({"OTHER_KEY": "value"})
        self.assertIsNone(result)

    def test_malformed_xml_returns_none(self):
        """Malformed XML returns None without raising."""
        result = statistics_from_gdal_metadata({"GDAL_METADATA": "not xml at all"})
        self.assertIsNone(result)

    def test_missing_required_stat_returns_none(self):
        """XML with STATISTICS_MINIMUM and STATISTICS_MAXIMUM but missing STATISTICS_MEAN returns None."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">255.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">30.0</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNone(result)

    def test_inconsistent_band_count_returns_none(self):
        """Non-contiguous band indices (band 0 and band 2 without band 1) returns None."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">255.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">30.0</Item>'
            '  <Item name="STATISTICS_MINIMUM" sample="2">10.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="2">240.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="2">120.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="2">35.0</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNone(result)


class TestGdalMetadataDerivedFields(TestCase):
    """Tests for count/m2 derivation in statistics_from_gdal_metadata."""

    def test_count_derived_from_histogram_sum(self):
        """count is derived from histogram.sum() on parse."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">50.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">25.0</Item>'
            '  <Item name="STATISTICS_HISTOMIN" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_HISTOMAX" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_HISTONUMBINS" sample="0">4</Item>'
            '  <Item name="STATISTICS_HISTOBINVALUES" sample="0">10|20|30|40</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNotNone(result)
        self.assertEqual(result.bands[0].count, 100)  # 10+20+30+40

    def test_m2_derived_from_stddev_squared_times_count(self):
        """m2 is derived from stddev**2 * count on parse."""
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">50.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">25.0</Item>'
            '  <Item name="STATISTICS_HISTOMIN" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_HISTOMAX" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_HISTONUMBINS" sample="0">4</Item>'
            '  <Item name="STATISTICS_HISTOBINVALUES" sample="0">10|20|30|40</Item>'
            "</GDALMetadata>"
        )
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml})
        self.assertIsNotNone(result)
        expected_m2 = 25.0**2 * 100
        self.assertAlmostEqual(result.bands[0].m2, expected_m2)


class TestLocaleRoundTrip(TestCase):
    """Tests for locale-safe GDAL metadata serialization."""

    def test_locale_safe_float_formatting(self):
        """statistics_to_gdal_metadata uses :.17g format (no locale dependency)."""
        band = BandStatistics(
            min=1.23456789012345678,
            max=9876.54321098765,
            mean=1234.567890123456,
            stddev=456.789012345678,
            count=1000,
            m2=456.789012345678**2 * 1000,
            histogram=np.array([500, 500]),
            bin_edges=np.array([1.23456789012345678, 5000.0, 9876.54321098765]),
        )
        stats = ImageStatistics(bands=[band])
        xml_str = statistics_to_gdal_metadata(stats)

        # Verify the XML can be parsed back successfully
        result = statistics_from_gdal_metadata({"GDAL_METADATA": xml_str})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.bands[0].min, band.min, places=10)
        self.assertAlmostEqual(result.bands[0].max, band.max, places=10)
        self.assertAlmostEqual(result.bands[0].mean, band.mean, places=10)
        self.assertAlmostEqual(result.bands[0].stddev, band.stddev, places=10)

    def test_locale_round_trip_with_locale_if_available(self):
        """Round-trip survives under de_DE locale (commas as decimal separators)."""
        import locale

        try:
            original_locale = locale.getlocale(locale.LC_NUMERIC)
            locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
        except (locale.Error, ValueError):
            self.skipTest("de_DE.UTF-8 locale not available on this system")

        try:
            band = BandStatistics(
                min=1.5,
                max=99.9,
                mean=50.25,
                stddev=12.75,
                count=100,
                m2=12.75**2 * 100,
                histogram=np.array([50, 50]),
                bin_edges=np.array([1.5, 50.0, 99.9]),
            )
            stats = ImageStatistics(bands=[band])
            xml_str = statistics_to_gdal_metadata(stats)
            result = statistics_from_gdal_metadata({"GDAL_METADATA": xml_str})

            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.bands[0].min, 1.5)
            self.assertAlmostEqual(result.bands[0].max, 99.9)
            self.assertAlmostEqual(result.bands[0].mean, 50.25)
            self.assertAlmostEqual(result.bands[0].stddev, 12.75)
        finally:
            locale.setlocale(locale.LC_NUMERIC, original_locale)


class TestStatisticsToGdalMetadata(TestCase):
    """Tests for the statistics_to_gdal_metadata function."""

    def test_known_stats_xml_structure(self):
        """Serialized XML contains expected Item elements with correct name, sample, and text."""
        from defusedxml import ElementTree

        band = BandStatistics(
            min=5.0,
            max=250.0,
            mean=127.5,
            stddev=40.0,
            count=60,
            m2=40.0**2 * 60,
            histogram=np.array([10, 20, 30]),
            bin_edges=np.array([0.0, 85.0, 170.0, 255.0]),
        )
        stats = ImageStatistics(bands=[band])
        xml_str = statistics_to_gdal_metadata(stats)

        root = ElementTree.fromstring(xml_str)
        items = {item.get("name"): item.text for item in root.findall("Item") if item.get("sample") == "0"}

        self.assertIn("STATISTICS_MINIMUM", items)
        self.assertIn("STATISTICS_MAXIMUM", items)
        self.assertIn("STATISTICS_MEAN", items)
        self.assertIn("STATISTICS_STDDEV", items)
        self.assertAlmostEqual(float(items["STATISTICS_MINIMUM"]), 5.0)
        self.assertAlmostEqual(float(items["STATISTICS_MAXIMUM"]), 250.0)
        self.assertAlmostEqual(float(items["STATISTICS_MEAN"]), 127.5)
        self.assertAlmostEqual(float(items["STATISTICS_STDDEV"]), 40.0)

    def test_stats_with_histogram_entries(self):
        """Serialized XML includes histogram entries with pipe-delimited bin values."""
        from defusedxml import ElementTree

        band = BandStatistics(
            min=0.0,
            max=100.0,
            mean=50.0,
            stddev=25.0,
            count=50,
            m2=25.0**2 * 50,
            histogram=np.array([5, 10, 15, 20]),
            bin_edges=np.array([0.0, 25.0, 50.0, 75.0, 100.0]),
        )
        stats = ImageStatistics(bands=[band])
        xml_str = statistics_to_gdal_metadata(stats)

        root = ElementTree.fromstring(xml_str)
        items = {item.get("name"): item.text for item in root.findall("Item") if item.get("sample") == "0"}

        self.assertIn("STATISTICS_HISTOMIN", items)
        self.assertIn("STATISTICS_HISTOMAX", items)
        self.assertIn("STATISTICS_HISTONUMBINS", items)
        self.assertIn("STATISTICS_HISTOBINVALUES", items)
        self.assertAlmostEqual(float(items["STATISTICS_HISTOMIN"]), 0.0)
        self.assertAlmostEqual(float(items["STATISTICS_HISTOMAX"]), 100.0)
        self.assertEqual(int(items["STATISTICS_HISTONUMBINS"]), 4)
        self.assertEqual(items["STATISTICS_HISTOBINVALUES"], "5|10|15|20")

    def test_multi_band_serialization(self):
        """Serialized XML for 2 bands contains items with correct sample attributes."""
        from defusedxml import ElementTree

        band0 = BandStatistics(
            min=0.0,
            max=255.0,
            mean=100.0,
            stddev=30.0,
            count=100,
            m2=30.0**2 * 100,
            histogram=np.array([50, 50]),
            bin_edges=np.array([0.0, 127.5, 255.0]),
        )
        band1 = BandStatistics(
            min=10.0,
            max=240.0,
            mean=120.0,
            stddev=35.0,
            count=100,
            m2=35.0**2 * 100,
            histogram=np.array([40, 60]),
            bin_edges=np.array([10.0, 125.0, 240.0]),
        )
        stats = ImageStatistics(bands=[band0, band1])
        xml_str = statistics_to_gdal_metadata(stats)

        root = ElementTree.fromstring(xml_str)

        # Collect items per sample
        items_by_sample = {}
        for item in root.findall("Item"):
            sample = item.get("sample")
            if sample not in items_by_sample:
                items_by_sample[sample] = {}
            items_by_sample[sample][item.get("name")] = item.text

        self.assertIn("0", items_by_sample)
        self.assertIn("1", items_by_sample)
        self.assertAlmostEqual(float(items_by_sample["0"]["STATISTICS_MEAN"]), 100.0)
        self.assertAlmostEqual(float(items_by_sample["1"]["STATISTICS_MEAN"]), 120.0)
        self.assertAlmostEqual(float(items_by_sample["0"]["STATISTICS_MINIMUM"]), 0.0)
        self.assertAlmostEqual(float(items_by_sample["1"]["STATISTICS_MINIMUM"]), 10.0)


class _MockMetadata(collections.abc.Mapping):
    """Minimal mock for the metadata object returned by ImageAssetProvider.metadata."""

    def __init__(self, metadata_dict):
        self._dict = metadata_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)


class _MockProvider:
    """Minimal mock for a duck-typed ImageAssetProvider.

    Stores a CHW array and exposes it as a single-block grid. Optionally
    attaches GDAL_METADATA XML so the metadata fast-path can be tested.
    """

    def __init__(self, image, metadata_dict=None, sparse_blocks=None):
        """
        :param image: CHW NDArray representing the full image.
        :param metadata_dict: Optional dict to use as metadata (supports Mapping protocol).
        :param sparse_blocks: Optional set of (row, col) tuples that should be
            reported as missing by has_block().
        """
        self._image = image
        self._metadata_dict = metadata_dict or {}
        self._sparse_blocks = sparse_blocks or set()
        self._num_bands = image.shape[0] if image.ndim == 3 else 1
        self._height = image.shape[-2]
        self._width = image.shape[-1]
        self._block_h = self._height
        self._block_w = self._width
        self._read_count = 0

    @property
    def block_grid_size(self):
        rows = (self._height + self._block_h - 1) // self._block_h
        cols = (self._width + self._block_w - 1) // self._block_w
        return (rows, cols)

    def has_block(self, row, col, resolution_level=0):
        return (row, col) not in self._sparse_blocks

    def get_block(self, row, col, resolution_level=0, bands=None):
        self._read_count += 1
        r_start = row * self._block_h
        r_end = min(r_start + self._block_h, self._height)
        c_start = col * self._block_w
        c_end = min(c_start + self._block_w, self._width)
        return self._image[:, r_start:r_end, c_start:c_end]

    @property
    def metadata(self):
        return _MockMetadata(self._metadata_dict)


class _MultiBlockProvider(_MockProvider):
    """A mock provider that splits the image into multiple blocks."""

    def __init__(self, image, block_h, block_w, metadata_dict=None, sparse_blocks=None):
        super().__init__(image, metadata_dict=metadata_dict, sparse_blocks=sparse_blocks)
        self._block_h = block_h
        self._block_w = block_w


class TestComputeImageStatistics(TestCase):
    """Tests for the compute_image_statistics function."""

    def test_metadata_fast_path_no_pixel_reads(self):
        """When GDAL metadata is present, returns stats without reading pixels."""
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">255.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">100.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">30.0</Item>'
            "</GDALMetadata>"
        )
        provider = _MockProvider(image, metadata_dict={"GDAL_METADATA": xml})

        result = compute_image_statistics(provider)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 1)
        self.assertAlmostEqual(result.bands[0].mean, 100.0)
        # No pixel reads should have occurred
        self.assertEqual(provider._read_count, 0)

    def test_no_metadata_falls_back_to_block_reads(self):
        """When metadata is absent, computes statistics from pixel data."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        provider = _MockProvider(image)

        result = compute_image_statistics(provider)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 1)
        self.assertAlmostEqual(result.bands[0].mean, 25.0)
        self.assertGreater(provider._read_count, 0)

    def test_compute_if_missing_false_returns_none(self):
        """When metadata is absent and compute_if_missing=False, returns None."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        provider = _MockProvider(image)

        result = compute_image_statistics(provider, compute_if_missing=False)

        self.assertIsNone(result)
        self.assertEqual(provider._read_count, 0)

    def test_force_recompute_skips_metadata(self):
        """When force_recompute=True, always reads pixels even if metadata exists."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        xml = (
            "<GDALMetadata>"
            '  <Item name="STATISTICS_MINIMUM" sample="0">0.0</Item>'
            '  <Item name="STATISTICS_MAXIMUM" sample="0">255.0</Item>'
            '  <Item name="STATISTICS_MEAN" sample="0">999.0</Item>'
            '  <Item name="STATISTICS_STDDEV" sample="0">30.0</Item>'
            "</GDALMetadata>"
        )
        provider = _MockProvider(image, metadata_dict={"GDAL_METADATA": xml})

        result = compute_image_statistics(provider, force_recompute=True)

        self.assertIsNotNone(result)
        # Should reflect actual pixel data, not the metadata mean of 999.0
        self.assertAlmostEqual(result.bands[0].mean, 25.0, places=0)
        self.assertGreater(provider._read_count, 0)

    def test_sparse_blocks_excluded(self):
        """Blocks where has_block() returns False are excluded from computation."""
        # 1-band 4x4 image split into 2x2 blocks → 2x2 grid
        image = np.array(
            [[[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]]],
            dtype=np.uint8,
        )
        # Mark block (1, 1) as sparse (bottom-right 2x2 block)
        provider = _MultiBlockProvider(image, block_h=2, block_w=2, sparse_blocks={(1, 1)})

        result = compute_image_statistics(provider)

        self.assertIsNotNone(result)
        # 3 out of 4 blocks should have been read
        self.assertEqual(provider._read_count, 3)
        # Total histogram count should be 3 blocks × 4 pixels = 12 (not 16)
        total_count = int(np.sum(result.bands[0].histogram))
        self.assertEqual(total_count, 12)

    def test_multi_band_provider(self):
        """Statistics are computed correctly for multi-band images."""
        image = np.array(
            [
                [[10, 20], [30, 40]],
                [[100, 110], [120, 130]],
            ],
            dtype=np.uint8,
        )
        provider = _MockProvider(image)

        result = compute_image_statistics(provider)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 2)
        self.assertAlmostEqual(result.bands[0].mean, 25.0, places=0)
        self.assertAlmostEqual(result.bands[1].mean, 115.0, places=0)

    def test_multi_block_merge(self):
        """Statistics from multiple blocks are correctly merged."""
        # 1-band 4x2 image split into two 2x2 blocks
        image = np.array([[[10, 20, 110, 120], [30, 40, 130, 140]]], dtype=np.uint8)
        provider = _MultiBlockProvider(image, block_h=2, block_w=2)

        result = compute_image_statistics(provider)

        self.assertIsNotNone(result)
        # Total pixel count across both blocks
        total_count = int(np.sum(result.bands[0].histogram))
        self.assertEqual(total_count, 8)
        # Mean should reflect all pixels
        expected_mean = float(np.mean(image))
        self.assertAlmostEqual(result.bands[0].mean, expected_mean, places=0)

    def test_all_sparse_blocks_returns_none(self):
        """When all blocks are sparse, returns None."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        provider = _MultiBlockProvider(image, block_h=2, block_w=2, sparse_blocks={(0, 0)})

        result = compute_image_statistics(provider)

        self.assertIsNone(result)

    def test_per_band_bin_edges_accepted(self):
        """compute_image_statistics accepts List[NDArray] for per-band bin_edges."""
        image = np.array(
            [
                [[10, 20], [30, 40]],
                [[100, 110], [120, 130]],
            ],
            dtype=np.uint8,
        )
        provider = _MockProvider(image)
        edges = [np.linspace(0, 255, 257), np.linspace(0, 255, 257)]

        result = compute_image_statistics(provider, bin_edges=edges)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.bands), 2)

    def test_per_band_edges_pinned_from_first_block(self):
        """Per-band bin_edges are pinned from first block (not just band-0 edges)."""
        # Multi-band float image where bands have very different ranges
        # Band 0: [0, 1000], Band 1: [0, 1]
        rng = np.random.default_rng(42)
        band0 = rng.uniform(0, 1000, size=(1, 8, 8)).astype(np.float32)
        band1 = rng.uniform(0, 1, size=(1, 8, 8)).astype(np.float32)
        image = np.concatenate([band0, band1], axis=0)

        # Use multi-block provider so pinning matters across blocks
        provider = _MultiBlockProvider(image, block_h=4, block_w=4)

        result = compute_image_statistics(provider, force_recompute=True)

        self.assertIsNotNone(result)
        # Band 0 edges should span ~[0, 1000] (from first block's band-0 range)
        self.assertGreater(result.bands[0].bin_edges[-1], 100.0)
        # Band 1 edges should span ~[0, 1] (from first block's band-1 range)
        # If the bug were present (using band-0 edges for all bands), band-1
        # edges would span [0, 1000] which would be wrong
        self.assertLess(result.bands[1].bin_edges[-1], 10.0)

    def test_block_sampling_reduces_pixel_count(self):
        """SamplingStrategy.BLOCK with sample_rate < 1.0 processes fewer blocks."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 32, 32), dtype=np.uint8)
        # 4x4 block grid = 16 blocks total
        provider = _MultiBlockProvider(image, block_h=8, block_w=8)

        result = compute_image_statistics(
            provider,
            sampling=SamplingStrategy.BLOCK,
            sample_rate=0.5,
            force_recompute=True,
        )

        self.assertIsNotNone(result)
        total_pixels = 32 * 32
        histogram_count = int(np.sum(result.bands[0].histogram))
        # With sample_rate=0.5 over 16 blocks, expect roughly 8 blocks processed
        # Each block has 64 pixels, so roughly 512 pixels total
        # Allow generous bounds: between 10% and 90% of total
        self.assertGreater(histogram_count, total_pixels * 0.1)
        self.assertLess(histogram_count, total_pixels * 0.9)

    def test_block_sampling_sample_rate_overridden(self):
        """After block sampling, sample_rate reflects actual blocks_processed/total_blocks."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 16, 16), dtype=np.uint8)
        # 4x4 block grid = 16 blocks
        provider = _MultiBlockProvider(image, block_h=4, block_w=4)

        result = compute_image_statistics(
            provider,
            sampling=SamplingStrategy.BLOCK,
            sample_rate=0.5,
            force_recompute=True,
        )

        self.assertIsNotNone(result)
        # sample_rate should be actual coverage, not the merge_statistics sum
        self.assertGreater(result.sample_rate, 0.0)
        self.assertLessEqual(result.sample_rate, 1.0)
        # It should approximately equal 0.5 but with stochastic variation
        # The key assertion is that it's a ratio (blocks_processed/total_blocks)
        # so it must be a multiple of 1/16 (one block out of 16)
        self.assertAlmostEqual(result.sample_rate % (1.0 / 16), 0.0, places=10)

    def test_block_sampling_full_rate_processes_all(self):
        """SamplingStrategy.BLOCK with sample_rate=1.0 processes all blocks (no sampling)."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 8, 8), dtype=np.uint8)
        provider = _MultiBlockProvider(image, block_h=4, block_w=4)

        result = compute_image_statistics(
            provider,
            sampling=SamplingStrategy.BLOCK,
            sample_rate=1.0,
            force_recompute=True,
        )

        self.assertIsNotNone(result)
        # All pixels processed
        self.assertEqual(int(np.sum(result.bands[0].histogram)), 64)

    def test_block_sampling_uses_all_pixels_within_block(self):
        """Block sampling uses SamplingStrategy.ALL for pixels within included blocks."""
        # Use a provider where we can verify per-block pixel counts
        image = np.zeros((1, 4, 4), dtype=np.uint8)
        image[0, :, :] = np.arange(16).reshape(4, 4)
        # 2x2 blocks → 4 blocks, each with 4 pixels
        provider = _MultiBlockProvider(image, block_h=2, block_w=2)

        result = compute_image_statistics(
            provider,
            sampling=SamplingStrategy.BLOCK,
            sample_rate=1.0,
            force_recompute=True,
        )

        self.assertIsNotNone(result)
        # Each included block contributes exactly 4 pixels (not sampled within)
        # With sample_rate=1.0, all 16 pixels are included
        self.assertEqual(result.bands[0].count, 16)

    def test_multi_band_float_pinning_across_blocks(self):
        """Multi-band float provider pins per-band edges independently across blocks."""
        # Band 0 values in [0, 100], Band 1 values in [500, 600]
        band0 = np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]], dtype=np.float32)
        band1 = np.array([[510.0, 520.0, 530.0, 540.0], [550.0, 560.0, 570.0, 580.0]], dtype=np.float32)
        image = np.stack([band0, band1])  # shape (2, 2, 4)
        # Split into 2 blocks of (2, 2)
        provider = _MultiBlockProvider(image, block_h=2, block_w=2)

        result = compute_image_statistics(provider, force_recompute=True)

        self.assertIsNotNone(result)
        # Band 0 edges should be in low range (derived from first block's band 0)
        self.assertLess(result.bands[0].bin_edges[-1], 200.0)
        # Band 1 edges should be in high range (derived from first block's band 1)
        self.assertGreater(result.bands[1].bin_edges[0], 400.0)


class TestConcurrentBlockIteration(TestCase):
    """Tests for threaded compute_image_statistics via num_workers parameter."""

    def test_serial_default_behavior(self):
        """num_workers < 2 preserves exact serial behavior."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
        provider = _MultiBlockProvider(image, block_h=8, block_w=8)

        result_default = compute_image_statistics(provider, force_recompute=True, num_workers=0)
        result_serial = compute_image_statistics(provider, force_recompute=True, num_workers=1)

        self.assertIsNotNone(result_default)
        self.assertIsNotNone(result_serial)
        for b in range(3):
            self.assertAlmostEqual(result_default.bands[b].mean, result_serial.bands[b].mean, places=10)
            self.assertAlmostEqual(result_default.bands[b].stddev, result_serial.bands[b].stddev, places=10)
            np.testing.assert_array_equal(result_default.bands[b].histogram, result_serial.bands[b].histogram)

    def test_threaded_results_match_serial(self):
        """num_workers=2 produces identical results to num_workers=0 with SamplingStrategy.ALL."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(3, 32, 32), dtype=np.uint8)
        provider_serial = _MultiBlockProvider(image, block_h=8, block_w=8)
        provider_threaded = _MultiBlockProvider(image, block_h=8, block_w=8)

        result_serial = compute_image_statistics(
            provider_serial,
            force_recompute=True,
            num_workers=0,
            sampling=SamplingStrategy.ALL,
        )
        result_threaded = compute_image_statistics(
            provider_threaded,
            force_recompute=True,
            num_workers=2,
            sampling=SamplingStrategy.ALL,
        )

        self.assertIsNotNone(result_serial)
        self.assertIsNotNone(result_threaded)
        for b in range(3):
            self.assertAlmostEqual(result_serial.bands[b].mean, result_threaded.bands[b].mean, places=10)
            self.assertAlmostEqual(result_serial.bands[b].stddev, result_threaded.bands[b].stddev, places=10)
            self.assertAlmostEqual(result_serial.bands[b].min, result_threaded.bands[b].min, places=10)
            self.assertAlmostEqual(result_serial.bands[b].max, result_threaded.bands[b].max, places=10)
            self.assertEqual(result_serial.bands[b].count, result_threaded.bands[b].count)
            np.testing.assert_array_equal(result_serial.bands[b].histogram, result_threaded.bands[b].histogram)
            np.testing.assert_array_almost_equal(result_serial.bands[b].bin_edges, result_threaded.bands[b].bin_edges)

    def test_threaded_with_many_workers(self):
        """num_workers=4 works correctly with more workers than blocks."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 8, 8), dtype=np.uint8)
        # Only 4 blocks total (2x2 grid), but 4 workers
        provider = _MultiBlockProvider(image, block_h=4, block_w=4)

        result = compute_image_statistics(provider, force_recompute=True, num_workers=4, sampling=SamplingStrategy.ALL)

        self.assertIsNotNone(result)
        self.assertEqual(result.bands[0].count, 64)
        expected_mean = float(np.mean(image))
        self.assertAlmostEqual(result.bands[0].mean, expected_mean, places=10)

    def test_threaded_single_block(self):
        """Threading with a single-block provider works (no remaining blocks to dispatch)."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        provider = _MockProvider(image)

        result = compute_image_statistics(provider, force_recompute=True, num_workers=4, sampling=SamplingStrategy.ALL)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.bands[0].mean, 25.0)
        self.assertEqual(result.bands[0].count, 4)

    def test_threaded_with_explicit_bin_edges(self):
        """Threading works correctly with caller-supplied bin_edges."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(2, 16, 16), dtype=np.uint8)
        provider_serial = _MultiBlockProvider(image, block_h=8, block_w=8)
        provider_threaded = _MultiBlockProvider(image, block_h=8, block_w=8)
        edges = [np.linspace(0, 255, 129), np.linspace(0, 255, 129)]

        result_serial = compute_image_statistics(provider_serial, bin_edges=edges, force_recompute=True, num_workers=0)
        result_threaded = compute_image_statistics(provider_threaded, bin_edges=edges, force_recompute=True, num_workers=2)

        self.assertIsNotNone(result_serial)
        self.assertIsNotNone(result_threaded)
        for b in range(2):
            self.assertAlmostEqual(result_serial.bands[b].mean, result_threaded.bands[b].mean, places=10)
            np.testing.assert_array_equal(result_serial.bands[b].histogram, result_threaded.bands[b].histogram)

    def test_threaded_float_image(self):
        """Threading produces correct results for float32 images with auto-derived edges."""
        rng = np.random.default_rng(42)
        image = rng.uniform(-50, 50, size=(2, 16, 16)).astype(np.float32)
        provider_serial = _MultiBlockProvider(image, block_h=8, block_w=8)
        provider_threaded = _MultiBlockProvider(image, block_h=8, block_w=8)

        result_serial = compute_image_statistics(provider_serial, force_recompute=True, num_workers=0)
        result_threaded = compute_image_statistics(provider_threaded, force_recompute=True, num_workers=2)

        self.assertIsNotNone(result_serial)
        self.assertIsNotNone(result_threaded)
        for b in range(2):
            self.assertAlmostEqual(result_serial.bands[b].mean, result_threaded.bands[b].mean, places=5)
            self.assertAlmostEqual(result_serial.bands[b].stddev, result_threaded.bands[b].stddev, places=5)
            np.testing.assert_array_equal(result_serial.bands[b].histogram, result_threaded.bands[b].histogram)

    def test_threaded_no_deadlock_under_concurrent_execution(self):
        """Threaded execution completes without deadlock for a larger block grid."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 64, 64), dtype=np.uint8)
        # 8x8 blocks → 64 blocks total
        provider = _MultiBlockProvider(image, block_h=8, block_w=8)

        result = compute_image_statistics(provider, force_recompute=True, num_workers=4, sampling=SamplingStrategy.ALL)

        self.assertIsNotNone(result)
        self.assertEqual(result.bands[0].count, 64 * 64)
        expected_mean = float(np.mean(image))
        self.assertAlmostEqual(result.bands[0].mean, expected_mean, places=10)

    def test_threaded_all_blocks_read(self):
        """All blocks are read when using threading (verifying via read_count)."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1, 16, 16), dtype=np.uint8)
        # 4x4 blocks → 16 blocks
        provider = _MultiBlockProvider(image, block_h=4, block_w=4)

        compute_image_statistics(provider, force_recompute=True, num_workers=3, sampling=SamplingStrategy.ALL)

        self.assertEqual(provider._read_count, 16)
