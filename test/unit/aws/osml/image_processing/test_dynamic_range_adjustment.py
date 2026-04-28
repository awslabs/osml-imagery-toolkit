#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import numpy as np

from aws.osml.image_processing.dynamic_range_adjustment import DRAParameters, dynamic_range_adjust


class TestDRAParameters(unittest.TestCase):
    """Tests for the migrated DRAParameters class."""

    def test_from_counts(self):
        """Copied from the original gdal test — verifies identical behavior."""
        counts = [0] * 1024
        counts[1:99] = [1] * (99 - 1)
        counts[100:400] = [200] * (400 - 100)
        counts[1022] = 1

        dra_parameters = DRAParameters.from_counts(counts=counts)

        self.assertEqual(dra_parameters.actual_min_value, 1)
        self.assertEqual(dra_parameters.actual_max_value, 1022)
        self.assertAlmostEqual(dra_parameters.suggested_min_value, 47, delta=1)
        self.assertAlmostEqual(dra_parameters.suggested_max_value, 506, delta=1)

    def test_from_counts_list_vs_ndarray_identical(self):
        """from_counts() produces identical results for list and NDArray input."""
        counts_list = [0] * 256
        counts_list[10:50] = [5] * 40
        counts_list[50:200] = [100] * 150
        counts_list[200:250] = [10] * 50

        counts_array = np.array(counts_list, dtype=np.float64)

        result_list = DRAParameters.from_counts(counts=counts_list)
        result_array = DRAParameters.from_counts(counts=counts_array)

        self.assertAlmostEqual(result_list.suggested_min_value, result_array.suggested_min_value)
        self.assertAlmostEqual(result_list.suggested_max_value, result_array.suggested_max_value)
        self.assertAlmostEqual(result_list.actual_min_value, result_array.actual_min_value)
        self.assertAlmostEqual(result_list.actual_max_value, result_array.actual_max_value)

    def test_from_counts_known_histogram(self):
        """Known histogram produces expected suggested min/max values."""
        # Uniform distribution across bins 50-200
        counts = [0] * 256
        counts[50:201] = [100] * 151

        result = DRAParameters.from_counts(counts=counts)

        # Actual min/max should be at the first/last non-zero bins
        self.assertAlmostEqual(result.actual_min_value, 50.0)
        self.assertAlmostEqual(result.actual_max_value, 200.0)

        # Suggested values should be within the actual range
        self.assertGreaterEqual(result.suggested_min_value, result.actual_min_value)
        self.assertLessEqual(result.suggested_max_value, result.actual_max_value)


class TestBuildLut(unittest.TestCase):
    """Tests for DRAParameters.build_lut()."""

    def test_dra_mode_lut_shape(self):
        """DRA mode LUT has 256 entries for uint8 input."""
        params = DRAParameters(
            suggested_min_value=20.0, suggested_max_value=230.0, actual_min_value=0.0, actual_max_value=255.0
        )
        lut = params.build_lut(np.uint8, np.uint8, range_adjustment="dra")

        self.assertEqual(lut.shape, (256,))
        self.assertEqual(lut.dtype, np.uint8)

    def test_minmax_mode_lut_shape(self):
        """Minmax mode LUT has 256 entries for uint8 input."""
        params = DRAParameters(
            suggested_min_value=20.0, suggested_max_value=230.0, actual_min_value=10.0, actual_max_value=240.0
        )
        lut = params.build_lut(np.uint8, np.uint8, range_adjustment="minmax")

        self.assertEqual(lut.shape, (256,))
        # actual_min (10) should map to 0, actual_max (240) should map to 255
        self.assertEqual(lut[10], 0)
        self.assertEqual(lut[240], 255)

    def test_uint16_lut_shape(self):
        """LUT for uint16 input has 65536 entries."""
        params = DRAParameters(
            suggested_min_value=1000.0, suggested_max_value=60000.0, actual_min_value=0.0, actual_max_value=65535.0
        )
        lut = params.build_lut(np.uint16, np.uint8, range_adjustment="dra")

        self.assertEqual(lut.shape, (65536,))
        self.assertEqual(lut.dtype, np.uint8)

    def test_zero_width_range_produces_constant(self):
        """Zero-width source range produces constant mid-range output."""
        params = DRAParameters(
            suggested_min_value=128.0, suggested_max_value=128.0, actual_min_value=128.0, actual_max_value=128.0
        )
        lut = params.build_lut(np.uint8, np.uint8, range_adjustment="dra")

        # All entries should be the same mid-range value
        self.assertTrue(np.all(lut == lut[0]))
        self.assertEqual(lut[0], 127)  # floor(255/2)

    def test_lut_values_clipped_to_output_range(self):
        """All LUT values are within the valid range of the output dtype."""
        params = DRAParameters(
            suggested_min_value=50.0, suggested_max_value=200.0, actual_min_value=0.0, actual_max_value=255.0
        )
        lut = params.build_lut(np.uint8, np.uint8, range_adjustment="dra")

        self.assertTrue(np.all(lut >= 0))
        self.assertTrue(np.all(lut <= 255))

    def test_unsupported_range_adjustment_raises(self):
        """Unsupported range_adjustment raises ValueError."""
        params = DRAParameters(
            suggested_min_value=0.0, suggested_max_value=255.0, actual_min_value=0.0, actual_max_value=255.0
        )

        with self.assertRaises(ValueError) as ctx:
            params.build_lut(np.uint8, np.uint8, range_adjustment="invalid")
        self.assertIn("Unsupported range_adjustment", str(ctx.exception))

    def test_float_input_dtype_raises(self):
        """Float input dtype raises ValueError."""
        params = DRAParameters(suggested_min_value=0.0, suggested_max_value=1.0, actual_min_value=0.0, actual_max_value=1.0)

        with self.assertRaises(ValueError):
            params.build_lut(np.float32, np.uint8)


class TestDynamicRangeAdjust(unittest.TestCase):
    """Tests for the dynamic_range_adjust function."""

    def _make_luts(self, image, num_bins=256, range_adjustment="dra"):
        """Helper: compute stats, build DRAParameters, then build LUTs."""
        from aws.osml.image_processing.statistics import compute_statistics

        stats = compute_statistics(image, num_bins=num_bins)
        luts = []
        for band in stats.bands:
            params = DRAParameters.from_counts(
                counts=band.histogram,
                first_bucket_value=float(band.bin_edges[0]),
                last_bucket_value=float(band.bin_edges[-1]),
            )
            luts.append(params.build_lut(image.dtype, np.uint8, range_adjustment=range_adjustment))
        return luts

    def test_dra_mode_known_array(self):
        """DRA mode on a known array produces uint8 output in valid range."""
        image = np.array([[[0, 50], [200, 255]]], dtype=np.uint8)
        luts = self._make_luts(image, range_adjustment="dra")

        result = dynamic_range_adjust(image, luts)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_minmax_mode_known_array(self):
        """Minmax mode maps actual min toward 0 and actual max toward 255."""
        image = np.array([[[50, 100], [150, 200]]], dtype=np.uint8)
        luts = self._make_luts(image, range_adjustment="minmax")

        result = dynamic_range_adjust(image, luts)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)
        self.assertLessEqual(result[0, 0, 0], 5)
        self.assertGreaterEqual(result[0, 1, 1], 250)

    def test_constant_band_no_exception(self):
        """A band with all identical values does not raise an exception."""
        image = np.full((1, 4, 4), 128, dtype=np.uint8)
        luts_dra = self._make_luts(image, range_adjustment="dra")
        luts_minmax = self._make_luts(image, range_adjustment="minmax")

        result_dra = dynamic_range_adjust(image, luts_dra)
        result_minmax = dynamic_range_adjust(image, luts_minmax)

        self.assertEqual(result_dra.shape, image.shape)
        self.assertEqual(result_minmax.shape, image.shape)

    def test_per_band_independence(self):
        """Each band is adjusted independently using its own LUT."""
        band0 = np.array([[0, 25], [25, 50]], dtype=np.uint8)
        band1 = np.array([[200, 220], [240, 255]], dtype=np.uint8)
        image = np.stack([band0, band1])

        luts = self._make_luts(image, range_adjustment="minmax")
        result = dynamic_range_adjust(image, luts)

        self.assertEqual(result.shape, (2, 2, 2))
        # Band 0: min pixel (0) → near 0, max pixel (50) → near 255
        self.assertLessEqual(result[0, 0, 0], 5)
        self.assertGreaterEqual(result[0, 1, 1], 250)
        # Band 1: min pixel (200) → near 0, max pixel (255) → near 255
        self.assertLessEqual(result[1, 0, 0], 10)
        self.assertGreaterEqual(result[1, 1, 1], 245)

    def test_output_dtype_matches_lut(self):
        """Output dtype matches the LUT's dtype."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint16)
        luts = self._make_luts(image, range_adjustment="minmax")

        result = dynamic_range_adjust(image, luts)
        self.assertEqual(result.dtype, np.uint8)

    def test_2d_input_treated_as_single_band(self):
        """A 2-D (H, W) input is treated as a single-band image."""
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        luts = self._make_luts(image[np.newaxis, :, :], range_adjustment="minmax")

        result = dynamic_range_adjust(image, luts)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (2, 2))

    def test_band_count_mismatch_raises(self):
        """Mismatched band count between image and luts raises ValueError."""
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        luts = self._make_luts(np.zeros((1, 4, 4), dtype=np.uint8))

        with self.assertRaises(ValueError) as ctx:
            dynamic_range_adjust(image, luts)
        self.assertIn("Band count mismatch", str(ctx.exception))

    def test_no_input_mutation(self):
        """dynamic_range_adjust does not mutate the input array."""
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        original = image.copy()
        luts = self._make_luts(image, range_adjustment="minmax")

        dynamic_range_adjust(image, luts)

        np.testing.assert_array_equal(image, original)

    def test_output_values_clipped_to_dtype_range(self):
        """All output values are within the valid range of the output dtype."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 65536, size=(3, 16, 16), dtype=np.uint16)
        luts = self._make_luts(image, range_adjustment="dra")

        result = dynamic_range_adjust(image, luts)

        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_multi_band_dra_mode(self):
        """DRA mode works correctly with multi-band images."""
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
        luts = self._make_luts(image, range_adjustment="dra")

        result = dynamic_range_adjust(image, luts)

        self.assertEqual(result.shape, (3, 8, 8))
        self.assertEqual(result.dtype, np.uint8)

    def test_luts_reusable_across_blocks(self):
        """Same LUTs can be applied to multiple blocks without recomputation."""
        full_image = np.array(
            [[[0, 50, 100, 150], [200, 255, 128, 64], [32, 96, 160, 224], [16, 48, 80, 112]]],
            dtype=np.uint8,
        )
        luts = self._make_luts(full_image, range_adjustment="dra")

        block_a = full_image[:, :2, :2]
        block_b = full_image[:, 2:, 2:]

        result_a = dynamic_range_adjust(block_a, luts)
        result_b = dynamic_range_adjust(block_b, luts)

        self.assertEqual(result_a.shape, block_a.shape)
        self.assertEqual(result_b.shape, block_b.shape)
        self.assertEqual(result_a.dtype, np.uint8)
        self.assertEqual(result_b.dtype, np.uint8)

    def test_lut_identity_is_object_reuse(self):
        """Verifies the LUT is not rebuilt — same object applied to different blocks."""
        params = DRAParameters(
            suggested_min_value=0.0,
            suggested_max_value=255.0,
            actual_min_value=0.0,
            actual_max_value=255.0,
        )
        lut = params.build_lut(np.uint8, np.uint8, range_adjustment="dra")

        block_a = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        block_b = np.array([[[200, 210], [220, 230]]], dtype=np.uint8)

        # Same lut object used for both calls
        result_a = dynamic_range_adjust(block_a, [lut])
        result_b = dynamic_range_adjust(block_b, [lut])

        self.assertEqual(result_a.shape, block_a.shape)
        self.assertEqual(result_b.shape, block_b.shape)
