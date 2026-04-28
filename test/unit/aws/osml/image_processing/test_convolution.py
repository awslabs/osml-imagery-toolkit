#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import numpy as np

from aws.osml.image_processing.convolution import sips_convolve, sips_correlate


class TestSipsConvolve(unittest.TestCase):
    """Tests for the sips_convolve function."""

    def test_5x5_sips_table_2_25(self):
        """SIPS Table 2.25: 5x5 convolution verification.

        Uses the asymmetric 5x5 sharpening kernel (Table 2.17) and the
        10x10 input matrix (Table 2.18) from SIPS Section 2.4.7.1.
        The expected output is Table 2.25 (clipped to 11-bit range).
        """
        # Table 2.17: 5x5 Convolution Kernel
        kernel = np.array(
            [
                [-0.0004, -0.0022, -0.0032, -0.0023, -0.0005],
                [-0.0021, -0.0112, -0.0220, -0.0122, -0.0016],
                [-0.0031, -0.0021, 1.1180, -0.0019, -0.0029],
                [-0.0020, -0.0102, -0.0200, -0.0092, -0.0017],
                [-0.0003, -0.0019, -0.0030, -0.0018, -0.0002],
            ],
            dtype=np.float64,
        )

        # Table 2.18: 5x5 Convolution Input Matrix
        image = np.array(
            [
                [139, 310, 147, 220, 205, 224, 215, 314, 223, 138],
                [221, 199, 198, 179, 201, 253, 223, 181, 1982, 2047],
                [178, 239, 151, 119, 151, 232, 1869, 1901, 1965, 1931],
                [278, 232, 181, 88, 119, 1901, 1929, 1869, 1482, 1588],
                [1889, 1928, 2047, 1989, 1897, 1931, 1597, 1655, 1648, 1642],
                [1971, 2028, 1949, 1982, 1948, 1545, 1582, 1536, 1551, 1563],
                [892, 1022, 879, 969, 935, 1970, 1693, 1656, 1619, 1579],
                [876, 992, 970, 873, 1022, 1838, 1955, 1665, 1618, 1608],
                [1001, 927, 971, 913, 933, 931, 1951, 1974, 1539, 1591],
                [944, 996, 950, 875, 982, 923, 963, 2003, 1973, 1499],
            ],
            dtype=np.float64,
        )

        # Table 2.25: 5x5 Convolution Example Output Matrix (clipped to [0, 2047])
        expected = np.array(
            [
                [131, 323, 142, 224, 204, 216, 189, 256, 77, 0],
                [222, 197, 200, 179, 195, 226, 144, 68, 2047, 2047],
                [158, 227, 132, 94, 105, 153, 1959, 1977, 2016, 1967],
                [187, 135, 77, 0, 0, 1980, 1982, 1893, 1451, 1567],
                [1973, 2019, 2047, 2047, 1972, 1978, 1587, 1652, 1653, 1648],
                [2044, 2047, 2018, 2047, 2003, 1528, 1566, 1521, 1542, 1557],
                [834, 978, 819, 916, 871, 2014, 1695, 1655, 1620, 1577],
                [859, 989, 966, 853, 1007, 1891, 1994, 1658, 1613, 1607],
                [1006, 923, 974, 909, 914, 884, 2003, 2008, 1516, 1579],
                [942, 1001, 950, 867, 979, 882, 880, 2031, 2010, 1483],
            ],
            dtype=np.float64,
        )

        result = sips_convolve(image, kernel)
        # Clip to 11-bit range as SIPS does
        result_clipped = np.clip(np.round(result), 0, 2047)

        np.testing.assert_array_almost_equal(
            result_clipped,
            expected,
            decimal=0,
            err_msg="5x5 convolution output does not match SIPS Table 2.25",
        )

    def test_4x4_sips_table_2_26(self):
        """SIPS Table 2.26: 4x4 convolution verification.

        Uses the 4x4 kernel (Table 2.19) and the 10x10 input matrix
        (Table 2.20) from SIPS Section 2.4.7.1. The expected output
        is Table 2.26 (clipped to 11-bit range).
        """
        # Table 2.19: 4x4 Convolution Kernel
        kernel = np.array(
            [
                [0.0034181250, -0.0307631250, -0.0307631250, 0.0034181250],
                [-0.0512693750, 0.4614243750, 0.4614243750, -0.0512693750],
                [-0.0170900000, 0.1538100000, 0.1538100000, -0.0170900000],
                [0.0024412500, -0.0219712500, -0.0219712500, 0.0024412500],
            ],
            dtype=np.float64,
        )

        # Table 2.20: 4x4 Input Convolution Matrix
        image = np.array(
            [
                [0, 0, 0, 0, 100, 200, 300, 400, 500, 600],
                [0, 0, 100, 200, 300, 400, 500, 600, 700, 800],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
                [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
                [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800],
                [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
                [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2047, 2047],
                [1500, 1600, 1700, 1800, 1900, 2000, 2047, 2047, 2047, 2047],
            ],
            dtype=np.float64,
        )

        # Table 2.26: 4x4 Convolution Example Output Matrix (clipped to [0, 2047])
        expected = np.array(
            [
                [0, 21, 96, 183, 284, 384, 484, 584, 697, 697],
                [93, 192, 298, 400, 500, 600, 700, 800, 912, 912],
                [285, 400, 500, 600, 700, 800, 900, 1000, 1112, 1112],
                [487, 600, 700, 800, 900, 1000, 1100, 1200, 1312, 1312],
                [687, 800, 900, 1000, 1100, 1200, 1300, 1400, 1512, 1512],
                [887, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1712, 1712],
                [1087, 1200, 1300, 1400, 1500, 1600, 1700, 1801, 1919, 1919],
                [1287, 1400, 1500, 1600, 1700, 1801, 1908, 1994, 2037, 2037],
                [1509, 1622, 1722, 1822, 1925, 2005, 2041, 2047, 2047, 2047],
                [1403, 1516, 1616, 1716, 1817, 1910, 1991, 2044, 2047, 2047],
            ],
            dtype=np.float64,
        )

        result = sips_convolve(image, kernel)
        result_clipped = np.clip(np.round(result), 0, 2047)

        np.testing.assert_array_almost_equal(
            result_clipped,
            expected,
            decimal=0,
            err_msg="4x4 convolution output does not match SIPS Table 2.26",
        )

    def test_single_band_hw_input(self):
        """A 2-D (H, W) input is handled correctly and returns 2-D output."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
        image = np.ones((5, 5), dtype=np.float64) * 100.0

        result = sips_convolve(image, kernel)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (5, 5))
        # Uniform image with unit-sum kernel → output ≈ input
        np.testing.assert_allclose(result, 100.0, atol=1e-10)

    def test_chw_input_output_layout(self):
        """CHW input produces CHW output with correct shape."""
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        rng = np.random.default_rng(42)
        image = rng.random((3, 8, 8), dtype=np.float64)

        result = sips_convolve(image, kernel)

        self.assertEqual(result.shape, (3, 8, 8))

    def test_identity_kernel(self):
        """Convolution with a 1x1 identity kernel returns the input unchanged."""
        kernel = np.array([[1.0]], dtype=np.float64)
        image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

        result = sips_convolve(image, kernel)

        np.testing.assert_array_equal(result, image)

    def test_non_2d_kernel_raises(self):
        """Non-2-D kernel raises ValueError."""
        kernel = np.ones((3, 3, 3), dtype=np.float64)
        image = np.ones((5, 5), dtype=np.float64)

        with self.assertRaises(ValueError) as ctx:
            sips_convolve(image, kernel)
        self.assertIn("2-D", str(ctx.exception))

    def test_invalid_ndim_raises(self):
        """4-D input raises ValueError."""
        kernel = np.ones((3, 3), dtype=np.float64)
        image = np.ones((1, 1, 5, 5), dtype=np.float64)

        with self.assertRaises(ValueError):
            sips_convolve(image, kernel)

    def test_no_input_mutation(self):
        """sips_convolve does not mutate the input array."""
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        image = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)
        original = image.copy()

        sips_convolve(image, kernel)

        np.testing.assert_array_equal(image, original)


class TestSipsCorrelate(unittest.TestCase):
    """Tests for the sips_correlate function."""

    def test_lagrange_interpolation_sips_table_2_15(self):
        """SIPS Table 2.15: LaGrange interpolation (correlation) verification.

        Uses the 4x4 LaGrange kernel (Table 2.9) and the 10x10 input
        matrix (Table 2.11) from SIPS Section 2.3.7.1. The expected
        output is Table 2.15 (clipped to 11-bit range). This is the
        SIPS-specified correlation verification example.
        """
        # Table 2.9: 4x4 LaGrange Interpolation Kernel (16/32 x 8/32)
        kernel = np.array(
            [
                [0.0034181250, -0.0307631250, -0.0307631250, 0.0034181250],
                [-0.0512693750, 0.4614243750, 0.4614243750, -0.0512693750],
                [-0.0170900000, 0.1538100000, 0.1538100000, -0.0170900000],
                [0.0024412500, -0.0219712500, -0.0219712500, 0.0024412500],
            ],
            dtype=np.float64,
        )

        # Table 2.11: Input 10x10 Matrix for Matrix Verification
        image = np.array(
            [
                [0, 0, 0, 0, 100, 200, 300, 400, 500, 600],
                [0, 0, 100, 200, 300, 400, 500, 600, 700, 800],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
                [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
                [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800],
                [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
                [1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2047, 2047],
                [1500, 1600, 1700, 1800, 1900, 2000, 2047, 2047, 2047, 2047],
            ],
            dtype=np.float64,
        )

        # Table 2.15: Output 10x10 LaGrange Interpolation Matrix (clipped to [0, 2047])
        expected = np.array(
            [
                [0, 0, 14, 73, 178, 278, 378, 478, 591, 591],
                [19, 87, 198, 300, 400, 500, 600, 700, 813, 813],
                [184, 300, 400, 500, 600, 700, 800, 900, 1013, 1013],
                [388, 500, 600, 700, 800, 900, 1000, 1100, 1213, 1213],
                [588, 700, 800, 900, 1000, 1100, 1200, 1300, 1413, 1413],
                [788, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1613, 1613],
                [988, 1100, 1200, 1300, 1400, 1500, 1600, 1701, 1817, 1817],
                [1188, 1300, 1400, 1500, 1600, 1701, 1805, 1902, 1994, 1994],
                [1403, 1516, 1616, 1716, 1817, 1910, 1991, 2044, 2047, 2047],
                [1509, 1622, 1722, 1822, 1925, 2005, 2041, 2047, 2047, 2047],
            ],
            dtype=np.float64,
        )

        result = sips_correlate(image, kernel)
        result_clipped = np.clip(np.round(result), 0, 2047)

        np.testing.assert_array_almost_equal(
            result_clipped,
            expected,
            decimal=0,
            err_msg="Correlation output does not match SIPS Table 2.15",
        )

    def test_single_band_hw_input(self):
        """A 2-D (H, W) input is handled correctly and returns 2-D output."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
        image = np.ones((5, 5), dtype=np.float64) * 100.0

        result = sips_correlate(image, kernel)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (5, 5))

    def test_chw_input_output_layout(self):
        """CHW input produces CHW output with correct shape."""
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        rng = np.random.default_rng(42)
        image = rng.random((3, 8, 8), dtype=np.float64)

        result = sips_correlate(image, kernel)

        self.assertEqual(result.shape, (3, 8, 8))

    def test_symmetric_kernel_matches_convolve(self):
        """For a symmetric kernel, correlation and convolution produce the same result."""
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        rng = np.random.default_rng(42)
        image = rng.random((2, 10, 10), dtype=np.float64) * 1000.0

        conv_result = sips_convolve(image, kernel)
        corr_result = sips_correlate(image, kernel)

        np.testing.assert_allclose(conv_result, corr_result, atol=1e-10)

    def test_non_2d_kernel_raises(self):
        """Non-2-D kernel raises ValueError."""
        kernel = np.ones((3,), dtype=np.float64)
        image = np.ones((5, 5), dtype=np.float64)

        with self.assertRaises(ValueError) as ctx:
            sips_correlate(image, kernel)
        self.assertIn("2-D", str(ctx.exception))

    def test_invalid_ndim_raises(self):
        """4-D input raises ValueError."""
        kernel = np.ones((3, 3), dtype=np.float64)
        image = np.ones((1, 1, 5, 5), dtype=np.float64)

        with self.assertRaises(ValueError):
            sips_correlate(image, kernel)

    def test_no_input_mutation(self):
        """sips_correlate does not mutate the input array."""
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        image = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)
        original = image.copy()

        sips_correlate(image, kernel)

        np.testing.assert_array_equal(image, original)


class TestConvolutionCorrelationRelationship(unittest.TestCase):
    """Tests verifying the relationship between convolution and correlation."""

    def test_correlate_equals_convolve_with_flipped_kernel(self):
        """correlate(image, kernel) == convolve(image, kernel[::-1, ::-1])."""
        kernel = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(123)
        image = rng.random((2, 10, 10), dtype=np.float64) * 1000.0

        corr_result = sips_correlate(image, kernel)
        conv_flipped = sips_convolve(image, kernel[::-1, ::-1])

        np.testing.assert_allclose(corr_result, conv_flipped, atol=1e-8)

    def test_per_band_independence(self):
        """Each band is processed independently."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
        rng = np.random.default_rng(42)
        image = rng.random((3, 8, 8), dtype=np.float64) * 1000.0

        # Process all bands at once
        result_all = sips_convolve(image, kernel)

        # Process each band individually
        for b in range(3):
            result_single = sips_convolve(image[b], kernel)
            np.testing.assert_allclose(
                result_all[b],
                result_single,
                atol=1e-10,
                err_msg=f"Band {b} result differs between CHW and single-band processing",
            )
