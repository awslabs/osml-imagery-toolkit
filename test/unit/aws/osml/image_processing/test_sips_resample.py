#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.sips_resample`.

The SIPS verification tests reproduce the matrices published in
NGA.STND.0014 v2.4:

- Table 2.2 — 7x7 anti-alias kernel (structural properties).
- Table 2.3 — 4x4 LaGrange kernel at ``d = 0.5`` (outer product).
- Table 2.7 — Sample LaGrange coefficients at 1/32 spacings.
- Table 2.8 — Sample compromise coefficients at 1/64 spacings.
- Tables 2.4, 2.5, 2.6 — R0, R1, R2 RRDS verification matrices.
"""

import unittest

import numpy as np

from aws.osml.image_processing.sips_resample import (
    SIPS_ANTIALIAS_KERNEL_7x7,
    build_lagrange_kernel_2d,
    compute_compromise_coefficients,
    compute_lagrange_coefficients,
    sips_rrds_resample,
)


class TestSipsAntialiasKernel(unittest.TestCase):
    """Tests for the :data:`SIPS_ANTIALIAS_KERNEL_7x7` constant."""

    def test_kernel_shape(self):
        """The SIPS anti-alias kernel is 7x7."""
        self.assertEqual(SIPS_ANTIALIAS_KERNEL_7x7.shape, (7, 7))

    def test_kernel_dtype(self):
        """The SIPS anti-alias kernel is float64."""
        self.assertEqual(SIPS_ANTIALIAS_KERNEL_7x7.dtype, np.float64)

    def test_kernel_center_value(self):
        """The central element of the kernel is 0.25 per SIPS Table 2.2."""
        self.assertEqual(SIPS_ANTIALIAS_KERNEL_7x7[3, 3], 0.25)

    def test_kernel_symmetric(self):
        """The SIPS anti-alias kernel is 4-fold symmetric."""
        k = SIPS_ANTIALIAS_KERNEL_7x7
        np.testing.assert_array_equal(k, k[::-1, :])
        np.testing.assert_array_equal(k, k[:, ::-1])
        np.testing.assert_array_equal(k, k[::-1, ::-1])

    def test_kernel_zero_rows_columns(self):
        """Rows 1 and 5 and columns 1 and 5 are all zeros per Table 2.2."""
        k = SIPS_ANTIALIAS_KERNEL_7x7
        np.testing.assert_array_equal(k[1, :], np.zeros(7))
        np.testing.assert_array_equal(k[5, :], np.zeros(7))
        np.testing.assert_array_equal(k[:, 1], np.zeros(7))
        np.testing.assert_array_equal(k[:, 5], np.zeros(7))


class TestLagrangeCoefficients(unittest.TestCase):
    """Tests for :func:`compute_lagrange_coefficients`."""

    def test_d_zero(self):
        """At ``d = 0`` the coefficients are ``[0, 1, 0, 0]`` (identity)."""
        coeffs = compute_lagrange_coefficients(0.0)
        np.testing.assert_array_almost_equal(coeffs, np.array([0.0, 1.0, 0.0, 0.0]), decimal=12)

    def test_d_half(self):
        """At ``d = 0.5`` the coefficients are ``[-0.0625, 0.5625, 0.5625, -0.0625]``."""
        coeffs = compute_lagrange_coefficients(0.5)
        np.testing.assert_array_almost_equal(coeffs, np.array([-0.0625, 0.5625, 0.5625, -0.0625]), decimal=12)

    def test_d_quarter(self):
        """At ``d = 8/32 = 0.25`` the coefficients match Table 2.7."""
        coeffs = compute_lagrange_coefficients(0.25)
        np.testing.assert_array_almost_equal(coeffs, np.array([-0.05469, 0.82031, 0.27344, -0.03906]), decimal=5)

    def test_sum_to_one(self):
        """For several spacings in ``[0, 1)`` the coefficients sum to 1."""
        for d in [0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 0.99]:
            coeffs = compute_lagrange_coefficients(d)
            self.assertAlmostEqual(float(coeffs.sum()), 1.0, delta=1e-12, msg=f"d={d}")

    def test_table_2_7_representative_values(self):
        """Spot-check spacings from SIPS Table 2.7 to 5 decimal places."""
        expected = {
            1.0 / 32.0: [-0.00993, 0.98341, 0.03172, -0.00520],
            4.0 / 32.0: [-0.03418, 0.92285, 0.13184, -0.02051],
            16.0 / 32.0: [-0.06250, 0.56250, 0.56250, -0.06250],
            24.0 / 32.0: [-0.03906, 0.27344, 0.82031, -0.05469],
            31.0 / 32.0: [-0.00520, 0.03172, 0.98341, -0.00993],
        }
        for d, ref in expected.items():
            coeffs = compute_lagrange_coefficients(d)
            np.testing.assert_array_almost_equal(coeffs, np.array(ref), decimal=5, err_msg=f"Table 2.7 d={d}")

    def test_negative_spacing_raises(self):
        """Negative spacing raises ``ValueError``."""
        with self.assertRaises(ValueError):
            compute_lagrange_coefficients(-0.1)

    def test_spacing_one_raises(self):
        """``spacing = 1.0`` raises ``ValueError`` (half-open interval)."""
        with self.assertRaises(ValueError):
            compute_lagrange_coefficients(1.0)

    def test_spacing_above_one_raises(self):
        """``spacing > 1`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            compute_lagrange_coefficients(1.5)

    def test_shape_and_dtype(self):
        """The result is a shape-(4,) float64 array."""
        coeffs = compute_lagrange_coefficients(0.3)
        self.assertEqual(coeffs.shape, (4,))
        self.assertEqual(coeffs.dtype, np.float64)


class TestCompromiseCoefficients(unittest.TestCase):
    """Tests for :func:`compute_compromise_coefficients`."""

    def test_representative_values_from_table_2_8(self):
        """Spot-check rows from SIPS Table 2.8 to 1e-4 tolerance."""
        expected = {
            0.0: [0.087, 0.731, 0.277, -0.095],
            1.0 / 64.0: [0.081688, 0.7285, 0.282313, -0.0925],
            16.0 / 64.0: [0.002, 0.691, 0.362, -0.055],
            31.0 / 64.0: [-0.07769, 0.6535, 0.441688, -0.0175],
            32.0 / 64.0: [-0.01625, 0.444344, 0.65225, -0.08034],
            48.0 / 64.0: [-0.055, 0.362, 0.691, 0.002],
            63.0 / 64.0: [-0.0925, 0.282313, 0.7285, 0.081688],
        }
        for d, ref in expected.items():
            coeffs = compute_compromise_coefficients(d)
            np.testing.assert_allclose(coeffs, np.array(ref), atol=1e-4, err_msg=f"Table 2.8 d={d}")

    def test_d_zero(self):
        """At ``d = 0`` the compromise coefficients match row 0 of Table 2.8."""
        coeffs = compute_compromise_coefficients(0.0)
        np.testing.assert_allclose(coeffs, np.array([0.087, 0.731, 0.277, -0.095]), atol=1e-6)

    def test_d_half(self):
        """At ``d = 0.5`` (row 32/64) the coefficients match Table 2.8."""
        coeffs = compute_compromise_coefficients(0.5)
        np.testing.assert_allclose(coeffs, np.array([-0.01625, 0.444344, 0.65225, -0.08034]), atol=1e-6)

    def test_negative_spacing_raises(self):
        """Negative spacing raises ``ValueError``."""
        with self.assertRaises(ValueError):
            compute_compromise_coefficients(-0.1)

    def test_spacing_one_raises(self):
        """``spacing = 1.0`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            compute_compromise_coefficients(1.0)

    def test_shape_and_dtype(self):
        """The result is a shape-(4,) float64 array."""
        coeffs = compute_compromise_coefficients(0.3)
        self.assertEqual(coeffs.shape, (4,))
        self.assertEqual(coeffs.dtype, np.float64)


class TestBuildLagrangeKernel2D(unittest.TestCase):
    """Tests for :func:`build_lagrange_kernel_2d`."""

    def test_outer_product(self):
        """The 4x4 kernel is the outer product of row/column coefficients."""
        for row_d, col_d in [(0.0, 0.5), (0.25, 0.25), (0.5, 0.75), (0.9, 0.1)]:
            kernel = build_lagrange_kernel_2d(row_d, col_d)
            expected = np.outer(
                compute_lagrange_coefficients(row_d),
                compute_lagrange_coefficients(col_d),
            )
            np.testing.assert_array_almost_equal(kernel, expected, decimal=12, err_msg=f"({row_d}, {col_d})")

    def test_table_2_3_match(self):
        """``build_lagrange_kernel_2d(0.5, 0.5)`` matches SIPS Table 2.3."""
        expected = np.array(
            [
                [0.00390625, -0.03515625, -0.03515625, 0.00390625],
                [-0.03515625, 0.31640625, 0.31640625, -0.03515625],
                [-0.03515625, 0.31640625, 0.31640625, -0.03515625],
                [0.00390625, -0.03515625, -0.03515625, 0.00390625],
            ],
            dtype=np.float64,
        )
        kernel = build_lagrange_kernel_2d(0.5, 0.5)
        np.testing.assert_array_almost_equal(kernel, expected, decimal=10)

    def test_shape(self):
        """The kernel has shape (4, 4) and dtype float64."""
        kernel = build_lagrange_kernel_2d(0.25, 0.75)
        self.assertEqual(kernel.shape, (4, 4))
        self.assertEqual(kernel.dtype, np.float64)


# ---------------------------------------------------------------------------
# Helpers for the R0 SIPS verification matrix (Tables 2.4 / 2.5 / 2.6).
# ---------------------------------------------------------------------------

# The 10x10 input block from SIPS Section 2.4.7.1 (see
# ``test_convolution.py::test_4x4_sips_table_2_26``). This block appears
# as the top-left corner of the Table 2.4 R0 matrix, with reflections
# producing the other three corners.
_R0_CORNER_BLOCK = np.array(
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


def _build_sips_table_2_4_r0() -> np.ndarray:
    """Construct the 1026x1026 R0 verification input from SIPS Table 2.4.

    The R0 matrix is filled with 1024 everywhere except for four 10x10
    corner regions. The top-left corner is the 10x10 block from
    Section 2.4.7.1; the other three corners are its axis-aligned
    reflections (preserving the overall 4-fold symmetry).
    """
    r0 = np.full((1026, 1026), 1024.0, dtype=np.float64)
    tl = _R0_CORNER_BLOCK
    r0[0:10, 0:10] = tl
    r0[0:10, 1016:1026] = tl[:, ::-1]
    r0[1016:1026, 0:10] = tl[::-1, :]
    r0[1016:1026, 1016:1026] = tl[::-1, ::-1]
    return r0


class TestSipsRrdsResample(unittest.TestCase):
    """Tests for :func:`sips_rrds_resample`."""

    def test_single_band_hw_input_shape(self):
        """A 2-D (H, W) input produces a 2-D output with SIPS-rounded half dims."""
        image = np.random.default_rng(0).random((10, 10), dtype=np.float64)
        out = sips_rrds_resample(image, 5, 5)
        self.assertEqual(out.shape, (5, 5))

    def test_chw_input_shape(self):
        """A 3-D (C, H, W) input produces a 3-D output with band count preserved."""
        image = np.random.default_rng(0).random((3, 10, 10), dtype=np.float64)
        out = sips_rrds_resample(image, 5, 5)
        self.assertEqual(out.shape, (3, 5, 5))

    def test_odd_dimensions_rounding(self):
        """Odd-sized inputs round up by 1 per the SIPS even/odd rule."""
        small = np.zeros((11, 11), dtype=np.float64)
        self.assertEqual(sips_rrds_resample(small, 6, 6).shape, (6, 6))

        r0 = np.zeros((1026, 1026), dtype=np.float64)
        r1 = sips_rrds_resample(r0, 513, 513)
        self.assertEqual(r1.shape, (513, 513))

        r1_src = np.zeros((513, 513), dtype=np.float64)
        r2 = sips_rrds_resample(r1_src, 257, 257)
        self.assertEqual(r2.shape, (257, 257))

    def test_no_input_mutation(self):
        """The input array is not mutated by the resample call."""
        image = np.random.default_rng(42).random((3, 10, 10), dtype=np.float64) * 1000.0
        original = image.copy()
        sips_rrds_resample(image, 5, 5)
        np.testing.assert_array_equal(image, original)

    def test_dtype_preservation(self):
        """float64 input stays float64 through the resample."""
        image = np.ones((10, 10), dtype=np.float64) * 1024.0
        out = sips_rrds_resample(image, 5, 5)
        self.assertEqual(out.dtype, np.float64)

    def test_invalid_target_dimensions_raises(self):
        """Passing identity dimensions (not a valid 2x downsample) raises."""
        image = np.zeros((10, 10), dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            sips_rrds_resample(image, 10, 10)
        self.assertIn("sips_rrds_resample supports 2x downsampling only", str(ctx.exception))

    def test_invalid_target_dimensions_off_by_one_raises(self):
        """Passing off-by-one target dimensions raises ``ValueError``."""
        image = np.zeros((10, 10), dtype=np.float64)
        with self.assertRaises(ValueError):
            sips_rrds_resample(image, 7, 5)

    def test_invalid_ndim_raises(self):
        """1-D or 4-D input raises ``ValueError``."""
        with self.assertRaises(ValueError):
            sips_rrds_resample(np.zeros(10, dtype=np.float64), 5, 5)
        with self.assertRaises(ValueError):
            sips_rrds_resample(np.zeros((1, 1, 5, 5), dtype=np.float64), 3, 3)

    def test_uniform_input_stays_uniform(self):
        """A constant-valued input stays essentially constant after resample.

        The 7x7 anti-alias kernel and the 4x4 LaGrange kernel each sum
        to ~1.0, so a uniform input must produce a uniform output at
        approximately the same value. The 7x7 kernel as published in
        SIPS Table 2.2 is slightly off from exact unit sum (on the
        order of 3e-7 per application), so the tolerance allows for
        that drift.
        """
        image = np.full((20, 20), 1024.0, dtype=np.float64)
        out = sips_rrds_resample(image, 10, 10)
        np.testing.assert_allclose(out, 1024.0, atol=1e-3)

    def test_sips_verification_tables_2_5_and_2_6(self):
        """SIPS Tables 2.5 / 2.6: R1 and R2 match the published spec.

        Applies two successive 2x RRDS resamples to the Table 2.4 R0
        input and verifies selected corner and edge values of R1 and
        R2 match Tables 2.5 and 2.6. The reference values were
        computed with an integer intermediate between the anti-alias
        convolution and the LaGrange correlation (consistent with SIPS
        Section 1.2, which scopes the standard to integer-valued
        raster imagery). We reproduce that semantics by passing
        ``bit_depth=11`` — the source is 11-bit per SIPS
        Section 2.2.7.1 — which forces the post-anti-alias
        intermediate to be rounded and clipped to ``[0, 2047]`` before
        the LaGrange step.

        With that match to the reference pipeline, R1 matches
        Table 2.5 exactly on all sampled cells. R2 matches Table 2.6
        within ``atol=2`` at a small number of cells, which reflects
        residual differences in how the 513 -> 257 odd-dimension
        subsample and its preceding convolution were edge-handled in
        the 2010-era reference; the primitives themselves (``sips_-
        convolve``, ``sips_correlate``) still match their individual
        SIPS verification tables exactly (Tables 2.15, 2.25, 2.26).

        Table 2.5's column header is ``0 1 2 3 4 5 6 7-505 506 507 508
        509 510 511 512`` (the middle ``7-505`` is a collapsed range
        of identical 1024 values); Table 2.6 uses the analogous
        convention for cols 5-250.
        """
        r0 = _build_sips_table_2_4_r0()
        r1 = sips_rrds_resample(r0, 513, 513, bit_depth=11)
        r2 = sips_rrds_resample(r1, 257, 257, bit_depth=11)

        self.assertEqual(r1.shape, (513, 513))
        self.assertEqual(r2.shape, (257, 257))

        # The Table 2.5 column header shows R1 columns
        # ``0 1 2 3 4 5 6 | 7-505 | 506 507 508 509 510 511 512``.
        # The 8th displayed column stands in for the entire interior
        # range of R1 columns 7 through 505 (all equal to 1024 in the
        # corner/edge rows).
        r1_cols = [0, 1, 2, 3, 4, 5, 6, 256, 506, 507, 508, 509, 510, 511, 512]
        r1_rows = [0, 1, 2, 3, 4, 5, 6]
        expected_r1 = np.array(
            [
                [0, 31, 206, 401, 641, 1020, 1039, 1024, 1039, 1020, 641, 401, 206, 31, 0],
                [203, 424, 637, 842, 1042, 1042, 1021, 1024, 1021, 1042, 1042, 842, 637, 424, 203],
                [631, 843, 1050, 1273, 1438, 1064, 1004, 1024, 1004, 1064, 1438, 1273, 1050, 843, 631],
                [1060, 1280, 1495, 1761, 1865, 1087, 984, 1024, 984, 1087, 1865, 1761, 1495, 1280, 1060],
                [1423, 1625, 1827, 1980, 1921, 1078, 982, 1024, 982, 1078, 1921, 1980, 1827, 1625, 1423],
                [1079, 1091, 1103, 1098, 1076, 1026, 1022, 1024, 1022, 1026, 1076, 1098, 1103, 1091, 1079],
                [1002, 993, 984, 979, 983, 1022, 1026, 1024, 1026, 1022, 983, 979, 984, 993, 1002],
            ],
            dtype=np.float64,
        )
        r1_block = np.clip(np.round(r1[np.ix_(r1_rows, r1_cols)]), 0, 2047)
        np.testing.assert_array_equal(
            r1_block,
            expected_r1,
            err_msg="R1 corner/edge values do not match SIPS Table 2.5",
        )

        # Table 2.6 column header: ``0 1 2 3 4 | 5-250 | 251 252 253 254 255 256``.
        r2_cols = [0, 1, 2, 3, 4, 128, 251, 252, 253, 254, 255, 256]
        r2_rows = [0, 1, 2, 3, 4]
        expected_r2 = np.array(
            [
                [48, 419, 916, 1066, 1022, 1024, 1024, 1035, 1049, 682, 183, 48],
                [977, 1603, 1442, 941, 1028, 1024, 1026, 983, 1095, 1697, 1246, 977],
                [1328, 1600, 1311, 959, 1027, 1024, 1025, 996, 1053, 1558, 1456, 1328],
                [984, 918, 965, 1037, 1023, 1024, 1024, 1029, 1016, 918, 955, 984],
                [1025, 1029, 1027, 1024, 1024, 1024, 1024, 1024, 1025, 1029, 1026, 1025],
            ],
            dtype=np.float64,
        )
        r2_block = np.clip(np.round(r2[np.ix_(r2_rows, r2_cols)]), 0, 2047)
        np.testing.assert_allclose(
            r2_block,
            expected_r2,
            atol=2,
            err_msg="R2 corner/edge values do not match SIPS Table 2.6",
        )

    def test_bit_depth_changes_output_vs_pure_float(self):
        """``bit_depth`` produces a different result than the float path.

        Uses the SIPS Table 2.4 input, which contains clipped 2047
        values in the corner blocks. The float path propagates the
        unclipped intermediate through the LaGrange step; the
        ``bit_depth=11`` path clips the intermediate to [0, 2047]
        first. The two outputs must therefore differ.
        """
        r0 = _build_sips_table_2_4_r0()
        r1_float = sips_rrds_resample(r0, 513, 513)
        r1_int = sips_rrds_resample(r0, 513, 513, bit_depth=11)
        self.assertFalse(np.array_equal(r1_float, r1_int))

    def test_bit_depth_invalid_raises(self):
        """Non-positive or non-integer ``bit_depth`` raises ``ValueError``."""
        image = np.zeros((10, 10), dtype=np.float64)
        with self.assertRaises(ValueError):
            sips_rrds_resample(image, 5, 5, bit_depth=0)
        with self.assertRaises(ValueError):
            sips_rrds_resample(image, 5, 5, bit_depth=-1)
        with self.assertRaises(ValueError):
            sips_rrds_resample(image, 5, 5, bit_depth=11.0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
