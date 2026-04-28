#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import numpy as np

from aws.osml.image_processing.lut import apply_lut


class TestApplyLut(unittest.TestCase):
    """Tests for the apply_lut function."""

    def test_identity_lut_uint8(self):
        """Identity LUT (lut[i]=i) produces output equal to input."""
        lut = np.arange(256, dtype=np.uint8)
        image = np.array([[[0, 50], [200, 255]]], dtype=np.uint8)

        result = apply_lut(image, lut)

        np.testing.assert_array_equal(result, image)

    def test_identity_lut_uint16(self):
        """Identity LUT for uint16 produces output equal to input."""
        lut = np.arange(65536, dtype=np.uint16)
        image = np.array([[[0, 1000], [30000, 65535]]], dtype=np.uint16)

        result = apply_lut(image, lut)

        np.testing.assert_array_equal(result, image)

    def test_inversion_lut(self):
        """Inversion LUT (lut[i]=255-i) inverts pixel values."""
        lut = np.arange(255, -1, -1, dtype=np.uint8)
        image = np.array([[[0, 127], [128, 255]]], dtype=np.uint8)

        result = apply_lut(image, lut)

        expected = np.array([[[255, 128], [127, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_out_of_range_clipping(self):
        """Input values outside [0, len(lut)-1] are clipped before lookup."""
        # LUT with only 10 entries
        lut = np.arange(10, dtype=np.uint8) * 25
        image = np.array([[[0, 5], [9, 255]]], dtype=np.uint8)

        result = apply_lut(image, lut)

        # Value 255 should be clipped to index 9 → lut[9] = 225
        self.assertEqual(result[0, 0, 0], 0)  # lut[0] = 0
        self.assertEqual(result[0, 0, 1], 125)  # lut[5] = 125
        self.assertEqual(result[0, 1, 0], 225)  # lut[9] = 225
        self.assertEqual(result[0, 1, 1], 225)  # clipped to lut[9] = 225

    def test_uint8_vs_uint16_same_logical_result(self):
        """uint8 and uint16 paths produce the same logical result for equivalent inputs."""
        lut = np.arange(256, dtype=np.uint8)
        lut_16 = np.arange(256, dtype=np.uint16)

        image_8 = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        image_16 = image_8.astype(np.uint16)

        result_8 = apply_lut(image_8, lut)
        result_16 = apply_lut(image_16, lut_16)

        np.testing.assert_array_equal(result_8.astype(np.uint16), result_16)

    def test_multi_band(self):
        """LUT is applied independently to each band."""
        lut = np.array([100, 200], dtype=np.uint8)
        image = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.uint8)

        result = apply_lut(image, lut)

        expected = np.array([[[100, 200], [200, 100]], [[200, 100], [100, 200]]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_2d_input_treated_as_single_band(self):
        """A 2-D (H, W) input is treated as a single-band image."""
        lut = np.arange(256, dtype=np.uint8)
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

        result = apply_lut(image, lut)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_array_equal(result, image)

    def test_output_dtype_matches_lut_dtype(self):
        """Output dtype matches the LUT's dtype."""
        lut_f32 = np.linspace(0.0, 1.0, 256).astype(np.float32)
        image = np.array([[[0, 128], [255, 64]]], dtype=np.uint8)

        result = apply_lut(image, lut_f32)

        self.assertEqual(result.dtype, np.float32)

    def test_non_1d_lut_raises(self):
        """Non-1-D LUT raises ValueError."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        image = np.array([[[10, 20]]], dtype=np.uint8)

        with self.assertRaises(ValueError) as ctx:
            apply_lut(image, lut)
        self.assertIn("1-D", str(ctx.exception))

    def test_invalid_ndim_raises(self):
        """4-D input raises ValueError."""
        lut = np.arange(256, dtype=np.uint8)
        image = np.zeros((1, 1, 2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            apply_lut(image, lut)

    def test_no_input_mutation(self):
        """apply_lut does not mutate the input array."""
        lut = np.arange(255, -1, -1, dtype=np.uint8)
        image = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        original = image.copy()

        apply_lut(image, lut)

        np.testing.assert_array_equal(image, original)

    def test_output_shape_preserved(self):
        """Output has same spatial dimensions and band count as input."""
        lut = np.arange(256, dtype=np.uint8)
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(3, 16, 16), dtype=np.uint8)

        result = apply_lut(image, lut)

        self.assertEqual(result.shape, image.shape)

    def test_constant_lut(self):
        """A constant LUT maps all pixels to the same value."""
        lut = np.full(256, 42, dtype=np.uint8)
        image = np.array([[[0, 128], [255, 64]]], dtype=np.uint8)

        result = apply_lut(image, lut)

        expected = np.full_like(image, 42, dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
