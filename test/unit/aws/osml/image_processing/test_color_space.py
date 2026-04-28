#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import numpy as np

from aws.osml.image_processing.color_space import color_space_transform


class TestColorSpaceTransform(unittest.TestCase):
    """Tests for the color_space_transform function."""

    def test_srgb_to_prophoto_rgb_table_2_44(self):
        """Verify sRGB→ProPhoto RGB against SIPS Table 2.44.

        Table 2.44 provides reference values for CMM conversion between
        sRGB and ProPhoto RGB. Values are in [0, 1] floating point.
        The SIPS spec notes results may differ by 1-2 digital counts
        (at 8-bit), so we use atol=0.01 for floating-point comparison.
        """
        # SIPS Table 2.44: sRGB → ProPhoto RGB
        # Each row: (sRGB_R, sRGB_G, sRGB_B, ProPhoto_R, ProPhoto_G, ProPhoto_B)
        table = [
            (1.0, 0.0, 0.0, 0.7019, 0.2751, 0.1031),
            (0.0, 1.0, 0.0, 0.5397, 0.9274, 0.3040),
            (0.0, 0.0, 1.0, 0.3356, 0.1371, 0.9229),
            (0.0, 1.0, 1.0, 0.6575, 0.9440, 0.9907),
            (1.0, 0.0, 1.0, 0.8002, 0.3165, 0.9328),
            (1.0, 1.0, 0.0, 0.9192, 0.9842, 0.3275),
            (0.5, 0.5, 0.5, 0.4241, 0.4241, 0.4241),
            (0.5, 0.0, 0.0, 0.2976, 0.1167, 0.0437),
            (0.0, 0.5, 0.0, 0.2289, 0.3933, 0.1289),
            (0.0, 0.0, 0.5, 0.1423, 0.0581, 0.3913),
            (0.8353, 0.5059, 0.1294, 0.6227, 0.4701, 0.1893),
            (0.2706, 0.7882, 0.8431, 0.5303, 0.7061, 0.7922),
        ]

        for srgb_r, srgb_g, srgb_b, pp_r, pp_g, pp_b in table:
            # Build a 3×1×1 CHW image
            image = np.array([[[srgb_r]], [[srgb_g]], [[srgb_b]]], dtype=np.float64)
            expected = np.array([pp_r, pp_g, pp_b], dtype=np.float64)

            result = color_space_transform(image, "srgb", "prophoto_rgb")

            np.testing.assert_allclose(
                result[:, 0, 0],
                expected,
                atol=0.01,
                err_msg=(f"sRGB ({srgb_r}, {srgb_g}, {srgb_b}) → ProPhoto RGB mismatch"),
            )

    def test_unsupported_source_raises_valueerror(self):
        """Unsupported source color space raises ValueError."""
        image = np.zeros((3, 2, 2), dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            color_space_transform(image, "unknown_space", "srgb")
        self.assertIn("Unsupported source", str(ctx.exception))
        self.assertIn("unknown_space", str(ctx.exception))

    def test_unsupported_destination_raises_valueerror(self):
        """Unsupported destination color space raises ValueError."""
        image = np.zeros((3, 2, 2), dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            color_space_transform(image, "srgb", "invalid_dest")
        self.assertIn("Unsupported destination", str(ctx.exception))
        self.assertIn("invalid_dest", str(ctx.exception))

    def test_non_3_band_input_raises_valueerror(self):
        """Input with != 3 bands raises ValueError."""
        # 1-band image
        image_1band = np.zeros((1, 4, 4), dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            color_space_transform(image_1band, "srgb", "prophoto_rgb")
        self.assertIn("3 bands", str(ctx.exception))

        # 4-band image
        image_4band = np.zeros((4, 4, 4), dtype=np.float64)
        with self.assertRaises(ValueError) as ctx:
            color_space_transform(image_4band, "srgb", "prophoto_rgb")
        self.assertIn("3 bands", str(ctx.exception))

    def test_2d_input_raises_valueerror(self):
        """2-D input raises ValueError (not 3-D CHW)."""
        image = np.zeros((4, 4), dtype=np.float64)
        with self.assertRaises(ValueError):
            color_space_transform(image, "srgb", "prophoto_rgb")

    def test_same_source_and_destination_returns_copy(self):
        """Same source and destination returns a copy of the input."""
        image = np.random.default_rng(42).random((3, 4, 4))
        result = color_space_transform(image, "srgb", "srgb")
        np.testing.assert_array_equal(result, image)
        # Verify it's a copy, not the same object
        self.assertFalse(np.shares_memory(result, image))

    def test_no_input_mutation(self):
        """color_space_transform does not mutate the input array."""
        image = np.random.default_rng(42).random((3, 4, 4))
        original = image.copy()
        color_space_transform(image, "srgb", "prophoto_rgb")
        np.testing.assert_array_equal(image, original)

    def test_output_shape_preserved(self):
        """Output has same shape as input."""
        image = np.random.default_rng(42).random((3, 8, 12))
        result = color_space_transform(image, "srgb", "adobe_rgb")
        self.assertEqual(result.shape, image.shape)

    def test_srgb_to_linear_srgb(self):
        """sRGB to linear_srgb applies gamma decode only."""
        # Pure white should remain white (within floating-point tolerance
        # from the matrix round-trip through XYZ)
        white = np.ones((3, 1, 1), dtype=np.float64)
        result = color_space_transform(white, "srgb", "linear_srgb")
        np.testing.assert_allclose(result, white, atol=1e-6)

        # Pure black should remain black
        black = np.zeros((3, 1, 1), dtype=np.float64)
        result = color_space_transform(black, "srgb", "linear_srgb")
        np.testing.assert_allclose(result, black, atol=1e-10)

    def test_gray_neutral_preserved(self):
        """Neutral gray should remain neutral after sRGB→ProPhoto RGB.

        SIPS Table 2.44 shows (0.5, 0.5, 0.5) → (0.4241, 0.4241, 0.4241).
        All three channels should be equal for a neutral gray.
        """
        gray = np.full((3, 1, 1), 0.5, dtype=np.float64)
        result = color_space_transform(gray, "srgb", "prophoto_rgb")
        # All channels should be equal
        self.assertAlmostEqual(result[0, 0, 0], result[1, 0, 0], places=4)
        self.assertAlmostEqual(result[1, 0, 0], result[2, 0, 0], places=4)
