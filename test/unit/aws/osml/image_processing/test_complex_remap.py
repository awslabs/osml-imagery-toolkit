#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from aws.osml.image_processing.complex_remap import (
    ROLE_AMPLITUDE_INDEX,
    ROLE_IMAGINARY,
    ROLE_MAGNITUDE,
    ROLE_PHASE,
    ROLE_REAL,
    complex_to_power,
    decode_to_iq,
    is_complex,
    magnitude_remap,
    power_to_decibels,
    quarter_power_remap,
)


class TestComplexToPower(TestCase):
    """Tests for complex_to_power() with all supported input layouts."""

    def test_native_complex64(self):
        """Native complex64 input produces correct power values."""
        data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
        result = complex_to_power(data)
        expected = np.array([[5.0, 25.0], [61.0, 113.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_native_complex64_chw(self):
        """Native complex64 in (1, H, W) shape produces correct power."""
        data = np.array([[[1 + 2j, 3 + 4j]]], dtype=np.complex64)
        result = complex_to_power(data)
        expected = np.array([[5.0, 25.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_2hw_float32(self):
        """(2, H, W) float32 I/Q produces I² + Q²."""
        iq = np.array([[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]], dtype=np.float32)
        result = complex_to_power(iq)
        expected = np.array([[5.0, 25.0], [61.0, 113.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_2hw_int16(self):
        """(2, H, W) int16 I/Q produces correct power with float32 output."""
        iq = np.array([[[3, 5]], [[4, 12]]], dtype=np.int16)
        result = complex_to_power(iq)
        expected = np.array([[25.0, 169.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_invalid_shape_raises(self):
        """1D or single-band non-complex array raises ValueError."""
        with self.assertRaises(ValueError):
            complex_to_power(np.array([1.0, 2.0, 3.0]))

    def test_single_band_non_complex_raises(self):
        """(1, H, W) float32 (not complex) raises ValueError."""
        data = np.ones((1, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            complex_to_power(data)

    def test_agrees_with_np_abs_squared(self):
        """Result agrees with np.abs(z)**2 for random complex input."""
        rng = np.random.default_rng(42)
        data = (rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))).astype(np.complex64)
        result = complex_to_power(data)
        expected = np.abs(data) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestDecodeToIQ(TestCase):
    """Tests for decode_to_iq() covering all band interpretations."""

    def test_real_imaginary_float32_noop(self):
        """Real/imaginary float32 is returned as-is (cast only)."""
        iq = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
        result = decode_to_iq(iq, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        np.testing.assert_array_equal(result, iq)
        self.assertEqual(result.dtype, np.float32)

    def test_real_imaginary_int16_cast(self):
        """Real/imaginary int16 is cast to float32."""
        iq = np.array([[[100, 200]], [[-50, 50]]], dtype=np.int16)
        result = decode_to_iq(iq, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[0], np.array([[100.0, 200.0]]))
        np.testing.assert_allclose(result[1], np.array([[-50.0, 50.0]]))

    def test_amplitude_index_phase_with_table(self):
        """AMP8I_PHS8I decoding applies amplitude table and phase scaling."""
        amp_table = np.arange(256, dtype=np.float32) * 0.5
        amp_idx = np.array([[10, 20]], dtype=np.uint8)
        phase_idx = np.array([[0, 128]], dtype=np.uint8)
        data = np.stack([amp_idx, phase_idx], axis=0)

        result = decode_to_iq(data, band_interpretation=[ROLE_AMPLITUDE_INDEX, ROLE_PHASE], amplitude_table=amp_table)

        self.assertEqual(result.shape, (2, 1, 2))
        self.assertEqual(result.dtype, np.float32)
        amp_10 = amp_table[10]
        amp_20 = amp_table[20]
        phase_0 = np.float32(0.0 / 256.0 * 2 * np.pi)
        phase_128 = np.float32(128.0 / 256.0 * 2 * np.pi)
        np.testing.assert_allclose(result[0, 0, 0], amp_10 * np.cos(phase_0), rtol=1e-5)
        np.testing.assert_allclose(result[1, 0, 0], amp_10 * np.sin(phase_0), atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 1], amp_20 * np.cos(phase_128), rtol=1e-5)
        np.testing.assert_allclose(result[1, 0, 1], amp_20 * np.sin(phase_128), rtol=1e-5)

    def test_amplitude_index_without_table_raises(self):
        """amplitude_index without amplitude_table raises ValueError."""
        data = np.zeros((2, 4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            decode_to_iq(data, band_interpretation=[ROLE_AMPLITUDE_INDEX, ROLE_PHASE])

    def test_amplitude_table_without_index_raises(self):
        """amplitude_table provided without amplitude_index in interpretation raises ValueError."""
        data = np.zeros((2, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            decode_to_iq(data, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], amplitude_table=np.arange(256))

    def test_magnitude_phase_float(self):
        """Magnitude/phase float: M*cos(P), M*sin(P) (radians)."""
        mag = np.array([[10.0, 5.0]], dtype=np.float32)
        phase = np.array([[0.0, np.pi / 4]], dtype=np.float32)
        data = np.stack([mag, phase], axis=0)

        result = decode_to_iq(data, band_interpretation=[ROLE_MAGNITUDE, ROLE_PHASE])

        self.assertEqual(result.shape, (2, 1, 2))
        np.testing.assert_allclose(result[0, 0, 0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(result[1, 0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 1], 5.0 * np.cos(np.pi / 4), rtol=1e-5)
        np.testing.assert_allclose(result[1, 0, 1], 5.0 * np.sin(np.pi / 4), rtol=1e-5)

    def test_magnitude_phase_integer(self):
        """Magnitude/phase integer: phase scaled by value / 2^nbits * 2pi."""
        mag = np.array([[100, 200]], dtype=np.int16)
        phase_int = np.array([[0, 16384]], dtype=np.int16)  # 16384/65536 * 2pi = pi/2
        data = np.stack([mag, phase_int], axis=0)

        result = decode_to_iq(data, band_interpretation=[ROLE_MAGNITUDE, ROLE_PHASE])

        expected_phase = np.float32(16384.0 / 65536.0 * 2 * np.pi)
        np.testing.assert_allclose(result[0, 0, 0], 100.0, rtol=1e-5)
        np.testing.assert_allclose(result[1, 0, 0], 0.0, atol=1e-3)
        np.testing.assert_allclose(result[0, 0, 1], 200.0 * np.cos(expected_phase), rtol=1e-4)
        np.testing.assert_allclose(result[1, 0, 1], 200.0 * np.sin(expected_phase), rtol=1e-4)

    def test_native_complex64_decode(self):
        """Native complex64 (H, W) → (2, H, W) float32 I/Q."""
        data = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
        result = decode_to_iq(data)
        self.assertEqual(result.shape, (2, 1, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[0], np.array([[1.0, 3.0]]))
        np.testing.assert_allclose(result[1], np.array([[2.0, 4.0]]))

    def test_native_complex64_1hw(self):
        """Native complex64 (1, H, W) → (2, H, W) float32 I/Q."""
        data = np.array([[[1 + 2j, 3 + 4j]]], dtype=np.complex64)
        result = decode_to_iq(data)
        self.assertEqual(result.shape, (2, 1, 2))
        np.testing.assert_allclose(result[0], np.array([[1.0, 3.0]]))
        np.testing.assert_allclose(result[1], np.array([[2.0, 4.0]]))

    def test_inferred_band_interpretation_2band(self):
        """When band_interpretation is None and data is (2, H, W) numeric, infers real/imag."""
        data = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
        result = decode_to_iq(data)
        np.testing.assert_array_equal(result, data)

    def test_no_interpretation_single_band_raises(self):
        """Single-band non-complex without interpretation raises ValueError."""
        data = np.ones((1, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            decode_to_iq(data)

    def test_invalid_role_raises(self):
        """Unrecognized role string raises ValueError."""
        data = np.zeros((2, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            decode_to_iq(data, band_interpretation=["real", "unknown_role"])

    def test_shape_validation_2d_raises(self):
        """2D (H, W) non-complex array raises ValueError."""
        data = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            decode_to_iq(data, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])

    def test_band_count_mismatch_raises(self):
        """Data with fewer bands than interpretation length raises ValueError."""
        data = np.zeros((1, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            decode_to_iq(data, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])


class TestDecodeToIQZeroAndNaN(TestCase):
    """Tests that zero and NaN inputs produce safe outputs from decode_to_iq."""

    def test_all_zero_input(self):
        """All-zero I/Q data decodes to all-zero output."""
        data = np.zeros((2, 4, 4), dtype=np.float32)
        result = decode_to_iq(data, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        np.testing.assert_array_equal(result, np.zeros((2, 4, 4), dtype=np.float32))

    def test_nan_input_preserved_in_decode(self):
        """NaN in real/imaginary decode passes through (guards are in remap presets)."""
        data = np.array([[[np.nan, 1.0]], [[np.nan, 2.0]]], dtype=np.float32)
        result = decode_to_iq(data, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        self.assertTrue(np.isnan(result[0, 0, 0]))
        self.assertTrue(np.isnan(result[1, 0, 0]))
        np.testing.assert_allclose(result[0, 0, 1], 1.0)
        np.testing.assert_allclose(result[1, 0, 1], 2.0)


class TestComplexToPowerZeroNaN(TestCase):
    """Tests that zero and NaN blocks don't produce inf/NaN in complex_to_power."""

    def test_all_zero_block(self):
        """All-zero I/Q block produces all-zero power (no inf/NaN)."""
        data = np.zeros((2, 8, 8), dtype=np.float32)
        result = complex_to_power(data)
        np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_nan_block(self):
        """NaN I/Q block produces NaN power (not inf)."""
        data = np.full((2, 4, 4), np.nan, dtype=np.float32)
        result = complex_to_power(data)
        self.assertTrue(np.all(np.isnan(result)))

    def test_all_zero_complex(self):
        """All-zero native complex produces all-zero power."""
        data = np.zeros((4, 4), dtype=np.complex64)
        result = complex_to_power(data)
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(result)))


class TestPowerToDecibels(TestCase):
    """Tests for power_to_decibels()."""

    def test_basic_conversion(self):
        """Known power values produce correct dB."""
        power = np.array([[1.0, 10.0, 100.0]], dtype=np.float32)
        result = power_to_decibels(power)
        expected = np.array([[0.0, 10.0, 20.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_zero_power_produces_neg_inf(self):
        """Zero power produces -inf (documented behavior)."""
        power = np.array([[0.0]], dtype=np.float32)
        result = power_to_decibels(power)
        self.assertTrue(np.isneginf(result[0, 0]))


# ===========================================================================
# Phase 2: Remap presets
# ===========================================================================


class TestQuarterPowerRemap(TestCase):
    """Tests for quarter_power_remap() preset."""

    def test_output_shape_and_dtype(self):
        """Output is (1, H, W) float32."""
        block = np.random.default_rng(42).standard_normal((2, 8, 8)).astype(np.float32)
        result = quarter_power_remap(block)
        self.assertEqual(result.shape, (1, 8, 8))
        self.assertEqual(result.dtype, np.float32)

    def test_known_values(self):
        """sqrt(sqrt(I² + Q²)) for known input."""
        block = np.array([[[3.0]], [[4.0]]], dtype=np.float32)
        result = quarter_power_remap(block)
        expected = np.sqrt(np.sqrt(np.float32(25.0)))
        np.testing.assert_allclose(result[0, 0, 0], expected, rtol=1e-5)

    def test_all_zero_block(self):
        """All-zero block produces all-zero output (no inf/NaN)."""
        block = np.zeros((2, 4, 4), dtype=np.float32)
        result = quarter_power_remap(block)
        np.testing.assert_array_equal(result, np.zeros((1, 4, 4), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_nan_block_produces_zeros(self):
        """NaN block produces zeros (guard against propagation)."""
        block = np.full((2, 4, 4), np.nan, dtype=np.float32)
        result = quarter_power_remap(block)
        np.testing.assert_array_equal(result, np.zeros((1, 4, 4), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_finite_output_for_valid_input(self):
        """Random valid input always produces finite output."""
        rng = np.random.default_rng(123)
        block = rng.standard_normal((2, 16, 16)).astype(np.float32)
        result = quarter_power_remap(block)
        self.assertTrue(np.all(np.isfinite(result)))


class TestMagnitudeRemap(TestCase):
    """Tests for magnitude_remap() preset."""

    def test_output_shape_and_dtype(self):
        """Output is (1, H, W) float32."""
        block = np.random.default_rng(42).standard_normal((2, 8, 8)).astype(np.float32)
        result = magnitude_remap(block)
        self.assertEqual(result.shape, (1, 8, 8))
        self.assertEqual(result.dtype, np.float32)

    def test_known_values(self):
        """sqrt(I² + Q²) for known input."""
        block = np.array([[[3.0]], [[4.0]]], dtype=np.float32)
        result = magnitude_remap(block)
        np.testing.assert_allclose(result[0, 0, 0], 5.0, rtol=1e-5)

    def test_all_zero_block(self):
        """All-zero block produces all-zero output."""
        block = np.zeros((2, 4, 4), dtype=np.float32)
        result = magnitude_remap(block)
        np.testing.assert_array_equal(result, np.zeros((1, 4, 4), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_nan_block_produces_zeros(self):
        """NaN block produces zeros (guard)."""
        block = np.full((2, 4, 4), np.nan, dtype=np.float32)
        result = magnitude_remap(block)
        np.testing.assert_array_equal(result, np.zeros((1, 4, 4), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_relationship_to_complex_abs(self):
        """magnitude_remap agrees with np.abs on complex input."""
        rng = np.random.default_rng(99)
        real = rng.standard_normal((1, 8)).astype(np.float32)
        imag = rng.standard_normal((1, 8)).astype(np.float32)
        block = np.stack([real, imag], axis=0)
        result = magnitude_remap(block)
        expected = np.abs(real + 1j * imag)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)


# ===========================================================================
# Phase 2: is_complex() detection logic
# ===========================================================================


def _make_source(pixel_value_type=None, num_bands=1, metadata=None):
    """Helper to create a mock source for is_complex() tests."""
    source = MagicMock()
    type(source).pixel_value_type = PropertyMock(return_value=pixel_value_type)
    type(source).num_bands = PropertyMock(return_value=num_bands)
    type(source).metadata = PropertyMock(return_value=metadata)
    return source


class TestIsComplexPositive(TestCase):
    """Tests that is_complex() returns True for complex imagery indicators."""

    def test_complex64_pixel_type(self):
        """pixel_value_type='COMPLEX64' → True."""
        source = _make_source(pixel_value_type="COMPLEX64")
        self.assertTrue(is_complex(source))

    def test_complex128_pixel_type(self):
        """pixel_value_type='Complex128' (mixed case) → True."""
        source = _make_source(pixel_value_type="Complex128")
        self.assertTrue(is_complex(source))

    def test_enum_style_complex_pixel_type(self):
        """pixel_value_type='PixelType.Complex64' (enum repr) → True."""
        source = _make_source(pixel_value_type="PixelType.Complex64")
        self.assertTrue(is_complex(source))

    def test_icat_sariq(self):
        """ICAT=SARIQ → True (definitionally I/Q complex)."""
        source = _make_source(num_bands=2, metadata={"ICAT": "SARIQ"})
        self.assertTrue(is_complex(source))

    def test_isubcat_i_q(self):
        """ISUBCAT contains I and Q → True."""
        source = _make_source(num_bands=2, metadata={"ISUBCAT": ["I", "Q"]})
        self.assertTrue(is_complex(source))

    def test_isubcat_i_q_string(self):
        """ISUBCAT as comma-separated string 'I,Q' → True."""
        source = _make_source(num_bands=2, metadata={"ISUBCAT": "I,Q"})
        self.assertTrue(is_complex(source))

    def test_isubcat_m_p_icat_sar(self):
        """ISUBCAT M/P with ICAT=SAR → True."""
        source = _make_source(num_bands=2, metadata={"ISUBCAT": ["M", "P"], "ICAT": "SAR"})
        self.assertTrue(is_complex(source))

    def test_isubcat_m_p_icat_isar(self):
        """ISUBCAT M/P with ICAT=ISAR → True."""
        source = _make_source(num_bands=2, metadata={"ISUBCAT": ["M", "P"], "ICAT": "ISAR"})
        self.assertTrue(is_complex(source))

    def test_isubcat_m_p_icat_sariq(self):
        """ISUBCAT M/P with ICAT=SARIQ → True (ICAT check fires first anyway)."""
        source = _make_source(num_bands=2, metadata={"ISUBCAT": ["M", "P"], "ICAT": "SARIQ"})
        self.assertTrue(is_complex(source))

    def test_irep_polar_icat_sar(self):
        """IREP=POLAR + ICAT=SAR → True."""
        source = _make_source(num_bands=2, metadata={"IREP": "POLAR", "ICAT": "SAR"})
        self.assertTrue(is_complex(source))

    def test_irep_nodisply_icat_sar_without_subheader_indicators(self):
        """IREP=NODISPLY + ICAT=SAR but no complex pixel type or band labels → False.

        Detection relies on image subheader indicators only. A file with
        NODISPLY/SAR but no complex pixel type or I/Q band labels is not
        identified as complex by is_complex() — use load_complex_remap()
        with the full DatasetReader if DES-based detection is needed.
        """
        source = _make_source(
            num_bands=2,
            metadata={"IREP": "NODISPLY", "ICAT": "SAR"},
        )
        self.assertFalse(is_complex(source))


class TestIsComplexNegative(TestCase):
    """Tests that is_complex() returns False for non-complex sources."""

    def test_icat_sar_single_band_float(self):
        """ICAT=SAR, single-band (SIDD-like) → False."""
        source = _make_source(num_bands=1, metadata={"ICAT": "SAR", "IREP": "MONO"})
        self.assertFalse(is_complex(source))

    def test_irep_mono_icat_sar(self):
        """IREP=MONO + ICAT=SAR, single-band (SIDD-like display product) → False."""
        source = _make_source(
            num_bands=1,
            metadata={"IREP": "MONO", "ICAT": "SAR"},
        )
        self.assertFalse(is_complex(source))

    def test_irep_rgb_icat_sar(self):
        """IREP=RGB + ICAT=SAR, 3-band (SIDD color product) → False."""
        source = _make_source(
            num_bands=3,
            metadata={"IREP": "RGB", "ICAT": "SAR"},
        )
        self.assertFalse(is_complex(source))

    def test_irep_polar_icat_ccd(self):
        """IREP=POLAR + ICAT=CCD (interferometric) → False."""
        source = _make_source(num_bands=2, metadata={"IREP": "POLAR", "ICAT": "CCD"})
        self.assertFalse(is_complex(source))

    def test_irep_polar_icat_wind(self):
        """IREP=POLAR + ICAT=WIND (vector quantity) → False."""
        source = _make_source(num_bands=2, metadata={"IREP": "POLAR", "ICAT": "WIND"})
        self.assertFalse(is_complex(source))

    def test_irep_polar_icat_current(self):
        """IREP=POLAR + ICAT=CURRENT → False."""
        source = _make_source(num_bands=2, metadata={"IREP": "POLAR", "ICAT": "CURRENT"})
        self.assertFalse(is_complex(source))

    def test_isubcat_m_p_icat_vis(self):
        """ISUBCAT M/P but ICAT=VIS (non-SAR) → False."""
        source = _make_source(num_bands=2, metadata={"ISUBCAT": ["M", "P"], "ICAT": "VIS"})
        self.assertFalse(is_complex(source))

    def test_icat_isar_single_band_mono(self):
        """ICAT=ISAR, IREP=MONO, single-band (detected ISAR magnitude) → False."""
        source = _make_source(num_bands=1, metadata={"ICAT": "ISAR", "IREP": "MONO"})
        self.assertFalse(is_complex(source))

    def test_irep_nodisply_icat_sar_single_band(self):
        """IREP=NODISPLY + ICAT=SAR, single-band, no complex indicators → False."""
        source = _make_source(
            num_bands=1,
            metadata={"IREP": "NODISPLY", "ICAT": "SAR"},
        )
        self.assertFalse(is_complex(source))


class TestIsComplexEdgeCases(TestCase):
    """Edge cases for is_complex()."""

    def test_none_metadata(self):
        """Source with None metadata doesn't crash."""
        source = _make_source(pixel_value_type="UINT16", num_bands=1, metadata=None)
        self.assertFalse(is_complex(source))

    def test_missing_fields(self):
        """Source with empty metadata dict returns False."""
        source = _make_source(num_bands=1, metadata={})
        self.assertFalse(is_complex(source))

    def test_pixel_value_type_throws(self):
        """Source where pixel_value_type raises exception → handled gracefully."""
        source = MagicMock()
        type(source).pixel_value_type = PropertyMock(side_effect=AttributeError)
        type(source).num_bands = PropertyMock(return_value=1)
        type(source).metadata = PropertyMock(return_value=None)
        self.assertFalse(is_complex(source))

    def test_band_info_subcategory_access(self):
        """ISUBCAT via BAND_INFO list of dicts → I/Q detected."""
        metadata = {"BAND_INFO": [{"ISUBCAT": "I"}, {"ISUBCAT": "Q"}]}
        source = _make_source(num_bands=2, metadata=metadata)
        self.assertTrue(is_complex(source))

    def test_icat_sar_with_no_complex_indicators(self):
        """ICAT=SAR with no complex pixel type, no ISUBCAT → False (SIDD path)."""
        source = _make_source(num_bands=1, metadata={"ICAT": "SAR", "IREP": "MONO"})
        self.assertFalse(is_complex(source))
