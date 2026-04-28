#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Integration tests for the complex imagery remap pipeline.

Tests the full end-to-end flow: complex asset → is_complex → ComplexRemapFactory →
compute_image_statistics → DisplayChainFactory → ProcessingChain producing uint8
output with reasonable contrast.
"""

from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from aws.osml.image_processing.complex_remap import (
    ROLE_AMPLITUDE_INDEX,
    ROLE_IMAGINARY,
    ROLE_MAGNITUDE,
    ROLE_PHASE,
    ROLE_REAL,
    ComplexRemapFactory,
    is_complex,
)
from aws.osml.image_processing.display_chain_factory import DisplayChainFactory
from aws.osml.image_processing.statistics import compute_image_statistics


def _make_complex_source(block_data, num_bands=2, pixel_value_type="FLOAT32", metadata=None):
    """Create a mock source that behaves like an ImageAssetProvider with complex data."""
    source = MagicMock()
    _, h, w = block_data.shape
    type(source).num_bands = PropertyMock(return_value=num_bands)
    type(source).pixel_value_type = PropertyMock(return_value=pixel_value_type)
    type(source).num_rows = PropertyMock(return_value=h)
    type(source).num_columns = PropertyMock(return_value=w)
    type(source).num_pixels_per_block_horizontal = PropertyMock(return_value=w)
    type(source).num_pixels_per_block_vertical = PropertyMock(return_value=h)
    type(source).num_resolution_levels = PropertyMock(return_value=1)
    type(source).block_grid_size = PropertyMock(return_value=(1, 1))
    type(source).key = PropertyMock(return_value="test_complex_source")
    type(source).metadata = PropertyMock(return_value=metadata)
    source.get_block = MagicMock(return_value=block_data)
    source.has_block = MagicMock(return_value=True)
    return source


def _make_sicd_like_iq_data(size=64, seed=42):
    """Generate synthetic SICD-like I/Q data with realistic SAR statistics.

    Real SAR data after focusing has Rayleigh-distributed magnitude with
    a clutter background and some bright scatterers. This approximation
    uses log-normal distributed magnitudes with uniform phase.
    """
    rng = np.random.default_rng(seed)
    magnitude = rng.lognormal(mean=3.0, sigma=1.5, size=(size, size)).astype(np.float32)
    # Add some bright scatterers
    num_scatterers = max(1, size * size // 100)
    scatter_rows = rng.integers(0, size, size=num_scatterers)
    scatter_cols = rng.integers(0, size, size=num_scatterers)
    magnitude[scatter_rows, scatter_cols] *= rng.uniform(10, 50, size=num_scatterers).astype(np.float32)

    phase = rng.uniform(-np.pi, np.pi, size=(size, size)).astype(np.float32)
    real = magnitude * np.cos(phase)
    imag = magnitude * np.sin(phase)
    return np.stack([real, imag], axis=0)


class TestEndToEndSICDLikeFloat32(TestCase):
    """End-to-end: synthetic SICD-like float32 I/Q → uint8 display output."""

    def setUp(self):
        self.iq_data = _make_sicd_like_iq_data(size=64, seed=42)
        self.source = _make_complex_source(
            self.iq_data,
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ICAT": "SAR", "IREP": "NODISPLY", "ISUBCAT": ["I", "Q"]},
        )

    def test_is_complex_detects_source(self):
        """is_complex correctly identifies the synthetic complex source."""
        self.assertTrue(is_complex(self.source))

    def test_remap_produces_scalar_magnitude(self):
        """ComplexRemapFactory.build() wraps the source with a remap to scalar."""
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        self.assertEqual(remapped.num_bands, 1)
        self.assertEqual(remapped.pixel_value_type, "FLOAT32")
        block = remapped.get_block(0, 0)
        self.assertEqual(block.shape, (1, 64, 64))
        self.assertEqual(block.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(block)))
        self.assertTrue(np.all(block >= 0))

    def test_statistics_on_remapped_data_are_sensible(self):
        """Statistics computed on remapped data have non-zero mean and reasonable stddev."""
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        stats = compute_image_statistics(remapped, force_recompute=True)
        self.assertIsNotNone(stats)
        self.assertEqual(len(stats.bands), 1)
        band_stat = stats.bands[0]
        self.assertGreater(band_stat.mean, 0.0)
        self.assertGreater(band_stat.stddev, 0.0)
        self.assertGreater(band_stat.max, band_stat.min)

    def test_full_pipeline_produces_uint8_with_distribution(self):
        """Full pipeline produces uint8 output with values distributed across [0, 255]."""
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape[0], 1)
        # Not all zeros or all 255
        self.assertFalse(np.all(output == 0), "Output is all zeros — no contrast")
        self.assertFalse(np.all(output == 255), "Output is all 255 — saturated")
        # Has some spread across the dynamic range
        unique_vals = np.unique(output)
        self.assertGreater(len(unique_vals), 10, f"Only {len(unique_vals)} unique values — poor contrast")
        # Values span a reasonable portion of [0, 255]
        output_range = int(output.max()) - int(output.min())
        self.assertGreater(output_range, 50, f"Output range is only {output_range} — insufficient spread")


class TestEndToEndInt16IQ(TestCase):
    """End-to-end: synthetic int16 I/Q data → uint8 display output."""

    def setUp(self):
        rng = np.random.default_rng(99)
        magnitude = rng.lognormal(mean=6.0, sigma=1.0, size=(32, 32))
        phase = rng.uniform(-np.pi, np.pi, size=(32, 32))
        real = (magnitude * np.cos(phase)).astype(np.int16)
        imag = (magnitude * np.sin(phase)).astype(np.int16)
        self.iq_data = np.stack([real, imag], axis=0)
        self.source = _make_complex_source(
            self.iq_data,
            num_bands=2,
            pixel_value_type="INT16",
            metadata={"ISUBCAT": ["I", "Q"]},
        )

    def test_full_pipeline_int16(self):
        """int16 I/Q data produces uint8 output with reasonable distribution."""
        self.assertTrue(is_complex(self.source))
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.all(output == 0))
        self.assertFalse(np.all(output == 255))
        unique_vals = np.unique(output)
        self.assertGreater(len(unique_vals), 5)


class TestEndToEndAMP8IPHS8I(TestCase):
    """End-to-end: synthetic AMP8I_PHS8I data → uint8 display output."""

    def setUp(self):
        self.amp_table = np.arange(256, dtype=np.float32) ** 1.5 / 64.0
        rng = np.random.default_rng(77)
        amp_indices = rng.integers(10, 200, size=(32, 32), dtype=np.uint8)
        phase_indices = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        self.block_data = np.stack([amp_indices, phase_indices], axis=0)
        self.source = _make_complex_source(
            self.block_data,
            num_bands=2,
            pixel_value_type="UINT8",
            metadata={"ICAT": "SARIQ"},
        )

    def test_full_pipeline_amp8i_phs8i(self):
        """AMP8I_PHS8I data with amplitude table produces uint8 output with distribution."""
        self.assertTrue(is_complex(self.source))
        remapped = ComplexRemapFactory.build(
            self.source,
            band_interpretation=[ROLE_AMPLITUDE_INDEX, ROLE_PHASE],
            amplitude_table=self.amp_table,
        )
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.all(output == 0))
        self.assertFalse(np.all(output == 255))
        unique_vals = np.unique(output)
        self.assertGreater(len(unique_vals), 5)


class TestEndToEndMagnitudePhase(TestCase):
    """End-to-end: magnitude/phase float data → uint8 display output."""

    def setUp(self):
        rng = np.random.default_rng(55)
        magnitude = rng.lognormal(mean=2.0, sigma=1.0, size=(32, 32)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(32, 32)).astype(np.float32)
        self.block_data = np.stack([magnitude, phase], axis=0)
        self.source = _make_complex_source(
            self.block_data,
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["M", "P"], "ICAT": "SAR"},
        )

    def test_full_pipeline_magnitude_phase(self):
        """Magnitude/phase float SAR data produces uint8 output with distribution."""
        self.assertTrue(is_complex(self.source))
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_MAGNITUDE, ROLE_PHASE])
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.all(output == 0))
        self.assertFalse(np.all(output == 255))


class TestEndToEndNativeComplex(TestCase):
    """End-to-end: native complex64 data → uint8 display output."""

    def setUp(self):
        rng = np.random.default_rng(33)
        magnitude = rng.lognormal(mean=3.0, sigma=1.5, size=(32, 32))
        phase = rng.uniform(-np.pi, np.pi, size=(32, 32))
        complex_data = (magnitude * np.exp(1j * phase)).astype(np.complex64)
        self.block_data = complex_data[np.newaxis, :, :]
        self.source = _make_complex_source(
            self.block_data,
            num_bands=1,
            pixel_value_type="COMPLEX64",
            metadata={},
        )

    def test_full_pipeline_native_complex(self):
        """Native complex64 data produces uint8 output with distribution."""
        self.assertTrue(is_complex(self.source))
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=None)
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.all(output == 0))
        self.assertFalse(np.all(output == 255))
        unique_vals = np.unique(output)
        self.assertGreater(len(unique_vals), 5)


class TestEndToEndMagnitudeRemap(TestCase):
    """End-to-end using magnitude remap preset instead of quarter_power."""

    def setUp(self):
        self.iq_data = _make_sicd_like_iq_data(size=32, seed=11)
        self.source = _make_complex_source(
            self.iq_data,
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["I", "Q"]},
        )

    def test_magnitude_remap_pipeline(self):
        """Magnitude remap also produces usable uint8 output."""
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], remap="magnitude")
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.all(output == 0))
        self.assertFalse(np.all(output == 255))


class TestEndToEndCustomRemap(TestCase):
    """End-to-end with a custom remap callable."""

    def setUp(self):
        self.iq_data = _make_sicd_like_iq_data(size=32, seed=22)
        self.source = _make_complex_source(
            self.iq_data,
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["I", "Q"]},
        )

    def test_custom_remap_pipeline(self):
        """Custom remap callable (log magnitude) produces usable uint8 output."""

        def log_magnitude_remap(block):
            power = block[0] ** 2 + block[1] ** 2
            magnitude = np.sqrt(power)
            result = np.log1p(magnitude)
            if not np.all(np.isfinite(result)):
                result = np.where(np.isfinite(result), result, np.float32(0.0))
            return result[np.newaxis, :, :]

        remapped = ComplexRemapFactory.build(
            self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], remap=log_magnitude_remap
        )
        stats = compute_image_statistics(remapped, force_recompute=True)
        chain = DisplayChainFactory.build(remapped, stats=stats)

        block = remapped.get_block(0, 0)
        output = chain(block)

        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.all(output == 0))
        self.assertFalse(np.all(output == 255))


class TestStatisticsQuality(TestCase):
    """Validate that statistics on remapped SAR data are meaningful for DRA."""

    def setUp(self):
        self.iq_data = _make_sicd_like_iq_data(size=64, seed=42)
        self.source = _make_complex_source(
            self.iq_data,
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["I", "Q"]},
        )

    def test_statistics_reflect_remapped_domain(self):
        """Statistics are computed in magnitude domain, not raw I/Q domain."""
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        stats = compute_image_statistics(remapped, force_recompute=True)
        band = stats.bands[0]

        # Quarter-power of log-normal magnitude should have all-positive values
        self.assertGreater(band.min, 0.0)
        self.assertGreater(band.mean, 0.0)
        self.assertGreater(band.stddev, 0.0)
        # Mean should be well above zero (not near-zero like raw I/Q)
        self.assertGreater(band.mean, band.stddev * 0.1)

    def test_histogram_has_non_trivial_distribution(self):
        """Histogram of remapped data is not concentrated in a single bin."""
        remapped = ComplexRemapFactory.build(self.source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        stats = compute_image_statistics(remapped, force_recompute=True)
        band = stats.bands[0]

        nonzero_bins = np.count_nonzero(band.histogram)
        total_bins = len(band.histogram)
        # At least 10% of bins should have some count (data is spread out)
        self.assertGreater(nonzero_bins, total_bins * 0.1)
