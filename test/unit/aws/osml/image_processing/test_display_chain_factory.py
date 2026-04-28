#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase

import numpy as np

from aws.osml.image_processing.display_chain_factory import DisplayChainFactory
from aws.osml.image_processing.processing_chain import ProcessingChain
from aws.osml.image_processing.statistics import BandStatistics, ImageStatistics


class MockSource:
    """A mock image source for testing DisplayChainFactory modality detection."""

    def __init__(self, pixel_value_type="UINT8", num_bands=3, metadata=None, data_extensions=None):
        self.pixel_value_type = pixel_value_type
        self.num_bands = num_bands
        self._metadata = metadata or {}
        self._data_extensions = data_extensions

    @property
    def metadata(self):
        return self._metadata

    def get_data_extensions(self):
        return self._data_extensions


class TestDisplayChainFactoryGeoTIFF(TestCase):
    """Tests GeoTIFF PhotometricInterpretation-based detection."""

    def test_photometric_3_palette(self):
        """GeoTIFF PhotometricInterpretation=3 → palette chain (empty steps)."""
        source = MockSource(
            pixel_value_type="UINT8",
            num_bands=1,
            metadata={"PhotometricInterpretation": 3},
        )
        chain = DisplayChainFactory.build(source)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertEqual(chain.output_bands, 3)
        self.assertEqual(chain.output_dtype, np.dtype(np.uint8))
        self.assertEqual(len(chain.steps), 0)

    def test_photometric_6_display_ready(self):
        """GeoTIFF PhotometricInterpretation=6 → display-ready (empty chain)."""
        source = MockSource(
            pixel_value_type="UINT8",
            num_bands=3,
            metadata={"PhotometricInterpretation": 6},
        )
        chain = DisplayChainFactory.build(source)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertEqual(len(chain.steps), 0)
        self.assertEqual(chain.output_dtype, np.dtype(np.uint8))


class TestDisplayChainFactoryEODetection(TestCase):
    """Tests EO chain for various source types that now fall through to EO path."""

    def test_float32_single_band_post_remap(self):
        """Float32 single-band source (post-remap) → builds DRA chain."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0.0, 100.0, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=100.0,
            mean=50.0,
            stddev=20.0,
            count=256,
            m2=20.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat])

        source = MockSource(
            pixel_value_type="FLOAT32",
            num_bands=1,
            metadata={},
        )
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)
        self.assertEqual(chain.output_bands, 1)

    def test_icat_sar_irep_mono_eo_chain(self):
        """NITF ICAT=SAR + IREP=MONO → EO chain (detected SAR product, not complex)."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat])

        source = MockSource(
            pixel_value_type="UINT16",
            num_bands=1,
            metadata={"ICAT": "SAR", "IREP": "MONO"},
        )
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)
        self.assertEqual(chain.output_bands, 1)

    def test_complex_pixel_type_falls_to_eo(self):
        """Complex pixel type now falls through to EO chain (caller's responsibility to remap)."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat])

        source = MockSource(pixel_value_type="COMPLEX64", num_bands=2, metadata={})
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)

    def test_irep_nodisply_icat_sar_falls_to_eo(self):
        """IREP=NODISPLY + ICAT=SAR now falls through to EO chain (no SAR-specific chain)."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat])

        source = MockSource(
            pixel_value_type="FLOAT32",
            num_bands=2,
            metadata={"IREP": "NODISPLY", "ICAT": "SAR"},
        )
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)


class TestDisplayChainFactoryDisplayReady(TestCase):
    """Tests display-ready classification for uint8 sources."""

    def test_uint8_1_band_display_ready(self):
        """uint8 + 1 band → display-ready (empty chain)."""
        source = MockSource(pixel_value_type="UINT8", num_bands=1, metadata={})
        chain = DisplayChainFactory.build(source)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertEqual(len(chain.steps), 0)
        self.assertEqual(chain.output_bands, 1)
        self.assertEqual(chain.output_dtype, np.dtype(np.uint8))

    def test_uint8_3_bands_display_ready(self):
        """uint8 + 3 bands → display-ready (empty chain)."""
        source = MockSource(pixel_value_type="UINT8", num_bands=3, metadata={})
        chain = DisplayChainFactory.build(source)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertEqual(len(chain.steps), 0)
        self.assertEqual(chain.output_bands, 3)
        self.assertEqual(chain.output_dtype, np.dtype(np.uint8))


class TestDisplayChainFactoryValidation(TestCase):
    """Tests ValueError for unknown parameters."""

    def test_sar_remap_parameter_not_in_signature(self):
        """sar_remap parameter is no longer part of the build() signature."""
        import inspect

        sig = inspect.signature(DisplayChainFactory.build)
        self.assertNotIn("sar_remap", sig.parameters)

    def test_unknown_range_adjustment_raises_valueerror(self):
        """Unknown range_adjustment string raises ValueError."""
        source = MockSource(pixel_value_type="UINT16", num_bands=1, metadata={})

        with self.assertRaises(ValueError) as ctx:
            DisplayChainFactory.build(source, range_adjustment="unknown_adj")

        self.assertIn("unknown_adj", str(ctx.exception))


class TestDetectRGBBandsIREPBAND(TestCase):
    """Tests NITF IREPBAND-based RGB band detection (priority 1)."""

    def test_irepband_rgb_at_known_indices(self):
        """NITF IREPBAND with R, G, B at known indices."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"IREPBAND": ["R", "G", "B", "M"]}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (0, 1, 2))

    def test_irepband_rgb_reordered(self):
        """NITF IREPBAND with R, G, B in non-standard order."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"IREPBAND": ["M", "B", "G", "R"]}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (3, 2, 1))

    def test_irepband_case_insensitive(self):
        """NITF IREPBAND matching is case-insensitive."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"IREPBAND": ["r", "g", "b", "m"]}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (0, 1, 2))

    def test_irepband_missing_blue_falls_through(self):
        """NITF IREPBAND without B falls through to next priority."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"IREPBAND": ["R", "G", "M", "M"], "PhotometricInterpretation": 2}
        result = _detect_rgb_bands(metadata, num_bands=4)
        # Falls through to PhotometricInterpretation=2
        self.assertEqual(result, (0, 1, 2))


class TestDetectRGBBandsPhotometric(TestCase):
    """Tests GeoTIFF PhotometricInterpretation=2 detection (priority 2)."""

    def test_photometric_2_returns_012(self):
        """GeoTIFF PhotometricInterpretation=2 → (0, 1, 2)."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"PhotometricInterpretation": 2}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (0, 1, 2))


class TestDetectRGBBandsGDALMetadata(TestCase):
    """Tests GDAL_METADATA ColorInterp detection (priority 3)."""

    def test_gdal_colorinterp_standard(self):
        """GDAL_METADATA ColorInterp → correct indices."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        xml = (
            "<GDALMetadata>"
            '  <Item name="ColorInterp" sample="0">Red</Item>'
            '  <Item name="ColorInterp" sample="1">Green</Item>'
            '  <Item name="ColorInterp" sample="2">Blue</Item>'
            '  <Item name="ColorInterp" sample="3">Alpha</Item>'
            "</GDALMetadata>"
        )
        metadata = {"GDAL_METADATA": xml}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (0, 1, 2))

    def test_gdal_colorinterp_reordered(self):
        """GDAL_METADATA ColorInterp with non-standard band order."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        xml = (
            "<GDALMetadata>"
            '  <Item name="ColorInterp" sample="0">Blue</Item>'
            '  <Item name="ColorInterp" sample="1">Green</Item>'
            '  <Item name="ColorInterp" sample="2">Red</Item>'
            '  <Item name="ColorInterp" sample="3">Alpha</Item>'
            "</GDALMetadata>"
        )
        metadata = {"GDAL_METADATA": xml}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (2, 1, 0))

    def test_gdal_colorinterp_malformed_xml(self):
        """Malformed GDAL_METADATA XML falls through gracefully."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"GDAL_METADATA": "<not valid xml"}
        result = _detect_rgb_bands(metadata, num_bands=4)
        # Falls through to fallback (0, 1, 2)
        self.assertEqual(result, (0, 1, 2))


class TestDetectRGBBandsBANDSB(TestCase):
    """Tests BANDSB TRE wavelength matching (priority 4)."""

    def test_bandsb_wavelength_matching(self):
        """BANDSB wavelength matching with known center wavelengths."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        # Band 0: 475nm (blue), Band 1: 550nm (green), Band 2: 650nm (red), Band 3: 850nm (NIR)
        metadata = {"BANDSB": {"center_wavelengths": [475.0, 550.0, 650.0, 850.0]}}
        result = _detect_rgb_bands(metadata, num_bands=4)
        # Red=band2 (650nm), Green=band1 (550nm), Blue=band0 (475nm)
        self.assertEqual(result, (2, 1, 0))

    def test_bandsb_flat_form(self):
        """BANDSB_CENTER_WAVELENGTHS flat form works."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"BANDSB_CENTER_WAVELENGTHS": [475.0, 550.0, 650.0, 850.0]}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (2, 1, 0))

    def test_bandsb_with_3_or_fewer_bands_skipped(self):
        """BANDSB with 3 or fewer wavelengths is skipped (needs >3 bands)."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"BANDSB": {"center_wavelengths": [475.0, 550.0, 650.0]}}
        result = _detect_rgb_bands(metadata, num_bands=4)
        # Falls through to fallback
        self.assertEqual(result, (0, 1, 2))


class TestDetectRGBBandsISUBCAT(TestCase):
    """Tests NITF ISUBCAT color code detection (priority 5)."""

    def test_isubcat_rgb(self):
        """NITF ISUBCAT R, G, B → correct indices."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"ISUBCAT": ["R", "G", "B", "NIR"]}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (0, 1, 2))

    def test_isubcat_reordered(self):
        """NITF ISUBCAT with reordered R, G, B."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"ISUBCAT": ["NIR", "B", "R", "G"]}
        result = _detect_rgb_bands(metadata, num_bands=4)
        self.assertEqual(result, (2, 3, 1))


class TestDetectRGBBandsFallback(TestCase):
    """Tests fallback to (0, 1, 2) with warning."""

    def test_unknown_multiband_fallback(self):
        """Unknown multiband → (0, 1, 2) with warning logged."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        metadata = {"SOME_OTHER_KEY": "value"}
        with self.assertLogs("aws.osml.image_processing.display_chain_factory", level="WARNING") as cm:
            result = _detect_rgb_bands(metadata, num_bands=5)

        self.assertEqual(result, (0, 1, 2))
        self.assertTrue(any("falling back to bands (0, 1, 2)" in msg for msg in cm.output))

    def test_none_metadata_fallback(self):
        """None metadata → (0, 1, 2) with warning logged."""
        from aws.osml.image_processing.display_chain_factory import _detect_rgb_bands

        with self.assertLogs("aws.osml.image_processing.display_chain_factory", level="WARNING") as cm:
            result = _detect_rgb_bands(None, num_bands=5)

        self.assertEqual(result, (0, 1, 2))
        self.assertTrue(any("falling back to bands (0, 1, 2)" in msg for msg in cm.output))


class TestHelperFunctions(TestCase):
    """Tests for individual helper functions."""

    def test_find_band_case_insensitive(self):
        """_find_band performs case-insensitive matching."""
        from aws.osml.image_processing.display_chain_factory import _find_band

        self.assertEqual(_find_band(["red", "GREEN", "Blue"], "Red"), 0)
        self.assertEqual(_find_band(["red", "GREEN", "Blue"], "green"), 1)
        self.assertEqual(_find_band(["red", "GREEN", "Blue"], "BLUE"), 2)

    def test_find_band_not_found(self):
        """_find_band returns None when target not found."""
        from aws.osml.image_processing.display_chain_factory import _find_band

        self.assertIsNone(_find_band(["M", "M", "M"], "R"))

    def test_get_nitf_band_representations_list(self):
        """_get_nitf_band_representations handles list input."""
        from aws.osml.image_processing.display_chain_factory import _get_nitf_band_representations

        result = _get_nitf_band_representations({"IREPBAND": ["R", "G", "B"]})
        self.assertEqual(result, ["R", "G", "B"])

    def test_get_nitf_band_representations_comma_separated(self):
        """_get_nitf_band_representations handles comma-separated string."""
        from aws.osml.image_processing.display_chain_factory import _get_nitf_band_representations

        result = _get_nitf_band_representations({"IREPBAND": "R,G,B,M"})
        self.assertEqual(result, ["R", "G", "B", "M"])

    def test_get_nitf_band_representations_missing(self):
        """_get_nitf_band_representations returns None when key missing."""
        from aws.osml.image_processing.display_chain_factory import _get_nitf_band_representations

        result = _get_nitf_band_representations({})
        self.assertIsNone(result)

    def test_get_gdal_color_interps_valid(self):
        """_get_gdal_color_interps parses valid XML."""
        from aws.osml.image_processing.display_chain_factory import _get_gdal_color_interps

        xml = (
            "<GDALMetadata>"
            '  <Item name="ColorInterp" sample="0">Red</Item>'
            '  <Item name="ColorInterp" sample="1">Green</Item>'
            '  <Item name="ColorInterp" sample="2">Blue</Item>'
            "</GDALMetadata>"
        )
        result = _get_gdal_color_interps({"GDAL_METADATA": xml})
        self.assertEqual(result, ["Red", "Green", "Blue"])

    def test_get_gdal_color_interps_missing(self):
        """_get_gdal_color_interps returns None when key missing."""
        from aws.osml.image_processing.display_chain_factory import _get_gdal_color_interps

        result = _get_gdal_color_interps({})
        self.assertIsNone(result)

    def test_get_bandsb_tre_dict_form(self):
        """_get_bandsb_tre handles dict form."""
        from aws.osml.image_processing.display_chain_factory import _get_bandsb_tre

        result = _get_bandsb_tre({"BANDSB": {"center_wavelengths": [475.0, 550.0, 650.0, 850.0]}})
        self.assertEqual(result, [475.0, 550.0, 650.0, 850.0])

    def test_get_bandsb_tre_flat_form(self):
        """_get_bandsb_tre handles flat form."""
        from aws.osml.image_processing.display_chain_factory import _get_bandsb_tre

        result = _get_bandsb_tre({"BANDSB_CENTER_WAVELENGTHS": [475.0, 550.0, 650.0]})
        self.assertEqual(result, [475.0, 550.0, 650.0])

    def test_get_bandsb_tre_missing(self):
        """_get_bandsb_tre returns None when no wavelength data."""
        from aws.osml.image_processing.display_chain_factory import _get_bandsb_tre

        result = _get_bandsb_tre({})
        self.assertIsNone(result)

    def test_match_bands_by_wavelength_standard(self):
        """_match_bands_by_wavelength selects correct bands for standard multispectral."""
        from aws.osml.image_processing.display_chain_factory import _match_bands_by_wavelength

        # Typical 4-band multispectral: Blue, Green, Red, NIR
        wavelengths = [475.0, 550.0, 650.0, 850.0]
        result = _match_bands_by_wavelength(wavelengths)
        # Red=band2 (650nm closest to 660), Green=band1 (550nm closest to 532.5), Blue=band0 (475nm closest to 472.5)
        self.assertEqual(result, (2, 1, 0))

    def test_match_bands_by_wavelength_all_same_returns_none(self):
        """_match_bands_by_wavelength returns None when all bands have same wavelength."""
        from aws.osml.image_processing.display_chain_factory import _match_bands_by_wavelength

        # All bands at same wavelength → all indices would be the same
        wavelengths = [550.0, 550.0, 550.0, 550.0]
        result = _match_bands_by_wavelength(wavelengths)
        self.assertIsNone(result)


class TestBandSelectionWiringInBuild(TestCase):
    """Tests that build() correctly wires band selection into the EO chain."""

    def test_explicit_band_selection_overrides_detection(self):
        """Explicit band_selection parameter overrides automatic detection."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat, band_stat])

        source = MockSource(
            pixel_value_type="UINT16",
            num_bands=5,
            metadata={"IREPBAND": ["R", "G", "B", "NIR", "SWIR"]},
        )
        chain = DisplayChainFactory.build(source, stats=stats, band_selection=(3, 2, 1))

        self.assertIsInstance(chain, ProcessingChain)
        self.assertEqual(chain.input_bands, (3, 2, 1))
        self.assertEqual(chain.output_bands, 3)

    def test_multiband_no_explicit_selection_triggers_detection(self):
        """Source with >3 bands and no explicit band_selection calls _detect_rgb_bands."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat, band_stat])

        # Source with 5 bands and IREPBAND metadata indicating R at 2, G at 3, B at 4
        source = MockSource(
            pixel_value_type="UINT16",
            num_bands=5,
            metadata={"IREPBAND": ["M", "M", "R", "G", "B"]},
        )
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        # _detect_rgb_bands should find R=2, G=3, B=4
        self.assertEqual(chain.input_bands, (2, 3, 4))
        self.assertEqual(chain.output_bands, 3)

    def test_1_band_source_no_band_selection(self):
        """Source with 1 band leaves input_bands as None (no band selection)."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat])

        source = MockSource(pixel_value_type="UINT16", num_bands=1, metadata={})
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertIsNone(chain.input_bands)
        self.assertEqual(chain.output_bands, 1)

    def test_3_band_source_no_band_selection(self):
        """Source with 3 bands leaves input_bands as None (no band selection)."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat, band_stat])

        source = MockSource(pixel_value_type="UINT16", num_bands=3, metadata={})
        chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertIsNone(chain.input_bands)
        self.assertEqual(chain.output_bands, 3)

    def test_multiband_fallback_detection_when_no_metadata(self):
        """Source with >3 bands and no metadata falls back to (0, 1, 2)."""
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat, band_stat])

        source = MockSource(pixel_value_type="UINT16", num_bands=5, metadata={})

        with self.assertLogs("aws.osml.image_processing.display_chain_factory", level="WARNING"):
            chain = DisplayChainFactory.build(source, stats=stats)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertEqual(chain.input_bands, (0, 1, 2))
        self.assertEqual(chain.output_bands, 3)


class MockBlockSource:
    """A mock image source that supports block-based pixel reads for integration tests."""

    def __init__(self, image, pixel_value_type="UINT16", metadata=None):
        self._image = image
        self.pixel_value_type = pixel_value_type
        self.num_bands = image.shape[0]
        self._metadata = metadata or {}

    @property
    def block_grid_size(self):
        return (1, 1)

    def has_block(self, row, col, resolution_level=0):
        return row == 0 and col == 0

    def get_block(self, row, col, resolution_level=0, bands=None):
        if bands is not None:
            return self._image[list(bands), :, :]
        return self._image.copy()

    @property
    def metadata(self):
        return self._metadata

    def get_data_extensions(self):
        return None


class TestDisplayChainFactoryIntegration(TestCase):
    """Integration tests for the full DisplayChainFactory → MappedImageProvider pipeline."""

    def test_end_to_end_build_mapped_provider_produces_uint8(self):
        """End-to-end: DisplayChainFactory.build() → MappedImageProvider → get_block() produces valid uint8 output."""
        from aws.osml.image_processing.mapped_provider import MappedImageProvider

        # Create a mock source with uint16 pixel data (simulating a high-bit-depth EO image)
        rng = np.random.default_rng(42)
        image = rng.integers(0, 65535, size=(3, 64, 64), dtype=np.uint16)
        source = MockBlockSource(image, pixel_value_type="UINT16")

        # Pre-compute stats for the source
        histogram = np.ones(256, dtype=np.float64)
        bin_edges = np.linspace(0, 65535, 257)
        band_stat = BandStatistics(
            min=0.0,
            max=65535.0,
            mean=32000.0,
            stddev=10000.0,
            count=256,
            m2=10000.0**2 * 256,
            histogram=histogram,
            bin_edges=bin_edges,
        )
        stats = ImageStatistics(bands=[band_stat, band_stat, band_stat])

        # Build the display chain
        chain = DisplayChainFactory.build(source, stats=stats)

        # Create MappedImageProvider with the chain
        mapped = MappedImageProvider(source, chain, source_bands=chain.input_bands)

        # Get a block and verify the output
        result = mapped.get_block(0, 0)

        # Output should be uint8
        self.assertEqual(result.dtype, np.uint8)
        # Output should have valid pixel values in [0, 255]
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))
        # Output should have the expected number of bands
        self.assertEqual(result.shape[0], chain.output_bands)
        # Output should have the same spatial dimensions
        self.assertEqual(result.shape[1], 64)
        self.assertEqual(result.shape[2], 64)

    def test_stats_auto_computation_when_stats_none(self):
        """Statistics auto-computation when stats=None triggers stats computation from source."""
        # Create a mock source with uint16 pixel data that supports block iteration
        rng = np.random.default_rng(123)
        image = rng.integers(100, 60000, size=(1, 32, 32), dtype=np.uint16)
        source = MockBlockSource(image, pixel_value_type="UINT16")

        # Call build with stats=None — this should trigger stats computation
        chain = DisplayChainFactory.build(source, stats=None)

        # Verify the chain was built successfully (has steps from EO chain)
        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)
        self.assertEqual(chain.output_bands, 1)
        self.assertEqual(chain.output_dtype, np.dtype(np.uint8))

    def test_adaptive_statistics_large_source_uses_block_sampling(self):
        """Large images (>100 blocks) trigger block sampling for faster stats computation."""
        from unittest.mock import patch

        from aws.osml.image_processing.statistics import SamplingStrategy

        # Create a source with many blocks (11x11 = 121 blocks, exceeds threshold of 100)
        rng = np.random.default_rng(99)
        block = rng.integers(100, 60000, size=(1, 32, 32), dtype=np.uint16)

        class LargeBlockSource:
            pixel_value_type = "UINT16"
            num_bands = 1

            @property
            def block_grid_size(self):
                return (11, 11)

            def has_block(self, row, col, resolution_level=0):
                return True

            def get_block(self, row, col, resolution_level=0, bands=None):
                return block.copy()

            @property
            def metadata(self):
                return {}

            def get_data_extensions(self):
                return None

        source = LargeBlockSource()

        with patch(
            "aws.osml.image_processing.statistics.compute_image_statistics",
            wraps=__import__(
                "aws.osml.image_processing.statistics", fromlist=["compute_image_statistics"]
            ).compute_image_statistics,
        ) as mock_compute:
            chain = DisplayChainFactory.build(source, stats=None)
            mock_compute.assert_called_once()
            call_kwargs = mock_compute.call_args[1]
            self.assertEqual(call_kwargs["sampling"], SamplingStrategy.BLOCK)
            self.assertLess(call_kwargs["sample_rate"], 1.0)
            self.assertGreater(call_kwargs["num_workers"], 0)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)
        self.assertEqual(chain.output_dtype, np.dtype(np.uint8))

    def test_adaptive_statistics_small_source_uses_full_scan(self):
        """Small images (≤100 blocks) use full scan (ALL strategy)."""
        from unittest.mock import patch

        from aws.osml.image_processing.statistics import SamplingStrategy

        rng = np.random.default_rng(42)
        image = rng.integers(100, 60000, size=(1, 32, 32), dtype=np.uint16)
        source = MockBlockSource(image, pixel_value_type="UINT16")

        with patch(
            "aws.osml.image_processing.statistics.compute_image_statistics",
            wraps=__import__(
                "aws.osml.image_processing.statistics", fromlist=["compute_image_statistics"]
            ).compute_image_statistics,
        ) as mock_compute:
            chain = DisplayChainFactory.build(source, stats=None)
            mock_compute.assert_called_once()
            call_kwargs = mock_compute.call_args[1]
            self.assertEqual(call_kwargs["sampling"], SamplingStrategy.ALL)
            self.assertEqual(call_kwargs["sample_rate"], 1.0)
            self.assertEqual(call_kwargs["num_workers"], 0)

        self.assertIsInstance(chain, ProcessingChain)
        self.assertGreater(len(chain.steps), 0)
