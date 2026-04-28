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
    ComplexRemapFactory,
    load_complex_remap,
)


def _make_source(num_bands=2, pixel_value_type="FLOAT32", num_rows=64, num_columns=64, block_data=None, metadata=None):
    """Create a mock ImageAssetProvider source for factory tests."""
    source = MagicMock()
    type(source).num_bands = PropertyMock(return_value=num_bands)
    type(source).pixel_value_type = PropertyMock(return_value=pixel_value_type)
    type(source).num_rows = PropertyMock(return_value=num_rows)
    type(source).num_columns = PropertyMock(return_value=num_columns)
    type(source).num_pixels_per_block_horizontal = PropertyMock(return_value=num_columns)
    type(source).num_pixels_per_block_vertical = PropertyMock(return_value=num_rows)
    type(source).num_resolution_levels = PropertyMock(return_value=1)
    type(source).block_grid_size = PropertyMock(return_value=(1, 1))
    type(source).key = PropertyMock(return_value="test_source")
    type(source).metadata = PropertyMock(return_value=metadata)

    if block_data is not None:
        source.get_block = MagicMock(return_value=block_data)
    else:
        rng = np.random.default_rng(42)
        default_block = rng.standard_normal((num_bands, num_rows, num_columns)).astype(np.float32)
        source.get_block = MagicMock(return_value=default_block)

    source.has_block = MagicMock(return_value=True)
    return source


class TestComplexRemapFactoryBuild(TestCase):
    """Tests for ComplexRemapFactory.build()."""

    def test_returns_mapped_provider_with_correct_properties(self):
        """Factory returns MappedImageProvider with num_bands=1, pixel_value_type='FLOAT32'."""
        source = _make_source(num_bands=2, pixel_value_type="FLOAT32")
        result = ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        self.assertEqual(result.num_bands, 1)
        self.assertEqual(result.pixel_value_type, "FLOAT32")

    def test_real_imaginary_float32(self):
        """Real/imaginary float32 source produces (1, H, W) float32 output."""
        block = np.array([[[3.0, 5.0]], [[4.0, 12.0]]], dtype=np.float32)
        source = _make_source(num_bands=2, block_data=block, num_rows=1, num_columns=2)
        provider = ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertEqual(output.dtype, np.float32)
        # quarter_power: sqrt(sqrt(I^2 + Q^2))
        expected_0 = np.sqrt(np.sqrt(np.float32(9.0 + 16.0)))
        expected_1 = np.sqrt(np.sqrt(np.float32(25.0 + 144.0)))
        np.testing.assert_allclose(output[0, 0, 0], expected_0, rtol=1e-5)
        np.testing.assert_allclose(output[0, 0, 1], expected_1, rtol=1e-5)

    def test_amplitude_index_phase_with_table(self):
        """amplitude_index/phase with amplitude table: table applied per block."""
        amp_table = np.arange(256, dtype=np.float32) * 2.0
        amp_idx = np.array([[10, 20]], dtype=np.uint8)
        phase_idx = np.array([[0, 64]], dtype=np.uint8)
        block = np.stack([amp_idx, phase_idx], axis=0)

        source = _make_source(num_bands=2, pixel_value_type="UINT8", block_data=block, num_rows=1, num_columns=2)
        provider = ComplexRemapFactory.build(
            source,
            band_interpretation=[ROLE_AMPLITUDE_INDEX, ROLE_PHASE],
            amplitude_table=amp_table,
        )
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertGreater(output[0, 0, 0], 0.0)

    def test_magnitude_phase_float(self):
        """magnitude/phase with float source: trig decode produces valid I/Q."""
        mag = np.array([[10.0, 5.0]], dtype=np.float32)
        phase = np.array([[0.0, np.pi / 4]], dtype=np.float32)
        block = np.stack([mag, phase], axis=0)

        source = _make_source(num_bands=2, pixel_value_type="FLOAT32", block_data=block, num_rows=1, num_columns=2)
        provider = ComplexRemapFactory.build(source, band_interpretation=[ROLE_MAGNITUDE, ROLE_PHASE])
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertTrue(np.all(np.isfinite(output)))
        # Magnitude of (10*cos(0), 10*sin(0)) is 10, quarter-power = sqrt(sqrt(100))
        expected_0 = np.sqrt(np.sqrt(np.float32(100.0)))
        np.testing.assert_allclose(output[0, 0, 0], expected_0, rtol=1e-4)

    def test_band_interpretation_length_mismatch_raises(self):
        """band_interpretation length != source.num_bands raises ValueError."""
        source = _make_source(num_bands=2)
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL])
        self.assertIn("1 roles", str(ctx.exception))
        self.assertIn("2 bands", str(ctx.exception))

    def test_amplitude_index_without_table_raises(self):
        """'amplitude_index' without amplitude_table raises ValueError."""
        source = _make_source(num_bands=2)
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(source, band_interpretation=[ROLE_AMPLITUDE_INDEX, ROLE_PHASE])
        self.assertIn("amplitude_table is required", str(ctx.exception))

    def test_amplitude_table_without_index_raises(self):
        """amplitude_table provided without 'amplitude_index' raises ValueError."""
        source = _make_source(num_bands=2)
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(
                source,
                band_interpretation=[ROLE_REAL, ROLE_IMAGINARY],
                amplitude_table=np.arange(256, dtype=np.float32),
            )
        self.assertIn("amplitude_table provided but", str(ctx.exception))

    def test_none_interpretation_native_complex(self):
        """band_interpretation=None with native complex64 source: inferred correctly."""
        block = np.array([[[1 + 2j, 3 + 4j]]], dtype=np.complex64)
        source = _make_source(num_bands=1, pixel_value_type="COMPLEX64", block_data=block, num_rows=1, num_columns=2)
        provider = ComplexRemapFactory.build(source, band_interpretation=None)
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertEqual(output.dtype, np.float32)
        # quarter_power of (1+2j): sqrt(sqrt(1+4)) = sqrt(sqrt(5))
        expected_0 = np.sqrt(np.sqrt(np.float32(5.0)))
        np.testing.assert_allclose(output[0, 0, 0], expected_0, rtol=1e-5)

    def test_none_interpretation_2band_numeric(self):
        """band_interpretation=None with 2-band numeric: defaults to real/imaginary."""
        block = np.array([[[3.0]], [[4.0]]], dtype=np.float32)
        source = _make_source(num_bands=2, pixel_value_type="FLOAT32", block_data=block, num_rows=1, num_columns=1)
        provider = ComplexRemapFactory.build(source, band_interpretation=None)
        output = provider.get_block(0, 0)
        expected = np.sqrt(np.sqrt(np.float32(25.0)))
        np.testing.assert_allclose(output[0, 0, 0], expected, rtol=1e-5)

    def test_none_interpretation_single_band_non_complex_raises(self):
        """band_interpretation=None, non-complex, num_bands=1 raises ValueError."""
        source = _make_source(num_bands=1, pixel_value_type="FLOAT32")
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(source, band_interpretation=None)
        self.assertIn("cannot infer interpretation", str(ctx.exception))

    def test_custom_callable_invoked(self):
        """Custom callable accepted and invoked per block."""
        block = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
        source = _make_source(num_bands=2, pixel_value_type="FLOAT32", block_data=block, num_rows=1, num_columns=2)

        def custom_remap(iq: np.ndarray) -> np.ndarray:
            return (iq[0] + iq[1])[np.newaxis, :, :]

        provider = ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], remap=custom_remap)
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        np.testing.assert_allclose(output[0, 0, 0], 4.0)
        np.testing.assert_allclose(output[0, 0, 1], 6.0)

    def test_magnitude_remap_preset(self):
        """remap='magnitude' uses magnitude preset."""
        block = np.array([[[3.0]], [[4.0]]], dtype=np.float32)
        source = _make_source(num_bands=2, pixel_value_type="FLOAT32", block_data=block, num_rows=1, num_columns=1)
        provider = ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], remap="magnitude")
        output = provider.get_block(0, 0)
        np.testing.assert_allclose(output[0, 0, 0], 5.0, rtol=1e-5)

    def test_unknown_remap_preset_raises(self):
        """Unknown remap preset name raises ValueError."""
        source = _make_source(num_bands=2)
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], remap="nonexistent")
        self.assertIn("Unknown remap preset", str(ctx.exception))

    def test_with_cache_second_call_hits_cache(self):
        """With cache provided, second get_block() hits cache."""
        from aws.osml.image_processing.tile_cache import TileCache

        block = np.array([[[3.0]], [[4.0]]], dtype=np.float32)
        source = _make_source(num_bands=2, pixel_value_type="FLOAT32", block_data=block, num_rows=1, num_columns=1)
        cache = TileCache(max_bytes=64 * 1024 * 1024)

        provider = ComplexRemapFactory.build(source, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY], cache=cache)
        output1 = provider.get_block(0, 0)
        output2 = provider.get_block(0, 0)
        np.testing.assert_array_equal(output1, output2)
        # Source should only be called once (cache hit on second call)
        self.assertEqual(source.get_block.call_count, 1)

    def test_integer_phase_scaling(self):
        """Integer phase with 'phase' role uses value / 2^nbits * 2π scaling."""
        mag = np.array([[100]], dtype=np.int16)
        # 16384 / 65536 * 2π = π/2
        phase_int = np.array([[16384]], dtype=np.int16)
        block = np.stack([mag, phase_int], axis=0)

        source = _make_source(num_bands=2, pixel_value_type="INT16", block_data=block, num_rows=1, num_columns=1)
        provider = ComplexRemapFactory.build(source, band_interpretation=[ROLE_MAGNITUDE, ROLE_PHASE], remap="magnitude")
        output = provider.get_block(0, 0)
        # M/P → I/Q: I = 100*cos(π/2) ≈ 0, Q = 100*sin(π/2) ≈ 100
        # magnitude: sqrt(I² + Q²) ≈ 100
        np.testing.assert_allclose(output[0, 0, 0], 100.0, rtol=1e-3)

    def test_unrecognized_role_raises(self):
        """Unrecognized role in band_interpretation raises ValueError."""
        source = _make_source(num_bands=2)
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(source, band_interpretation=["real", "bogus"])
        self.assertIn("Unrecognized band role", str(ctx.exception))

    def test_amplitude_table_with_none_interpretation_raises(self):
        """amplitude_table with band_interpretation=None raises ValueError."""
        source = _make_source(num_bands=2, pixel_value_type="FLOAT32")
        with self.assertRaises(ValueError) as ctx:
            ComplexRemapFactory.build(source, band_interpretation=None, amplitude_table=np.arange(256, dtype=np.float32))
        self.assertIn("amplitude_table provided but band_interpretation is None", str(ctx.exception))


# ===========================================================================
# Tests for load_complex_remap()
# ===========================================================================


def _make_reader(
    image_asset=None,
    des_assets=None,
    asset_keys=None,
):
    """Create a mock reader for load_complex_remap() tests."""
    reader = MagicMock()

    if asset_keys is None:
        asset_keys = ["image:0"]
        if des_assets:
            asset_keys.extend(des_assets.keys())

    reader.get_asset_keys = MagicMock(return_value=asset_keys)

    if image_asset is None:
        image_asset = _make_source(num_bands=2, pixel_value_type="FLOAT32")

    assets_map = {"image:0": image_asset}
    if des_assets:
        assets_map.update(des_assets)

    reader.get_asset = MagicMock(side_effect=lambda k: assets_map.get(k))
    return reader


def _make_des_asset(desid="XML_DATA_CONTENT", desshtn="urn:SICD:1.3.0", xml_content=None):
    """Create a mock DES asset."""
    des_asset = MagicMock()
    des_asset.metadata = {"DESID": desid, "DESSHTN": desshtn}
    if xml_content is None:
        xml_content = ""
    raw_asset = MagicMock()
    raw_asset.read = MagicMock(return_value=xml_content.encode("utf-8"))
    des_asset.raw_asset = raw_asset
    return des_asset


_SICD_RE32F_XML = """<?xml version="1.0" encoding="UTF-8"?>
<SICD xmlns="urn:SICD:1.3.0">
  <ImageData>
    <PixelType>RE32F_IM32F</PixelType>
    <NumRows>64</NumRows>
    <NumCols>64</NumCols>
    <FirstRow>0</FirstRow>
    <FirstCol>0</FirstCol>
    <FullImage><NumRows>64</NumRows><NumCols>64</NumCols></FullImage>
    <SCPPixel><Row>32</Row><Col>32</Col></SCPPixel>
  </ImageData>
</SICD>"""

_SICD_AMP8I_XML = """<?xml version="1.0" encoding="UTF-8"?>
<SICD xmlns="urn:SICD:1.3.0">
  <ImageData>
    <PixelType>AMP8I_PHS8I</PixelType>
    <AmpTable size="256">
{amp_entries}
    </AmpTable>
    <NumRows>64</NumRows>
    <NumCols>64</NumCols>
    <FirstRow>0</FirstRow>
    <FirstCol>0</FirstCol>
    <FullImage><NumRows>64</NumRows><NumCols>64</NumCols></FullImage>
    <SCPPixel><Row>32</Row><Col>32</Col></SCPPixel>
  </ImageData>
</SICD>"""


def _make_amp_table_xml():
    """Generate AmpTable XML entries for 256 values."""
    entries = []
    for i in range(256):
        entries.append(f'      <Amplitude index="{i}">{float(i) * 0.5}</Amplitude>')
    return "\n".join(entries)


class TestLoadComplexRemapSICD(TestCase):
    """Tests for load_complex_remap() with SICD DES."""

    def test_sicd_re32f_im32f(self):
        """SICD DES with PixelType=RE32F_IM32F → band_interpretation=['real','imaginary']."""
        des_asset = _make_des_asset(desshtn="urn:SICD:1.3.0", xml_content=_SICD_RE32F_XML)
        image_asset = _make_source(num_bands=2, pixel_value_type="FLOAT32")
        reader = _make_reader(
            image_asset=image_asset,
            des_assets={"des:0": des_asset},
            asset_keys=["image:0", "des:0"],
        )
        provider = load_complex_remap(reader)
        self.assertEqual(provider.num_bands, 1)
        self.assertEqual(provider.pixel_value_type, "FLOAT32")

    def test_sicd_amp8i_phs8i_with_amp_table(self):
        """SICD DES with AMP8I_PHS8I and AmpTable → amplitude table extracted."""
        xml = _SICD_AMP8I_XML.format(amp_entries=_make_amp_table_xml())
        des_asset = _make_des_asset(desshtn="urn:SICD:1.3.0", xml_content=xml)

        block = np.stack([np.array([[10, 20]], dtype=np.uint8), np.array([[0, 128]], dtype=np.uint8)], axis=0)
        image_asset = _make_source(num_bands=2, pixel_value_type="UINT8", block_data=block, num_rows=1, num_columns=2)

        reader = _make_reader(
            image_asset=image_asset,
            des_assets={"des:0": des_asset},
            asset_keys=["image:0", "des:0"],
        )
        provider = load_complex_remap(reader)
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertGreater(output[0, 0, 0], 0.0)

    def test_sidd_des_ignored_sicd_des_found(self):
        """When both SIDD and SICD DES exist, correctly identifies SICD by DESSHTN."""
        sidd_des = _make_des_asset(desid="XML_DATA_CONTENT", desshtn="urn:SIDD:3.0", xml_content="<SIDD/>")
        sicd_des = _make_des_asset(desshtn="urn:SICD:1.3.0", xml_content=_SICD_RE32F_XML)
        image_asset = _make_source(num_bands=2, pixel_value_type="FLOAT32")

        reader = _make_reader(
            image_asset=image_asset,
            des_assets={"des:0": sidd_des, "des:1": sicd_des},
            asset_keys=["image:0", "des:0", "des:1"],
        )
        provider = load_complex_remap(reader)
        self.assertEqual(provider.num_bands, 1)

    def test_sidd_des_only_falls_back_to_subheader(self):
        """SIDD DES only (no SICD) → falls back to NITF subheader fields."""
        sidd_des = _make_des_asset(desid="XML_DATA_CONTENT", desshtn="urn:SIDD:3.0", xml_content="<SIDD/>")
        image_asset = _make_source(
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["I", "Q"]},
        )

        reader = _make_reader(
            image_asset=image_asset,
            des_assets={"des:0": sidd_des},
            asset_keys=["image:0", "des:0"],
        )
        provider = load_complex_remap(reader)
        self.assertEqual(provider.num_bands, 1)


class TestLoadComplexRemapNITFSubheader(TestCase):
    """Tests for load_complex_remap() fallback to NITF subheader fields."""

    def test_isubcat_iq(self):
        """No DES, ISUBCAT=['I','Q'] → band_interpretation=['real','imaginary']."""
        image_asset = _make_source(
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["I", "Q"]},
        )
        reader = _make_reader(image_asset=image_asset, asset_keys=["image:0"])
        provider = load_complex_remap(reader)
        self.assertEqual(provider.num_bands, 1)
        self.assertEqual(provider.pixel_value_type, "FLOAT32")

    def test_isubcat_mp_icat_sar(self):
        """No DES, ISUBCAT=['M','P'], ICAT=SAR → band_interpretation=['magnitude','phase']."""
        image_asset = _make_source(
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["M", "P"], "ICAT": "SAR"},
        )
        reader = _make_reader(image_asset=image_asset, asset_keys=["image:0"])
        provider = load_complex_remap(reader)
        self.assertEqual(provider.num_bands, 1)

    def test_no_des_no_usable_metadata_raises(self):
        """No DES, no usable NITF metadata → raises ValueError."""
        image_asset = _make_source(num_bands=2, pixel_value_type="FLOAT32", metadata={"ICAT": "VIS"})
        reader = _make_reader(image_asset=image_asset, asset_keys=["image:0"])
        with self.assertRaises(ValueError) as ctx:
            load_complex_remap(reader)
        self.assertIn("Cannot determine complex band interpretation", str(ctx.exception))

    def test_no_image_assets_raises(self):
        """Reader with no image assets raises ValueError."""
        reader = MagicMock()
        reader.get_asset_keys = MagicMock(return_value=["des:0"])
        with self.assertRaises(ValueError) as ctx:
            load_complex_remap(reader)
        self.assertIn("No image assets found", str(ctx.exception))

    def test_custom_asset_key(self):
        """Explicit asset_key is used instead of auto-detection."""
        image_asset = _make_source(
            num_bands=2,
            pixel_value_type="FLOAT32",
            metadata={"ISUBCAT": ["I", "Q"]},
        )
        reader = MagicMock()
        reader.get_asset_keys = MagicMock(return_value=["image:0", "image:1"])
        reader.get_asset = MagicMock(return_value=image_asset)

        provider = load_complex_remap(reader, asset_key="image:1")
        reader.get_asset.assert_called_with("image:1")
        self.assertEqual(provider.num_bands, 1)

    def test_native_complex_pixel_type_fallback(self):
        """Native complex pixel type without ISUBCAT: band_interpretation=None passed to factory."""
        block = np.array([[[1 + 2j, 3 + 4j]]], dtype=np.complex64)
        image_asset = _make_source(
            num_bands=1,
            pixel_value_type="COMPLEX64",
            block_data=block,
            num_rows=1,
            num_columns=2,
            metadata={},
        )
        reader = _make_reader(image_asset=image_asset, asset_keys=["image:0"])
        provider = load_complex_remap(reader)
        self.assertEqual(provider.num_bands, 1)
        output = provider.get_block(0, 0)
        self.assertEqual(output.shape, (1, 1, 2))
        self.assertTrue(np.all(np.isfinite(output)))
