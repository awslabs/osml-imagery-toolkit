#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from pathlib import Path
from unittest.mock import MagicMock

from aws.osml.metadata.dataset_utils import (
    _derive_proj_wkt,
    _extract_igeolo_gcps,
    derive_geotiff_georeference,
    load_sensor_model,
)
from aws.osml.photogrammetry import ProjectiveSensorModel, SICDSensorModel
from aws.osml.photogrammetry.rpc_sensor_model import RPCSensorModel

# ---- Inline RPC00B TRE dict (same values as test_rpc_sensor_model_builder.py) ----
SAMPLE_RPC00B_DICT = {
    "SUCCESS": "1",
    "ERR_BIAS": "0005.18",
    "ERR_RAND": "0000.98",
    "LINE_OFF": "002606",
    "SAMP_OFF": "04409",
    "LAT_OFF": "+24.9697",
    "LONG_OFF": "+121.5875",
    "HEIGHT_OFF": "+0377",
    "LINE_SCALE": "002606",
    "SAMP_SCALE": "04410",
    "LAT_SCALE": "+00.0593",
    "LONG_SCALE": "+000.1008",
    "HEIGHT_SCALE": "+0500",
    "LINE_NUM_COEFF": [
        "-1.219784E-2",
        "-1.779120E-1",
        "-1.197441E+0",
        "-1.962294E-2",
        "-2.400754E-3",
        "-1.266875E-4",
        "-2.864113E-4",
        "-9.364538E-4",
        "+9.023196E-3",
        "-6.372315E-6",
        "-3.353148E-6",
        "-9.192860E-6",
        "+1.114250E-6",
        "-9.605230E-6",
        "-4.486697E-5",
        "-2.875052E-4",
        "-6.413221E-5",
        "-1.735976E-6",
        "-2.999621E-8",
        "-1.049317E-6",
    ],
    "LINE_DEN_COEFF": [
        "+1.000000E+0",
        "-5.210514E-3",
        "-4.132499E-3",
        "-6.048495E-4",
        "-5.553299E-5",
        "-3.766012E-6",
        "-8.264991E-6",
        "+2.892116E-5",
        "-1.516595E-4",
        "+5.320691E-5",
        "-6.859532E-6",
        "-1.898968E-6",
        "-2.098215E-4",
        "-5.094778E-7",
        "-3.142094E-5",
        "-4.513800E-4",
        "-2.112928E-7",
        "-5.952780E-7",
        "-2.304619E-5",
        "-5.413752E-8",
    ],
    "SAMP_NUM_COEFF": [
        "-6.780123E-3",
        "+1.024478E+0",
        "+2.011171E-5",
        "-2.239644E-2",
        "-1.306157E-3",
        "+1.651806E-4",
        "+6.467390E-5",
        "+5.991485E-3",
        "-1.995634E-4",
        "-3.633161E-6",
        "-1.540867E-6",
        "+1.551750E-5",
        "-6.259453E-5",
        "-9.389538E-6",
        "-4.054396E-5",
        "-6.342832E-5",
        "+2.809450E-8",
        "+2.041979E-6",
        "-3.292629E-6",
        "+2.079533E-7",
    ],
    "SAMP_DEN_COEFF": [
        "+1.000000E+0",
        "+8.111176E-4",
        "+1.340323E-3",
        "-3.211588E-4",
        "+2.661355E-5",
        "-1.689184E-6",
        "+2.372103E-6",
        "+2.060620E-6",
        "+5.500269E-5",
        "-9.237594E-6",
        "+1.666669E-7",
        "+7.025150E-8",
        "+2.152491E-6",
        "+4.632533E-8",
        "+1.002425E-6",
        "+1.169637E-6",
        "-1.738953E-8",
        "+0.000000E+0",
        "+2.386543E-7",
        "+0.000000E+0",
    ],
}

# Path to a real SICD XML file for DES XML testing
SICD_PFA_XML_PATH = Path("./test/data/sicd/example.sicd121.pfa.xml")


def _read_xml(path: Path) -> str:
    """Read XML file content as a string."""
    return path.read_text(encoding="utf-8")


def _make_mock_reader(
    image_width=8820,
    image_height=5212,
    metadata_dict=None,
    des_xml_strings=None,
):
    """
    Create a mock DatasetReader matching the osml-imagery-io API.

    Uses get_asset_keys() / get_asset() pattern with "image:" and "des:" prefixed keys.
    """
    reader = MagicMock()

    # Build asset key list
    asset_keys = ["image:0"]
    if des_xml_strings:
        for i in range(len(des_xml_strings)):
            asset_keys.append(f"des:{i}")  # noqa: E231
    reader.get_asset_keys.return_value = asset_keys

    # Mock image asset
    mock_image_asset = MagicMock()
    mock_image_asset.num_columns = image_width
    mock_image_asset.num_rows = image_height
    mock_image_asset.num_bands = 1

    # Mock metadata on the image asset (supports dict() conversion via Mapping protocol)
    mock_image_asset.metadata = metadata_dict if metadata_dict is not None else {}

    # Mock DES assets
    mock_des_assets = {}
    if des_xml_strings:
        for i, xml_str in enumerate(des_xml_strings):
            des_asset = MagicMock()
            des_asset.metadata = {"DESID": "XML_DATA_CONTENT"}
            raw = MagicMock()
            if isinstance(xml_str, Exception):
                raw.read.side_effect = xml_str
            else:
                raw.read.return_value = xml_str.encode("utf-8")
            des_asset.raw_asset = raw
            mock_des_assets[f"des:{i}"] = des_asset

    def get_asset_side_effect(key):
        if key == "image:0":
            return mock_image_asset
        if key in mock_des_assets:
            return mock_des_assets[key]
        raise KeyError(f"No asset with key: {key}")

    reader.get_asset.side_effect = get_asset_side_effect

    return reader


def _make_mock_reader_no_images():
    """Create a mock DatasetReader with no image assets."""
    reader = MagicMock()
    reader.get_asset_keys.return_value = []
    return reader


class TestLoadSensorModel:
    """Unit tests for load_sensor_model convenience function."""

    def test_rpc00b_metadata_produces_rpc_sensor_model(self):
        """DatasetReader with RPC00B TRE metadata produces an RPCSensorModel."""
        metadata_dict = {"RPC00B": SAMPLE_RPC00B_DICT}
        reader = _make_mock_reader(metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None
        assert isinstance(model, RPCSensorModel)

    def test_no_metadata_returns_none(self):
        """DatasetReader with no metadata returns None."""
        reader = _make_mock_reader(metadata_dict={})

        model = load_sensor_model(reader)

        assert model is None

    def test_empty_metadata_dict_returns_none(self):
        """DatasetReader with empty metadata dict returns None."""
        reader = _make_mock_reader(metadata_dict=None)

        model = load_sensor_model(reader)

        assert model is None

    def test_sicd_des_xml_produces_sicd_sensor_model(self):
        """DatasetReader with SICD DES XML produces a SICDSensorModel."""
        sicd_xml = _read_xml(SICD_PFA_XML_PATH)
        reader = _make_mock_reader(des_xml_strings=[sicd_xml])

        model = load_sensor_model(reader)

        assert model is not None
        assert isinstance(model, SICDSensorModel)

    def test_extracts_image_dimensions(self):
        """load_sensor_model extracts image dimensions from the image asset."""
        metadata_dict = {"RPC00B": SAMPLE_RPC00B_DICT}
        reader = _make_mock_reader(image_width=1024, image_height=768, metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None
        reader.get_asset_keys.assert_called()
        reader.get_asset.assert_called()

    def test_extracts_des_xml_strings(self):
        """load_sensor_model reads all DES segments via des: asset keys."""
        sicd_xml = _read_xml(SICD_PFA_XML_PATH)
        reader = _make_mock_reader(des_xml_strings=[sicd_xml])

        model = load_sensor_model(reader)

        assert model is not None
        reader.get_asset_keys.assert_called()

    def test_no_image_assets_returns_none(self):
        """DatasetReader with no image assets returns None."""
        reader = _make_mock_reader_no_images()

        model = load_sensor_model(reader)

        assert model is None

    def test_geo_transform_derived_from_tiff_tags(self):
        """ModelPixelScale (33550) + ModelTiepoint (33922) are used to derive geo_transform."""
        metadata_dict = {
            "33550": [0.0001, 0.0001, 0.0],
            "33922": [0.0, 0.0, 0.0, 121.0, 25.0, 0.0],
        }
        reader = _make_mock_reader(metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None

    def test_multiple_tiepoints_as_gcps(self):
        """Multiple ModelTiepoints without ModelPixelScale are treated as GCPs."""
        # 4 tiepoints: each is [pixel_x, pixel_y, pixel_z, geo_x, geo_y, geo_z]
        tiepoints = [
            0.0,
            0.0,
            0.0,
            121.48749,
            25.02860,
            0.0,
            8820.0,
            0.0,
            0.0,
            121.68566,
            25.01000,
            0.0,
            8820.0,
            5212.0,
            0.0,
            121.68595,
            24.91148,
            0.0,
            0.0,
            5212.0,
            0.0,
            121.48975,
            24.92772,
            0.0,
        ]
        metadata_dict = {"33922": tiepoints}
        reader = _make_mock_reader(metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_model_transformation_produces_affine(self):
        """ModelTransformation (tag 34264) for rotated images produces an affine model."""
        # 4x4 row-major: [a, b, 0, tx, d, e, 0, ty, 0, 0, 0, 0, 0, 0, 0, 1]
        transform = [
            0.0001,
            0.00001,
            0.0,
            121.0,
            0.00001,
            -0.0001,
            0.0,
            25.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        metadata_dict = {"34264": transform}
        reader = _make_mock_reader(metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None

    def test_proj_wkt_derived_from_geokeys(self):
        """GeoKeyDirectory (tag 34735) with EPSG code produces proj_wkt for affine model."""
        # GeoKeyDirectory: [version=1, rev=1, minor=1, num_keys=3,
        #   GTModelTypeGeoKey(1024)=2(Geographic), GTRasterTypeGeoKey(1025)=1,
        #   GeographicTypeGeoKey(2048)=4326]
        geokey_dir = [1, 1, 1, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]
        metadata_dict = {
            "33550": [0.0001, 0.0001, 0.0],
            "33922": [0.0, 0.0, 0.0, 121.0, 25.0, 0.0],
            "34735": geokey_dir,
        }
        reader = _make_mock_reader(metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None

    def test_reader_exception_returns_none(self):
        """If the reader raises an exception, load_sensor_model returns None."""
        reader = MagicMock()
        reader.get_asset_keys.side_effect = RuntimeError("Reader failed")

        model = load_sensor_model(reader)

        assert model is None

    def test_des_read_failure_skips_segment(self):
        """If a DES asset read fails, it is skipped and other segments still processed."""
        sicd_xml = _read_xml(SICD_PFA_XML_PATH)

        reader = MagicMock()
        reader.get_asset_keys.return_value = ["image:0", "des:0", "des:1"]

        mock_image_asset = MagicMock()
        mock_image_asset.num_columns = 1024
        mock_image_asset.num_rows = 768
        mock_image_asset.num_bands = 1
        mock_image_asset.metadata = {}

        # First DES fails, second succeeds
        mock_des_bad = MagicMock()
        mock_des_bad.metadata = {"DESID": "XML_DATA_CONTENT"}
        mock_des_bad.raw_asset.read.side_effect = RuntimeError("Read failed")
        mock_des_good = MagicMock()
        mock_des_good.metadata = {"DESID": "XML_DATA_CONTENT"}
        mock_des_good.raw_asset.read.return_value = sicd_xml.encode("utf-8")

        def get_asset_side_effect(key):
            if key == "image:0":
                return mock_image_asset
            if key == "des:0":
                return mock_des_bad
            if key == "des:1":
                return mock_des_good
            raise KeyError(key)

        reader.get_asset.side_effect = get_asset_side_effect

        model = load_sensor_model(reader)

        assert model is not None
        assert isinstance(model, SICDSensorModel)

    def test_null_metadata_on_asset_returns_none(self):
        """If image asset metadata is None, load_sensor_model handles it gracefully."""
        reader = MagicMock()
        reader.get_asset_keys.return_value = ["image:0"]

        mock_image_asset = MagicMock()
        mock_image_asset.num_columns = 100
        mock_image_asset.num_rows = 100
        mock_image_asset.metadata = None

        reader.get_asset.return_value = mock_image_asset

        model = load_sensor_model(reader)

        assert model is None

    def test_igeolo_fallback_produces_projective_model(self):
        """IGEOLO corner coordinates are used as GCPs when no other model sources exist."""
        # Geographic IGEOLO: 4 corners at known lat/lon (ddmmssXdddmmssY format)
        # UL=25N 121E, UR=25N 122E, LR=24N 122E, LL=24N 121E
        igeolo = "250000N1210000E250000N1220000E240000N1220000E240000N1210000E"
        metadata_dict = {"IGEOLO": igeolo, "ICORDS": "G"}
        reader = _make_mock_reader(image_width=1000, image_height=1000, metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_igeolo_with_rpc_produces_composite(self):
        """IGEOLO + RPC produces a composite with RPC as precision model."""
        from aws.osml.photogrammetry import CompositeSensorModel

        igeolo = "250000N1210000E250000N1220000E240000N1220000E240000N1210000E"
        metadata_dict = {"RPC00B": SAMPLE_RPC00B_DICT, "IGEOLO": igeolo, "ICORDS": "G"}
        reader = _make_mock_reader(metadata_dict=metadata_dict)

        model = load_sensor_model(reader)

        assert model is not None
        assert isinstance(model, CompositeSensorModel)
        assert isinstance(model.precision_sensor_model, RPCSensorModel)

    def test_asset_key_parameter(self):
        """Explicit asset_key parameter selects the specified image asset."""
        reader = MagicMock()
        reader.get_asset_keys.return_value = ["image:0", "image:1"]

        mock_image_asset = MagicMock()
        mock_image_asset.num_columns = 512
        mock_image_asset.num_rows = 512
        mock_image_asset.metadata = {"RPC00B": SAMPLE_RPC00B_DICT}

        reader.get_asset.return_value = mock_image_asset

        model = load_sensor_model(reader, asset_key="image:1")

        assert model is not None
        reader.get_asset.assert_called_with("image:1")


class TestExtractIgeoloGcps:
    """Unit tests for _extract_igeolo_gcps helper."""

    def test_geographic_igeolo_parsed(self):
        """Geographic (ICORDS=G) IGEOLO is parsed into 4 GCPs."""
        igeolo = "250000N1210000E250000N1220000E240000N1220000E240000N1210000E"
        meta = {"IGEOLO": igeolo, "ICORDS": "G"}

        gcps = _extract_igeolo_gcps(meta, 1000, 1000)

        assert gcps is not None
        assert len(gcps) == 4
        assert gcps[0].image_x == 0.0
        assert gcps[0].image_y == 0.0
        assert gcps[0].world_latitude == 25.0
        assert gcps[0].world_longitude == 121.0
        assert gcps[2].image_x == 1000.0
        assert gcps[2].image_y == 1000.0

    def test_decimal_igeolo_parsed(self):
        """Decimal degrees (ICORDS=D) IGEOLO is parsed into 4 GCPs."""
        igeolo = "+25.000+121.000+25.000+122.000+24.000+122.000+24.000+121.000"
        meta = {"IGEOLO": igeolo, "ICORDS": "D"}

        gcps = _extract_igeolo_gcps(meta, 500, 500)

        assert gcps is not None
        assert len(gcps) == 4
        assert gcps[0].world_latitude == 25.0
        assert gcps[1].world_longitude == 122.0

    def test_utm_igeolo_parsed(self):
        """UTM (ICORDS=N) IGEOLO with valid coordinates is parsed into 4 GCPs."""
        # Zone 12N, 4 corners around Tucson AZ area
        # UL: zone=12, easting=616743, northing=3494066
        # UR: zone=12, easting=626233, northing=3494072
        # LR: zone=12, easting=626303, northing=3492963
        # LL: zone=12, easting=616813, northing=3492958
        igeolo = "125616743494066125626233494072125626303492963125616813492958"
        meta = {"IGEOLO": igeolo, "ICORDS": "N"}

        gcps = _extract_igeolo_gcps(meta, 1024, 1024)

        assert gcps is not None
        assert len(gcps) == 4
        # Should be in the Tucson, AZ area (~31.57°N, ~-110.35°W)
        assert 31.0 < gcps[0].world_latitude < 32.0
        assert -111.0 < gcps[0].world_longitude < -110.0

    def test_mgrs_igeolo_returns_none(self):
        """MGRS (ICORDS=U) IGEOLO returns None (not yet supported)."""
        igeolo = "12RWV616749406612RWV626239407212RWV626309296312RWV6168192958"
        meta = {"IGEOLO": igeolo, "ICORDS": "U"}

        gcps = _extract_igeolo_gcps(meta, 100, 100)

        assert gcps is None

    def test_missing_igeolo_returns_none(self):
        """Missing IGEOLO key returns None."""
        meta = {"ICORDS": "G"}

        gcps = _extract_igeolo_gcps(meta, 100, 100)

        assert gcps is None

    def test_missing_icords_returns_none(self):
        """Missing ICORDS key returns None."""
        igeolo = "250000N1210000E" * 4
        meta = {"IGEOLO": igeolo}

        gcps = _extract_igeolo_gcps(meta, 100, 100)

        assert gcps is None


class TestDeriveGeotiffGeoreference:
    """Unit tests for derive_geotiff_georeference helper."""

    def test_scale_and_single_tiepoint(self):
        """ModelPixelScale + single tiepoint → affine geo_transform."""
        meta = {
            "33550": [0.001, 0.001, 0.0],
            "33922": [0.0, 0.0, 0.0, -77.0, 39.0, 0.0],
        }
        geo_transform, gcps = derive_geotiff_georeference(meta)

        assert geo_transform is not None
        assert len(geo_transform) == 6
        assert geo_transform[0] == -77.0  # x-origin
        assert geo_transform[1] == 0.001  # pixel width
        assert geo_transform[5] == -0.001  # pixel height (negated)
        assert gcps is None

    def test_model_transformation(self):
        """ModelTransformation (34264) → affine geo_transform."""
        # Row-major 4x4: [a, b, 0, tx, d, e, 0, ty, ...]
        transform = [
            0.001,
            0.0,
            0.0,
            -77.0,
            0.0,
            -0.001,
            0.0,
            39.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        meta = {"34264": transform}
        geo_transform, gcps = derive_geotiff_georeference(meta)

        assert geo_transform is not None
        # geo_transform = [tx, a, b, ty, d, e]
        assert geo_transform[0] == -77.0
        assert geo_transform[1] == 0.001
        assert geo_transform[3] == 39.0
        assert geo_transform[5] == -0.001
        assert gcps is None

    def test_multiple_tiepoints_without_scale(self):
        """Multiple tiepoints without ModelPixelScale → GCPs."""
        tiepoints = [
            0.0,
            0.0,
            0.0,
            -77.0,
            39.0,
            0.0,
            100.0,
            0.0,
            0.0,
            -76.9,
            39.0,
            0.0,
            100.0,
            100.0,
            0.0,
            -76.9,
            38.9,
            0.0,
            0.0,
            100.0,
            0.0,
            -77.0,
            38.9,
            0.0,
        ]
        meta = {"33922": tiepoints}
        geo_transform, gcps = derive_geotiff_georeference(meta)

        assert geo_transform is None
        assert gcps is not None
        assert len(gcps) == 4
        assert gcps[0].image_x == 0.0
        assert gcps[0].world_longitude == -77.0
        assert gcps[0].world_latitude == 39.0

    def test_empty_metadata(self):
        """Empty metadata returns (None, None)."""
        geo_transform, gcps = derive_geotiff_georeference({})

        assert geo_transform is None
        assert gcps is None


class TestDeriveProjWkt:
    """Unit tests for _derive_proj_wkt helper."""

    def test_geographic_crs(self):
        """GeoKeyDirectory with GeographicTypeGeoKey=4326 produces WGS84 WKT."""
        geokey_dir = [1, 1, 1, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326]
        meta = {"34735": geokey_dir}

        wkt = _derive_proj_wkt(meta)

        assert wkt is not None
        assert "4326" in wkt or "WGS 84" in wkt

    def test_projected_crs(self):
        """GeoKeyDirectory with ProjectedCSTypeGeoKey=32618 produces UTM WKT."""
        geokey_dir = [1, 1, 1, 2, 1024, 0, 1, 1, 3072, 0, 1, 32618]
        meta = {"34735": geokey_dir}

        wkt = _derive_proj_wkt(meta)

        assert wkt is not None
        assert "UTM" in wkt or "32618" in wkt

    def test_no_geokeys_returns_none(self):
        """Missing GeoKeyDirectory returns None."""
        wkt = _derive_proj_wkt({})

        assert wkt is None

    def test_user_defined_crs_returns_none(self):
        """GeoKey value 32767 (user-defined) returns None."""
        geokey_dir = [1, 1, 1, 1, 2048, 0, 1, 32767]
        meta = {"34735": geokey_dir}

        wkt = _derive_proj_wkt(meta)

        assert wkt is None
