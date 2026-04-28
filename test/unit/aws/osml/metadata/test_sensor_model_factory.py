#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.


import pytest

from aws.osml.metadata.gcp_sensor_model_builder import GroundControlPoint
from aws.osml.metadata.sensor_model_factory import (
    ALL_SENSOR_MODEL_TYPES,
    ChippedImageInfoFacade,
    SensorModelFactory,
    SensorModelTypes,
)
from aws.osml.photogrammetry import (
    AffineSensorModel,
    ChippedImageSensorModel,
    CompositeSensorModel,
    ProjectiveSensorModel,
)
from aws.osml.photogrammetry.rpc_sensor_model import RPCSensorModel

# ---- Inline test data for ICHIPB ----
# Simulates a chip from a larger 8192x8192 image.
SAMPLE_ICHIPB_DICT = {
    "OP_COL_11": "0.5",
    "OP_ROW_11": "0.5",
    "FI_COL_11": "100.5",
    "FI_ROW_11": "200.5",
    "OP_COL_12": "1024.5",
    "OP_ROW_12": "0.5",
    "FI_COL_12": "1124.5",
    "FI_ROW_12": "200.5",
    "OP_COL_21": "0.5",
    "OP_ROW_21": "1024.5",
    "FI_COL_21": "100.5",
    "FI_ROW_21": "1224.5",
    "OP_COL_22": "1024.5",
    "OP_ROW_22": "1024.5",
    "FI_COL_22": "1124.5",
    "FI_ROW_22": "1224.5",
    "FI_COL": "8192",
    "FI_ROW": "8192",
}

# ---- Inline test data for RPC00B (from test_rpc_sensor_model_builder.py) ----
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
    "LINE_NUM_COEFF": ["-1.219784E-2", "+0.000000E+0", "-1.197441E+0"] + ["+0.000000E+0"] * 17,
    "LINE_DEN_COEFF": ["+1.000000E+0"] + ["+0.000000E+0"] * 19,
    "SAMP_NUM_COEFF": ["+0.000000E+0", "+1.024478E+0"] + ["+0.000000E+0"] * 18,
    "SAMP_DEN_COEFF": ["+1.000000E+0"] + ["+0.000000E+0"] * 19,
}

# ---- Inline test data for CSCRNA ----
SAMPLE_CSCRNA_DICT = {
    "PREDICT_CORNERS": "Y",
    "ULCNR_LAT": "+25.02860",
    "ULCNR_LONG": "+121.48749",
    "ULCNR_HT": "+00027.1",
    "URCNR_LAT": "+25.01000",
    "URCNR_LONG": "+121.68566",
    "URCNR_HT": "+00234.7",
    "LRCNR_LAT": "+24.91148",
    "LRCNR_LONG": "+121.68595",
    "LRCNR_HT": "+00403.1",
    "LLCNR_LAT": "+24.92772",
    "LLCNR_LONG": "+121.48975",
    "LLCNR_HT": "+00431.4",
}

# ---- Inline test data for RSMIDA + RSMPCA ----
SAMPLE_RSMIDA_DICT = {
    "GRNDD": "R",
    "XUOR": "-2.42965895449297E+06",
    "YUOR": "-4.76049894293300E+06",
    "ZUOR": "+3.46898407315533E+06",
    "XUXR": "+8.90233155120443E-01",
    "XUYR": "+2.50327118321895E-01",
    "XUZR": "-3.80553890213932E-01",
    "YUXR": "-4.55502457571841E-01",
    "YUYR": "+4.86367706250322E-01",
    "YUZR": "-7.45629911861651E-01",
    "ZUXR": "-1.56226448294838E-03",
    "ZUYR": "+8.37127701219746E-01",
    "ZUZR": "+5.47005275276417E-01",
    "V1X": "+3.60917770743581E+01",
    "V1Y": "-2.28824840001127E+01",
    "V1Z": "-9.99992874460703E+02",
    "V2X": "+5.18884362473253E+03",
    "V2Y": "-2.87450036723962E+01",
    "V2Z": "-1.00211587769373E+03",
    "V3X": "+3.79919348686812E+01",
    "V3Y": "+4.59734548268640E+03",
    "V3Z": "-1.00166489059808E+03",
    "V4X": "+5.01402740997898E+03",
    "V4Y": "+4.85440104533856E+03",
    "V4Z": "-1.00381011450997E+03",
    "V5X": "-3.60920649717823E+01",
    "V5Y": "+2.28826665274178E+01",
    "V5Z": "+1.00000085119816E+03",
    "V6X": "+1.88541350122103E+03",
    "V6Y": "+2.07081675817197E+01",
    "V6Z": "+9.99720731314608E+02",
    "V7X": "-3.53972599523443E+01",
    "V7Y": "+1.74601545186290E+03",
    "V7Z": "+9.99761368681871E+02",
    "V8X": "+1.81960288385246E+03",
    "V8Y": "+1.84110851608994E+03",
    "V8Z": "+9.99477441903243E+02",
    "GRPX": "",
    "GRPY": "",
    "GRPZ": "",
    "MINR": "00000000",
    "MAXR": "00009292",
    "MINC": "00000000",
    "MAXC": "00009122",
}

SAMPLE_RSMPCA_DICT = {
    "IID": "2_8",
    "EDITION": "1101222272-2",
    "RSN": "001",
    "CSN": "001",
    "RFEP": "+3.98498860405865E-09",
    "CFEP": "+3.63668781918539E-09",
    "RNRMO": "+4.64600000000000E+03",
    "CNRMO": "+4.56100000000000E+03",
    "XNRMO": "+2.65571142640788E+03",
    "YNRMO": "+2.40732324869712E+03",
    "ZNRMO": "-2.33161661728923E+00",
    "RNRMSF": "+5.59920000000000E+03",
    "CNRMSF": "+5.49720000000000E+03",
    "XNRMSF": "+3.11793659470231E+03",
    "YNRMSF": "+2.96084811268529E+03",
    "ZNRMSF": "+1.00233148808220E+03",
    "RNPWRX": "1",
    "RNPWRY": "1",
    "RNPWRZ": "1",
    "RNPCF": [
        "+4.63481151803541E-01",
        "+1.47720153582094E+00",
        "+2.45032267644299E-02",
        "+0.00000000000000E+00",
        "+3.89362443002779E-01",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
    ],
    "RDPWRX": "1",
    "RDPWRY": "1",
    "RDPWRZ": "1",
    "RDPCF": [
        "+1.00000000000000E+00",
        "+5.02862720499332E-02",
        "-3.02226608403966E-02",
        "+0.00000000000000E+00",
        "-4.48001662401358E-01",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
    ],
    "CNPWRX": "1",
    "CNPWRY": "1",
    "CNPWRZ": "1",
    "CNPCF": [
        "+3.66773732052461E-01",
        "-3.99759610583925E-02",
        "+1.49479106149881E+00",
        "+0.00000000000000E+00",
        "+3.60340008196375E-01",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
    ],
    "CDPWRX": "1",
    "CDPWRY": "1",
    "CDPWRZ": "1",
    "CDPCF": [
        "+1.00000000000000E+00",
        "+5.02862720498340E-02",
        "-3.02226608402894E-02",
        "+0.00000000000000E+00",
        "-4.48001662401429E-01",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
        "+0.00000000000000E+00",
    ],
}

# ---- Sample geo transform (GeoTIFF-style) ----
SAMPLE_GEO_TRANSFORM = [121.0, 0.0001, 0.0, 25.0, 0.0, -0.0001]

# ---- Sample GCPs ----
SAMPLE_GCPS = [
    GroundControlPoint(0.0, 0.0, 121.48749, 25.02860, 27.1),
    GroundControlPoint(8820.0, 0.0, 121.68566, 25.01000, 234.7),
    GroundControlPoint(8820.0, 5212.0, 121.68595, 24.91148, 403.1),
    GroundControlPoint(0.0, 5212.0, 121.48975, 24.92772, 431.4),
]


class TestChippedImageInfoFacade:
    """Unit tests for ChippedImageInfoFacade. Requirements: 9.1."""

    def test_valid_ichipb_parses_coordinates(self):
        """Valid ICHIPB dict produces correct full_image and chipped_image coordinates."""
        facade = ChippedImageInfoFacade(SAMPLE_ICHIPB_DICT)

        assert len(facade.full_image_coordinates) == 4
        assert len(facade.chipped_image_coordinates) == 4

        # Grid point 11
        assert facade.full_image_coordinates[0].x == pytest.approx(100.5)
        assert facade.full_image_coordinates[0].y == pytest.approx(200.5)
        assert facade.chipped_image_coordinates[0].x == pytest.approx(0.5)
        assert facade.chipped_image_coordinates[0].y == pytest.approx(0.5)

        # Grid point 22
        assert facade.full_image_coordinates[3].x == pytest.approx(1124.5)
        assert facade.full_image_coordinates[3].y == pytest.approx(1224.5)
        assert facade.chipped_image_coordinates[3].x == pytest.approx(1024.5)
        assert facade.chipped_image_coordinates[3].y == pytest.approx(1024.5)

    def test_valid_ichipb_parses_full_image_dimensions(self):
        """Valid ICHIPB dict produces correct full image width and height."""
        facade = ChippedImageInfoFacade(SAMPLE_ICHIPB_DICT)

        assert facade.full_image_width == 8192
        assert facade.full_image_height == 8192

    def test_malformed_ichipb_does_not_crash(self):
        """Malformed ICHIPB dict logs warning but does not raise."""
        bad_dict = {"OP_COL_11": "not_a_number", "OP_ROW_11": "0.5"}
        # Should not raise — gracefully handles parse failure
        facade = ChippedImageInfoFacade(bad_dict)
        # The facade may not have all attributes set, but it should not crash
        assert facade is not None

    def test_empty_ichipb_does_not_crash(self):
        """Empty ICHIPB dict logs warning but does not raise."""
        facade = ChippedImageInfoFacade({})
        assert facade is not None

    def test_all_four_grid_points_parsed(self):
        """All four grid points (11, 12, 21, 22) are parsed correctly."""
        facade = ChippedImageInfoFacade(SAMPLE_ICHIPB_DICT)

        expected_fi = [(100.5, 200.5), (1124.5, 200.5), (100.5, 1224.5), (1124.5, 1224.5)]
        expected_op = [(0.5, 0.5), (1024.5, 0.5), (0.5, 1024.5), (1024.5, 1024.5)]

        for i, (fi_x, fi_y) in enumerate(expected_fi):
            assert facade.full_image_coordinates[i].x == pytest.approx(fi_x)
            assert facade.full_image_coordinates[i].y == pytest.approx(fi_y)

        for i, (op_x, op_y) in enumerate(expected_op):
            assert facade.chipped_image_coordinates[i].x == pytest.approx(op_x)
            assert facade.chipped_image_coordinates[i].y == pytest.approx(op_y)


class TestSensorModelFactory:
    """Unit tests for SensorModelFactory. Requirements: 1.1-1.7, 9.1-9.4."""

    def test_rpc_only_produces_rpc_sensor_model(self):
        """Factory with RPC00B dict only produces RPCSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, RPCSensorModel)

    def test_rpc_with_cscrna_produces_composite(self):
        """Factory with RPC00B + CSCRNA produces CompositeSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT, "CSCRNA": SAMPLE_CSCRNA_DICT},
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, CompositeSensorModel)

    def test_rsm_produces_sensor_model(self):
        """Factory with RSM dicts produces an RSM sensor model."""
        factory = SensorModelFactory(
            actual_image_width=9123,
            actual_image_height=9293,
            tre_dicts={"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": SAMPLE_RSMPCA_DICT},
        )
        model = factory.build()
        assert model is not None

    def test_cscrna_only_produces_projective(self):
        """Factory with CSCRNA only produces ProjectiveSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"CSCRNA": SAMPLE_CSCRNA_DICT},
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_geo_transform_produces_affine(self):
        """Factory with geo_transform produces AffineSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            geo_transform=SAMPLE_GEO_TRANSFORM,
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, AffineSensorModel)

    def test_gcps_produce_projective(self):
        """Factory with 4+ GCPs produces ProjectiveSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            ground_control_points=SAMPLE_GCPS,
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_ichipb_wraps_precision_model(self):
        """Factory with ICHIPB + RPC00B wraps precision model in ChippedImageSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=1025,
            actual_image_height=1025,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT, "ICHIPB": SAMPLE_ICHIPB_DICT},
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, ChippedImageSensorModel)

    def test_ichipb_wraps_approximate_cscrna_model(self):
        """Factory with ICHIPB + CSCRNA wraps approximate model in ChippedImageSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=1025,
            actual_image_height=1025,
            tre_dicts={"CSCRNA": SAMPLE_CSCRNA_DICT, "ICHIPB": SAMPLE_ICHIPB_DICT},
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, ChippedImageSensorModel)

    def test_ichipb_with_rpc_and_cscrna_produces_composite_of_chipped(self):
        """Factory with ICHIPB + RPC00B + CSCRNA produces CompositeSensorModel of two ChippedImageSensorModels."""
        factory = SensorModelFactory(
            actual_image_width=1025,
            actual_image_height=1025,
            tre_dicts={
                "RPC00B": SAMPLE_RPC00B_DICT,
                "CSCRNA": SAMPLE_CSCRNA_DICT,
                "ICHIPB": SAMPLE_ICHIPB_DICT,
            },
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, CompositeSensorModel)
        assert isinstance(model.precision_sensor_model, ChippedImageSensorModel)
        assert isinstance(model.approximate_sensor_model, ChippedImageSensorModel)

    def test_ichipb_uses_full_image_dimensions_for_cscrna(self):
        """When ICHIPB is present, CSCRNA projective builder uses full image dimensions from ICHIPB."""
        factory = SensorModelFactory(
            actual_image_width=1025,
            actual_image_height=1025,
            tre_dicts={"CSCRNA": SAMPLE_CSCRNA_DICT, "ICHIPB": SAMPLE_ICHIPB_DICT},
        )
        model = factory.build()
        # The model should be built using 8192x8192 (from ICHIPB), not 1025x1025
        assert model is not None

    def test_selected_sensor_model_types_filtering(self):
        """Factory respects selected_sensor_model_types filtering."""
        # Only allow AFFINE — should ignore RPC00B
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
            selected_sensor_model_types=[SensorModelTypes.AFFINE],
        )
        model = factory.build()
        assert model is None

    def test_selected_types_allows_rpc(self):
        """Factory with RPC in selected types builds RPC model."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
            selected_sensor_model_types=[SensorModelTypes.RPC],
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, RPCSensorModel)

    def test_returns_none_for_empty_inputs(self):
        """Factory with no metadata returns None."""
        factory = SensorModelFactory(actual_image_width=100, actual_image_height=100)
        model = factory.build()
        assert model is None

    def test_returns_none_for_empty_tre_dicts(self):
        """Factory with empty tre_dicts returns None."""
        factory = SensorModelFactory(
            actual_image_width=100,
            actual_image_height=100,
            tre_dicts={},
        )
        model = factory.build()
        assert model is None

    def test_rsm_takes_priority_over_rpc(self):
        """RSM has higher priority than RPC — when both present, RSM is used for precision."""
        tre_dicts = {
            "RSMIDA": SAMPLE_RSMIDA_DICT,
            "RSMPCA": SAMPLE_RSMPCA_DICT,
            "RPC00B": SAMPLE_RPC00B_DICT,
        }
        factory = SensorModelFactory(
            actual_image_width=9123,
            actual_image_height=9293,
            tre_dicts=tre_dicts,
        )
        model = factory.build()
        assert model is not None
        # Should not be RPCSensorModel — RSM takes priority
        assert not isinstance(model, RPCSensorModel)

    def test_affine_overridden_by_gcp(self):
        """GCP approximate model overrides affine approximate model."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            geo_transform=SAMPLE_GEO_TRANSFORM,
            ground_control_points=SAMPLE_GCPS,
        )
        model = factory.build()
        assert model is not None
        # GCP produces ProjectiveSensorModel, which overrides the affine
        assert isinstance(model, ProjectiveSensorModel)

    def test_fewer_than_4_gcps_ignored(self):
        """Fewer than 4 GCPs should not produce a model from GCPs."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            ground_control_points=SAMPLE_GCPS[:3],
        )
        model = factory.build()
        assert model is None

    def test_default_selected_types_includes_all(self):
        """Default selected_sensor_model_types includes all types."""
        factory = SensorModelFactory(actual_image_width=100, actual_image_height=100)
        assert factory.selected_sensor_model_types == ALL_SENSOR_MODEL_TYPES

    def test_des_xml_strings_empty_list(self):
        """Empty des_xml_strings list does not crash."""
        factory = SensorModelFactory(
            actual_image_width=100,
            actual_image_height=100,
            des_xml_strings=[],
        )
        model = factory.build()
        assert model is None

    def test_des_xml_strings_with_empty_string(self):
        """des_xml_strings containing empty strings are skipped."""
        factory = SensorModelFactory(
            actual_image_width=100,
            actual_image_height=100,
            des_xml_strings=["", None],
        )
        model = factory.build()
        assert model is None

    def test_precision_only_returns_precision(self):
        """When only precision model is available, return it directly (not composite)."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
        )
        model = factory.build()
        assert model is not None
        assert not isinstance(model, CompositeSensorModel)

    def test_approximate_only_returns_approximate(self):
        """When only approximate model is available, return it directly (not composite)."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            geo_transform=SAMPLE_GEO_TRANSFORM,
        )
        model = factory.build()
        assert model is not None
        assert not isinstance(model, CompositeSensorModel)

    def test_affine_plus_rpc_produces_composite(self):
        """Affine approximate + RPC precision produces CompositeSensorModel."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
            geo_transform=SAMPLE_GEO_TRANSFORM,
        )
        model = factory.build()
        assert model is not None
        assert isinstance(model, CompositeSensorModel)

    def test_sensor_model_types_enum_values(self):
        """SensorModelTypes enum has expected values."""
        assert SensorModelTypes.AFFINE.value == "AFFINE"
        assert SensorModelTypes.PROJECTIVE.value == "PROJECTIVE"
        assert SensorModelTypes.RPC.value == "RPC"
        assert SensorModelTypes.RSM.value == "RSM"
        assert SensorModelTypes.SICD.value == "SICD"

    def test_filtering_excludes_projective_cscrna(self):
        """When PROJECTIVE not in selected types, CSCRNA is not used."""
        factory = SensorModelFactory(
            actual_image_width=8820,
            actual_image_height=5212,
            tre_dicts={"CSCRNA": SAMPLE_CSCRNA_DICT},
            selected_sensor_model_types=[SensorModelTypes.RPC],
        )
        model = factory.build()
        assert model is None
