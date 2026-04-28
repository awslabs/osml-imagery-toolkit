#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from math import radians

import pytest

from aws.osml.metadata.rsm_sensor_model_builder import RSMSensorModelBuilder
from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
from aws.osml.photogrammetry.replacement_sensor_model import RSMPolynomialSensorModel

# Real RSMIDA TRE field values extracted from test/data/i_6130a_truncated_tres.xml.
# Hardcoded as an inline dict so this test does not depend on GDAL XML parsing.
SAMPLE_RSMIDA_DICT = {
    "IID": "2_8",
    "EDITION": "1101222272-2",
    "ISID": "",
    "SID": "",
    "STID": "FRAME",
    "YEAR": "1970",
    "MONTH": "01",
    "DAY": "01",
    "HOUR": "00",
    "MINUTE": "00",
    "SECOND": "00.000000",
    "NRG": "",
    "NCG": "",
    "TRG": "",
    "TCG": "",
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
    "FULLR": "",
    "FULLC": "",
    "MINR": "00000000",
    "MAXR": "00009292",
    "MINC": "00000000",
    "MAXC": "00009122",
    "IE0": "",
    "IER": "",
    "IEC": "",
    "IERR": "",
    "IERC": "",
    "IECC": "",
    "IA0": "",
    "IAR": "",
    "IAC": "",
    "IARR": "",
    "IARC": "",
    "IACC": "",
    "SPX": "",
    "SVX": "",
    "SAX": "",
    "SPY": "",
    "SVY": "",
    "SAY": "",
    "SPZ": "",
    "SVZ": "",
    "SAZ": "",
}

# Real RSMPCA TRE field values extracted from test/data/i_6130a_truncated_tres.xml.
# Polynomial powers are 1,1,1 so each polynomial has (1+1)*(1+1)*(1+1) = 8 coefficients.
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
    # Row numerator polynomial (RN): powers 1,1,1 → 8 coefficients
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
    # Row denominator polynomial (RD): powers 1,1,1 → 8 coefficients
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
    # Column numerator polynomial (CN): powers 1,1,1 → 8 coefficients
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
    # Column denominator polynomial (CD): powers 1,1,1 → 8 coefficients
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


class TestRSMSensorModelBuilder:
    """Unit tests for RSMSensorModelBuilder. Requirements: 3.1, 3.2, 3.4, 3.5, 3.6."""

    def test_single_rsmpca_produces_polynomial_sensor_model(self):
        """A single RSMPCA dict should produce an RSMPolynomialSensorModel. Req 3.2."""
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": SAMPLE_RSMPCA_DICT}
        builder = RSMSensorModelBuilder(tre_dicts)
        model = builder.build()

        assert model is not None
        assert isinstance(model, RSMPolynomialSensorModel)

    def test_constructed_model_has_correct_normalization_params(self):
        """Verify the constructed model's normalization parameters match the RSMPCA dict values."""
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": SAMPLE_RSMPCA_DICT}
        model = RSMSensorModelBuilder(tre_dicts).build()

        assert model is not None
        assert model.section_row == 1
        assert model.section_col == 1
        assert model.row_norm_offset == pytest.approx(4646.0)
        assert model.column_norm_offset == pytest.approx(4561.0)
        assert model.x_norm_offset == pytest.approx(2655.71142640788)
        assert model.y_norm_offset == pytest.approx(2407.32324869712)
        assert model.z_norm_offset == pytest.approx(-2.33161661728923)
        assert model.row_norm_scale == pytest.approx(5599.2)
        assert model.column_norm_scale == pytest.approx(5497.2)
        assert model.x_norm_scale == pytest.approx(3117.93659470231)
        assert model.y_norm_scale == pytest.approx(2960.84811268529)
        assert model.z_norm_scale == pytest.approx(1002.3314880822)

    def test_world_to_image_ground_domain_origin(self):
        """Ground domain origin: lon=-117.03881, lat=33.16173, elev=-6.7 → image ~(0.5, 0.5). Req 3.1, 3.2."""
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": SAMPLE_RSMPCA_DICT}
        model = RSMSensorModelBuilder(tre_dicts).build()
        assert model is not None

        geodetic_origin = GeodeticWorldCoordinate([radians(-117.03881), radians(33.16173), -6.7])
        image_coord = model.world_to_image(geodetic_origin)
        assert image_coord.x == pytest.approx(0.5, abs=1.0)
        assert image_coord.y == pytest.approx(0.5, abs=1.0)

    def test_image_to_world_round_trip(self):
        """Round-trip: image (0.5, 0.5) → world → should match lon=-117.03881, lat=33.16173. Req 3.1, 3.2."""
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": SAMPLE_RSMPCA_DICT}
        model = RSMSensorModelBuilder(tre_dicts).build()
        assert model is not None

        geodetic_origin = GeodeticWorldCoordinate([radians(-117.03881), radians(33.16173), -6.7])
        round_trip_world = model.image_to_world(ImageCoordinate([0.5, 0.5]))
        assert round_trip_world.longitude == pytest.approx(geodetic_origin.longitude, abs=0.00001)
        assert round_trip_world.latitude == pytest.approx(geodetic_origin.latitude, abs=0.00001)

    def test_returns_none_when_rsmida_missing(self):
        """No RSMIDA key in tre_dicts should return None. Req 3.5."""
        tre_dicts = {"RSMPCA": SAMPLE_RSMPCA_DICT}
        builder = RSMSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_returns_none_when_rsmida_missing_empty_dict(self):
        """Empty tre_dicts should return None. Req 3.5."""
        builder = RSMSensorModelBuilder({})
        assert builder.build() is None

    def test_returns_none_when_required_rsmida_field_missing(self):
        """RSMIDA dict missing a required field should return None. Req 3.6."""
        rsmida = dict(SAMPLE_RSMIDA_DICT)
        del rsmida["GRNDD"]
        tre_dicts = {"RSMIDA": rsmida, "RSMPCA": SAMPLE_RSMPCA_DICT}
        builder = RSMSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_returns_none_when_required_rsmpca_field_missing(self):
        """RSMPCA dict missing a required field should return None. Req 3.6."""
        rsmpca = dict(SAMPLE_RSMPCA_DICT)
        del rsmpca["RNRMO"]
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": rsmpca}
        builder = RSMSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_returns_none_when_rsmpca_coefficient_missing(self):
        """RSMPCA dict missing a polynomial coefficient list should return None. Req 3.6."""
        rsmpca = dict(SAMPLE_RSMPCA_DICT)
        del rsmpca["RNPCF"]
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": rsmpca}
        builder = RSMSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_multiple_rsmpca_without_rsmpia_returns_none(self):
        """Multiple RSMPCA dicts without RSMPIA should return None. Req 3.4."""
        rsmpca_2 = dict(SAMPLE_RSMPCA_DICT)
        rsmpca_2["RSN"] = "001"
        rsmpca_2["CSN"] = "002"
        tre_dicts = {
            "RSMIDA": SAMPLE_RSMIDA_DICT,
            "RSMPCA": [SAMPLE_RSMPCA_DICT, rsmpca_2],
        }
        builder = RSMSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_rsmpca_as_list_with_single_element(self):
        """A single RSMPCA in a list should still produce an RSMPolynomialSensorModel. Req 3.2."""
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT, "RSMPCA": [SAMPLE_RSMPCA_DICT]}
        builder = RSMSensorModelBuilder(tre_dicts)
        model = builder.build()

        assert model is not None
        assert isinstance(model, RSMPolynomialSensorModel)

    def test_rsmida_only_no_rsmpca_returns_none(self):
        """RSMIDA present but no RSMPCA should return None (no polynomials)."""
        tre_dicts = {"RSMIDA": SAMPLE_RSMIDA_DICT}
        builder = RSMSensorModelBuilder(tre_dicts)
        assert builder.build() is None
