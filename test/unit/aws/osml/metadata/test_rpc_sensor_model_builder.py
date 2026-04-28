#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from math import radians

import numpy as np
import pytest

from aws.osml.metadata.rpc_sensor_model_builder import RPCSensorModelBuilder
from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
from aws.osml.photogrammetry.rpc_sensor_model import RPCSensorModel

# Real RPC00B TRE field values extracted from test/data/sample-metadata-ms-rpc00b.xml.
# Hardcoded as an inline dict so this test does not depend on GDAL XML parsing.
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


class TestRPCSensorModelBuilder:
    """Unit tests for RPCSensorModelBuilder. Requirements: 2.1, 2.3, 2.4, 2.5, 2.6."""

    def test_successful_construction_produces_rpc_sensor_model(self):
        """Build from valid RPC00B dict produces an RPCSensorModel instance."""
        tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
        builder = RPCSensorModelBuilder(tre_dicts)
        model = builder.build()

        assert model is not None
        assert isinstance(model, RPCSensorModel)

    def test_constructed_model_has_correct_parameters(self):
        """Verify the constructed model's scalar parameters match the input dict values."""
        tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
        builder = RPCSensorModelBuilder(tre_dicts)
        model = builder.build()

        assert model is not None
        assert model.err_bias == pytest.approx(5.18)
        assert model.err_rand == pytest.approx(0.98)
        assert model.line_off == pytest.approx(2606.0)
        assert model.samp_off == pytest.approx(4409.0)
        assert model.lat_off == pytest.approx(24.9697)
        assert model.long_off == pytest.approx(121.5875)
        assert model.height_off == pytest.approx(377.0)
        assert model.line_scale == pytest.approx(2606.0)
        assert model.samp_scale == pytest.approx(4410.0)
        assert model.lat_scale == pytest.approx(0.0593)
        assert model.long_scale == pytest.approx(0.1008)
        assert model.height_scale == pytest.approx(500.0)

    def test_world_to_image_ul_corner(self):
        """UL corner: lon=121.48749, lat=25.02860, ht=27.1 → image ~(0, 0)."""
        tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
        model = RPCSensorModelBuilder(tre_dicts).build()
        assert model is not None

        world_coord = GeodeticWorldCoordinate([radians(121.48749), radians(25.02860), 27.1])
        image_coord = model.world_to_image(world_coord)
        assert np.allclose(image_coord.coordinate, np.array([0.0, 0.0]), atol=1.0)

    def test_world_to_image_ur_corner(self):
        """UR corner: lon=121.68566, lat=25.01000, ht=234.7 → image ~(8819, 0)."""
        tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
        model = RPCSensorModelBuilder(tre_dicts).build()
        assert model is not None

        world_coord = GeodeticWorldCoordinate([radians(121.68566), radians(25.01000), 234.7])
        image_coord = model.world_to_image(world_coord)
        assert np.allclose(image_coord.coordinate, np.array([8819.0, 0.0]), atol=1.0)

    def test_world_to_image_lr_corner(self):
        """LR corner: lon=121.68595, lat=24.91148, ht=403.1 → image ~(8819, 5211)."""
        tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
        model = RPCSensorModelBuilder(tre_dicts).build()
        assert model is not None

        world_coord = GeodeticWorldCoordinate([radians(121.68595), radians(24.91148), 403.1])
        image_coord = model.world_to_image(world_coord)
        assert np.allclose(image_coord.coordinate, np.array([8819.0, 5211.0]), atol=1.0)

    def test_world_to_image_ll_corner(self):
        """LL corner: lon=121.48975, lat=24.92772, ht=431.4 → image ~(0, 5211)."""
        tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
        model = RPCSensorModelBuilder(tre_dicts).build()
        assert model is not None

        world_coord = GeodeticWorldCoordinate([radians(121.48975), radians(24.92772), 431.4])
        image_coord = model.world_to_image(world_coord)
        assert np.allclose(image_coord.coordinate, np.array([0.0, 5211.0]), atol=1.0)

    def test_returns_none_when_success_not_1(self):
        """RPC00B with SUCCESS != 1 should return None."""
        rpc_dict = dict(SAMPLE_RPC00B_DICT)
        rpc_dict["SUCCESS"] = "0"
        tre_dicts = {"RPC00B": rpc_dict}
        builder = RPCSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_returns_none_when_rpc00b_key_missing(self):
        """No RPC00B key in tre_dicts should return None."""
        tre_dicts = {"CSCRNA": {"ULCNR_LAT": "+25.02860"}}
        builder = RPCSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_returns_none_when_rpc00b_empty_dict(self):
        """Empty tre_dicts should return None."""
        builder = RPCSensorModelBuilder({})
        assert builder.build() is None

    def test_returns_none_when_required_field_missing(self):
        """RPC00B dict missing a required field should return None."""
        rpc_dict = dict(SAMPLE_RPC00B_DICT)
        del rpc_dict["ERR_BIAS"]
        tre_dicts = {"RPC00B": rpc_dict}
        builder = RPCSensorModelBuilder(tre_dicts)
        assert builder.build() is None

    def test_returns_none_when_coefficient_missing(self):
        """RPC00B dict missing a polynomial coefficient list should return None."""
        rpc_dict = dict(SAMPLE_RPC00B_DICT)
        del rpc_dict["LINE_NUM_COEFF"]
        tre_dicts = {"RPC00B": rpc_dict}
        builder = RPCSensorModelBuilder(tre_dicts)
        assert builder.build() is None
