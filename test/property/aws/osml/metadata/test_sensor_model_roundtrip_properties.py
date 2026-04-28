#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

"""Property-based tests for sensor model world-to-image/image-to-world round-trip consistency.

# Feature: sensor-model-factory, Property 11: Sensor model world-to-image/image-to-world round-trip
"""

from math import radians

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aws.osml.metadata.affine_sensor_model_builder import AffineSensorModelBuilder
from aws.osml.metadata.projective_sensor_model_builder import ProjectiveSensorModelBuilder
from aws.osml.metadata.rpc_sensor_model_builder import RPCSensorModelBuilder
from aws.osml.photogrammetry import GeodeticWorldCoordinate
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Inline test data (same as unit tests — no GDAL dependency)
# ---------------------------------------------------------------------------

# Real RPC00B TRE field values from test/data/sample-metadata-ms-rpc00b.xml.
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

# Real CSCRNA TRE field values from test/data/sample-metadata-ms-rpc00b.xml.
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

CSCRNA_IMAGE_WIDTH = 8820.0
CSCRNA_IMAGE_HEIGHT = 5212.0

# Simple affine geo transform: origin at (121.0, 25.0) degrees, 0.001 deg/pixel east, -0.001 deg/pixel south.
SAMPLE_GEO_TRANSFORM = [121.0, 0.001, 0.0, 25.0, 0.0, -0.001]
AFFINE_IMAGE_WIDTH = 1000
AFFINE_IMAGE_HEIGHT = 1000

# Round-trip tolerance: 0.00001 radians (~0.57 millidegrees, ~63 meters at equator)
ROUND_TRIP_TOLERANCE_RAD = 0.00001


# ---------------------------------------------------------------------------
# Build sensor models once (module-level fixtures)
# ---------------------------------------------------------------------------


def _build_rpc_model():
    """Build an RPC sensor model from the sample RPC00B dict."""
    tre_dicts = {"RPC00B": SAMPLE_RPC00B_DICT}
    return RPCSensorModelBuilder(tre_dicts).build()


def _build_projective_model():
    """Build a projective sensor model from the sample CSCRNA dict."""
    tre_dicts = {"CSCRNA": SAMPLE_CSCRNA_DICT}
    return ProjectiveSensorModelBuilder(tre_dicts, CSCRNA_IMAGE_WIDTH, CSCRNA_IMAGE_HEIGHT).build()


def _build_affine_model():
    """Build an affine sensor model from a simple geo transform."""
    return AffineSensorModelBuilder(SAMPLE_GEO_TRANSFORM).build()


RPC_MODEL = _build_rpc_model()
PROJECTIVE_MODEL = _build_projective_model()
AFFINE_MODEL = _build_affine_model()


# ---------------------------------------------------------------------------
# Hypothesis strategies for valid world coordinates within each model's domain
# ---------------------------------------------------------------------------

# RPC valid domain: lat=[24.91, 25.03], lon=[121.49, 121.69].
# The RPC model's image_to_world uses a default ConstantElevationModel at HEIGHT_OFF (377m).
# For a consistent round-trip, the input height must match the default elevation model.
RPC_DEFAULT_HEIGHT = 377.0

rpc_world_coordinates = st.builds(
    lambda lat, lon: GeodeticWorldCoordinate([radians(lon), radians(lat), RPC_DEFAULT_HEIGHT]),
    lat=st.floats(min_value=24.92, max_value=25.02, allow_nan=False, allow_infinity=False),
    lon=st.floats(min_value=121.50, max_value=121.68, allow_nan=False, allow_infinity=False),
)

# Projective valid domain: image extent maps to the CSCRNA corners.
# Generate image coordinates within the image, then convert to world via image_to_world.
# Instead, generate world coordinates within the bounding box of the corners.
projective_world_coordinates = st.builds(
    lambda lat, lon: GeodeticWorldCoordinate([radians(lon), radians(lat), 0.0]),
    lat=st.floats(min_value=24.92, max_value=25.02, allow_nan=False, allow_infinity=False),
    lon=st.floats(min_value=121.50, max_value=121.68, allow_nan=False, allow_infinity=False),
)

# Affine valid domain: image pixels [0, 1000] x [0, 1000] map to
# lon=[121.0, 122.0], lat=[24.0, 25.0] degrees (no CRS projection).
affine_world_coordinates = st.builds(
    lambda lat, lon: GeodeticWorldCoordinate([radians(lon), radians(lat), 0.0]),
    lat=st.floats(min_value=24.05, max_value=24.95, allow_nan=False, allow_infinity=False),
    lon=st.floats(min_value=121.05, max_value=121.95, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Property 11: Sensor model world-to-image/image-to-world round-trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(world_coord=rpc_world_coordinates)
@settings(pbt_settings)
def test_rpc_world_to_image_to_world_round_trip(world_coord):
    """**Validates: Requirements 14.4**

    For any world coordinate within the RPC model's valid domain,
    applying world_to_image followed by image_to_world SHALL return
    a world coordinate within 0.00001 radians of the original.
    """
    # Feature: sensor-model-factory, Property 11: Sensor model world-to-image/image-to-world round-trip
    assert RPC_MODEL is not None

    image_coord = RPC_MODEL.world_to_image(world_coord)
    round_tripped = RPC_MODEL.image_to_world(image_coord)

    assert abs(round_tripped.longitude - world_coord.longitude) < ROUND_TRIP_TOLERANCE_RAD, (
        f"Longitude mismatch: original={world_coord.longitude}, round_tripped={round_tripped.longitude}, "
        f"diff={abs(round_tripped.longitude - world_coord.longitude)}"
    )
    assert abs(round_tripped.latitude - world_coord.latitude) < ROUND_TRIP_TOLERANCE_RAD, (
        f"Latitude mismatch: original={world_coord.latitude}, round_tripped={round_tripped.latitude}, "
        f"diff={abs(round_tripped.latitude - world_coord.latitude)}"
    )


@pytest.mark.property
@given(world_coord=projective_world_coordinates)
@settings(pbt_settings)
def test_projective_world_to_image_to_world_round_trip(world_coord):
    """**Validates: Requirements 14.4**

    For any world coordinate within the projective model's valid domain,
    applying world_to_image followed by image_to_world SHALL return
    a world coordinate within 0.00001 radians of the original.
    """
    # Feature: sensor-model-factory, Property 11: Sensor model world-to-image/image-to-world round-trip
    assert PROJECTIVE_MODEL is not None

    image_coord = PROJECTIVE_MODEL.world_to_image(world_coord)
    round_tripped = PROJECTIVE_MODEL.image_to_world(image_coord)

    assert abs(round_tripped.longitude - world_coord.longitude) < ROUND_TRIP_TOLERANCE_RAD, (
        f"Longitude mismatch: original={world_coord.longitude}, round_tripped={round_tripped.longitude}, "
        f"diff={abs(round_tripped.longitude - world_coord.longitude)}"
    )
    assert abs(round_tripped.latitude - world_coord.latitude) < ROUND_TRIP_TOLERANCE_RAD, (
        f"Latitude mismatch: original={world_coord.latitude}, round_tripped={round_tripped.latitude}, "
        f"diff={abs(round_tripped.latitude - world_coord.latitude)}"
    )


@pytest.mark.property
@given(world_coord=affine_world_coordinates)
@settings(pbt_settings)
def test_affine_world_to_image_to_world_round_trip(world_coord):
    """**Validates: Requirements 14.4**

    For any world coordinate within the affine model's valid domain,
    applying world_to_image followed by image_to_world SHALL return
    a world coordinate within 0.00001 radians of the original.
    """
    # Feature: sensor-model-factory, Property 11: Sensor model world-to-image/image-to-world round-trip
    assert AFFINE_MODEL is not None

    image_coord = AFFINE_MODEL.world_to_image(world_coord)
    round_tripped = AFFINE_MODEL.image_to_world(image_coord)

    assert abs(round_tripped.longitude - world_coord.longitude) < ROUND_TRIP_TOLERANCE_RAD, (
        f"Longitude mismatch: original={world_coord.longitude}, round_tripped={round_tripped.longitude}, "
        f"diff={abs(round_tripped.longitude - world_coord.longitude)}"
    )
    assert abs(round_tripped.latitude - world_coord.latitude) < ROUND_TRIP_TOLERANCE_RAD, (
        f"Latitude mismatch: original={world_coord.latitude}, round_tripped={round_tripped.latitude}, "
        f"diff={abs(round_tripped.latitude - world_coord.latitude)}"
    )
