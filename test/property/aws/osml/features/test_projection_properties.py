#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

"""Property-based tests for Projector invariants.

Validates:
- All projected imageGeometry coordinates fall within image_bounds
- Polygon ring closure is preserved after projection (first coord == last coord)
"""

import geojson
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aws.osml.features import Projector
from aws.osml.features.imaged_feature_property_accessor import ImagedFeaturePropertyAccessor
from aws.osml.metadata.affine_sensor_model_builder import AffineSensorModelBuilder
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Affine sensor model: origin at (121.0, 25.0) degrees, 0.001 deg/pixel east,
# -0.001 deg/pixel south. Image domain: 1000x1000 pixels.
# Geographic footprint: lon=[121.0, 122.0], lat=[24.0, 25.0]
# ---------------------------------------------------------------------------

SAMPLE_GEO_TRANSFORM = [121.0, 0.001, 0.0, 25.0, 0.0, -0.001]
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000
IMAGE_BOUNDS = (0.0, 0.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT))

AFFINE_MODEL = AffineSensorModelBuilder(SAMPLE_GEO_TRANSFORM).build()

# Constrain coordinates to the interior of the image footprint to ensure all
# features project within bounds. A small margin (5%) avoids edge effects.
LON_MIN = 121.05
LON_MAX = 121.95
LAT_MIN = 24.05
LAT_MAX = 24.95


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating GeoJSON features within the footprint
# ---------------------------------------------------------------------------


def _lon():
    return st.floats(min_value=LON_MIN, max_value=LON_MAX, allow_nan=False, allow_infinity=False)


def _lat():
    return st.floats(min_value=LAT_MIN, max_value=LAT_MAX, allow_nan=False, allow_infinity=False)


def _coord():
    return st.tuples(_lon(), _lat())


point_features = st.builds(
    lambda coord: geojson.Feature(geometry=geojson.Point(coordinates=list(coord))),
    coord=_coord(),
)


linestring_features = st.builds(
    lambda coords: geojson.Feature(geometry=geojson.LineString(coordinates=[list(c) for c in coords])),
    coords=st.lists(_coord(), min_size=2, max_size=10),
)


def _polygon_ring(min_size=4, max_size=8):
    """Generate a closed polygon ring (first == last) within the geographic extent."""
    return st.lists(_coord(), min_size=min_size, max_size=max_size).map(lambda pts: [list(c) for c in pts] + [list(pts[0])])


polygon_features = st.builds(
    lambda ring: geojson.Feature(geometry=geojson.Polygon(coordinates=[ring])),
    ring=_polygon_ring(),
)


multipoint_features = st.builds(
    lambda coords: geojson.Feature(geometry=geojson.MultiPoint(coordinates=[list(c) for c in coords])),
    coords=st.lists(_coord(), min_size=1, max_size=10),
)


all_geometry_features = st.one_of(point_features, linestring_features, polygon_features, multipoint_features)


# ---------------------------------------------------------------------------
# Property: All projected coordinates fall within image_bounds
# ---------------------------------------------------------------------------


def _extract_coords_from_geojson_dict(geom_dict: dict) -> list:
    """Recursively extract all (x, y) coordinates from a GeoJSON-like dict."""
    geom_type = geom_dict.get("type")
    coords = geom_dict.get("coordinates")

    if geom_type == "Point":
        return [tuple(coords[:2])]
    elif geom_type in ("LineString", "MultiPoint"):
        return [tuple(c[:2]) for c in coords]
    elif geom_type == "Polygon":
        result = []
        for ring in coords:
            result.extend(tuple(c[:2]) for c in ring)
        return result
    elif geom_type == "MultiLineString":
        result = []
        for line in coords:
            result.extend(tuple(c[:2]) for c in line)
        return result
    elif geom_type == "MultiPolygon":
        result = []
        for polygon in coords:
            for ring in polygon:
                result.extend(tuple(c[:2]) for c in ring)
        return result
    elif geom_type == "GeometryCollection":
        result = []
        for sub_geom in geom_dict.get("geometries", []):
            result.extend(_extract_coords_from_geojson_dict(sub_geom))
        return result
    return []


@pytest.mark.property
@given(feature=all_geometry_features)
@settings(pbt_settings)
def test_projected_coordinates_within_image_bounds(feature):
    """All projected imageGeometry coordinates must fall within image_bounds.

    For features generated within the geographic footprint of the affine model,
    the Projector must produce pixel coordinates that lie within [0, IMAGE_WIDTH]
    x [0, IMAGE_HEIGHT].
    """
    property_accessor = ImagedFeaturePropertyAccessor()
    projector = Projector(
        property_accessor=property_accessor,
        sensor_model=AFFINE_MODEL,
        image_bounds=IMAGE_BOUNDS,
        force=True,
    )

    results = projector.project_features([feature])
    assert len(results) == 1, "Feature within footprint should project successfully"

    image_geom = feature["properties"].get("imageGeometry")
    assert image_geom is not None, "imageGeometry must be set after projection"

    pixel_coords = _extract_coords_from_geojson_dict(image_geom)
    assert len(pixel_coords) > 0, "Must have at least one coordinate"

    for x, y in pixel_coords:
        assert IMAGE_BOUNDS[0] <= x <= IMAGE_BOUNDS[2], f"x={x} outside bounds [{IMAGE_BOUNDS[0]}, {IMAGE_BOUNDS[2]}]"
        assert IMAGE_BOUNDS[1] <= y <= IMAGE_BOUNDS[3], f"y={y} outside bounds [{IMAGE_BOUNDS[1]}, {IMAGE_BOUNDS[3]}]"


# ---------------------------------------------------------------------------
# Property: Polygon ring closure preserved after projection
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(feature=polygon_features)
@settings(pbt_settings)
def test_polygon_ring_closure_preserved(feature):
    """Polygon ring closure must be preserved after projection.

    GeoJSON requires polygon rings to have identical first and last coordinates.
    Since world_to_image() is deterministic, identical input coordinates must
    produce identical output coordinates, preserving this invariant.
    """
    property_accessor = ImagedFeaturePropertyAccessor()
    projector = Projector(
        property_accessor=property_accessor,
        sensor_model=AFFINE_MODEL,
        image_bounds=IMAGE_BOUNDS,
        force=True,
    )

    results = projector.project_features([feature])
    assert len(results) == 1, "Polygon within footprint should project successfully"

    image_geom = feature["properties"].get("imageGeometry")
    assert image_geom is not None
    assert image_geom["type"] == "Polygon"

    for ring in image_geom["coordinates"]:
        assert len(ring) >= 4, "Ring must have at least 4 coordinates (triangle + closing)"
        first = ring[0]
        last = ring[-1]
        assert first[0] == last[0], f"Ring x not closed: first={first[0]}, last={last[0]}"
        assert first[1] == last[1], f"Ring y not closed: first={first[1]}, last={last[1]}"
