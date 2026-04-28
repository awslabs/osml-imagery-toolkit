#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import time
import unittest
from unittest.mock import MagicMock

import geojson
import numpy as np

from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor, Projector
from aws.osml.metadata import SensorModelFactory, SensorModelTypes
from aws.osml.photogrammetry import AffineSensorModel, ConstantElevationModel, ElevationModel

logger = logging.getLogger(__name__)

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


def make_affine_sensor_model():
    """Create a simple affine sensor model for testing.

    Uses a transform where:
    - origin at lon=0, lat=0 (degrees)
    - 0.001 degrees per pixel in x
    - -0.001 degrees per pixel in y (north-up)

    So pixel (100, 200) maps to lon=0.1, lat=-0.2 degrees.
    """
    geo_transform = [0.0, 0.001, 0.0, 0.0, 0.0, -0.001]
    return AffineSensorModel(geo_transform)


class TestProjectorBasic(unittest.TestCase):
    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 1000.0, 1000.0)
        self.projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
        )

    def test_empty_input_returns_empty_list(self):
        result = self.projector.project_features([])
        assert result == []

    def test_none_geometry_skipped(self):
        feature = geojson.Feature(geometry=None, properties={})
        result = self.projector.project_features([feature])
        assert result == []

    def test_point_projection(self):
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "Point"
        coords = image_geom["coordinates"]
        assert abs(coords[0] - 500.0) < 0.01
        assert abs(coords[1] - 500.0) < 0.01

    def test_linestring_projection(self):
        feature = geojson.Feature(
            geometry=geojson.LineString([(0.1, -0.1), (0.2, -0.2), (0.3, -0.3)]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "LineString"
        coords = image_geom["coordinates"]
        assert len(coords) == 3
        assert abs(coords[0][0] - 100.0) < 0.01
        assert abs(coords[0][1] - 100.0) < 0.01
        assert abs(coords[1][0] - 200.0) < 0.01
        assert abs(coords[1][1] - 200.0) < 0.01

    def test_polygon_projection(self):
        feature = geojson.Feature(
            geometry=geojson.Polygon([[(0.1, -0.1), (0.3, -0.1), (0.3, -0.3), (0.1, -0.3), (0.1, -0.1)]]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "Polygon"
        coords = image_geom["coordinates"]
        # Exterior ring
        assert len(coords[0]) == 5
        # First and last should match (ring closure)
        assert coords[0][0] == coords[0][-1]

    def test_polygon_with_hole(self):
        exterior = [(0.1, -0.1), (0.5, -0.1), (0.5, -0.5), (0.1, -0.5), (0.1, -0.1)]
        hole = [(0.2, -0.2), (0.4, -0.2), (0.4, -0.4), (0.2, -0.4), (0.2, -0.2)]
        feature = geojson.Feature(
            geometry=geojson.Polygon([exterior, hole]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "Polygon"
        coords = image_geom["coordinates"]
        assert len(coords) == 2  # exterior + 1 hole
        # Ring closure preserved
        assert coords[0][0] == coords[0][-1]
        assert coords[1][0] == coords[1][-1]

    def test_multipoint_projection(self):
        feature = geojson.Feature(
            geometry=geojson.MultiPoint([(0.1, -0.1), (0.2, -0.2)]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "MultiPoint"
        coords = image_geom["coordinates"]
        assert len(coords) == 2

    def test_multilinestring_projection(self):
        feature = geojson.Feature(
            geometry=geojson.MultiLineString(
                [
                    [(0.1, -0.1), (0.2, -0.2)],
                    [(0.3, -0.3), (0.4, -0.4)],
                ]
            ),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "MultiLineString"
        assert len(image_geom["coordinates"]) == 2

    def test_multipolygon_projection(self):
        poly1 = [[(0.1, -0.1), (0.2, -0.1), (0.2, -0.2), (0.1, -0.2), (0.1, -0.1)]]
        poly2 = [[(0.3, -0.3), (0.4, -0.3), (0.4, -0.4), (0.3, -0.4), (0.3, -0.3)]]
        feature = geojson.Feature(
            geometry=geojson.MultiPolygon([poly1, poly2]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "MultiPolygon"
        assert len(image_geom["coordinates"]) == 2

    def test_geometry_collection_projection(self):
        feature = geojson.Feature(
            geometry=geojson.GeometryCollection(
                [
                    geojson.Point((0.1, -0.1)),
                    geojson.LineString([(0.2, -0.2), (0.3, -0.3)]),
                ]
            ),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_geom = result[0]["properties"]["imageGeometry"]
        assert image_geom["type"] == "GeometryCollection"
        assert len(image_geom["geometries"]) == 2


class TestProjectorBoundsFiltering(unittest.TestCase):
    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 500.0, 500.0)
        self.projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
        )

    def test_feature_inside_bounds_included(self):
        feature = geojson.Feature(
            geometry=geojson.Point((0.1, -0.1)),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1

    def test_feature_outside_bounds_excluded(self):
        # lon=0.8, lat=-0.8 -> pixel (800, 800) which is outside (0,0,500,500)
        feature = geojson.Feature(
            geometry=geojson.Point((0.8, -0.8)),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 0
        # Feature should be unmodified
        assert "imageGeometry" not in feature["properties"]

    def test_feature_partially_overlapping_bounds(self):
        # Polygon spans from pixel (100,100) to (600,600) - partially overlaps (0,0,500,500)
        feature = geojson.Feature(
            geometry=geojson.Polygon([[(0.1, -0.1), (0.6, -0.1), (0.6, -0.6), (0.1, -0.6), (0.1, -0.1)]]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1

    def test_two_pass_filtering_bbox_rejects(self):
        """Feature with large imageBBox that doesn't intersect bounds is rejected in broadphase."""
        # lon=0.6 to 0.9, lat=-0.6 to -0.9 -> pixels (600,600) to (900,900)
        feature = geojson.Feature(
            geometry=geojson.Point((0.75, -0.75)),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 0


class TestProjectorForceParameter(unittest.TestCase):
    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 1000.0, 1000.0)

    def test_force_false_skips_existing_imagegeometry(self):
        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=False,
        )
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            properties={
                "imageGeometry": {"type": "Point", "coordinates": [123.0, 456.0]},
            },
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        # Should retain the original imageGeometry, not re-project
        assert feature["properties"]["imageGeometry"]["coordinates"] == [123.0, 456.0]

    def test_force_true_reprojects_existing_imagegeometry(self):
        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=True,
        )
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            properties={
                "imageGeometry": {"type": "Point", "coordinates": [123.0, 456.0]},
            },
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        # Should have re-projected from geometry
        coords = feature["properties"]["imageGeometry"]["coordinates"]
        assert abs(coords[0] - 500.0) < 0.01
        assert abs(coords[1] - 500.0) < 0.01

    def test_force_false_existing_outside_bounds_excluded(self):
        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=(0.0, 0.0, 100.0, 100.0),
            force=False,
        )
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            properties={
                "imageGeometry": {"type": "Point", "coordinates": [500.0, 500.0]},
            },
        )
        result = projector.project_features([feature])
        assert len(result) == 0


class TestProjectorElevation(unittest.TestCase):
    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 1000.0, 1000.0)

    def test_explicit_z_used(self):
        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
        )
        # AffineSensorModel ignores elevation, so the projection result is the same
        # but we verify it doesn't crash with explicit Z
        feature = geojson.Feature(
            geometry={"type": "Point", "coordinates": [0.5, -0.5, 100.0]},
            properties={},
        )
        result = projector.project_features([feature])
        assert len(result) == 1

    def test_elevation_model_used_when_no_z(self):
        elevation_model = ConstantElevationModel(500.0)
        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            elevation_model=elevation_model,
        )
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            properties={},
        )
        result = projector.project_features([feature])
        assert len(result) == 1

    def test_elevation_model_failure_falls_back_to_zero(self):
        elevation_model = MagicMock(spec=ElevationModel)
        elevation_model.set_elevation.return_value = False

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            elevation_model=elevation_model,
        )
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            properties={},
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        elevation_model.set_elevation.assert_called_once()

    def test_explicit_z_takes_precedence_over_elevation_model(self):
        elevation_model = MagicMock(spec=ElevationModel)
        elevation_model.set_elevation.return_value = True

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            elevation_model=elevation_model,
        )
        feature = geojson.Feature(
            geometry={"type": "Point", "coordinates": [0.5, -0.5, 100.0]},
            properties={},
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        # Elevation model should NOT be called when Z is explicit
        elevation_model.set_elevation.assert_not_called()


class TestProjectorBBox(unittest.TestCase):
    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 1000.0, 1000.0)
        self.projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
        )

    def test_feature_bbox_projected_via_corners(self):
        feature = geojson.Feature(
            geometry=geojson.Point((0.5, -0.5)),
            bbox=[0.4, -0.6, 0.6, -0.4],
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_bbox = result[0]["properties"]["imageBBox"]
        # bbox corners: (0.4,-0.6)->(400,600), (0.4,-0.4)->(400,400), (0.6,-0.4)->(600,400), (0.6,-0.6)->(600,600)
        assert abs(image_bbox[0] - 400.0) < 0.01
        assert abs(image_bbox[1] - 400.0) < 0.01
        assert abs(image_bbox[2] - 600.0) < 0.01
        assert abs(image_bbox[3] - 600.0) < 0.01

    def test_no_feature_bbox_derives_from_geometry(self):
        feature = geojson.Feature(
            geometry=geojson.Polygon([[(0.1, -0.1), (0.3, -0.1), (0.3, -0.3), (0.1, -0.3), (0.1, -0.1)]]),
            properties={},
        )
        result = self.projector.project_features([feature])
        assert len(result) == 1
        image_bbox = result[0]["properties"]["imageBBox"]
        assert abs(image_bbox[0] - 100.0) < 0.01
        assert abs(image_bbox[1] - 100.0) < 0.01
        assert abs(image_bbox[2] - 300.0) < 0.01
        assert abs(image_bbox[3] - 300.0) < 0.01


class TestProjectorMultipleFeatures(unittest.TestCase):
    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 500.0, 500.0)
        self.projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
        )

    def test_mixed_in_and_out_of_bounds(self):
        inside = geojson.Feature(geometry=geojson.Point((0.1, -0.1)), properties={})
        outside = geojson.Feature(geometry=geojson.Point((0.8, -0.8)), properties={})
        result = self.projector.project_features([inside, outside])
        assert len(result) == 1
        assert result[0] is inside
        # Outside feature should not be mutated
        assert "imageGeometry" not in outside["properties"]


class TestProjectorRoundTripAffine(unittest.TestCase):
    """Round-trip: imageGeometry -> Geolocator(force=True) -> Projector(force=True) -> verify match."""

    def setUp(self):
        self.sensor_model = make_affine_sensor_model()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 1000.0, 1000.0)

    def test_round_trip_point(self):
        original_coords = [500.0, 300.0]
        feature = geojson.Feature(
            geometry=None,
            properties={"imageGeometry": {"type": "Point", "coordinates": original_coords}},
        )

        geolocator = Geolocator(property_accessor=self.accessor, sensor_model=self.sensor_model, force=True)
        geolocator.geolocate_features([feature])
        assert feature["geometry"] is not None

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=True,
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        recovered = result[0]["properties"]["imageGeometry"]["coordinates"]
        assert abs(recovered[0] - original_coords[0]) < 0.01
        assert abs(recovered[1] - original_coords[1]) < 0.01

    def test_round_trip_polygon(self):
        original_coords = [[[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0], [100.0, 100.0]]]
        feature = geojson.Feature(
            geometry=None,
            properties={"imageGeometry": {"type": "Polygon", "coordinates": original_coords}},
        )

        geolocator = Geolocator(property_accessor=self.accessor, sensor_model=self.sensor_model, force=True)
        geolocator.geolocate_features([feature])
        assert feature["geometry"] is not None

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=True,
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        recovered = result[0]["properties"]["imageGeometry"]["coordinates"]
        for orig_ring, rec_ring in zip(original_coords, recovered):
            for orig_pt, rec_pt in zip(orig_ring, rec_ring):
                assert abs(rec_pt[0] - orig_pt[0]) < 0.01
                assert abs(rec_pt[1] - orig_pt[1]) < 0.01

    def test_round_trip_linestring(self):
        original_coords = [[200.0, 200.0], [400.0, 400.0], [600.0, 200.0]]
        feature = geojson.Feature(
            geometry=None,
            properties={"imageGeometry": {"type": "LineString", "coordinates": original_coords}},
        )

        geolocator = Geolocator(property_accessor=self.accessor, sensor_model=self.sensor_model, force=True)
        geolocator.geolocate_features([feature])

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=True,
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        recovered = result[0]["properties"]["imageGeometry"]["coordinates"]
        for orig_pt, rec_pt in zip(original_coords, recovered):
            assert abs(rec_pt[0] - orig_pt[0]) < 0.01
            assert abs(rec_pt[1] - orig_pt[1]) < 0.01


class TestProjectorRoundTripRPC(unittest.TestCase):
    """Round-trip tests using a real-world RPC sensor model (tolerance < 1.0 pixel)."""

    def setUp(self):
        self.sensor_model = SensorModelFactory(
            8819,
            5211,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
            selected_sensor_model_types=[SensorModelTypes.RPC],
        ).build()
        self.accessor = ImagedFeaturePropertyAccessor()
        self.image_bounds = (0.0, 0.0, 8819.0, 5211.0)

    def test_round_trip_point_rpc(self):
        original_coords = [4000.0, 2500.0]
        feature = geojson.Feature(
            geometry=None,
            properties={"imageGeometry": {"type": "Point", "coordinates": original_coords}},
        )

        geolocator = Geolocator(property_accessor=self.accessor, sensor_model=self.sensor_model, force=True)
        geolocator.geolocate_features([feature])
        assert feature["geometry"] is not None

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=True,
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        recovered = result[0]["properties"]["imageGeometry"]["coordinates"]
        assert abs(recovered[0] - original_coords[0]) < 1.0
        assert abs(recovered[1] - original_coords[1]) < 1.0

    def test_round_trip_polygon_rpc(self):
        original_coords = [[[2000.0, 1000.0], [3000.0, 1000.0], [3000.0, 2000.0], [2000.0, 2000.0], [2000.0, 1000.0]]]
        feature = geojson.Feature(
            geometry=None,
            properties={"imageGeometry": {"type": "Polygon", "coordinates": original_coords}},
        )

        geolocator = Geolocator(property_accessor=self.accessor, sensor_model=self.sensor_model, force=True)
        geolocator.geolocate_features([feature])

        projector = Projector(
            property_accessor=self.accessor,
            sensor_model=self.sensor_model,
            image_bounds=self.image_bounds,
            force=True,
        )
        result = projector.project_features([feature])
        assert len(result) == 1
        recovered = result[0]["properties"]["imageGeometry"]["coordinates"]
        for orig_ring, rec_ring in zip(original_coords, recovered):
            for orig_pt, rec_pt in zip(orig_ring, rec_ring):
                assert abs(rec_pt[0] - orig_pt[0]) < 1.0
                assert abs(rec_pt[1] - orig_pt[1]) < 1.0


class TestProjectorPerformance(unittest.TestCase):
    """Performance benchmark: 10K Point features through an RPC sensor model."""

    def test_10k_points_rpc_benchmark(self):
        sensor_model = SensorModelFactory(
            8819,
            5211,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
            selected_sensor_model_types=[SensorModelTypes.RPC],
        ).build()
        accessor = ImagedFeaturePropertyAccessor()
        image_bounds = (0.0, 0.0, 8819.0, 5211.0)
        projector = Projector(
            property_accessor=accessor,
            sensor_model=sensor_model,
            image_bounds=image_bounds,
        )

        rng = np.random.default_rng(42)
        lon_min, lon_max = 121.487, 121.688
        lat_min, lat_max = 24.911, 25.029
        lons = rng.uniform(lon_min, lon_max, 10000)
        lats = rng.uniform(lat_min, lat_max, 10000)

        features = [
            geojson.Feature(geometry=geojson.Point((float(lons[i]), float(lats[i]))), properties={}) for i in range(10000)
        ]

        start = time.perf_counter()
        result = projector.project_features(features)
        elapsed = time.perf_counter() - start

        logger.info("Performance benchmark: projected %d/%d features in %.3f seconds", len(result), 10000, elapsed)
        assert len(result) > 0


if __name__ == "__main__":
    unittest.main()
