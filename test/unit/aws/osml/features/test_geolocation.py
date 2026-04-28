#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import geojson
import numpy as np

from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor
from aws.osml.metadata import SensorModelFactory, SensorModelTypes

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


class TestGeolocation(unittest.TestCase):
    def setUp(self):
        sensor_model = SensorModelFactory(
            2048,
            2048,
            tre_dicts={"RPC00B": SAMPLE_RPC00B_DICT},
            selected_sensor_model_types=[SensorModelTypes.RPC],
        ).build()

        self.geolocator = Geolocator(ImagedFeaturePropertyAccessor(), sensor_model)

    def test_geolocate_missing_features(self):
        features = []
        self.geolocator.geolocate_features(features)
        # Nothing to assert; just make sure it doesn't raise an exception.

    def test_geolocate_bbox_feature(self):
        feature = geojson.Feature(
            geometry=None, properties={ImagedFeaturePropertyAccessor.IMAGE_BBOX: [0, 0, 8819.0, 5211.0]}
        )
        self.geolocator.geolocate_features([feature])
        assert feature.bbox is not None
        assert feature.geometry is None
        assert np.allclose(feature.bbox, np.array([121.48749, 24.91148, 121.68595, 25.02860]), atol=1e-2)

    def test_geolocate_point_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Point)
        assert np.allclose(feature.geometry.coordinates, np.array([121.48749, 25.02860, 377.0]), atol=1e-3)

    def test_geolocate_linestring_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    "type": "LineString",
                    "coordinates": [[0, 0], [8819.0, 0.0], [8819.0, 5211.0]],
                }
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.LineString)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array([[121.48749, 25.02860, 377.0], [121.68566, 25.01000, 377.0], [121.68595, 24.91148, 377.0]]),
            atol=1e-3,
        )

    def test_geolocate_linearring_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    "type": "LinearRing",
                    "coordinates": [[0, 0], [8819.0, 0.0], [8819.0, 5211.0], [0, 0]],
                }
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.LineString)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.48749, 25.02860, 377.0],
                        [121.68566, 25.01000, 377.0],
                        [121.68595, 24.91148, 377.0],
                        [121.48749, 25.02860, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_geolocate_polygon_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [8819.0, 0.0], [8819.0, 5211.0], [0, 0]]],
                }
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Polygon)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.48749, 25.02860, 377.0],
                        [121.68566, 25.01000, 377.0],
                        [121.68595, 24.91148, 377.0],
                        [121.48749, 25.02860, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_geolocate_multipoint_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "MultiPoint", "coordinates": [[0, 0], [8819.0, 0.0]]}
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.MultiPoint)
        assert np.allclose(
            feature.geometry.coordinates, np.array([[121.48749, 25.02860, 377.0], [121.68566, 25.01000, 377.0]]), atol=1e-3
        )

    def test_geolocate_bounds_imcoords_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                "bounds_imcoords": [0, 0, 10, 10],
                "detection_score": 0.95,
                "feature_types": {"foo": 0.7},
                "image_id": "fake-image-id",
            },
        )

        self.geolocator.geolocate_features([feature])

        # Check to make sure the geolocation capability finds the bounds_imcoord property and creates a
        # polygon feature
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Polygon)

        # Check to ensure the exterior boundary of the polygon has all 5 points (4 corners + repeat of 1st corner)
        assert len(feature.geometry.coordinates[0]) == 5
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.489307, 25.027718, 377.0],
                        [121.489308, 25.027526, 377.0],
                        [121.489082, 25.027546, 377.0],
                        [121.489081, 25.027739, 377.0],
                        [121.489307, 25.027718, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_geolocate_geom_imcoords_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                "geom_imcoords": [[0, 0], [8819.0, 0.0], [8819.0, 5211.0], [0, 0]],
                "detection_score": 0.95,
                "feature_types": {"aircraft": 0.7},
                "image_id": "fake-image-id",
            },
        )

        self.geolocator.geolocate_features([feature])

        # Check to make sure the geolocation capability finds the geom_imcoord property and creates a
        # polygon feature
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Polygon)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.48749, 25.02860, 377.0],
                        [121.68566, 25.01000, 377.0],
                        [121.68595, 24.91148, 377.0],
                        [121.48749, 25.02860, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_force_false_skips_features_with_existing_geometry(self):
        existing_geometry = geojson.Point([10.0, 20.0, 0.0])
        feature = geojson.Feature(
            geometry=existing_geometry,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )

        geolocator = Geolocator(ImagedFeaturePropertyAccessor(), self.geolocator.sensor_model, force=False)
        geolocator.geolocate_features([feature])

        # geometry should remain unchanged because force=False and geometry already exists
        assert feature.geometry == existing_geometry

    def test_force_true_re_geolocates_features_with_existing_geometry(self):
        existing_geometry = geojson.Point([10.0, 20.0, 0.0])
        feature = geojson.Feature(
            geometry=existing_geometry,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )

        geolocator = Geolocator(ImagedFeaturePropertyAccessor(), self.geolocator.sensor_model, force=True)
        geolocator.geolocate_features([feature])

        # geometry should be re-computed because force=True
        assert feature.geometry is not None
        assert feature.geometry != existing_geometry
        assert isinstance(feature.geometry, geojson.Point)
        assert np.allclose(feature.geometry.coordinates, np.array([121.48749, 25.02860, 377.0]), atol=1e-3)

    def test_force_false_processes_features_without_geometry(self):
        feature = geojson.Feature(
            geometry=None,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )

        geolocator = Geolocator(ImagedFeaturePropertyAccessor(), self.geolocator.sensor_model, force=False)
        geolocator.geolocate_features([feature])

        # geometry should be computed because it was None
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Point)
        assert np.allclose(feature.geometry.coordinates, np.array([121.48749, 25.02860, 377.0]), atol=1e-3)

    def test_force_false_mixed_features(self):
        existing_geometry = geojson.Point([10.0, 20.0, 0.0])
        feature_with_geom = geojson.Feature(
            geometry=existing_geometry,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )
        feature_without_geom = geojson.Feature(
            geometry=None,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )

        geolocator = Geolocator(ImagedFeaturePropertyAccessor(), self.geolocator.sensor_model, force=False)
        geolocator.geolocate_features([feature_with_geom, feature_without_geom])

        # feature with existing geometry should be untouched
        assert feature_with_geom.geometry == existing_geometry
        # feature without geometry should be processed
        assert feature_without_geom.geometry is not None
        assert isinstance(feature_without_geom.geometry, geojson.Point)
