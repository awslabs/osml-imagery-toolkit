#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import geojson
import shapely


class TestFeatureIndex(unittest.TestCase):
    def setUp(self):
        from aws.osml.features import STRFeature2DSpatialIndex

        test_features = []
        for r in range(0, 30, 10):
            for c in range(0, 30, 10):
                test_features.append(geojson.Feature(geometry=None, properties={"imageBBox": [c, r, c + 5, r + 5]}))
        test_fc = geojson.FeatureCollection(features=test_features)

        self.index = STRFeature2DSpatialIndex(test_fc, use_image_geometries=True)

    def test_find_partial_intersects(self):
        results = self.index.find_intersects(shapely.box(-1, -1, 11, 11))
        assert len(list(results)) == 4

    def test_find_contains(self):
        results = self.index.find_intersects(shapely.box(-1, -1, 31, 31))
        assert len(list(results)) == 9

    def test_find_nearest(self):
        results = self.index.find_nearest(shapely.Point(1, 1), max_distance=5)
        assert len(list(results)) == 1
