#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import geojson


class TestSensorModel(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        self.sample_geojson_detections = self.build_geojson_detections()

    @staticmethod
    def build_geojson_detections():
        with open("./test/data/detections.geojson", "r") as geojson_file:
            return geojson.load(geojson_file)


if __name__ == "__main__":
    unittest.main()
