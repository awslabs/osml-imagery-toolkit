#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest

from aws.osml.photogrammetry import (
    ConditionalElevationModel,
    ConstantElevationModel,
    ConstantOffsetProvider,
    EMConditionFalse,
    EMConditionTrue,
    GeodeticWorldCoordinate,
    MultiElevationModel,
    OffsetElevationModel,
)


class TestMultiElevationModelDescribeRegion(unittest.TestCase):
    def test_returns_first_non_none(self):
        empty = MultiElevationModel([])
        constant = ConstantElevationModel(100.0)
        model = MultiElevationModel([empty, constant])

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        summary = model.describe_region(coord)

        self.assertIsNotNone(summary)
        self.assertEqual(summary.min_elevation, 100.0)
        self.assertEqual(summary.max_elevation, 100.0)

    def test_returns_none_when_all_models_return_none(self):
        empty1 = MultiElevationModel([])
        empty2 = MultiElevationModel([])
        model = MultiElevationModel([empty1, empty2])

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        self.assertIsNone(model.describe_region(coord))

    def test_first_model_wins(self):
        first = ConstantElevationModel(50.0)
        second = ConstantElevationModel(200.0)
        model = MultiElevationModel([first, second])

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        summary = model.describe_region(coord)

        self.assertIsNotNone(summary)
        self.assertEqual(summary.min_elevation, 50.0)


class TestOffsetElevationModelDescribeRegion(unittest.TestCase):
    def test_adjusts_min_max_by_offset(self):
        inner = ConstantElevationModel(100.0)
        offset_provider = ConstantOffsetProvider(25.0)
        model = OffsetElevationModel(inner, offset_provider)

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        summary = model.describe_region(coord)

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.min_elevation, 125.0)
        self.assertAlmostEqual(summary.max_elevation, 125.0)
        self.assertEqual(summary.no_data_value, -32767)
        self.assertEqual(summary.post_spacing, 30.0)

    def test_returns_none_when_inner_returns_none(self):
        inner = MultiElevationModel([])
        offset_provider = ConstantOffsetProvider(25.0)
        model = OffsetElevationModel(inner, offset_provider)

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        self.assertIsNone(model.describe_region(coord))

    def test_negative_offset(self):
        inner = ConstantElevationModel(100.0)
        offset_provider = ConstantOffsetProvider(-10.0)
        model = OffsetElevationModel(inner, offset_provider)

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        summary = model.describe_region(coord)

        self.assertAlmostEqual(summary.min_elevation, 90.0)
        self.assertAlmostEqual(summary.max_elevation, 90.0)


class TestConditionalElevationModelDescribeRegion(unittest.TestCase):
    def test_delegates_when_condition_true(self):
        inner = ConstantElevationModel(100.0)
        model = ConditionalElevationModel(inner, EMConditionTrue())

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        summary = model.describe_region(coord)

        self.assertIsNotNone(summary)
        self.assertEqual(summary.min_elevation, 100.0)

    def test_returns_none_when_condition_false(self):
        inner = ConstantElevationModel(100.0)
        model = ConditionalElevationModel(inner, EMConditionFalse())

        coord = GeodeticWorldCoordinate([0.0, 0.5, 0.0])
        self.assertIsNone(model.describe_region(coord))


if __name__ == "__main__":
    unittest.main()
