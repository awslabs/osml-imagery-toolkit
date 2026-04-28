#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import unittest
from typing import Any, Optional, Tuple
from unittest.mock import patch

import numpy as np

from aws.osml.elevation import ElevationModelBuilder, GeometryCondition, StoredDEMTileFactory
from aws.osml.photogrammetry import (
    ConditionalElevationModel,
    ConstantElevationModel,
    DigitalElevationModel,
    DigitalElevationModelTileFactory,
    DigitalElevationModelTileSet,
    ElevationModelCondition,
    ElevationRegionSummary,
    GeodeticWorldCoordinate,
    GeometryQuery,
    MultiElevationModel,
    NormalizedElevationModel,
    OffsetElevationModel,
    SensorModel,
)


class _StubTileSet(DigitalElevationModelTileSet):
    def find_tile_id(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[str]:
        return "stub_tile.tif"


class _StubTileFactory(DigitalElevationModelTileFactory):
    def get_tile(self, tile_path: str) -> Tuple[Optional[Any], Optional[SensorModel], Optional[ElevationRegionSummary]]:
        return None, None, None


class _StubCondition(ElevationModelCondition):
    def __init__(self, value: bool = True):
        super().__init__()
        self.value = value

    def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        return self.value


class _StubGeometryQuery(GeometryQuery):
    def get_geometry(self, world_coordinate: GeodeticWorldCoordinate):
        return None


class TestElevationModelBuilder(unittest.TestCase):
    def test_empty_builder_raises(self):
        builder = ElevationModelBuilder()
        with self.assertRaises(ValueError):
            builder.build()

    def test_single_source_no_multi_wrapper(self):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet())
        model = builder.build(normalize=False)
        self.assertIsInstance(model, DigitalElevationModel)

    def test_single_source_with_normalize(self):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet())
        model = builder.build(normalize=True)
        self.assertIsInstance(model, NormalizedElevationModel)
        self.assertIsInstance(model.inner_elevation_model, DigitalElevationModel)

    def test_multiple_sources_produces_multi_model(self):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet())
        builder.add_source(_StubTileFactory(), _StubTileSet())
        model = builder.build(normalize=False)
        self.assertIsInstance(model, MultiElevationModel)
        self.assertEqual(len(model.elevation_models), 2)

    def test_source_with_condition_produces_conditional_wrapper(self):
        builder = ElevationModelBuilder()
        condition = _StubCondition(True)
        builder.add_source(_StubTileFactory(), _StubTileSet(), condition=condition)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, ConditionalElevationModel)
        self.assertIsInstance(model.inner_elevation_model, DigitalElevationModel)
        self.assertIs(model.em_condition, condition)

    def test_source_with_geometry_query_wraps_in_geometry_condition(self):
        builder = ElevationModelBuilder()
        query = _StubGeometryQuery()
        builder.add_source(_StubTileFactory(), _StubTileSet(), condition=query)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, ConditionalElevationModel)
        self.assertIsInstance(model.em_condition, GeometryCondition)
        self.assertIs(model.em_condition.geometry_query, query)
        self.assertFalse(model.em_condition.invert)

    def test_source_with_geometry_query_inverted(self):
        builder = ElevationModelBuilder()
        query = _StubGeometryQuery()
        builder.add_source(_StubTileFactory(), _StubTileSet(), condition=query, invert_condition=True)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, ConditionalElevationModel)
        self.assertIsInstance(model.em_condition, GeometryCondition)
        self.assertTrue(model.em_condition.invert)

    @patch("aws.osml.elevation.builder.RasterOffsetProvider")
    def test_with_geoid_wraps_in_offset_model(self, mock_provider_cls):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet())
        builder.with_geoid("/path/to/geoid.tif", scale_factor=2.0)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, OffsetElevationModel)
        mock_provider_cls.assert_called_once_with("/path/to/geoid.tif", 2.0)

    @patch("aws.osml.elevation.builder.RasterOffsetProvider")
    def test_with_geoid_and_normalize(self, mock_provider_cls):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet())
        builder.with_geoid("/path/to/geoid.tif")
        model = builder.build(normalize=True)
        self.assertIsInstance(model, NormalizedElevationModel)
        self.assertIsInstance(model.inner_elevation_model, OffsetElevationModel)

    def test_add_fallback(self):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet())
        builder.add_fallback(elevation=42.0)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, MultiElevationModel)
        self.assertEqual(len(model.elevation_models), 2)
        self.assertIsInstance(model.elevation_models[1], ConstantElevationModel)
        self.assertEqual(model.elevation_models[1].constant_elevation, 42.0)

    def test_add_elevation_model(self):
        builder = ElevationModelBuilder()
        custom_model = ConstantElevationModel(99.0)
        builder.add_elevation_model(custom_model)
        model = builder.build(normalize=False)
        self.assertIs(model, custom_model)

    def test_add_elevation_model_with_condition(self):
        builder = ElevationModelBuilder()
        custom_model = ConstantElevationModel(99.0)
        condition = _StubCondition(True)
        builder.add_elevation_model(custom_model, condition=condition)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, ConditionalElevationModel)
        self.assertIs(model.inner_elevation_model, custom_model)

    def test_fluent_chaining(self):
        builder = ElevationModelBuilder()
        result = builder.add_source(_StubTileFactory(), _StubTileSet()).add_fallback()
        self.assertIs(result, builder)

    def test_normalize_false_skips_normalization(self):
        builder = ElevationModelBuilder()
        builder.add_fallback(0.0)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, ConstantElevationModel)
        self.assertNotIsInstance(model, NormalizedElevationModel)

    def test_multiple_sources_with_mixed_conditions(self):
        builder = ElevationModelBuilder()
        builder.add_source(_StubTileFactory(), _StubTileSet(), condition=_StubCondition(True))
        builder.add_source(_StubTileFactory(), _StubTileSet())
        builder.add_fallback(0.0)
        model = builder.build(normalize=False)
        self.assertIsInstance(model, MultiElevationModel)
        self.assertEqual(len(model.elevation_models), 3)
        self.assertIsInstance(model.elevation_models[0], ConditionalElevationModel)
        self.assertIsInstance(model.elevation_models[1], DigitalElevationModel)
        self.assertIsInstance(model.elevation_models[2], ConstantElevationModel)


class TestElevationModelBuilderEndToEnd(unittest.TestCase):
    """End-to-end test using real DEM data from data/unit/."""

    def test_builder_produced_model_sets_elevation(self):
        tile_set = _FixedTileSet("dem_chip_10x10.tif")
        factory = StoredDEMTileFactory("data/unit")
        model = ElevationModelBuilder().add_source(factory, tile_set).add_fallback(0.0).build()

        coord = GeodeticWorldCoordinate([np.radians(-77.955), np.radians(38.955), 0.0])
        success = model.set_elevation(coord)
        self.assertTrue(success)
        self.assertNotEqual(coord.elevation, 0.0)

    def test_builder_fallback_used_when_dem_misses(self):
        tile_set = _EmptyTileSet()
        factory = StoredDEMTileFactory("data/unit")
        model = ElevationModelBuilder().add_source(factory, tile_set).add_fallback(42.0).build()

        coord = GeodeticWorldCoordinate([np.radians(10.0), np.radians(50.0), 0.0])
        success = model.set_elevation(coord)
        self.assertTrue(success)
        self.assertAlmostEqual(coord.elevation, 42.0)


class _FixedTileSet(DigitalElevationModelTileSet):
    def __init__(self, tile_id: str):
        super().__init__()
        self.tile_id = tile_id

    def find_tile_id(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[str]:
        return self.tile_id


class _EmptyTileSet(DigitalElevationModelTileSet):
    def find_tile_id(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[str]:
        return None


if __name__ == "__main__":
    unittest.main()
