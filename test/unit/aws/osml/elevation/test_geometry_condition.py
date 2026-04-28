#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from typing import Optional
from unittest.mock import MagicMock

import shapely

from aws.osml.elevation import GeometryCondition
from aws.osml.photogrammetry import GeodeticWorldCoordinate, GeometryQuery


class _MockGeometryQuery(GeometryQuery):
    """A mock GeometryQuery that returns a geometry for a configured set of coordinates."""

    def __init__(self, has_geometry: bool = True) -> None:
        super().__init__()
        self.has_geometry = has_geometry

    def get_geometry(self, world_coordinate: GeodeticWorldCoordinate) -> Optional[shapely.Geometry]:
        if self.has_geometry:
            return shapely.Point(0, 0)
        return None


class TestGeometryCondition:
    def test_returns_true_when_geometry_found(self):
        query = _MockGeometryQuery(has_geometry=True)
        condition = GeometryCondition(query)
        coord = GeodeticWorldCoordinate([0.5, 0.5, 0.0])
        assert condition.is_true(coord) is True

    def test_returns_false_when_no_geometry(self):
        query = _MockGeometryQuery(has_geometry=False)
        condition = GeometryCondition(query)
        coord = GeodeticWorldCoordinate([0.5, 0.5, 0.0])
        assert condition.is_true(coord) is False

    def test_invert_true_flips_result_when_geometry_found(self):
        query = _MockGeometryQuery(has_geometry=True)
        condition = GeometryCondition(query, invert=True)
        coord = GeodeticWorldCoordinate([0.5, 0.5, 0.0])
        assert condition.is_true(coord) is False

    def test_invert_true_flips_result_when_no_geometry(self):
        query = _MockGeometryQuery(has_geometry=False)
        condition = GeometryCondition(query, invert=True)
        coord = GeodeticWorldCoordinate([0.5, 0.5, 0.0])
        assert condition.is_true(coord) is True

    def test_passes_coordinate_to_query(self):
        query = MagicMock(spec=GeometryQuery)
        query.get_geometry.return_value = None
        condition = GeometryCondition(query)
        coord = GeodeticWorldCoordinate([1.0, 2.0, 100.0])
        condition.is_true(coord)
        query.get_geometry.assert_called_once_with(coord)
