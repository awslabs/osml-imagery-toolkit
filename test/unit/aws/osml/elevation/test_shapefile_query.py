#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import pickle
from math import radians

import pytest
import shapefile
import shapely

from aws.osml.elevation import ShapefileQuery
from aws.osml.photogrammetry import GeodeticWorldCoordinate


@pytest.fixture
def simple_shapefile(tmp_path):
    """Create a shapefile with a single polygon covering [10,20] x [30,31] degrees."""
    shp_path = str(tmp_path / "test_poly")
    with shapefile.Writer(shp_path) as w:
        w.field("name", "C")
        w.poly([[[10, 30], [20, 30], [20, 31], [10, 31], [10, 30]]])
        w.record("region_a")
    return shp_path + ".shp"


@pytest.fixture
def multi_polygon_shapefile(tmp_path):
    """Create a shapefile with a MULTIPOLYGON geometry."""
    shp_path = str(tmp_path / "test_multi")
    with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as w:
        w.field("name", "C")
        # Two separate polygons as parts of one shape record
        w.poly(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                [[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]],
            ]
        )
        w.record("multi_region")
    return shp_path + ".shp"


@pytest.fixture
def two_polygon_shapefile(tmp_path):
    """Create a shapefile with two separate polygon features in adjacent tiles."""
    shp_path = str(tmp_path / "test_two")
    with shapefile.Writer(shp_path) as w:
        w.field("name", "C")
        # Polygon in tile (5, 10)
        w.poly([[[5.2, 10.2], [5.8, 10.2], [5.8, 10.8], [5.2, 10.8], [5.2, 10.2]]])
        w.record("poly_a")
        # Polygon in tile (6, 10)
        w.poly([[[6.2, 10.2], [6.8, 10.2], [6.8, 10.8], [6.2, 10.8], [6.2, 10.2]]])
        w.record("poly_b")
    return shp_path + ".shp"


class TestShapefileQuery:
    def test_point_inside_polygon_returns_geometry(self, simple_shapefile):
        query = ShapefileQuery(simple_shapefile)
        coord = GeodeticWorldCoordinate([radians(15.0), radians(30.5), 0.0])
        result = query.get_geometry(coord)
        assert result is not None
        assert isinstance(result, shapely.Geometry)

    def test_point_outside_polygon_returns_none(self, simple_shapefile):
        query = ShapefileQuery(simple_shapefile)
        coord = GeodeticWorldCoordinate([radians(5.0), radians(30.5), 0.0])
        result = query.get_geometry(coord)
        assert result is None

    def test_point_outside_tile_returns_none(self, simple_shapefile):
        query = ShapefileQuery(simple_shapefile)
        coord = GeodeticWorldCoordinate([radians(15.0), radians(50.0), 0.0])
        result = query.get_geometry(coord)
        assert result is None

    def test_bbox_filtering_restricts_loaded_features(self, two_polygon_shapefile):
        """Only features overlapping the 1-degree tile should be loaded."""
        query = ShapefileQuery(two_polygon_shapefile)
        # Query in tile (5, 10)
        coord_a = GeodeticWorldCoordinate([radians(5.5), radians(10.5), 0.0])
        result_a = query.get_geometry(coord_a)
        assert result_a is not None

        # Query in tile (6, 10)
        coord_b = GeodeticWorldCoordinate([radians(6.5), radians(10.5), 0.0])
        result_b = query.get_geometry(coord_b)
        assert result_b is not None

        # The cached STRtrees should be different (different tiles)
        tree_a = query._get_geometry(5, 10)
        tree_b = query._get_geometry(6, 10)
        assert len(tree_a.geometries) == 1
        assert len(tree_b.geometries) == 1

    def test_multi_geometry_flattened(self, multi_polygon_shapefile):
        """Multi-part geometries should be flattened into individual geometries."""
        query = ShapefileQuery(multi_polygon_shapefile)
        # Both parts are in tile (0, 0) through (2, 2), query the tile (0, 0)
        tree = query._get_geometry(0, 0)
        # The first polygon [0,0]-[1,1] is fully in tile (0,0)
        assert len(tree.geometries) >= 1

        # Point inside first part
        coord = GeodeticWorldCoordinate([radians(0.5), radians(0.5), 0.0])
        result = query.get_geometry(coord)
        assert result is not None

    def test_multi_geometry_second_part(self, multi_polygon_shapefile):
        """Second part of a multi-polygon should also be queryable."""
        query = ShapefileQuery(multi_polygon_shapefile)
        coord = GeodeticWorldCoordinate([radians(2.5), radians(2.5), 0.0])
        result = query.get_geometry(coord)
        assert result is not None

    def test_lru_cache_returns_same_tree(self, simple_shapefile):
        """Second query to same tile should return cached STRtree."""
        query = ShapefileQuery(simple_shapefile)
        tree1 = query._get_geometry(10, 30)
        tree2 = query._get_geometry(10, 30)
        assert tree1 is tree2

    def test_lru_cache_evicts_old_entries(self, two_polygon_shapefile):
        """LRU cache with size 1 should evict old entries."""
        query = ShapefileQuery(two_polygon_shapefile, geom_cache_size=1)
        tree_a = query._get_geometry(5, 10)
        # Load a different tile to evict the first
        query._get_geometry(6, 10)
        # Re-query the first tile — should be a new object (not cached)
        tree_a2 = query._get_geometry(5, 10)
        assert tree_a is not tree_a2

    def test_pickle_excludes_file_handle(self, simple_shapefile):
        """ShapefileQuery should be picklable; file handle excluded from state."""
        query = ShapefileQuery(simple_shapefile)
        # Trigger reader initialization
        coord = GeodeticWorldCoordinate([radians(15.0), radians(30.5), 0.0])
        query.get_geometry(coord)
        assert query._reader is not None

        # Pickle and unpickle
        data = pickle.dumps(query)
        restored = pickle.loads(data)
        assert restored._reader is None
        assert restored.vector_filepath == simple_shapefile

        # Restored instance should still work
        result = restored.get_geometry(coord)
        assert result is not None

    def test_empty_shapefile_returns_none(self, tmp_path):
        """Query against empty shapefile returns None."""
        shp_path = str(tmp_path / "empty")
        with shapefile.Writer(shp_path) as w:
            w.field("name", "C")
            # No features added
        query = ShapefileQuery(shp_path + ".shp")
        coord = GeodeticWorldCoordinate([radians(0.0), radians(0.0), 0.0])
        result = query.get_geometry(coord)
        assert result is None

    def test_point_on_polygon_boundary(self, simple_shapefile):
        """Point exactly on polygon edge should be found (distance = 0)."""
        query = ShapefileQuery(simple_shapefile)
        # Point on left edge of polygon at x=10
        coord = GeodeticWorldCoordinate([radians(10.0), radians(30.5), 0.0])
        result = query.get_geometry(coord)
        assert result is not None
