#  Copyright 2024 Amazon.com, Inc. or its affiliates.

import math
from typing import Any, Dict, Optional
from unittest import TestCase

import pyproj

from aws.osml.image_processing.map_tileset import MapTileBounds, MapTileId
from aws.osml.image_processing.projected_image_tileset import ProjectedImageTileSet
from aws.osml.photogrammetry import ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, SensorModel
from aws.osml.photogrammetry.elevation_model import ElevationRegionSummary


class _AffineSensorModel(SensorModel):
    """Simple affine sensor model for testing."""

    def __init__(self, origin_lon_deg: float, origin_lat_deg: float, scale_x_deg: float, scale_y_deg: float) -> None:
        super().__init__()
        self._origin_lon = math.radians(origin_lon_deg)
        self._origin_lat = math.radians(origin_lat_deg)
        self._scale_x = math.radians(scale_x_deg)
        self._scale_y = math.radians(scale_y_deg)

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        lon = self._origin_lon + image_coordinate.x * self._scale_x
        lat = self._origin_lat - image_coordinate.y * self._scale_y
        world = GeodeticWorldCoordinate([lon, lat, 0.0])
        if elevation_model:
            elevation_model.set_elevation(world)
        return world

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        x = (world_coordinate.longitude - self._origin_lon) / self._scale_x
        y = (self._origin_lat - world_coordinate.latitude) / self._scale_y
        return ImageCoordinate([x, y])


class _ConstantElevation(ElevationModel):
    """Elevation model returning a constant value."""

    def __init__(self, elevation: float = 100.0) -> None:
        super().__init__()
        self._elevation = elevation

    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        world_coordinate.elevation = self._elevation
        return True

    def describe_region(self, world_coordinate: GeodeticWorldCoordinate) -> Optional[ElevationRegionSummary]:
        return ElevationRegionSummary(
            min_elevation=self._elevation,
            max_elevation=self._elevation,
            no_data_value=-32767,
            post_spacing=30.0,
        )


class TestProjectedImageTileSetConstruction(TestCase):
    """Tests for ProjectedImageTileSet factory construction."""

    def _make_sensor_model(self):
        return _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)

    def test_from_sensor_model_default_crs(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
        )
        self.assertIsNotNone(ts)
        self.assertEqual(ts.crs_id, "EPSG:4326")

    def test_from_sensor_model_utm_crs(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(32636),
            gsd=100.0,
        )
        self.assertIsNotNone(ts)
        self.assertEqual(ts.crs_id, "EPSG:32636")

    def test_from_sensor_model_auto_gsd(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
        )
        self.assertGreater(ts._gsd, 0)

    def test_from_sensor_model_explicit_gsd(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            gsd=0.005,
        )
        self.assertAlmostEqual(ts._gsd, 0.005)

    def test_from_sensor_model_custom_block_size(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(512, 256),
        )
        self.assertEqual(ts._block_width, 512)
        self.assertEqual(ts._block_height, 256)

    def test_from_sensor_model_with_elevation(self):
        sm = self._make_sensor_model()
        elev = _ConstantElevation(200.0)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            elevation_model=elev,
        )
        self.assertIsNotNone(ts)

    def test_from_sensor_model_explicit_num_levels(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            num_levels=3,
        )
        self.assertEqual(ts.num_tile_matrices, 3)


class TestProjectedImageTileSetCrsId(TestCase):
    """Tests for crs_id property."""

    def _make_sensor_model(self):
        return _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)

    def test_crs_id_epsg_4326(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(4326),
        )
        self.assertEqual(ts.crs_id, "EPSG:4326")

    def test_crs_id_epsg_3857(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(3857),
            gsd=100.0,
        )
        self.assertEqual(ts.crs_id, "EPSG:3857")

    def test_crs_id_utm(self):
        sm = self._make_sensor_model()
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(32637),
            gsd=100.0,
        )
        self.assertEqual(ts.crs_id, "EPSG:32637")


class TestProjectedImageTileSetMultiLevel(TestCase):
    """Tests for multi-level tile matrix support."""

    def _make_tileset(self, grid_cols=4, grid_rows=4, num_levels=None, block_size=(1024, 1024)):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        gsd = 0.001 * 1000 / block_size[0]
        return ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=int(grid_cols * block_size[0]),
            source_height=int(grid_rows * block_size[1]),
            gsd=gsd,
            block_size=block_size,
            num_levels=num_levels,
        )

    def test_level_0_dimensions(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=4096,
            source_height=2048,
            gsd=0.001,
            block_size=(1024, 1024),
        )
        cols, rows = ts.tile_matrix_dimensions(0)
        self.assertEqual(cols, 4)
        self.assertEqual(rows, 2)

    def test_level_n_halving(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=8192,
            source_height=8192,
            gsd=0.001,
            block_size=(1024, 1024),
            num_levels=5,
        )
        cols_0, rows_0 = ts.tile_matrix_dimensions(0)
        self.assertEqual(cols_0, 8)
        self.assertEqual(rows_0, 8)

        cols_1, rows_1 = ts.tile_matrix_dimensions(1)
        self.assertEqual(cols_1, 4)
        self.assertEqual(rows_1, 4)

        cols_2, rows_2 = ts.tile_matrix_dimensions(2)
        self.assertEqual(cols_2, 2)
        self.assertEqual(rows_2, 2)

        cols_3, rows_3 = ts.tile_matrix_dimensions(3)
        self.assertEqual(cols_3, 1)
        self.assertEqual(rows_3, 1)

    def test_auto_num_levels_single_tile_at_coarsest(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=8192,
            source_height=8192,
            gsd=0.001,
            block_size=(1024, 1024),
        )
        coarsest = ts.num_tile_matrices - 1
        cols, rows = ts.tile_matrix_dimensions(coarsest)
        self.assertEqual(cols, 1)
        self.assertEqual(rows, 1)

    def test_auto_num_levels_formula(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=8192,
            source_height=8192,
            gsd=0.001,
            block_size=(1024, 1024),
        )
        expected = math.ceil(math.log2(8)) + 1
        self.assertEqual(ts.num_tile_matrices, expected)

    def test_odd_grid_dimensions_halving(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=5120,
            source_height=3072,
            gsd=0.001,
            block_size=(1024, 1024),
            num_levels=4,
        )
        cols_0, rows_0 = ts.tile_matrix_dimensions(0)
        self.assertEqual(cols_0, 5)
        self.assertEqual(rows_0, 3)

        cols_1, rows_1 = ts.tile_matrix_dimensions(1)
        self.assertEqual(cols_1, 3)
        self.assertEqual(rows_1, 2)

        cols_2, rows_2 = ts.tile_matrix_dimensions(2)
        self.assertEqual(cols_2, 2)
        self.assertEqual(rows_2, 1)

    def test_non_square_block_size(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=2048,
            source_height=1024,
            gsd=0.001,
            block_size=(512, 256),
            num_levels=3,
        )
        cols_0, rows_0 = ts.tile_matrix_dimensions(0)
        self.assertEqual(cols_0, 4)
        self.assertEqual(rows_0, 4)


class TestProjectedImageTileSetGetTile(TestCase):
    """Tests for get_tile() method."""

    def _make_tileset_4326(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        return ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=4096,
            source_height=4096,
            gsd=0.001,
            block_size=(1024, 1024),
            num_levels=3,
        )

    def _make_tileset_utm(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        return ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(32636),
            gsd=100.0,
            block_size=(256, 256),
            num_levels=3,
        )

    def test_tile_has_native_bounds(self):
        ts = self._make_tileset_4326()
        tile = ts.get_tile(MapTileId(0, 0, 0))
        self.assertEqual(len(tile.native_bounds), 4)
        xmin, ymin, xmax, ymax = tile.native_bounds
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

    def test_tile_has_wgs84_bounds(self):
        ts = self._make_tileset_4326()
        tile = ts.get_tile(MapTileId(0, 0, 0))
        self.assertIsInstance(tile.bounds, MapTileBounds)
        self.assertLess(tile.bounds.min_lon, tile.bounds.max_lon)
        self.assertLess(tile.bounds.min_lat, tile.bounds.max_lat)

    def test_tile_size_correct(self):
        ts = self._make_tileset_4326()
        tile = ts.get_tile(MapTileId(0, 0, 0))
        self.assertEqual(tile.size.width, 1024)
        self.assertEqual(tile.size.height, 1024)

    def test_tile_id_preserved(self):
        ts = self._make_tileset_4326()
        tile_id = MapTileId(1, 2, 1)
        tile = ts.get_tile(tile_id)
        self.assertEqual(tile.id, tile_id)

    def test_adjacent_tiles_share_edges(self):
        ts = self._make_tileset_4326()
        tile_00 = ts.get_tile(MapTileId(0, 0, 0))
        tile_01 = ts.get_tile(MapTileId(0, 0, 1))
        tile_10 = ts.get_tile(MapTileId(0, 1, 0))

        self.assertAlmostEqual(tile_00.native_bounds[2], tile_01.native_bounds[0], places=10)
        self.assertAlmostEqual(tile_00.native_bounds[1], tile_10.native_bounds[3], places=10)

    def test_level_0_tiles_cover_footprint(self):
        ts = self._make_tileset_4326()
        cols, rows = ts.tile_matrix_dimensions(0)
        first = ts.get_tile(MapTileId(0, 0, 0))
        last = ts.get_tile(MapTileId(0, rows - 1, cols - 1))

        total_xmin = first.native_bounds[0]
        total_ymax = first.native_bounds[3]
        total_xmax = last.native_bounds[2]
        total_ymin = last.native_bounds[1]

        x_extent = total_xmax - total_xmin
        y_extent = total_ymax - total_ymin
        self.assertGreater(x_extent, 0)
        self.assertGreater(y_extent, 0)

    def test_utm_native_bounds_in_meters(self):
        ts = self._make_tileset_utm()
        tile = ts.get_tile(MapTileId(0, 0, 0))
        xmin, ymin, xmax, ymax = tile.native_bounds
        self.assertGreater(xmax - xmin, 1000)
        self.assertGreater(ymax - ymin, 1000)

    def test_utm_wgs84_bounds_in_radians(self):
        ts = self._make_tileset_utm()
        tile = ts.get_tile(MapTileId(0, 0, 0))
        self.assertGreater(tile.bounds.min_lon, 0)
        self.assertLess(tile.bounds.max_lon, math.pi)
        self.assertGreater(tile.bounds.min_lat, 0)
        self.assertLess(tile.bounds.max_lat, math.pi / 2)

    def test_coarser_level_tiles_larger_in_crs_units(self):
        ts = self._make_tileset_4326()
        tile_l0 = ts.get_tile(MapTileId(0, 0, 0))
        tile_l1 = ts.get_tile(MapTileId(1, 0, 0))

        span_l0_x = tile_l0.native_bounds[2] - tile_l0.native_bounds[0]
        span_l1_x = tile_l1.native_bounds[2] - tile_l1.native_bounds[0]
        self.assertAlmostEqual(span_l1_x, span_l0_x * 2, places=10)


class TestProjectedImageTileSetGetTileForLocation(TestCase):
    """Tests for get_tile_for_location() method."""

    def _make_tileset(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        return ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=4096,
            source_height=4096,
            gsd=0.001,
            block_size=(1024, 1024),
            num_levels=3,
        )

    def test_origin_maps_to_first_tile(self):
        ts = self._make_tileset()
        origin_world = GeodeticWorldCoordinate([math.radians(35.0), math.radians(48.0), 0.0])
        tile = ts.get_tile_for_location(origin_world, 0)
        self.assertEqual(tile.id.tile_row, 0)
        self.assertEqual(tile.id.tile_col, 0)

    def test_location_inside_footprint(self):
        ts = self._make_tileset()
        mid_world = GeodeticWorldCoordinate([math.radians(35.5), math.radians(47.5), 0.0])
        tile = ts.get_tile_for_location(mid_world, 0)
        self.assertGreaterEqual(tile.id.tile_row, 0)
        self.assertGreaterEqual(tile.id.tile_col, 0)

    def test_location_at_different_levels(self):
        ts = self._make_tileset()
        world = GeodeticWorldCoordinate([math.radians(35.5), math.radians(47.5), 0.0])
        tile_l0 = ts.get_tile_for_location(world, 0)
        tile_l1 = ts.get_tile_for_location(world, 1)
        self.assertGreaterEqual(tile_l0.id.tile_col, tile_l1.id.tile_col)


class TestProjectedImageTileSetLimitsForArea(TestCase):
    """Tests for get_tile_matrix_limits_for_area() method."""

    def _make_tileset(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        return ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=4096,
            source_height=4096,
            gsd=0.001,
            block_size=(1024, 1024),
            num_levels=3,
        )

    def test_returns_row_first_ordering(self):
        ts = self._make_tileset()
        corners = [
            GeodeticWorldCoordinate([math.radians(35.0), math.radians(48.0), 0.0]),
            GeodeticWorldCoordinate([math.radians(36.0), math.radians(48.0), 0.0]),
            GeodeticWorldCoordinate([math.radians(36.0), math.radians(47.0), 0.0]),
            GeodeticWorldCoordinate([math.radians(35.0), math.radians(47.0), 0.0]),
        ]
        min_row, min_col, max_row, max_col = ts.get_tile_matrix_limits_for_area(corners, 0)
        self.assertLessEqual(min_row, max_row)
        self.assertLessEqual(min_col, max_col)

    def test_full_extent_covers_all_tiles(self):
        ts = self._make_tileset()
        corners = [
            GeodeticWorldCoordinate([math.radians(35.0), math.radians(48.0), 0.0]),
            GeodeticWorldCoordinate([math.radians(39.096), math.radians(48.0), 0.0]),
            GeodeticWorldCoordinate([math.radians(39.096), math.radians(43.904), 0.0]),
            GeodeticWorldCoordinate([math.radians(35.0), math.radians(43.904), 0.0]),
        ]
        min_row, min_col, max_row, max_col = ts.get_tile_matrix_limits_for_area(corners, 0)
        cols, rows = ts.tile_matrix_dimensions(0)
        self.assertEqual(min_row, 0)
        self.assertEqual(min_col, 0)
        self.assertEqual(max_row, rows - 1)
        self.assertEqual(max_col, cols - 1)


class TestProjectedImageTileSetTileMatrixSetId(TestCase):
    """Tests for tile_matrix_set_id property."""

    def test_contains_crs_id(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        ts = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
        )
        self.assertIn("EPSG:4326", ts.tile_matrix_set_id)
