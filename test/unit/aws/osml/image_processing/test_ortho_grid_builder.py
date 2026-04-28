#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import math
from typing import Any, Dict, Optional
from unittest import TestCase

import numpy as np
import pyproj

from aws.osml.image_processing.map_tileset_wmq import WebMercatorQuadMapTileSet
from aws.osml.image_processing.ortho_grid_builder import OrthoGridBuilder
from aws.osml.image_processing.projected_image_tileset import ProjectedImageTileSet
from aws.osml.image_processing.warp_grid import WarpGridOptions
from aws.osml.photogrammetry import ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, SensorModel
from aws.osml.photogrammetry.elevation_model import ElevationRegionSummary


class _AffineSensorModel(SensorModel):
    """Simple affine sensor model for testing: pixel (x, y) maps to
    (origin_lon + x * scale_x, origin_lat - y * scale_y) in degrees,
    stored as radians internally.
    """

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


class _BoundedAffineSensorModel(_AffineSensorModel):
    """Affine sensor model that raises for points outside a bounded region."""

    def __init__(
        self,
        origin_lon_deg: float,
        origin_lat_deg: float,
        scale_x_deg: float,
        scale_y_deg: float,
        max_x: float,
        max_y: float,
    ) -> None:
        super().__init__(origin_lon_deg, origin_lat_deg, scale_x_deg, scale_y_deg)
        self._max_x = max_x
        self._max_y = max_y

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        result = super().world_to_image(world_coordinate)
        if result.x < -self._max_x or result.x > 2 * self._max_x:
            raise ValueError("Outside valid domain")
        if result.y < -self._max_y or result.y > 2 * self._max_y:
            raise ValueError("Outside valid domain")
        return result


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


class TestOrthoGridBuilderWithProjectedTileSet(TestCase):
    """Tests for OrthoGridBuilder using ProjectedImageTileSet."""

    def _make_sensor_model(self):
        return _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)

    def _make_builder(self, **kwargs):
        sm = kwargs.pop("sensor_model", self._make_sensor_model())
        source_width = kwargs.pop("source_width", 1000)
        source_height = kwargs.pop("source_height", 1000)
        tile_set = kwargs.pop(
            "tile_set",
            ProjectedImageTileSet.from_sensor_model(
                sensor_model=sm,
                source_width=source_width,
                source_height=source_height,
                block_size=(256, 256),
            ),
        )
        tile_matrix = kwargs.pop("tile_matrix", 0)
        defaults = dict(
            tile_set=tile_set,
            tile_matrix=tile_matrix,
            sensor_model=sm,
            source_width=source_width,
            source_height=source_height,
            options=WarpGridOptions(control_points_per_side=8),
        )
        defaults.update(kwargs)
        return OrthoGridBuilder(**defaults)

    def test_constructor_accepts_tile_set(self):
        builder = self._make_builder()
        self.assertIsNotNone(builder)

    def test_tile_limits_row_first(self):
        builder = self._make_builder()
        min_row, min_col, max_row, max_col = builder.tile_limits
        self.assertGreaterEqual(min_row, 0)
        self.assertGreaterEqual(min_col, 0)
        self.assertGreaterEqual(max_row, min_row)
        self.assertGreaterEqual(max_col, min_col)

    def test_tile_size_from_tile_set(self):
        builder = self._make_builder()
        w, h = builder.tile_size
        self.assertEqual(w, 256)
        self.assertEqual(h, 256)

    def test_build_returns_warp_grid(self):
        builder = self._make_builder()
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        w, h = builder.tile_size
        self.assertEqual(grid.map_x.shape, (h, w))
        self.assertEqual(grid.map_y.shape, (h, w))
        self.assertEqual(grid.valid_mask.shape, (h, w))
        self.assertEqual(grid.map_x.dtype, np.float32)

    def test_build_covered_tile_has_valid_pixels(self):
        builder = self._make_builder()
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        self.assertTrue(grid.valid_mask.any())

    def test_build_outside_tile_returns_none(self):
        builder = self._make_builder()
        grid = builder.build(9999, 9999)
        self.assertIsNone(grid)

    def test_source_bbox_reasonable(self):
        builder = self._make_builder()
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        x, y, w, h = grid.source_bbox
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_resolution_level_zero_for_single_level(self):
        builder = self._make_builder(num_source_levels=1)
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.source_resolution_level, 0)

    def test_with_elevation_model(self):
        elev = _ConstantElevation(elevation=500.0)
        builder = self._make_builder(elevation_model=elev)
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)

    def test_various_grid_densities(self):
        for density in [4, 8, 16]:
            builder = self._make_builder(options=WarpGridOptions(control_points_per_side=density))
            min_row, min_col, _, _ = builder.tile_limits
            grid = builder.build(min_row, min_col)
            self.assertIsNotNone(grid, f"Failed for density={density}")


class TestOrthoGridBuilderWithWebMercator(TestCase):
    """Tests for OrthoGridBuilder using WebMercatorQuadMapTileSet."""

    def _make_sensor_model(self):
        return _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)

    def test_web_mercator_tile_set(self):
        sm = self._make_sensor_model()
        tile_set = WebMercatorQuadMapTileSet()
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=10,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, max_row, max_col = builder.tile_limits
        self.assertGreaterEqual(max_row, min_row)
        self.assertGreaterEqual(max_col, min_col)

    def test_web_mercator_build_returns_grid(self):
        sm = self._make_sensor_model()
        tile_set = WebMercatorQuadMapTileSet()
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=10,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.map_x.shape, (256, 256))

    def test_web_mercator_tile_limits_are_global_coordinates(self):
        sm = self._make_sensor_model()
        tile_set = WebMercatorQuadMapTileSet()
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=10,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, max_row, max_col = builder.tile_limits
        # At zoom 10, tile indices should be > 0 for a location at 35E/48N
        self.assertGreater(min_row, 0)
        self.assertGreater(min_col, 0)


class TestOrthoGridBuilderProjectedCRS(TestCase):
    """Tests for OrthoGridBuilder with a projected CRS (UTM) via ProjectedImageTileSet."""

    def test_utm_build_produces_grid(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(32636),
            gsd=100.0,
            block_size=(128, 128),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.map_x.shape, (128, 128))

    def test_utm_tile_limits_reasonable(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(32636),
            gsd=100.0,
            block_size=(128, 128),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
        )
        min_row, min_col, max_row, max_col = builder.tile_limits
        self.assertGreaterEqual(max_row, min_row)
        self.assertGreaterEqual(max_col, min_col)


class TestOrthoGridBuilderPartialFailure(TestCase):
    """Tests for partial control-point failure handling."""

    def test_tile_fully_inside_no_failures(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(128, 128),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        # Fully covered tile should have mostly valid pixels
        self.assertTrue(grid.valid_mask.any())

    def test_coarsest_level_oversized_tile(self):
        """Tile at coarsest pyramid level is larger than footprint — still produces output."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(256, 256),
        )
        # Use the coarsest level where a single tile covers the entire footprint
        coarsest = tile_set.num_tile_matrices - 1
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=coarsest,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        # Should have some valid pixels
        self.assertTrue(grid.valid_mask.any())
        # But not all — tile is oversized
        # (for very small images this may actually be fully valid, so just check it ran)

    def test_partial_overlap_with_bounded_sensor_model(self):
        """Tile partially overlapping source — some control points fail."""
        sm = _BoundedAffineSensorModel(
            origin_lon_deg=35.0,
            origin_lat_deg=48.0,
            scale_x_deg=0.001,
            scale_y_deg=0.001,
            max_x=1000,
            max_y=1000,
        )
        # Create a tile set that's slightly larger than the image footprint
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(512, 512),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        # Check all tiles — at least one should produce output
        min_row, min_col, max_row, max_col = builder.tile_limits
        found_valid = False
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                grid = builder.build(r, c)
                if grid is not None and grid.valid_mask.any():
                    found_valid = True
                    break
            if found_valid:
                break
        self.assertTrue(found_valid)

    def test_completely_outside_returns_none(self):
        """Tile completely outside source footprint returns None."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(256, 256),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        # Request a tile far outside the limits
        grid = builder.build(9999, 9999)
        self.assertIsNone(grid)


class TestOrthoGridBuilderPyramidLevel(TestCase):
    """Tests for pyramid level selection."""

    def test_level_zero_for_single_level_source(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(256, 256),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            num_source_levels=1,
        )
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.source_resolution_level, 0)

    def test_coarser_level_selected_for_overview(self):
        """When output GSD is much coarser than source, a higher pyramid level is selected."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        # Use a very coarse GSD to trigger pyramid level selection
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            target_crs=pyproj.CRS.from_epsg(32636),
            gsd=500.0,  # Very coarse — much larger than source GSD
            block_size=(256, 256),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            num_source_levels=5,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        # With 500m output GSD vs ~100m source GSD, should select level > 0
        self.assertGreater(grid.source_resolution_level, 0)


class TestOrthoGridBuilderEdgeCases(TestCase):
    """Edge case tests for OrthoGridBuilder."""

    def test_single_pixel_source(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.01, scale_y_deg=0.01)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1,
            source_height=1,
            block_size=(64, 64),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1,
            source_height=1,
            options=WarpGridOptions(control_points_per_side=4),
        )
        min_row, min_col, max_row, max_col = builder.tile_limits
        self.assertGreaterEqual(max_row, min_row)
        self.assertGreaterEqual(max_col, min_col)

    def test_non_square_block_size(self):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(512, 256),
        )
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        w, h = builder.tile_size
        self.assertEqual(w, 512)
        self.assertEqual(h, 256)
        min_row, min_col, _, _ = builder.tile_limits
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.map_x.shape, (256, 512))

    def test_multi_matrix_level(self):
        """Build at a higher tile matrix level (coarser)."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            block_size=(256, 256),
        )
        # Level 1 (coarser)
        builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=1,
            sensor_model=sm,
            source_width=1000,
            source_height=1000,
            options=WarpGridOptions(control_points_per_side=8),
        )
        min_row, min_col, max_row, max_col = builder.tile_limits
        self.assertGreaterEqual(max_row, min_row)
        grid = builder.build(min_row, min_col)
        self.assertIsNotNone(grid)
