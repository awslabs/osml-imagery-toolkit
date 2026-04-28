#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import math
from typing import Any, Dict, Optional
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.image_to_image_grid_builder import ImageToImageGridBuilder
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


class TestImageToImageGridBuilderConstruction(TestCase):
    """Tests for ImageToImageGridBuilder initialization."""

    def _make_source_model(self):
        return _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)

    def _make_target_model(self):
        return _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)

    def test_basic_construction(self):
        builder = ImageToImageGridBuilder(
            source_sensor_model=self._make_source_model(),
            target_sensor_model=self._make_target_model(),
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
        )
        self.assertIsNotNone(builder)

    def test_tile_limits(self):
        builder = ImageToImageGridBuilder(
            source_sensor_model=self._make_source_model(),
            target_sensor_model=self._make_target_model(),
            source_width=1000,
            source_height=1000,
            target_width=2048,
            target_height=1536,
            block_width=1024,
            block_height=1024,
        )
        self.assertEqual(builder.tile_limits, (0, 0, 1, 1))

    def test_tile_size(self):
        builder = ImageToImageGridBuilder(
            source_sensor_model=self._make_source_model(),
            target_sensor_model=self._make_target_model(),
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            block_width=512,
            block_height=256,
        )
        self.assertEqual(builder.tile_size, (512, 256))

    def test_options_respected(self):
        builder = ImageToImageGridBuilder(
            source_sensor_model=self._make_source_model(),
            target_sensor_model=self._make_target_model(),
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            options=WarpGridOptions.FAST,
        )
        self.assertEqual(builder._options.control_points_per_side, 4)

    def test_with_elevation_model(self):
        elev = _ConstantElevation(elevation=200.0)
        builder = ImageToImageGridBuilder(
            source_sensor_model=self._make_source_model(),
            target_sensor_model=self._make_target_model(),
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            elevation_model=elev,
        )
        self.assertIsNotNone(builder)


class TestImageToImageGridBuilderBuild(TestCase):
    """Tests for ImageToImageGridBuilder.build() method."""

    def _make_builder(self, **kwargs):
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        defaults = dict(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )
        defaults.update(kwargs)
        return ImageToImageGridBuilder(**defaults)

    def test_build_returns_warp_grid(self):
        builder = self._make_builder()
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.map_x.shape, (256, 256))
        self.assertEqual(grid.map_y.shape, (256, 256))
        self.assertEqual(grid.valid_mask.shape, (256, 256))
        self.assertEqual(grid.map_x.dtype, np.float32)
        self.assertEqual(grid.map_y.dtype, np.float32)

    def test_identity_mapping_produces_near_identity_grid(self):
        """When source and target have the same sensor model, the warp grid
        should map each pixel approximately to itself."""
        builder = self._make_builder()
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        self.assertTrue(grid.valid_mask.all())
        x, y, w, h = grid.source_bbox
        center_map_x = grid.map_x[128, 128]
        center_map_y = grid.map_y[128, 128]
        self.assertAlmostEqual(center_map_x + x, 128, delta=2.0)
        self.assertAlmostEqual(center_map_y + y, 128, delta=2.0)

    def test_shifted_target_produces_offset(self):
        """When the target model is offset from the source, the grid should
        reflect the shift."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.1, origin_lat_deg=47.9, scale_x_deg=0.001, scale_y_deg=0.001)
        builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        x, y, w, h = grid.source_bbox
        center_source_x = grid.map_x[128, 128] + x
        center_source_y = grid.map_y[128, 128] + y
        self.assertAlmostEqual(center_source_x, 228, delta=2.0)
        self.assertAlmostEqual(center_source_y, 228, delta=2.0)

    def test_non_overlapping_returns_none(self):
        """When target covers a region completely outside source, returns None."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=100.0, origin_lat_deg=10.0, scale_x_deg=0.001, scale_y_deg=0.001)
        builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )
        grid = builder.build(0, 0)
        self.assertIsNone(grid)

    def test_valid_mask_marks_valid_pixels(self):
        builder = self._make_builder()
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        self.assertTrue(grid.valid_mask.any())

    def test_source_bbox_within_image(self):
        builder = self._make_builder()
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        x, y, w, h = grid.source_bbox
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_resolution_level_is_zero(self):
        builder = self._make_builder()
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.source_resolution_level, 0)

    def test_with_elevation_model(self):
        elev = _ConstantElevation(elevation=500.0)
        builder = self._make_builder(elevation_model=elev)
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)

    def test_without_elevation_model(self):
        builder = self._make_builder(elevation_model=None)
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)

    def test_various_grid_densities(self):
        for density in [4, 8, 16, 32]:
            builder = self._make_builder(
                block_width=128,
                block_height=128,
                options=WarpGridOptions(control_points_per_side=density),
            )
            grid = builder.build(0, 0)
            self.assertIsNotNone(grid, f"Failed for density={density}")
            self.assertEqual(grid.map_x.shape, (128, 128))

    def test_out_of_bounds_block_returns_none(self):
        builder = self._make_builder()
        grid = builder.build(100, 100)
        self.assertIsNone(grid)


class TestImageToImageGridBuilderEdgeCases(TestCase):
    """Edge case tests for ImageToImageGridBuilder."""

    def test_partial_block_at_edge(self):
        """Last block may be smaller than block_width x block_height."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1000,
            source_height=1000,
            target_width=300,
            target_height=300,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )
        grid = builder.build(1, 1)
        if grid is not None:
            self.assertEqual(grid.map_x.shape, (256, 256))

    def test_different_scale_models(self):
        """Source and target with different GSDs produce valid grids."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.002, scale_y_deg=0.002)
        builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1000,
            source_height=1000,
            target_width=500,
            target_height=500,
            block_width=128,
            block_height=128,
            options=WarpGridOptions(control_points_per_side=8),
        )
        grid = builder.build(0, 0)
        self.assertIsNotNone(grid)
        self.assertTrue(grid.valid_mask.any())

    def test_multiple_blocks_consistent(self):
        """Adjacent blocks should have consistent coverage."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1000,
            source_height=1000,
            target_width=1000,
            target_height=1000,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )
        grid_00 = builder.build(0, 0)
        grid_01 = builder.build(0, 1)
        grid_10 = builder.build(1, 0)
        self.assertIsNotNone(grid_00)
        self.assertIsNotNone(grid_01)
        self.assertIsNotNone(grid_10)
