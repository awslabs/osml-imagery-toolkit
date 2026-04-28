#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from typing import Optional
from unittest import TestCase

import cv2
import numpy as np

from aws.osml.image_processing.warp_grid import GridBuilder, OcclusionMode, WarpGrid, WarpGridOptions


class TestOcclusionMode(TestCase):
    """Tests for OcclusionMode enum."""

    def test_none_value(self):
        self.assertEqual(OcclusionMode.NONE.value, "none")

    def test_z_buffer_value(self):
        self.assertEqual(OcclusionMode.Z_BUFFER.value, "z_buffer")


class TestWarpGridOptions(TestCase):
    """Tests for WarpGridOptions dataclass and presets."""

    def test_defaults(self):
        opts = WarpGridOptions()
        self.assertEqual(opts.control_points_per_side, 16)
        self.assertEqual(opts.remap_interpolation, cv2.INTER_LINEAR)
        self.assertEqual(opts.occlusion_mode, OcclusionMode.NONE)

    def test_custom_values(self):
        opts = WarpGridOptions(control_points_per_side=8, remap_interpolation=cv2.INTER_CUBIC)
        self.assertEqual(opts.control_points_per_side, 8)
        self.assertEqual(opts.remap_interpolation, cv2.INTER_CUBIC)

    def test_frozen(self):
        opts = WarpGridOptions()
        with self.assertRaises(Exception):
            opts.control_points_per_side = 32

    def test_preset_fast(self):
        self.assertEqual(WarpGridOptions.FAST.control_points_per_side, 4)
        self.assertEqual(WarpGridOptions.FAST.occlusion_mode, OcclusionMode.NONE)

    def test_preset_terrain_corrected(self):
        self.assertEqual(WarpGridOptions.TERRAIN_CORRECTED.control_points_per_side, 16)
        self.assertEqual(WarpGridOptions.TERRAIN_CORRECTED.occlusion_mode, OcclusionMode.NONE)

    def test_preset_visibility_aware(self):
        self.assertEqual(WarpGridOptions.VISIBILITY_AWARE.control_points_per_side, 32)
        self.assertEqual(WarpGridOptions.VISIBILITY_AWARE.occlusion_mode, OcclusionMode.Z_BUFFER)


class TestWarpGrid(TestCase):
    """Tests for WarpGrid frozen dataclass."""

    def test_construction(self):
        map_x = np.zeros((256, 256), dtype=np.float32)
        map_y = np.ones((256, 256), dtype=np.float32)
        valid_mask = np.ones((256, 256), dtype=np.bool_)
        source_bbox = (10, 20, 100, 100)

        grid = WarpGrid(
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid_mask,
            source_bbox=source_bbox,
            source_resolution_level=0,
        )

        np.testing.assert_array_equal(grid.map_x, map_x)
        np.testing.assert_array_equal(grid.map_y, map_y)
        np.testing.assert_array_equal(grid.valid_mask, valid_mask)
        self.assertEqual(grid.source_bbox, (10, 20, 100, 100))
        self.assertEqual(grid.source_resolution_level, 0)

    def test_frozen(self):
        map_x = np.zeros((4, 4), dtype=np.float32)
        map_y = np.zeros((4, 4), dtype=np.float32)
        valid_mask = np.ones((4, 4), dtype=np.bool_)

        grid = WarpGrid(
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid_mask,
            source_bbox=(0, 0, 4, 4),
            source_resolution_level=0,
        )

        with self.assertRaises(Exception):
            grid.source_resolution_level = 1

    def test_shapes_match(self):
        h, w = 512, 1024
        grid = WarpGrid(
            map_x=np.zeros((h, w), dtype=np.float32),
            map_y=np.zeros((h, w), dtype=np.float32),
            valid_mask=np.ones((h, w), dtype=np.bool_),
            source_bbox=(0, 0, w, h),
            source_resolution_level=2,
        )

        self.assertEqual(grid.map_x.shape, (h, w))
        self.assertEqual(grid.map_y.shape, (h, w))
        self.assertEqual(grid.valid_mask.shape, (h, w))


class _ConcreteGridBuilder(GridBuilder):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self, options: WarpGridOptions = WarpGridOptions.TERRAIN_CORRECTED) -> None:
        super().__init__(options)

    @property
    def tile_limits(self):
        return (0, 0, 3, 3)

    @property
    def tile_size(self):
        return (512, 512)

    def build(self, tile_row: int, tile_col: int) -> Optional[WarpGrid]:
        return None


class TestGridBuilder(TestCase):
    """Tests for GridBuilder ABC."""

    def test_concrete_implementation(self):
        builder = _ConcreteGridBuilder()
        self.assertEqual(builder.tile_limits, (0, 0, 3, 3))
        self.assertEqual(builder.tile_size, (512, 512))

    def test_build_returns_none(self):
        builder = _ConcreteGridBuilder()
        self.assertIsNone(builder.build(0, 0))

    def test_z_buffer_raises_not_implemented(self):
        opts = WarpGridOptions(occlusion_mode=OcclusionMode.Z_BUFFER)
        with self.assertRaises(NotImplementedError):
            _ConcreteGridBuilder(options=opts)

    def test_none_occlusion_mode_accepted(self):
        opts = WarpGridOptions(occlusion_mode=OcclusionMode.NONE)
        builder = _ConcreteGridBuilder(options=opts)
        self.assertIsNotNone(builder)

    def test_cannot_instantiate_abc_directly(self):
        with self.assertRaises(TypeError):
            GridBuilder()

    def test_options_stored(self):
        opts = WarpGridOptions(control_points_per_side=8)
        builder = _ConcreteGridBuilder(options=opts)
        self.assertEqual(builder._options.control_points_per_side, 8)
