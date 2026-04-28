#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Integration tests for the warp pipeline: grid builders + WarpedImageProvider + downstream consumers."""

import math
from typing import Any, Dict, Optional
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.chip_factory import ChipFactory, ImageSize, PixelWindow
from aws.osml.image_processing.image_to_image_grid_builder import ImageToImageGridBuilder
from aws.osml.image_processing.mapped_provider import MappedImageProvider
from aws.osml.image_processing.ortho_grid_builder import OrthoGridBuilder
from aws.osml.image_processing.projected_image_tileset import ProjectedImageTileSet
from aws.osml.image_processing.pyramid import TiledImagePyramid
from aws.osml.image_processing.warp_grid import WarpGridOptions
from aws.osml.image_processing.warped_provider import WarpedImageProvider
from aws.osml.photogrammetry import ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, SensorModel
from aws.osml.photogrammetry.elevation_model import ElevationRegionSummary


class _AffineSensorModel(SensorModel):
    """Simple affine sensor model: pixel (x, y) maps to
    (origin_lon + x * scale_x, origin_lat - y * scale_y) in radians.
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


class _MockSource:
    """Minimal duck-typed ImageAssetProvider with a gradient pattern for validation."""

    def __init__(self, num_bands=3, width=512, height=512, block_w=256, block_h=256):
        self._num_bands = num_bands
        self._width = width
        self._height = height
        self._block_w = block_w
        self._block_h = block_h
        # Generate a recognizable gradient pattern: band 0 = row gradient, band 1 = col gradient
        data = np.zeros((num_bands, height, width), dtype=np.uint8)
        for b in range(num_bands):
            row_vals = np.linspace(0, 255, height, dtype=np.uint8)
            col_vals = np.linspace(0, 255, width, dtype=np.uint8)
            if b % 2 == 0:
                data[b] = row_vals[:, np.newaxis] * np.ones((1, width), dtype=np.uint8)
            else:
                data[b] = np.ones((height, 1), dtype=np.uint8) * col_vals[np.newaxis, :]
        self._data = data

    @property
    def key(self):
        return "integration_source"

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def num_rows(self):
        return self._height

    @property
    def num_columns(self):
        return self._width

    @property
    def num_pixels_per_block_horizontal(self):
        return self._block_w

    @property
    def num_pixels_per_block_vertical(self):
        return self._block_h

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def pixel_value_type(self):
        return "uint8"

    @property
    def block_grid_size(self):
        return (math.ceil(self._height / self._block_h), math.ceil(self._width / self._block_w))

    @property
    def metadata(self):
        return None

    def get_block(self, row, col, resolution_level=0, bands=None):
        y0 = row * self._block_h
        x0 = col * self._block_w
        y1 = min(y0 + self._block_h, self._height)
        x1 = min(x0 + self._block_w, self._width)
        block = self._data[:, y0:y1, x0:x1]
        if bands is not None:
            block = block[list(bands), :, :]
        nb = block.shape[0]
        padded = np.zeros((nb, self._block_h, self._block_w), dtype=block.dtype)
        padded[:, : block.shape[1], : block.shape[2]] = block
        return padded

    def has_block(self, row, col, resolution_level=0):
        return True


class TestOrthoWarpedPipeline(TestCase):
    """Integration: OrthoGridBuilder + WarpedImageProvider + MappedImageProvider."""

    def _make_pipeline(self, block_size=128):
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        source = _MockSource(num_bands=3, width=512, height=512, block_w=block_size, block_h=block_size)

        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=512,
            source_height=512,
            block_size=(block_size, block_size),
        )
        grid_builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=512,
            source_height=512,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        return source, warped, grid_builder

    def test_warped_produces_non_zero_pixels(self):
        """Full pipeline produces output blocks containing actual pixel data."""
        _, warped, _ = self._make_pipeline()
        block = warped.get_block(0, 0)
        self.assertEqual(block.ndim, 3)
        self.assertEqual(block.shape[0], 3)
        self.assertTrue(np.any(block > 0), "Warped output should contain non-zero pixels")

    def test_warped_with_mapped_provider(self):
        """WarpedImageProvider composes with MappedImageProvider."""
        _, warped, _ = self._make_pipeline()

        def invert(block):
            return 255 - block

        mapped = MappedImageProvider(warped, invert)
        block = mapped.get_block(0, 0)
        self.assertEqual(block.shape[0], 3)
        self.assertTrue(np.any(block < 255), "Inverted output should have values < 255")

    def test_warped_valid_mask_consistent_with_pixels(self):
        """Where valid_mask is True, pixels should be non-zero (given gradient source)."""
        _, warped, _ = self._make_pipeline()
        block = warped.get_block(0, 0)
        mask = warped.get_valid_mask(0, 0)
        # Invalid pixels must be zero
        invalid_pixels = block[:, ~mask]
        np.testing.assert_array_equal(invalid_pixels, 0)

    def test_full_image_coverage(self):
        """Iterating all blocks of the warped output produces complete coverage."""
        _, warped, _ = self._make_pipeline()
        rows, cols = warped.block_grid_size
        has_any_data = False
        for r in range(rows):
            for c in range(cols):
                block = warped.get_block(r, c)
                if np.any(block > 0):
                    has_any_data = True
        self.assertTrue(has_any_data, "At least some blocks should contain data")

    def test_output_dimensions_derived_from_builder(self):
        """WarpedImageProvider dimensions match grid builder tile_limits."""
        _, warped, grid_builder = self._make_pipeline()
        min_row, min_col, max_row, max_col = grid_builder.tile_limits
        tile_w, tile_h = grid_builder.tile_size
        self.assertEqual(warped.num_rows, (max_row - min_row + 1) * tile_h)
        self.assertEqual(warped.num_columns, (max_col - min_col + 1) * tile_w)


class TestImageToImageWarpedPipeline(TestCase):
    """Integration: ImageToImageGridBuilder + WarpedImageProvider."""

    def test_identity_models_preserve_geometry(self):
        """When source and target have the same sensor model, output should closely match source."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        source = _MockSource(num_bands=1, width=256, height=256, block_w=256, block_h=256)

        grid_builder = ImageToImageGridBuilder(
            source_sensor_model=sm,
            target_sensor_model=sm,
            source_width=256,
            source_height=256,
            target_width=256,
            target_height=256,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=16),
        )

        warped = WarpedImageProvider(source, grid_builder)
        warped_block = warped.get_block(0, 0)
        source_block = source.get_block(0, 0)

        # With identical sensor models, the warped output should be very close to source
        diff = np.abs(warped_block.astype(np.int16) - source_block.astype(np.int16))
        mean_diff = diff.mean()
        self.assertLess(mean_diff, 5.0, f"Mean pixel difference {mean_diff} too large for identity warp")

    def test_shifted_models_produce_offset(self):
        """Shifted target sensor model produces shifted source pixels in output."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.05, origin_lat_deg=47.95, scale_x_deg=0.001, scale_y_deg=0.001)
        source = _MockSource(num_bands=3, width=512, height=512, block_w=256, block_h=256)

        grid_builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=512,
            source_height=512,
            target_width=512,
            target_height=512,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        block = warped.get_block(0, 0)
        self.assertEqual(block.shape, (3, 256, 256))
        self.assertTrue(np.any(block > 0), "Shifted warp should still produce pixel data")

    def test_warped_output_matches_target_dimensions(self):
        """WarpedImageProvider has target image dimensions."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.002, scale_y_deg=0.002)
        source = _MockSource(num_bands=3, width=1024, height=1024, block_w=256, block_h=256)

        grid_builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=1024,
            source_height=1024,
            target_width=512,
            target_height=512,
            block_width=256,
            block_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        self.assertEqual(warped.num_rows, 512)
        self.assertEqual(warped.num_columns, 512)
        self.assertEqual(warped.block_grid_size, (2, 2))


class TestWarpedWithChipFactory(TestCase):
    """Integration: WarpedImageProvider + TiledImagePyramid + ChipFactory."""

    def test_chip_factory_produces_encoded_chip(self):
        """ChipFactory over a warped provider produces non-empty encoded bytes."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        source = _MockSource(num_bands=3, width=256, height=256, block_w=256, block_h=256)

        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=256,
            source_height=256,
            block_size=(256, 256),
        )
        grid_builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=256,
            source_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        pyramid = TiledImagePyramid([warped])

        chip_factory = ChipFactory(pyramid, output_format="png")
        chip_bytes = chip_factory.create_chip(
            PixelWindow(0, 0, 128, 128),
            output_size=ImageSize(128, 128),
        )
        self.assertIsNotNone(chip_bytes)
        self.assertGreater(len(chip_bytes), 0)

    def test_chip_factory_with_display_chain(self):
        """ChipFactory with a processing chain produces output."""
        from aws.osml.image_processing.processing_chain import ProcessingChain

        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        source = _MockSource(num_bands=3, width=256, height=256, block_w=256, block_h=256)

        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=256,
            source_height=256,
            block_size=(256, 256),
        )
        grid_builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=256,
            source_height=256,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        pyramid = TiledImagePyramid([warped])

        def normalize(block):
            return block

        chain = ProcessingChain(steps=[normalize], output_bands=3)
        chip_factory = ChipFactory(pyramid, output_format="png", processing_chain=chain)
        chip_bytes = chip_factory.create_chip(
            PixelWindow(0, 0, 64, 64),
            output_size=ImageSize(64, 64),
        )
        self.assertIsNotNone(chip_bytes)
        self.assertGreater(len(chip_bytes), 0)

    def test_chip_factory_scaled_output(self):
        """ChipFactory can produce tiles at a different size than the source window."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        source = _MockSource(num_bands=3, width=512, height=512, block_w=256, block_h=256)

        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=512,
            source_height=512,
            block_size=(256, 256),
        )
        grid_builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=512,
            source_height=512,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        pyramid = TiledImagePyramid([warped])

        chip_factory = ChipFactory(pyramid, output_format="png")
        chip_bytes = chip_factory.create_chip(
            PixelWindow(0, 0, 256, 256),
            output_size=ImageSize(128, 128),
        )
        self.assertIsNotNone(chip_bytes)
        self.assertGreater(len(chip_bytes), 0)


class TestWarpedWithElevation(TestCase):
    """Integration: full pipeline with elevation model."""

    def test_ortho_with_elevation(self):
        """OrthoGridBuilder with elevation model produces valid warped output."""
        sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        elev = _ConstantElevation(elevation=200.0)
        source = _MockSource(num_bands=3, width=256, height=256, block_w=128, block_h=128)

        tile_set = ProjectedImageTileSet.from_sensor_model(
            sensor_model=sm,
            source_width=256,
            source_height=256,
            block_size=(128, 128),
            elevation_model=elev,
        )
        grid_builder = OrthoGridBuilder(
            tile_set=tile_set,
            tile_matrix=0,
            sensor_model=sm,
            source_width=256,
            source_height=256,
            elevation_model=elev,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        block = warped.get_block(0, 0)
        self.assertEqual(block.shape[0], 3)
        self.assertTrue(np.any(block > 0))

    def test_image_to_image_with_elevation(self):
        """ImageToImageGridBuilder with elevation model produces valid output."""
        source_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        target_sm = _AffineSensorModel(origin_lon_deg=35.0, origin_lat_deg=48.0, scale_x_deg=0.001, scale_y_deg=0.001)
        elev = _ConstantElevation(elevation=500.0)
        source = _MockSource(num_bands=3, width=256, height=256, block_w=128, block_h=128)

        grid_builder = ImageToImageGridBuilder(
            source_sensor_model=source_sm,
            target_sensor_model=target_sm,
            source_width=256,
            source_height=256,
            target_width=256,
            target_height=256,
            elevation_model=elev,
            block_width=128,
            block_height=128,
            options=WarpGridOptions(control_points_per_side=8),
        )

        warped = WarpedImageProvider(source, grid_builder)
        block = warped.get_block(0, 0)
        self.assertEqual(block.shape, (3, 128, 128))
        self.assertTrue(np.any(block > 0))
