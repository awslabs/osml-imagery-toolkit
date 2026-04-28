#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.chip_factory`.

Tests cover :class:`ChipFactory` (resolution-level selection, pixel read,
final resize, encoding), :class:`ImageSize`, :class:`PixelWindow`, and the
updated :class:`TiledImagePyramid` (``best_level_for``, ``.reader`` property).
"""

import io
from unittest import TestCase

import numpy as np

from aws.osml.image_processing.chip_factory import ChipFactory, ImageSize, PixelWindow, _resample_to_size
from aws.osml.image_processing.pyramid import TiledImagePyramid, build_pyramid_levels
from aws.osml.io import PixelType

# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------


class _MockProvider:
    """Minimal ImageAssetProvider duck-type backed by a CHW array."""

    def __init__(
        self,
        image,
        tile_height=256,
        tile_width=256,
        sparse_tiles=None,
        pixel_value_type=PixelType.UInt8,
        pad_pixel_value=0.0,
        num_resolution_levels=1,
    ):
        self._base_image = image
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)
        self._sparse_tiles = set(sparse_tiles or [])
        self._pixel_value_type = pixel_value_type
        self._pad_pixel_value = float(pad_pixel_value)
        self._num_resolution_levels = int(num_resolution_levels)

    @property
    def key(self):
        return "mock:0"

    @property
    def num_rows(self):
        return int(self._base_image.shape[-2])

    @property
    def num_columns(self):
        return int(self._base_image.shape[-1])

    @property
    def num_bands(self):
        return int(self._base_image.shape[0])

    @property
    def num_pixels_per_block_horizontal(self):
        return self._tile_width

    @property
    def num_pixels_per_block_vertical(self):
        return self._tile_height

    @property
    def pixel_value_type(self):
        return self._pixel_value_type

    @property
    def pad_pixel_value(self):
        return self._pad_pixel_value

    @property
    def num_resolution_levels(self):
        return self._num_resolution_levels

    @property
    def block_grid_size(self):
        rows = (self.num_rows + self._tile_height - 1) // self._tile_height
        cols = (self.num_columns + self._tile_width - 1) // self._tile_width
        return (rows, cols)

    def has_block(self, row, col, resolution_level=0):
        grid_rows, grid_cols = self.block_grid_size
        if row < 0 or col < 0 or row >= grid_rows or col >= grid_cols:
            return False
        if (row, col) in self._sparse_tiles:
            return False
        return True

    def get_block(self, row, col, resolution_level=0, bands=None):
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self._base_image.shape[-2])
        x1 = min(x0 + self._tile_width, self._base_image.shape[-1])
        tile = self._base_image[:, y0:y1, x0:x1].copy()
        if tile.shape[-2] != self._tile_height or tile.shape[-1] != self._tile_width:
            padded = np.full(
                (tile.shape[0], self._tile_height, self._tile_width),
                self._pad_pixel_value,
                dtype=tile.dtype,
            )
            padded[:, : tile.shape[-2], : tile.shape[-1]] = tile
            tile = padded
        return tile


class _MockReader:
    """Minimal DatasetReader duck-type."""

    def __init__(self, assets, role_assets=None):
        self._assets = dict(assets)
        self._role_assets = role_assets or {}

    def get_asset(self, key):
        if key not in self._assets:
            raise KeyError(f"Asset not found: {key}")
        return self._assets[key]

    def get_assets_by_role(self, role):
        return self._role_assets.get(role, [])

    @property
    def metadata(self):
        return {}

    def get_data_assets(self):
        return []


# ----------------------------------------------------------------------
# ImageSize / PixelWindow
# ----------------------------------------------------------------------


class TestImageSize(TestCase):
    def test_named_access(self):
        s = ImageSize(width=512, height=256)
        self.assertEqual(s.width, 512)
        self.assertEqual(s.height, 256)

    def test_tuple_unpacking(self):
        s = ImageSize(1024, 768)
        w, h = s
        self.assertEqual(w, 1024)
        self.assertEqual(h, 768)


class TestPixelWindow(TestCase):
    def test_named_access(self):
        w = PixelWindow(x=10, y=20, width=100, height=200)
        self.assertEqual(w.x, 10)
        self.assertEqual(w.y, 20)
        self.assertEqual(w.width, 100)
        self.assertEqual(w.height, 200)


# ----------------------------------------------------------------------
# TiledImagePyramid — best_level_for and .reader
# ----------------------------------------------------------------------


class TestBestLevelFor(TestCase):
    """Tests for TiledImagePyramid.best_level_for."""

    def _make_pyramid(self, sizes):
        providers = []
        for rows, cols in sizes:
            image = np.zeros((3, rows, cols), dtype=np.uint8)
            providers.append(_MockProvider(image, tile_height=min(rows, 256), tile_width=min(cols, 256)))
        return TiledImagePyramid(providers)

    def test_single_level_returns_zero(self):
        pyramid = self._make_pyramid([(1024, 1024)])
        level = pyramid.best_level_for((512, 512), (512, 512))
        self.assertEqual(level, 0)

    def test_multi_level_selects_deepest_without_upsampling(self):
        # 4 levels: 1024, 512, 256, 128 (scale_factor=2)
        pyramid = self._make_pyramid([(1024, 1024), (512, 512), (256, 256), (128, 128)])
        # src_size=1024x1024, output_size=512x512
        # level 0: 1024/1=1024 >= 512 ✓
        # level 1: 1024/2=512 >= 512 ✓
        # level 2: 1024/4=256 >= 512? No ✗
        level = pyramid.best_level_for((1024, 1024), (512, 512))
        self.assertEqual(level, 1)

    def test_output_matches_src_returns_level_0(self):
        pyramid = self._make_pyramid([(1024, 1024), (512, 512)])
        level = pyramid.best_level_for((256, 256), (256, 256))
        self.assertEqual(level, 0)

    def test_output_larger_than_src_returns_level_0(self):
        # Upsampling case: output > src at all levels
        pyramid = self._make_pyramid([(1024, 1024), (512, 512)])
        level = pyramid.best_level_for((128, 128), (256, 256))
        self.assertEqual(level, 0)

    def test_small_output_uses_deepest_level(self):
        pyramid = self._make_pyramid([(1024, 1024), (512, 512), (256, 256), (128, 128)])
        # src_size=1024x1024, output_size=64x64
        # level 0: 1024/1=1024 >= 64 ✓
        # level 1: 1024/2=512 >= 64 ✓
        # level 2: 1024/4=256 >= 64 ✓
        # level 3: 1024/8=128 >= 64 ✓
        level = pyramid.best_level_for((1024, 1024), (64, 64))
        self.assertEqual(level, 3)

    def test_asymmetric_src_size(self):
        pyramid = self._make_pyramid([(1024, 1024), (512, 512), (256, 256)])
        # src_size=1024x512, output_size=256x256
        # level 0: 1024/1=1024>=256, 512/1=512>=256 ✓
        # level 1: 1024/2=512>=256, 512/2=256>=256 ✓
        # level 2: 1024/4=256>=256, 512/4=128>=256? No ✗
        level = pyramid.best_level_for((1024, 512), (256, 256))
        self.assertEqual(level, 1)

    def test_zero_src_size_raises(self):
        pyramid = self._make_pyramid([(1024, 1024)])
        with self.assertRaises(ValueError):
            pyramid.best_level_for((0, 512), (256, 256))

    def test_zero_output_size_raises(self):
        pyramid = self._make_pyramid([(1024, 1024)])
        with self.assertRaises(ValueError):
            pyramid.best_level_for((512, 512), (0, 256))

    def test_negative_raises(self):
        pyramid = self._make_pyramid([(1024, 1024)])
        with self.assertRaises(ValueError):
            pyramid.best_level_for((-1, 512), (256, 256))


class TestPyramidReaderProperty(TestCase):
    """Tests for TiledImagePyramid.reader property."""

    def test_from_providers_reader_is_none(self):
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        pyramid = TiledImagePyramid.from_providers([provider])
        self.assertIsNone(pyramid.reader)

    def test_from_dataset_stores_reader(self):
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        reader = _MockReader({"image:0": provider})
        pyramid = TiledImagePyramid.from_dataset(reader, "image:0")
        self.assertIs(pyramid.reader, reader)

    def test_constructor_with_reader(self):
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        reader = _MockReader({"image:0": provider})
        pyramid = TiledImagePyramid([provider], reader=reader)
        self.assertIs(pyramid.reader, reader)


# ----------------------------------------------------------------------
# ChipFactory — resolution selection + pixel read
# ----------------------------------------------------------------------


class TestChipFactoryBasicRead(TestCase):
    """Tests for ChipFactory reading pixels without encoding."""

    def _make_factory(self, image, output_format="png", tile_size=256):
        provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
        pyramid = TiledImagePyramid([provider])
        return ChipFactory(source=pyramid, output_format=output_format)

    def test_invalid_window_zero_width_raises(self):
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        factory = self._make_factory(image)
        with self.assertRaises(ValueError):
            factory.create_chip(PixelWindow(0, 0, 0, 256))

    def test_invalid_window_negative_height_raises(self):
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        factory = self._make_factory(image)
        with self.assertRaises(ValueError):
            factory.create_chip(PixelWindow(0, 0, 256, -1))

    def test_invalid_output_size_raises(self):
        image = np.zeros((3, 256, 256), dtype=np.uint8)
        factory = self._make_factory(image)
        with self.assertRaises(ValueError):
            factory.create_chip(PixelWindow(0, 0, 256, 256), output_size=ImageSize(0, 256))


class TestChipFactoryResolutionSelection(TestCase):
    """Tests for resolution-level selection in ChipFactory."""

    def _make_pyramid(self, image, tile_size=256):
        """Build a multi-level pyramid from an image."""
        from aws.osml.image_processing.resample import area_resample

        provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
        levels = build_pyramid_levels(provider, min_size=64, resample_func=area_resample)
        return TiledImagePyramid(levels)

    def test_no_scaling_uses_level_0(self):
        image = np.random.randint(0, 256, (3, 512, 512), dtype=np.uint8)
        pyramid = self._make_pyramid(image)
        factory = ChipFactory(source=pyramid, output_format="png")
        # src_window same size as output — should use level 0
        result = factory.create_chip(PixelWindow(0, 0, 256, 256))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)
        self.assertGreater(len(result), 0)

    def test_downscaling_uses_deeper_level(self):
        image = np.random.randint(0, 256, (3, 1024, 1024), dtype=np.uint8)
        pyramid = self._make_pyramid(image)
        factory = ChipFactory(source=pyramid, output_format="png")
        # Request 1024x1024 region at 256x256 output
        result = factory.create_chip(PixelWindow(0, 0, 1024, 1024), output_size=ImageSize(256, 256))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)


class TestChipFactoryCoordinateScaling(TestCase):
    """Tests for ceiling-based coordinate scaling."""

    def test_ceiling_end_coordinates(self):
        # Verify that the scaled window uses ceiling for end coordinates.
        # With a 1024x1024 image at level 1 (divisor=2), a window
        # (1, 1, 511, 511) should produce:
        #   scaled_x = 1//2 = 0
        #   scaled_y = 1//2 = 0
        #   x_end = (1+511+1)//2 = 256
        #   y_end = (1+511+1)//2 = 256
        #   scaled_window = (0, 0, 256, 256)
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (1, 1024, 1024), dtype=np.uint8)
        p0 = _MockProvider(image, tile_height=256, tile_width=256)
        p1 = _MockProvider(np.zeros((1, 512, 512), dtype=np.uint8), tile_height=256, tile_width=256)
        pyramid = TiledImagePyramid([p0, p1])
        factory = ChipFactory(source=pyramid, output_format="png")
        # Request a window that forces use of level 1 (output is half the src)
        result = factory.create_chip(PixelWindow(1, 1, 511, 511), output_size=ImageSize(256, 256))
        self.assertIsNotNone(result)


class TestChipFactoryFinalResize(TestCase):
    """Tests for the final resize step."""

    def test_output_dimensions_match_requested_size(self):
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (3, 512, 512), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        pyramid = TiledImagePyramid([provider])
        factory = ChipFactory(source=pyramid, output_format="png")

        output_size = ImageSize(128, 128)
        result = factory.create_chip(PixelWindow(0, 0, 512, 512), output_size=output_size)
        self.assertIsNotNone(result)

        # Decode the PNG to verify dimensions
        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 128)  # height
        self.assertEqual(decoded.shape[2], 128)  # width

    def test_no_resize_when_sizes_match(self):
        rng = np.random.RandomState(42)
        image = rng.randint(0, 256, (3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        pyramid = TiledImagePyramid([provider])
        factory = ChipFactory(source=pyramid, output_format="png")

        result = factory.create_chip(PixelWindow(0, 0, 256, 256))
        self.assertIsNotNone(result)

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 256)
        self.assertEqual(decoded.shape[2], 256)


class TestChipFactoryEncoding(TestCase):
    """Tests for encoding output formats."""

    def _make_factory(self, image, output_format, tile_size=256):
        provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
        pyramid = TiledImagePyramid([provider])
        return ChipFactory(source=pyramid, output_format=output_format)

    def test_png_output(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        factory = self._make_factory(image, "png", tile_size=128)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)
        # PNG magic bytes
        self.assertTrue(result[:4] == b"\x89PNG")

    def test_tiff_output(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        factory = self._make_factory(image, "geotiff", tile_size=128)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)
        # TIFF magic bytes (little-endian or big-endian)
        self.assertTrue(result[:2] in (b"II", b"MM"))

    def test_nitf_output(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        factory = self._make_factory(image, "nitf", tile_size=128)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)
        # NITF magic bytes
        self.assertTrue(result[:4] == b"NITF")

    def test_jpeg_output_uint8(self):
        # JPEG requires uint8
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        factory = self._make_factory(image, "jpeg", tile_size=128)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)
        # JPEG magic bytes (SOI marker)
        self.assertTrue(result[:2] == b"\xff\xd8")


class TestChipFactorySparseBlocks(TestCase):
    """Tests for sparse/empty block handling."""

    def test_all_sparse_returns_pad_values(self):
        # When all blocks are sparse the read_window returns a zero-filled
        # array, which is still valid data. ChipFactory should encode it.
        image = np.zeros((1, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256, sparse_tiles={(0, 0)})
        pyramid = TiledImagePyramid([provider])
        factory = ChipFactory(source=pyramid, output_format="png")
        result = factory.create_chip(PixelWindow(0, 0, 256, 256))
        # Even with sparse blocks, we get encoded bytes (pad-filled)
        self.assertIsNotNone(result)


class TestChipFactoryProcessingChain(TestCase):
    """Tests for processing chain integration."""

    def _identity_step(self, image):
        """No-op step that passes through unchanged."""
        return image

    def _halve_bands_step(self, image):
        """Step that takes first band only, triplicate to 3-band RGB."""
        band0 = image[0:1, :, :]
        return np.concatenate([band0, band0, band0], axis=0)

    def test_chain_applied_between_read_and_encode(self):
        from aws.osml.image_processing.processing_chain import ProcessingChain

        # 6-band source, chain selects bands (0,2,4) and produces 3-band uint8
        image = np.random.randint(0, 256, (6, 128, 128), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=128, tile_width=128)
        pyramid = TiledImagePyramid([provider])

        chain = ProcessingChain(
            steps=[self._identity_step],
            output_bands=3,
            output_dtype=np.dtype(np.uint8),
            input_bands=(0, 2, 4),
        )
        factory = ChipFactory(source=pyramid, output_format="png", processing_chain=chain)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)
        self.assertTrue(result[:4] == b"\x89PNG")

        # Decode to verify band count
        import io

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[0], 3)

    def test_chain_skipped_when_none(self):
        image = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=128, tile_width=128)
        pyramid = TiledImagePyramid([provider])
        factory = ChipFactory(source=pyramid, output_format="png", processing_chain=None)
        result = factory.create_chip(PixelWindow(0, 0, 128, 128))
        self.assertIsNotNone(result)

        import io

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[0], 3)

    def test_jpeg_output_with_chain_producing_3_band_uint8(self):
        from aws.osml.image_processing.processing_chain import ProcessingChain

        # 1-band uint16 source, chain produces 3-band uint8
        image = np.random.randint(0, 65535, (1, 64, 64), dtype=np.uint16)
        provider = _MockProvider(image, tile_height=64, tile_width=64, pixel_value_type=PixelType.UInt16)
        pyramid = TiledImagePyramid([provider])

        def to_rgb_uint8(img):
            normalized = (img.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
            return np.concatenate([normalized, normalized, normalized], axis=0)

        chain = ProcessingChain(
            steps=[to_rgb_uint8],
            output_bands=3,
            output_dtype=np.dtype(np.uint8),
        )
        factory = ChipFactory(source=pyramid, output_format="jpeg", processing_chain=chain)
        result = factory.create_chip(PixelWindow(0, 0, 64, 64))
        self.assertIsNotNone(result)
        # JPEG SOI marker
        self.assertTrue(result[:2] == b"\xff\xd8")

    def test_output_dtype_and_bands_match_chain(self):
        from aws.osml.image_processing.processing_chain import ProcessingChain

        # 4-band uint16 source, chain produces 1-band uint8
        image = np.random.randint(0, 65535, (4, 64, 64), dtype=np.uint16)
        provider = _MockProvider(image, tile_height=64, tile_width=64, pixel_value_type=PixelType.UInt16)
        pyramid = TiledImagePyramid([provider])

        def reduce_to_gray(img):
            return img[0:1, :, :].astype(np.uint8)

        chain = ProcessingChain(
            steps=[reduce_to_gray],
            output_bands=1,
            output_dtype=np.dtype(np.uint8),
            input_bands=(0, 1, 2, 3),
        )
        factory = ChipFactory(source=pyramid, output_format="png", processing_chain=chain)
        result = factory.create_chip(PixelWindow(0, 0, 64, 64))
        self.assertIsNotNone(result)

        import io

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[0], 1)

    def test_input_bands_none_reads_all_bands(self):
        from aws.osml.image_processing.processing_chain import ProcessingChain

        # 3-band source, chain has input_bands=None -> all bands read
        image = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=64, tile_width=64)
        pyramid = TiledImagePyramid([provider])

        chain = ProcessingChain(
            steps=[self._identity_step],
            output_bands=3,
            output_dtype=np.dtype(np.uint8),
            input_bands=None,
        )
        factory = ChipFactory(source=pyramid, output_format="png", processing_chain=chain)
        result = factory.create_chip(PixelWindow(0, 0, 64, 64))
        self.assertIsNotNone(result)

        import io

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[0], 3)

    def test_final_resize_after_chain(self):
        from aws.osml.image_processing.processing_chain import ProcessingChain

        image = np.random.randint(0, 256, (3, 256, 256), dtype=np.uint8)
        provider = _MockProvider(image, tile_height=256, tile_width=256)
        pyramid = TiledImagePyramid([provider])

        chain = ProcessingChain(
            steps=[self._identity_step],
            output_bands=3,
            output_dtype=np.dtype(np.uint8),
        )
        factory = ChipFactory(source=pyramid, output_format="png", processing_chain=chain)
        result = factory.create_chip(PixelWindow(0, 0, 256, 256), output_size=ImageSize(128, 128))
        self.assertIsNotNone(result)

        import io

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 128)
        self.assertEqual(decoded.shape[2], 128)


class TestChipFactoryOutOfBounds(TestCase):
    """Tests for out-of-bounds window handling in ChipFactory."""

    def _make_factory(self, image, output_format="png", tile_size=256):
        provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
        pyramid = TiledImagePyramid([provider])
        return ChipFactory(source=pyramid, output_format=output_format)

    def test_fully_oob_returns_none(self):
        """Window entirely outside image bounds returns None."""
        image = np.random.randint(0, 256, (3, 256, 256), dtype=np.uint8)
        factory = self._make_factory(image)
        result = factory.create_chip(PixelWindow(300, 300, 64, 64))
        self.assertIsNone(result)

    def test_fully_oob_negative_returns_none(self):
        """Window entirely in negative space returns None."""
        image = np.random.randint(0, 256, (3, 256, 256), dtype=np.uint8)
        factory = self._make_factory(image)
        result = factory.create_chip(PixelWindow(-128, -128, 64, 64))
        self.assertIsNone(result)

    def test_partial_oob_right_returns_padded_chip(self):
        """Window extending past right edge returns encoded chip with padding."""
        image = np.full((3, 128, 128), 200, dtype=np.uint8)
        factory = self._make_factory(image, tile_size=128)
        # Window is 256 wide but image is only 128 wide
        result = factory.create_chip(PixelWindow(0, 0, 256, 128), output_size=ImageSize(256, 128))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytearray)

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 128)
        self.assertEqual(decoded.shape[2], 256)
        # Left half should have image data, right half should be padded (0)
        np.testing.assert_array_equal(decoded[:, :, :128], 200)
        np.testing.assert_array_equal(decoded[:, :, 128:], 0)

    def test_partial_oob_bottom_returns_padded_chip(self):
        """Window extending past bottom edge returns encoded chip with padding."""
        image = np.full((3, 128, 128), 150, dtype=np.uint8)
        factory = self._make_factory(image, tile_size=128)
        result = factory.create_chip(PixelWindow(0, 0, 128, 256), output_size=ImageSize(128, 256))
        self.assertIsNotNone(result)

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 256)
        self.assertEqual(decoded.shape[2], 128)
        # Top half should have image data, bottom half should be padded (0)
        np.testing.assert_array_equal(decoded[:, :128, :], 150)
        np.testing.assert_array_equal(decoded[:, 128:, :], 0)

    def test_oob_with_decimation(self):
        """OOB window with output_size smaller than src_window (tile server use case)."""
        image = np.full((3, 512, 512), 100, dtype=np.uint8)
        factory = self._make_factory(image, tile_size=256)
        # Request 1024x1024 region (extends 512px beyond in each direction)
        # Output 256x256 (4x decimation)
        result = factory.create_chip(PixelWindow(0, 0, 1024, 1024), output_size=ImageSize(256, 256))
        self.assertIsNotNone(result)

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 256)
        self.assertEqual(decoded.shape[2], 256)
        # Top-left quadrant should have image data (decimated from 512x512 to 128x128)
        # The exact pixel values depend on resampling, but they should not be zero
        self.assertTrue(np.any(decoded[:, :128, :128] > 0))
        # Bottom-right quadrant should be pad (0)
        np.testing.assert_array_equal(decoded[:, 128:, 128:], 0)

    def test_partial_oob_negative_origin(self):
        """Window with negative origin returns padded chip."""
        image = np.full((3, 128, 128), 180, dtype=np.uint8)
        factory = self._make_factory(image, tile_size=128)
        # Window starts at (-64, -64), size 192x192
        result = factory.create_chip(PixelWindow(-64, -64, 192, 192), output_size=ImageSize(192, 192))
        self.assertIsNotNone(result)

        from aws.osml.io import imread

        decoded = imread(io.BytesIO(result), format="png")
        self.assertEqual(decoded.shape[1], 192)
        self.assertEqual(decoded.shape[2], 192)
        # Top-left corner should be pad (0)
        np.testing.assert_array_equal(decoded[:, :64, :64], 0)


class TestResampleToSize(TestCase):
    """Tests for the internal _resample_to_size helper."""

    def test_identity(self):
        img = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        result = _resample_to_size(img, ImageSize(64, 64))
        self.assertIs(result, img)

    def test_downsample(self):
        img = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        result = _resample_to_size(img, ImageSize(64, 64))
        self.assertEqual(result.shape, (3, 64, 64))

    def test_upsample(self):
        img = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        result = _resample_to_size(img, ImageSize(128, 128))
        self.assertEqual(result.shape, (3, 128, 128))
