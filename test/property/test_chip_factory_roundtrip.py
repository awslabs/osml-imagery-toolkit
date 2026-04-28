#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based round-trip tests for ChipFactory.

Encodes tiles from pyramids of varying types, decodes the output with
osml-imagery-io's DatasetReader, and verifies pixel/metadata
correctness. Also includes thread-safety integration tests.
"""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.chip_factory import ChipFactory, ImageSize, PixelWindow
from aws.osml.image_processing.processing_chain import ProcessingChain
from aws.osml.image_processing.pyramid import TiledImagePyramid, build_pyramid_levels
from aws.osml.io import IO, PixelType
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Mock Provider
# ---------------------------------------------------------------------------


class _MockProvider:
    """Minimal ImageAssetProvider duck-type backed by a CHW array."""

    def __init__(self, image, tile_height=256, tile_width=256):
        self._base_image = image
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)

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
        return PixelType.UInt8

    @property
    def pad_pixel_value(self):
        return 0.0

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def block_grid_size(self):
        rows = (self.num_rows + self._tile_height - 1) // self._tile_height
        cols = (self.num_columns + self._tile_width - 1) // self._tile_width
        return (rows, cols)

    def has_block(self, row, col, resolution_level=0):
        grid_rows, grid_cols = self.block_grid_size
        return 0 <= row < grid_rows and 0 <= col < grid_cols

    def get_block(self, row, col, resolution_level=0, bands=None):
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self._base_image.shape[-2])
        x1 = min(x0 + self._tile_width, self._base_image.shape[-1])
        tile = self._base_image[:, y0:y1, x0:x1].copy()
        if tile.shape[-2] != self._tile_height or tile.shape[-1] != self._tile_width:
            padded = np.full(
                (tile.shape[0], self._tile_height, self._tile_width),
                0,
                dtype=tile.dtype,
            )
            padded[:, : tile.shape[-2], : tile.shape[-1]] = tile
            tile = padded
        return tile


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_output_formats = st.sampled_from(["png", "geotiff", "nitf", "jpeg"])
_tile_sizes = st.sampled_from([64, 128, 256])


@st.composite
def tile_extraction_scenario(draw):
    """Generate a random tile extraction scenario.

    Returns a dict with keys: image, tile_size, output_format, src_window,
    output_size.
    """
    num_bands = draw(st.integers(min_value=1, max_value=3))
    height = draw(st.integers(min_value=64, max_value=256))
    width = draw(st.integers(min_value=64, max_value=256))
    tile_size = draw(_tile_sizes)
    output_format = draw(_output_formats)

    image = draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=(num_bands, height, width),
            elements=st.integers(min_value=0, max_value=255),
        )
    )

    # Generate a valid src_window within image bounds
    max_w = min(width, 128)
    max_h = min(height, 128)
    win_w = draw(st.integers(min_value=16, max_value=max_w))
    win_h = draw(st.integers(min_value=16, max_value=max_h))
    win_x = draw(st.integers(min_value=0, max_value=width - win_w))
    win_y = draw(st.integers(min_value=0, max_value=height - win_h))

    # Output size: same as src_window or scaled
    use_scaling = draw(st.booleans())
    if use_scaling:
        out_w = draw(st.integers(min_value=16, max_value=128))
        out_h = draw(st.integers(min_value=16, max_value=128))
        output_size = ImageSize(out_w, out_h)
    else:
        output_size = None

    # JPEG requires 3 bands — if format is jpeg and bands != 3, use chain
    needs_chain = output_format == "jpeg" and num_bands != 3

    return {
        "image": image,
        "tile_size": tile_size,
        "output_format": output_format,
        "src_window": PixelWindow(win_x, win_y, win_w, win_h),
        "output_size": output_size,
        "needs_chain": needs_chain,
    }


def _make_jpeg_chain():
    """Create a processing chain that produces 3-band uint8 for JPEG."""

    def to_rgb(img):
        if img.shape[0] == 3:
            return img.astype(np.uint8)
        band = img[0:1, :, :].astype(np.uint8)
        return np.concatenate([band, band, band], axis=0)

    return ProcessingChain(
        steps=[to_rgb],
        output_bands=3,
        output_dtype=np.dtype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(scenario=tile_extraction_scenario())
@settings(pbt_settings)
def test_roundtrip_encode_decode_preserves_shape(scenario):
    """Encoded tiles decode to the expected spatial dimensions.

    For any valid image and extraction parameters, the encoded tile
    when decoded must have height and width matching the requested
    output_size (or src_window dimensions when output_size is None).
    """
    image = scenario["image"]
    tile_size = scenario["tile_size"]
    output_format = scenario["output_format"]
    src_window = scenario["src_window"]
    output_size = scenario["output_size"]
    needs_chain = scenario["needs_chain"]

    provider = _MockProvider(image, tile_height=tile_size, tile_width=tile_size)
    pyramid = TiledImagePyramid([provider])

    chain = _make_jpeg_chain() if needs_chain else None
    factory = ChipFactory(
        source=pyramid,
        output_format=output_format,
        processing_chain=chain,
    )

    result = factory.create_chip(src_window, output_size=output_size)
    assert result is not None
    assert isinstance(result, bytearray)
    assert len(result) > 0

    # Decode and verify dimensions
    io_format = "tiff" if output_format == "geotiff" else output_format
    with IO.open(io.BytesIO(result), "r", io_format) as reader:
        asset = reader.get_asset("image:0")
        expected_w = output_size.width if output_size else src_window.width
        expected_h = output_size.height if output_size else src_window.height
        assert asset.num_columns == expected_w, f"Width mismatch: got {asset.num_columns}, expected {expected_w}"
        assert asset.num_rows == expected_h, f"Height mismatch: got {asset.num_rows}, expected {expected_h}"


@pytest.mark.property
@given(
    num_bands=st.integers(min_value=1, max_value=3),
    height=st.integers(min_value=64, max_value=128),
    width=st.integers(min_value=64, max_value=128),
)
@settings(pbt_settings)
def test_roundtrip_pixel_fidelity_png(num_bands, height, width):
    """PNG round-trip preserves pixel values exactly (lossless).

    For any uint8 image encoded to PNG, decoding must recover the
    exact same pixel values.
    """
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (num_bands, height, width), dtype=np.uint8)

    provider = _MockProvider(image, tile_height=height, tile_width=width)
    pyramid = TiledImagePyramid([provider])
    factory = ChipFactory(source=pyramid, output_format="png")

    result = factory.create_chip(PixelWindow(0, 0, width, height))
    assert result is not None

    with IO.open(io.BytesIO(result), "r", "png") as reader:
        asset = reader.get_asset("image:0")
        block = asset.get_block(0, 0)
        decoded = block[:num_bands, :height, :width]

    np.testing.assert_array_equal(decoded, image)


@pytest.mark.property
@given(
    height=st.integers(min_value=64, max_value=128),
    width=st.integers(min_value=64, max_value=128),
)
@settings(pbt_settings)
def test_roundtrip_nitf_format_validity(height, width):
    """NITF output has valid format header."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (3, height, width), dtype=np.uint8)

    provider = _MockProvider(image, tile_height=height, tile_width=width)
    pyramid = TiledImagePyramid([provider])
    factory = ChipFactory(source=pyramid, output_format="nitf")

    result = factory.create_chip(PixelWindow(0, 0, width, height))
    assert result is not None
    assert result[:4] == b"NITF"

    with IO.open(io.BytesIO(result), "r", "nitf") as reader:
        asset = reader.get_asset("image:0")
        assert asset.num_columns == width
        assert asset.num_rows == height
        assert asset.num_bands == 3


@pytest.mark.property
@given(
    height=st.integers(min_value=64, max_value=128),
    width=st.integers(min_value=64, max_value=128),
)
@settings(pbt_settings)
def test_roundtrip_geotiff_metadata_preserved(height, width):
    """GeoTIFF output preserves GeoTransform-derived tags."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (3, height, width), dtype=np.uint8)

    # Build a mock reader with GeoTIFF metadata
    class _MetaReader:
        def get_asset(self, key):
            return _MockProvider(image, tile_height=height, tile_width=width)

        @property
        def metadata(self):
            return _DictMeta(
                {
                    "33550": [0.001, 0.001, 0.0],
                    "33922": [0.0, 0.0, 0.0, -77.0, 39.0, 0.0],
                    "34735": [1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 2048, 0, 1, 4326],
                }
            )

        def get_asset_keys(self):
            return ["image:0"]

    class _DictMeta(dict):
        def __init__(self, d):
            super().__init__(d)

        def entries(self, prefix=None):
            if prefix:
                return {k: v for k, v in self.items() if k.startswith(prefix)}
            return dict(self)

    reader = _MetaReader()
    provider = reader.get_asset("image:0")
    pyramid = TiledImagePyramid([provider], reader=reader)
    factory = ChipFactory(source=pyramid, output_format="geotiff")

    result = factory.create_chip(PixelWindow(0, 0, width, height))
    assert result is not None

    with IO.open(io.BytesIO(result), "r", "tiff") as out_reader:
        asset = out_reader.get_asset("image:0")
        meta = dict(asset.metadata)
        assert "33550" in meta or "33922" in meta, "GeoTIFF metadata (ModelPixelScale or ModelTiepoint) not preserved"


@pytest.mark.property
@given(
    num_levels=st.integers(min_value=1, max_value=3),
    output_format=st.sampled_from(["png", "geotiff", "nitf"]),
)
@settings(pbt_settings)
def test_multi_level_pyramid_uses_correct_level(num_levels, output_format):
    """Multi-level pyramids select an appropriate resolution level."""
    from aws.osml.image_processing.resample import area_resample

    rng = np.random.RandomState(42)
    base_size = 256
    image = rng.randint(0, 256, (3, base_size, base_size), dtype=np.uint8)

    provider = _MockProvider(image, tile_height=128, tile_width=128)
    levels = build_pyramid_levels(provider, min_size=64, resample_func=area_resample)
    actual_levels = min(num_levels, len(levels))
    pyramid = TiledImagePyramid(levels[:actual_levels])

    factory = ChipFactory(source=pyramid, output_format=output_format)

    # Request the full image at reduced output
    output_size = ImageSize(64, 64)
    result = factory.create_chip(PixelWindow(0, 0, base_size, base_size), output_size=output_size)
    assert result is not None

    io_format = "tiff" if output_format == "geotiff" else output_format
    with IO.open(io.BytesIO(result), "r", io_format) as reader:
        asset = reader.get_asset("image:0")
        assert asset.num_columns == 64
        assert asset.num_rows == 64


# ---------------------------------------------------------------------------
# Thread-Safety Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_thread_safety_concurrent_create_chip():
    """Concurrent create_chip calls from a thread pool produce correct results.

    Multiple threads extract different regions from the same ChipFactory
    instance. Each result must be a valid encoding with correct dimensions.
    """
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (3, 512, 512), dtype=np.uint8)
    provider = _MockProvider(image, tile_height=128, tile_width=128)
    pyramid = TiledImagePyramid([provider])
    factory = ChipFactory(source=pyramid, output_format="png")

    # Define non-overlapping tile windows
    windows = [
        (PixelWindow(0, 0, 128, 128), ImageSize(64, 64)),
        (PixelWindow(128, 0, 128, 128), ImageSize(64, 64)),
        (PixelWindow(0, 128, 128, 128), ImageSize(64, 64)),
        (PixelWindow(128, 128, 128, 128), ImageSize(64, 64)),
        (PixelWindow(256, 0, 128, 128), ImageSize(64, 64)),
        (PixelWindow(0, 256, 128, 128), ImageSize(64, 64)),
        (PixelWindow(256, 256, 128, 128), ImageSize(64, 64)),
        (PixelWindow(384, 384, 128, 128), ImageSize(64, 64)),
    ]

    results = {}

    def _extract(idx, window, output_size):
        result = factory.create_chip(window, output_size=output_size)
        return idx, result

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_extract, i, w, s) for i, (w, s) in enumerate(windows)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    assert len(results) == len(windows)

    for idx, result in results.items():
        assert result is not None, f"Window {idx} returned None"
        assert isinstance(result, bytearray)
        assert result[:4] == b"\x89PNG", f"Window {idx} not valid PNG"

        with IO.open(io.BytesIO(result), "r", "png") as reader:
            asset = reader.get_asset("image:0")
            assert asset.num_columns == 64, f"Window {idx}: width != 64"
            assert asset.num_rows == 64, f"Window {idx}: height != 64"


@pytest.mark.property
def test_thread_safety_with_processing_chain():
    """Concurrent calls with a processing chain produce independent results."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (6, 256, 256), dtype=np.uint8)
    provider = _MockProvider(image, tile_height=128, tile_width=128)
    pyramid = TiledImagePyramid([provider])

    def select_and_normalize(img):
        return img[:3, :, :]

    chain = ProcessingChain(
        steps=[select_and_normalize],
        output_bands=3,
        output_dtype=np.dtype(np.uint8),
        input_bands=(0, 1, 2, 3, 4, 5),
    )

    factory = ChipFactory(source=pyramid, output_format="png", processing_chain=chain)

    windows = [PixelWindow(i * 64, j * 64, 64, 64) for i in range(4) for j in range(4)]

    results = {}

    def _extract(idx, window):
        return idx, factory.create_chip(window)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_extract, i, w) for i, w in enumerate(windows)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    assert len(results) == len(windows)
    for idx, result in results.items():
        assert result is not None
        assert result[:4] == b"\x89PNG"
        with IO.open(io.BytesIO(result), "r", "png") as reader:
            asset = reader.get_asset("image:0")
            assert asset.num_bands == 3
            assert asset.num_columns == 64
            assert asset.num_rows == 64


@pytest.mark.property
def test_thread_safety_mixed_formats():
    """Concurrent factories with different output formats produce valid tiles."""
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (3, 128, 128), dtype=np.uint8)
    provider = _MockProvider(image, tile_height=128, tile_width=128)
    pyramid = TiledImagePyramid([provider])

    factories = {
        "png": ChipFactory(source=pyramid, output_format="png"),
        "geotiff": ChipFactory(source=pyramid, output_format="geotiff"),
        "nitf": ChipFactory(source=pyramid, output_format="nitf"),
        "jpeg": ChipFactory(source=pyramid, output_format="jpeg"),
    }

    magic_bytes = {
        "png": b"\x89PNG",
        "geotiff": None,  # TIFF: "II" or "MM"
        "nitf": b"NITF",
        "jpeg": b"\xff\xd8",
    }

    window = PixelWindow(0, 0, 128, 128)
    results = {}

    def _extract(fmt, factory):
        return fmt, factory.create_chip(window)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_extract, fmt, f) for fmt, f in factories.items()]
        for future in as_completed(futures):
            fmt, result = future.result()
            results[fmt] = result

    for fmt, result in results.items():
        assert result is not None, f"{fmt} returned None"
        expected = magic_bytes[fmt]
        if expected is not None:
            assert result[: len(expected)] == expected, f"{fmt} has wrong magic bytes"
        else:
            assert result[:2] in (b"II", b"MM"), "GeoTIFF has wrong magic bytes"
