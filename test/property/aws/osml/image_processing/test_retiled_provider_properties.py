#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for RetiledImageProvider.

Validates the core invariant: reading all virtual tiles from a
RetiledImageProvider and stitching them together produces the same
pixel data as the original source image.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.retiled_provider import RetiledImageProvider
from property.conftest import pbt_settings


class _MockProvider:
    """Minimal mock ImageAssetProvider for property tests."""

    def __init__(self, image, tile_height, tile_width):
        self._image = image
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._grid_rows = math.ceil(image.shape[1] / tile_height)
        self._grid_cols = math.ceil(image.shape[2] / tile_width)

    @property
    def key(self):
        return "prop-test"

    @property
    def num_rows(self):
        return self._image.shape[1]

    @property
    def num_columns(self):
        return self._image.shape[2]

    @property
    def num_pixels_per_block_horizontal(self):
        return self._tile_width

    @property
    def num_pixels_per_block_vertical(self):
        return self._tile_height

    @property
    def num_resolution_levels(self):
        return 1

    @property
    def block_grid_size(self):
        return (self._grid_rows, self._grid_cols)

    @property
    def num_bands(self):
        return self._image.shape[0]

    @property
    def pixel_value_type(self):
        return "uint8"

    @property
    def metadata(self):
        return {}

    def has_block(self, row, col, resolution_level=0):
        return 0 <= row < self._grid_rows and 0 <= col < self._grid_cols

    def get_block(self, row, col, resolution_level=0, bands=None):
        y0 = row * self._tile_height
        x0 = col * self._tile_width
        y1 = min(y0 + self._tile_height, self._image.shape[1])
        x1 = min(x0 + self._tile_width, self._image.shape[2])
        block = self._image[:, y0:y1, x0:x1].copy()
        if bands is not None:
            block = block[list(bands), :, :]
        return block


@st.composite
def source_and_virtual_grid(draw):
    """Generate a random source image with source tile dims and virtual tile dims."""
    num_bands = draw(st.integers(min_value=1, max_value=3))
    img_h = draw(st.integers(min_value=4, max_value=128))
    img_w = draw(st.integers(min_value=4, max_value=128))

    image = draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=(num_bands, img_h, img_w),
            elements=st.integers(min_value=0, max_value=255),
        )
    )

    src_tile_h = draw(st.integers(min_value=2, max_value=max(2, img_h)))
    src_tile_w = draw(st.integers(min_value=2, max_value=max(2, img_w)))

    virt_tile_h = draw(st.integers(min_value=2, max_value=max(2, img_h)))
    virt_tile_w = draw(st.integers(min_value=2, max_value=max(2, img_w)))

    return image, src_tile_h, src_tile_w, virt_tile_h, virt_tile_w


@pytest.mark.property
@given(data=source_and_virtual_grid())
@settings(pbt_settings)
def test_stitched_virtual_tiles_equal_source_image(data):
    """Stitching all virtual tiles recovers the full source image.

    For any valid source grid and virtual tile size, reading all
    virtual tiles and assembling them into a single array produces
    exactly the same pixel data as the original source image.
    """
    image, src_tile_h, src_tile_w, virt_tile_h, virt_tile_w = data

    source = _MockProvider(image, tile_height=src_tile_h, tile_width=src_tile_w)
    retiled = RetiledImageProvider(source, tile_width=virt_tile_w, tile_height=virt_tile_h, pad_edges=False)

    reconstructed = np.zeros_like(image)

    grid_rows, grid_cols = retiled.block_grid_size
    for r in range(grid_rows):
        for c in range(grid_cols):
            tile = retiled.get_block(r, c)
            y0 = r * virt_tile_h
            x0 = c * virt_tile_w
            h = tile.shape[1]
            w = tile.shape[2]
            reconstructed[:, y0 : y0 + h, x0 : x0 + w] = tile

    np.testing.assert_array_equal(
        reconstructed,
        image,
        err_msg=(
            f"Reconstructed image differs from source. "
            f"Source: {image.shape}, src_tile: {src_tile_h}x{src_tile_w}, "
            f"virt_tile: {virt_tile_h}x{virt_tile_w}"
        ),
    )


@pytest.mark.property
@given(data=source_and_virtual_grid())
@settings(pbt_settings)
def test_padded_tiles_contain_source_pixels(data):
    """With pad_edges=True, the original image pixels are preserved in the padded result."""
    image, src_tile_h, src_tile_w, virt_tile_h, virt_tile_w = data

    source = _MockProvider(image, tile_height=src_tile_h, tile_width=src_tile_w)
    retiled = RetiledImageProvider(source, tile_width=virt_tile_w, tile_height=virt_tile_h, pad_edges=True)

    img_h = image.shape[1]
    img_w = image.shape[2]

    # Reconstruct by reading all tiles — the image region should match
    grid_rows, grid_cols = retiled.block_grid_size
    padded_h = grid_rows * virt_tile_h
    padded_w = grid_cols * virt_tile_w
    reconstructed = np.zeros((image.shape[0], padded_h, padded_w), dtype=image.dtype)

    for r in range(grid_rows):
        for c in range(grid_cols):
            tile = retiled.get_block(r, c)
            assert tile.shape == (image.shape[0], virt_tile_h, virt_tile_w), (
                f"Tile shape {tile.shape} doesn't match expected ({image.shape[0]}, {virt_tile_h}, {virt_tile_w})"
            )
            y0 = r * virt_tile_h
            x0 = c * virt_tile_w
            reconstructed[:, y0 : y0 + virt_tile_h, x0 : x0 + virt_tile_w] = tile

    # The source image region should be identical
    np.testing.assert_array_equal(
        reconstructed[:, :img_h, :img_w],
        image,
        err_msg="Source pixels not preserved in padded reconstruction",
    )
