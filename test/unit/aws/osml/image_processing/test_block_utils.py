#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.block_utils`.

Tests cover ``stitch_source_blocks``, ``read_block_or_pad``, and
``read_window`` with various window geometries: block-aligned,
multi-block, partial-edge, sparse, single-block sub-region, and
zero-dimension windows.

The tests use a minimal ``_MockProvider`` consistent with the mocks in
``test_pyramid_builder.py`` and ``test_downsample.py``.  Each block's
pixel values encode the block's grid position so that stitching
correctness can be verified deterministically.
"""

import numpy as np

from aws.osml.image_processing.block_utils import read_block_or_pad, read_window, stitch_source_blocks

# ----------------------------------------------------------------------
# Mock — minimal duck-typed ImageAssetProvider for unit tests
# ----------------------------------------------------------------------


class _MockProvider:
    """Minimal ImageAssetProvider duck-type for block_utils tests.

    Generates deterministic CHW blocks whose pixel values encode the
    block's grid position: ``fill_value = block_row * grid_cols + block_col + 1``.
    This lets tests verify that the correct block data lands in the
    correct region of the stitched output.

    Supports a ``sparse_tiles`` set — positions listed there cause
    ``has_block`` to return False, simulating missing blocks.
    """

    def __init__(
        self,
        grid_rows,
        grid_cols,
        tile_height=64,
        tile_width=64,
        num_bands=3,
        dtype=np.uint8,
        sparse_tiles=None,
        pad_pixel_value=0.0,
        image_rows=None,
        image_cols=None,
    ):
        self._grid_rows = int(grid_rows)
        self._grid_cols = int(grid_cols)
        self._tile_height = int(tile_height)
        self._tile_width = int(tile_width)
        self._num_bands = int(num_bands)
        self._dtype = np.dtype(dtype)
        self._sparse_tiles = set(sparse_tiles or [])
        self._pad_pixel_value = float(pad_pixel_value)
        self._image_rows = int(image_rows) if image_rows is not None else self._grid_rows * self._tile_height
        self._image_cols = int(image_cols) if image_cols is not None else self._grid_cols * self._tile_width
        self.get_block_calls = []

    # --- ImageAssetProvider interface ------------------------------------

    @property
    def num_rows(self):
        return self._image_rows

    @property
    def num_columns(self):
        return self._image_cols

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def num_pixels_per_block_vertical(self):
        return self._tile_height

    @property
    def num_pixels_per_block_horizontal(self):
        return self._tile_width

    @property
    def pixel_value_type(self):
        return self._dtype

    @property
    def pad_pixel_value(self):
        return self._pad_pixel_value

    # --- block access ----------------------------------------------------

    def _fill_value(self, row, col):
        """Deterministic fill value encoding the block position."""
        return row * self._grid_cols + col + 1

    def has_block(self, row, col, resolution_level=0):
        if row < 0 or col < 0 or row >= self._grid_rows or col >= self._grid_cols:
            return False
        return (row, col) not in self._sparse_tiles

    def get_block(self, row, col, resolution_level=0):
        self.get_block_calls.append((row, col, resolution_level))
        val = self._fill_value(row, col)
        return np.full(
            (self._num_bands, self._tile_height, self._tile_width),
            val,
            dtype=self._dtype,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# Common parameters used across tests.
TILE_H = 64
TILE_W = 64
NUM_BANDS = 3
DTYPE = np.uint8
PAD = 0.0
RES_LEVEL = 0


def _make_provider(grid_rows=4, grid_cols=4, sparse_tiles=None, pad=PAD):
    """Shortcut to build a ``_MockProvider`` with sensible defaults."""
    return _MockProvider(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        dtype=DTYPE,
        sparse_tiles=sparse_tiles,
        pad_pixel_value=pad,
    )


# ----------------------------------------------------------------------
# 5.2 — Window aligned exactly to block boundaries
# ----------------------------------------------------------------------


def test_stitch_block_aligned_single_block():
    """Window covers exactly one full block — output equals that block."""
    provider = _make_provider()
    result = stitch_source_blocks(
        provider,
        row_range=(0, TILE_H),
        col_range=(0, TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, TILE_H, TILE_W)
    assert result.dtype == DTYPE
    # Block (0, 0) fill value = 0 * 4 + 0 + 1 = 1
    np.testing.assert_array_equal(result, 1)


def test_stitch_block_aligned_2x2():
    """Window covers a 2x2 group of blocks exactly on boundaries."""
    provider = _make_provider()
    result = stitch_source_blocks(
        provider,
        row_range=(0, 2 * TILE_H),
        col_range=(0, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 2 * TILE_H, 2 * TILE_W)
    # Verify each quadrant has the expected fill value.
    # Block (0,0)=1, (0,1)=2, (1,0)=5, (1,1)=6  (grid_cols=4)
    np.testing.assert_array_equal(result[:, :TILE_H, :TILE_W], 1)
    np.testing.assert_array_equal(result[:, :TILE_H, TILE_W:], 2)
    np.testing.assert_array_equal(result[:, TILE_H:, :TILE_W], 5)
    np.testing.assert_array_equal(result[:, TILE_H:, TILE_W:], 6)


# ----------------------------------------------------------------------
# 5.3 — Window spanning multiple blocks in both dimensions
# ----------------------------------------------------------------------


def test_stitch_multi_block_cross_boundary():
    """Window spans parts of 4 blocks (not aligned to boundaries)."""
    provider = _make_provider()
    # Window starts at the midpoint of block (0,0) and extends into
    # blocks (0,1), (1,0), and (1,1).
    half_h = TILE_H // 2
    half_w = TILE_W // 2
    result = stitch_source_blocks(
        provider,
        row_range=(half_h, half_h + TILE_H),
        col_range=(half_w, half_w + TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, TILE_H, TILE_W)
    # Top-left quadrant comes from block (0,0) = 1
    np.testing.assert_array_equal(result[:, :half_h, :half_w], 1)
    # Top-right quadrant comes from block (0,1) = 2
    np.testing.assert_array_equal(result[:, :half_h, half_w:], 2)
    # Bottom-left quadrant comes from block (1,0) = 5
    np.testing.assert_array_equal(result[:, half_h:, :half_w], 5)
    # Bottom-right quadrant comes from block (1,1) = 6
    np.testing.assert_array_equal(result[:, half_h:, half_w:], 6)


def test_stitch_spans_3x3_blocks():
    """Window spans a 3x3 block region (9 blocks touched)."""
    provider = _make_provider()
    quarter_h = TILE_H // 4
    quarter_w = TILE_W // 4
    # Start inside block (0,0), end inside block (2,2).
    result = stitch_source_blocks(
        provider,
        row_range=(quarter_h, quarter_h + 2 * TILE_H),
        col_range=(quarter_w, quarter_w + 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 2 * TILE_H, 2 * TILE_W)
    # Verify that 9 blocks were read (3 rows x 3 cols).
    assert len(provider.get_block_calls) == 9


# ----------------------------------------------------------------------
# 5.4 — Window at image edges (partial blocks)
# ----------------------------------------------------------------------


def test_stitch_edge_window_bottom_right():
    """Window at the bottom-right corner reads only the last block."""
    provider = _make_provider(grid_rows=2, grid_cols=2)
    # Read the last 10 rows and 10 cols from the bottom-right block.
    y_start = 2 * TILE_H - 10
    x_start = 2 * TILE_W - 10
    result = stitch_source_blocks(
        provider,
        row_range=(y_start, 2 * TILE_H),
        col_range=(x_start, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 10, 10)
    # Block (1,1) fill value = 1 * 2 + 1 + 1 = 4
    np.testing.assert_array_equal(result, 4)


def test_stitch_edge_window_top_left_partial():
    """Small window at the very start of the image (partial block)."""
    provider = _make_provider()
    result = stitch_source_blocks(
        provider,
        row_range=(0, 5),
        col_range=(0, 5),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 5, 5)
    # Block (0,0) fill value = 1
    np.testing.assert_array_equal(result, 1)


# ----------------------------------------------------------------------
# 5.5 — Sparse blocks (has_block returns False)
# ----------------------------------------------------------------------


def test_stitch_sparse_block_filled_with_pad():
    """Sparse block region is filled with pad_value, not read."""
    pad_val = 42.0
    provider = _make_provider(grid_rows=2, grid_cols=2, sparse_tiles={(0, 0)}, pad=pad_val)
    result = stitch_source_blocks(
        provider,
        row_range=(0, 2 * TILE_H),
        col_range=(0, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=pad_val,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    # Block (0,0) is sparse -> pad_val (42 as uint8)
    np.testing.assert_array_equal(result[:, :TILE_H, :TILE_W], np.uint8(pad_val))
    # Block (0,1) exists -> fill value = 0*2 + 1 + 1 = 2
    np.testing.assert_array_equal(result[:, :TILE_H, TILE_W:], 2)
    # Sparse block should NOT have triggered get_block.
    read_positions = {(r, c) for r, c, _ in provider.get_block_calls}
    assert (0, 0) not in read_positions


def test_stitch_all_sparse_blocks():
    """All blocks sparse -> entire output is pad_value."""
    pad_val = 99.0
    provider = _make_provider(
        grid_rows=2,
        grid_cols=2,
        sparse_tiles={(0, 0), (0, 1), (1, 0), (1, 1)},
        pad=pad_val,
    )
    result = stitch_source_blocks(
        provider,
        row_range=(0, 2 * TILE_H),
        col_range=(0, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=pad_val,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    np.testing.assert_array_equal(result, np.uint8(pad_val))
    assert len(provider.get_block_calls) == 0


# ----------------------------------------------------------------------
# 5.6 — Single-block sub-region read
# ----------------------------------------------------------------------


def test_stitch_single_block_sub_region():
    """Window fits entirely within one block but does not cover it fully."""
    provider = _make_provider()
    # Read a 10x20 sub-region from the interior of block (1, 2).
    y_start = 1 * TILE_H + 10
    x_start = 2 * TILE_W + 5
    result = stitch_source_blocks(
        provider,
        row_range=(y_start, y_start + 10),
        col_range=(x_start, x_start + 20),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 10, 20)
    # Block (1, 2) fill value = 1 * 4 + 2 + 1 = 7
    np.testing.assert_array_equal(result, 7)
    # Only one block should have been read.
    assert len(provider.get_block_calls) == 1
    assert provider.get_block_calls[0][:2] == (1, 2)


# ----------------------------------------------------------------------
# 5.7 — Zero-dimension window
# ----------------------------------------------------------------------


def test_stitch_zero_height_window():
    """Zero-height window returns an empty array without reading blocks."""
    provider = _make_provider()
    result = stitch_source_blocks(
        provider,
        row_range=(10, 10),
        col_range=(0, TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 0, TILE_W)
    assert len(provider.get_block_calls) == 0


def test_stitch_zero_width_window():
    """Zero-width window returns an empty array without reading blocks."""
    provider = _make_provider()
    result = stitch_source_blocks(
        provider,
        row_range=(0, TILE_H),
        col_range=(10, 10),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, TILE_H, 0)
    assert len(provider.get_block_calls) == 0


# ----------------------------------------------------------------------
# 5.8 — read_window equivalence with stitch_source_blocks
# ----------------------------------------------------------------------


def test_read_window_equivalence():
    """read_window produces the same result as stitch_source_blocks
    with manually-extracted provider parameters."""
    provider = _make_provider()
    x, y, w, h = 10, 20, 100, 80
    res_level = 0
    fill = 0.0

    result_rw = read_window(provider, (x, y, w, h), resolution_level=res_level, fill_value=fill)

    result_ssb = stitch_source_blocks(
        provider,
        row_range=(y, y + h),
        col_range=(x, x + w),
        tile_height=provider.num_pixels_per_block_vertical,
        tile_width=provider.num_pixels_per_block_horizontal,
        num_bands=provider.num_bands,
        pad_value=fill,
        dtype=np.dtype(provider.pixel_value_type),
        resolution_level=res_level,
    )

    np.testing.assert_array_equal(result_rw, result_ssb)


def test_read_window_defaults():
    """read_window uses resolution_level=0 and fill_value=0 by default."""
    provider = _make_provider()
    result = read_window(provider, (0, 0, TILE_W, TILE_H))
    assert result.shape == (NUM_BANDS, TILE_H, TILE_W)
    assert result.dtype == DTYPE
    # Block (0,0) fill value = 1
    np.testing.assert_array_equal(result, 1)


# ----------------------------------------------------------------------
# 5.8b — Band-selective reads
# ----------------------------------------------------------------------


def test_read_window_bands_subset():
    """read_window with bands parameter returns only selected bands."""
    provider = _make_provider(grid_rows=2, grid_cols=2)
    result = read_window(provider, (0, 0, TILE_W, TILE_H), bands=(0, 2))
    assert result.shape == (2, TILE_H, TILE_W)
    # All values come from block (0,0) = 1
    np.testing.assert_array_equal(result, 1)


def test_read_window_bands_none_returns_all():
    """read_window with bands=None returns all bands (existing behavior)."""
    provider = _make_provider()
    result = read_window(provider, (0, 0, TILE_W, TILE_H), bands=None)
    assert result.shape == (NUM_BANDS, TILE_H, TILE_W)


def test_read_window_bands_single_band():
    """read_window with a single band returns shape (1, H, W)."""
    provider = _make_provider()
    result = read_window(provider, (0, 0, TILE_W, TILE_H), bands=(1,))
    assert result.shape == (1, TILE_H, TILE_W)
    np.testing.assert_array_equal(result, 1)


def test_stitch_source_blocks_bands_parameter():
    """stitch_source_blocks with bands selects from each block."""
    provider = _make_provider()
    result = stitch_source_blocks(
        provider,
        row_range=(0, TILE_H),
        col_range=(0, TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
        bands=(0, 2),
    )
    assert result.shape == (2, TILE_H, TILE_W)
    np.testing.assert_array_equal(result, 1)


def test_stitch_source_blocks_bands_with_sparse():
    """Band selection works with sparse blocks (pad-filled)."""
    pad_val = 77.0
    provider = _make_provider(grid_rows=2, grid_cols=2, sparse_tiles={(0, 0)}, pad=pad_val)
    result = stitch_source_blocks(
        provider,
        row_range=(0, TILE_H),
        col_range=(0, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=pad_val,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
        bands=(0,),
    )
    assert result.shape == (1, TILE_H, 2 * TILE_W)
    # Block (0,0) sparse -> pad_val
    np.testing.assert_array_equal(result[:, :, :TILE_W], np.uint8(pad_val))
    # Block (0,1) exists -> fill value = 2
    np.testing.assert_array_equal(result[:, :, TILE_W:], 2)


def test_read_window_bands_cross_boundary():
    """Band-selective read across multiple blocks returns correct shape."""
    provider = _make_provider(grid_rows=2, grid_cols=2)
    half_h = TILE_H // 2
    half_w = TILE_W // 2
    result = read_window(
        provider,
        (half_w, half_h, TILE_W, TILE_H),
        bands=(0, 1),
    )
    assert result.shape == (2, TILE_H, TILE_W)


# ----------------------------------------------------------------------
# 5.9 — read_block_or_pad: existing-block and sparse-block paths
# ----------------------------------------------------------------------


def test_read_block_or_pad_existing_block():
    """Existing block -> returns provider.get_block() result."""
    provider = _make_provider()
    result = read_block_or_pad(
        provider,
        tile_row=1,
        tile_col=2,
        resolution_level=RES_LEVEL,
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
    )
    assert result.shape == (NUM_BANDS, TILE_H, TILE_W)
    # Block (1, 2) fill value = 1 * 4 + 2 + 1 = 7
    np.testing.assert_array_equal(result, 7)
    assert len(provider.get_block_calls) == 1


def test_read_block_or_pad_sparse_block():
    """Sparse block -> returns pad-filled array, get_block not called."""
    pad_val = 55.0
    provider = _make_provider(sparse_tiles={(0, 0)}, pad=pad_val)
    result = read_block_or_pad(
        provider,
        tile_row=0,
        tile_col=0,
        resolution_level=RES_LEVEL,
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=pad_val,
        dtype=DTYPE,
    )
    assert result.shape == (NUM_BANDS, TILE_H, TILE_W)
    np.testing.assert_array_equal(result, np.uint8(pad_val))
    assert len(provider.get_block_calls) == 0


# ----------------------------------------------------------------------
# 5.10 — Out-of-bounds windows (padding for regions beyond image extent)
# ----------------------------------------------------------------------


def test_stitch_oob_extends_right():
    """Window extends past right edge — OOB region filled with pad."""
    provider = _make_provider(grid_rows=1, grid_cols=2)
    # Image is 1x2 blocks = 64x128 pixels. Request extends 32px past right edge.
    result = stitch_source_blocks(
        provider,
        row_range=(0, TILE_H),
        col_range=(0, 2 * TILE_W + 32),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, TILE_H, 2 * TILE_W + 32)
    # Block (0,0) = 1, Block (0,1) = 2, OOB = pad (0)
    np.testing.assert_array_equal(result[:, :, :TILE_W], 1)
    np.testing.assert_array_equal(result[:, :, TILE_W : 2 * TILE_W], 2)
    np.testing.assert_array_equal(result[:, :, 2 * TILE_W :], 0)
    assert len(provider.get_block_calls) == 2


def test_stitch_oob_extends_bottom():
    """Window extends past bottom edge — OOB region filled with pad."""
    provider = _make_provider(grid_rows=2, grid_cols=1)
    # Image is 2x1 blocks = 128x64 pixels. Request extends 48px past bottom.
    result = stitch_source_blocks(
        provider,
        row_range=(0, 2 * TILE_H + 48),
        col_range=(0, TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 2 * TILE_H + 48, TILE_W)
    # Block (0,0) = 1, Block (1,0) = 2, OOB = pad (0)
    np.testing.assert_array_equal(result[:, :TILE_H, :], 1)
    np.testing.assert_array_equal(result[:, TILE_H : 2 * TILE_H, :], 2)
    np.testing.assert_array_equal(result[:, 2 * TILE_H :, :], 0)
    assert len(provider.get_block_calls) == 2


def test_stitch_oob_extends_both_axes():
    """Window extends past both right and bottom edges."""
    provider = _make_provider(grid_rows=1, grid_cols=1)
    # Image is 1x1 block = 64x64 pixels. Request is 128x128 (2x in each direction).
    result = stitch_source_blocks(
        provider,
        row_range=(0, 2 * TILE_H),
        col_range=(0, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 2 * TILE_H, 2 * TILE_W)
    # Top-left quadrant = block (0,0) = 1; rest = pad (0)
    np.testing.assert_array_equal(result[:, :TILE_H, :TILE_W], 1)
    np.testing.assert_array_equal(result[:, :TILE_H, TILE_W:], 0)
    np.testing.assert_array_equal(result[:, TILE_H:, :TILE_W], 0)
    np.testing.assert_array_equal(result[:, TILE_H:, TILE_W:], 0)
    assert len(provider.get_block_calls) == 1


def test_stitch_oob_fully_outside():
    """Window is completely outside image bounds — all pad."""
    provider = _make_provider(grid_rows=2, grid_cols=2)
    # Image is 128x128. Window starts at (200, 200).
    result = stitch_source_blocks(
        provider,
        row_range=(200, 264),
        col_range=(200, 264),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 64, 64)
    np.testing.assert_array_equal(result, 0)
    assert len(provider.get_block_calls) == 0


def test_stitch_oob_negative_origin():
    """Window with negative origin — left/top OOB region filled with pad."""
    provider = _make_provider(grid_rows=2, grid_cols=2)
    # Request starts at (-32, -32) and covers 96x96 pixels.
    result = stitch_source_blocks(
        provider,
        row_range=(-32, 64),
        col_range=(-32, 64),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 96, 96)
    # Top 32 rows and left 32 cols are OOB (pad=0).
    np.testing.assert_array_equal(result[:, :32, :], 0)
    np.testing.assert_array_equal(result[:, :, :32], 0)
    # Bottom-right 64x64 region comes from block (0,0) = 1.
    np.testing.assert_array_equal(result[:, 32:, 32:], 1)
    assert len(provider.get_block_calls) == 1


def test_stitch_oob_fully_negative():
    """Window entirely in negative coordinates — all pad."""
    provider = _make_provider(grid_rows=2, grid_cols=2)
    result = stitch_source_blocks(
        provider,
        row_range=(-128, -64),
        col_range=(-128, -64),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 64, 64)
    np.testing.assert_array_equal(result, 0)
    assert len(provider.get_block_calls) == 0


def test_stitch_oob_partial_overlap_with_custom_pad():
    """OOB region uses the specified pad_value, not 0."""
    pad_val = 128.0
    provider = _make_provider(grid_rows=1, grid_cols=1, pad=pad_val)
    # Image is 64x64. Window is 64x128 (extends 64px right).
    result = stitch_source_blocks(
        provider,
        row_range=(0, TILE_H),
        col_range=(0, 2 * TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=pad_val,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, TILE_H, 2 * TILE_W)
    # Block (0,0) = 1, OOB region = pad_val (128)
    np.testing.assert_array_equal(result[:, :, :TILE_W], 1)
    np.testing.assert_array_equal(result[:, :, TILE_W:], np.uint8(pad_val))


def test_read_window_oob_extends_right():
    """read_window handles OOB windows via stitch_source_blocks."""
    provider = _make_provider(grid_rows=1, grid_cols=1)
    # Image is 64x64. Request 64x128 (extends 64px right).
    result = read_window(provider, (0, 0, 2 * TILE_W, TILE_H))
    assert result.shape == (NUM_BANDS, TILE_H, 2 * TILE_W)
    np.testing.assert_array_equal(result[:, :, :TILE_W], 1)
    np.testing.assert_array_equal(result[:, :, TILE_W:], 0)


def test_read_window_oob_negative_origin():
    """read_window handles negative-origin windows."""
    provider = _make_provider(grid_rows=1, grid_cols=1)
    # Request starts at (-32, -32), size 96x96.
    result = read_window(provider, (-32, -32, 96, 96))
    assert result.shape == (NUM_BANDS, 96, 96)
    # Top 32 rows and left 32 cols are pad.
    np.testing.assert_array_equal(result[:, :32, :], 0)
    np.testing.assert_array_equal(result[:, :, :32], 0)
    # Bottom-right 64x64 = block (0,0) = 1.
    np.testing.assert_array_equal(result[:, 32:, 32:], 1)


# ----------------------------------------------------------------------
# 5.11 — Non-aligned image dimensions (edge blocks partially filled)
# ----------------------------------------------------------------------


def test_stitch_oob_non_aligned_image_extends_past_edge():
    """Image size doesn't align with tile grid; OOB window still pads correctly.

    Image is 100x100 pixels with 64x64 tiles (2x2 grid, edge blocks only
    36px wide/tall of real data). A window extending past 100px must pad.
    """
    provider = _MockProvider(
        grid_rows=2,
        grid_cols=2,
        tile_height=TILE_H,
        tile_width=TILE_W,
        image_rows=100,
        image_cols=100,
    )
    # Request 128x128 starting at origin — extends 28px past image edge.
    result = stitch_source_blocks(
        provider,
        row_range=(0, 128),
        col_range=(0, 128),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 128, 128)
    # All 4 blocks in the 2x2 grid should be read (they all contain some data).
    assert len(provider.get_block_calls) == 4
    # Block (0,0)=1 fills the top-left 64x64.
    np.testing.assert_array_equal(result[:, :TILE_H, :TILE_W], 1)
    # Block (0,1)=2 fills the top-right tile region.
    np.testing.assert_array_equal(result[:, :TILE_H, TILE_W : 2 * TILE_W], 2)


def test_stitch_oob_non_aligned_window_starts_in_last_partial_block():
    """Window starting in the last (partial) block row extends past image.

    Image is 100x100 with 64px tiles. Block row 1 covers pixels 64-127
    but only 64-99 contain real data. A window from row 80 to 140 must
    read block row 1 and pad everything beyond the grid.
    """
    provider = _MockProvider(
        grid_rows=2,
        grid_cols=1,
        tile_height=TILE_H,
        tile_width=TILE_W,
        image_rows=100,
        image_cols=64,
    )
    result = stitch_source_blocks(
        provider,
        row_range=(80, 140),
        col_range=(0, TILE_W),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 60, TILE_W)
    # Block (1,0) fill value = 1*1 + 0 + 1 = 2 (grid_cols=1)
    # First 48 rows come from block (1,0) [rows 80-127 of image, i.e. src_y0=16..63]
    np.testing.assert_array_equal(result[:, :48, :], 2)
    # Last 12 rows (140-128=12) are beyond the grid — padded.
    np.testing.assert_array_equal(result[:, 48:, :], 0)
    # Only block row 1 should be read.
    assert len(provider.get_block_calls) == 1
    assert provider.get_block_calls[0][:2] == (1, 0)


def test_stitch_oob_non_aligned_fully_outside_partial_image():
    """Window entirely past a non-aligned image edge — all pad.

    Image is 100x100 (grid 2x2 with 64px tiles). Window starts at (100, 100).
    Block (1,1) is the last block (covers 64-127 in each axis), but the
    image only has 100 pixels. The window starts at pixel 100 which is
    within block (1,1)'s range but past the image's declared num_rows/cols.
    The clamping logic should exclude this block since max_block = (100-1)//64 = 1,
    but the window starts at row 100 which yields first_block = 100//64 = 1.
    Block (1,1) is valid, so it should be read.
    """
    provider = _MockProvider(
        grid_rows=2,
        grid_cols=2,
        tile_height=TILE_H,
        tile_width=TILE_W,
        image_rows=100,
        image_cols=100,
    )
    # Window starts at pixel (100, 100), extends to (160, 160).
    # max_block_row = (100-1)//64 = 1, max_block_col = (100-1)//64 = 1
    # first_block_row = 100//64 = 1, first_block_col = 100//64 = 1
    # So block (1,1) IS within the valid grid and will be read.
    result = stitch_source_blocks(
        provider,
        row_range=(100, 160),
        col_range=(100, 160),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 60, 60)
    # Block (1,1) fill value = 1*2 + 1 + 1 = 4
    # The window 100-127 overlaps block (1,1) range 64-127.
    # src_y0 = 100-64=36, src_y1 = min(160,128)-64=64 -> 28 rows from block
    # The remaining 32 rows (128-160) are OOB.
    np.testing.assert_array_equal(result[:, :28, :28], 4)
    np.testing.assert_array_equal(result[:, 28:, :], 0)
    np.testing.assert_array_equal(result[:, :, 28:], 0)


def test_stitch_oob_non_aligned_window_completely_past_image():
    """Window starts well past the non-aligned image extent — all pad.

    Image is 100x100 (grid 2x2 with 64px tiles). Window starts at (200, 200).
    max_block_row = (100-1)//64 = 1, first_block_row = 200//64 = 3 > 1.
    No blocks iterated.
    """
    provider = _MockProvider(
        grid_rows=2,
        grid_cols=2,
        tile_height=TILE_H,
        tile_width=TILE_W,
        image_rows=100,
        image_cols=100,
    )
    result = stitch_source_blocks(
        provider,
        row_range=(200, 264),
        col_range=(200, 264),
        tile_height=TILE_H,
        tile_width=TILE_W,
        num_bands=NUM_BANDS,
        pad_value=PAD,
        dtype=DTYPE,
        resolution_level=RES_LEVEL,
    )
    assert result.shape == (NUM_BANDS, 64, 64)
    np.testing.assert_array_equal(result, 0)
    assert len(provider.get_block_calls) == 0


def test_read_window_non_aligned_oob():
    """read_window with non-aligned image and OOB window."""
    provider = _MockProvider(
        grid_rows=2,
        grid_cols=2,
        tile_height=TILE_H,
        tile_width=TILE_W,
        image_rows=100,
        image_cols=100,
    )
    # Request 150x150 from origin on a 100x100 image.
    result = read_window(provider, (0, 0, 150, 150))
    assert result.shape == (NUM_BANDS, 150, 150)
    # All 4 grid blocks should be read.
    assert len(provider.get_block_calls) == 4
    # Block (0,0) = 1
    np.testing.assert_array_equal(result[:, :TILE_H, :TILE_W], 1)
