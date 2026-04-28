#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import numpy as np


def read_single_band(image_asset) -> np.ndarray:
    """
    Read all blocks from a single-band image asset into a 2D array.

    :param image_asset: an osml-imagery-io image asset with block-oriented access
    :return: 2D numpy array of shape (height, width)
    """
    grid_rows, grid_cols = image_asset.block_grid_size
    block_h = image_asset.num_pixels_per_block_vertical
    block_w = image_asset.num_pixels_per_block_horizontal
    height = image_asset.num_rows
    width = image_asset.num_columns

    result = np.empty((height, width), dtype=np.float64)
    for r in range(grid_rows):
        for c in range(grid_cols):
            block = image_asset.get_block(r, c)
            band_block = block[0] if block.ndim == 3 else block
            row_start = r * block_h
            col_start = c * block_w
            row_end = min(row_start + band_block.shape[0], height)
            col_end = min(col_start + band_block.shape[1], width)
            result[row_start:row_end, col_start:col_end] = band_block[: row_end - row_start, : col_end - col_start]
    return result
