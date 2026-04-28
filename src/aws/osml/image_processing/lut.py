#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Look-Up Table application for image processing.

This module provides the :func:`apply_lut` function for applying a 1-D
look-up table to each band of a CHW image array. It implements the SIPS
Section 2.5 per-pixel mapping operation.

For uint8 input with a 256-entry uint8 LUT, ``cv2.LUT()`` is used for
SIMD-accelerated performance. For wider bit depths (uint16, etc.), NumPy
fancy indexing is used.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def apply_lut(image: NDArray, lut: NDArray) -> NDArray:
    """Apply a 1-D look-up table independently to each band of an image.

    Maps each pixel value through the LUT: ``output[b, y, x] = lut[image[b, y, x]]``.
    Input values outside ``[0, len(lut)-1]`` are clipped to the LUT
    boundaries before lookup.

    :param image: Input array in CHW (bands, height, width) layout.
        A 2-D array (H, W) is treated as a single band.
    :param lut: 1-D look-up table array. The same LUT is applied to
        every band.
    :return: A new CHW NDArray with the same spatial dimensions and
        number of bands as the input. The output dtype matches the
        LUT's dtype.
    :raises ValueError: If ``lut`` is not 1-D, or if the input image
        is not 2-D or 3-D.
    """
    if lut.ndim != 1:
        raise ValueError(f"LUT must be 1-D, got {lut.ndim}-D")

    # Normalize to 3-D CHW
    squeeze = False
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
        squeeze = True
    elif image.ndim != 3:
        raise ValueError(f"Expected 2-D (H, W) or 3-D (C, H, W) array, got {image.ndim}-D")

    lut_len = len(lut)

    if image.dtype == np.uint8 and lut_len == 256 and lut.dtype == np.uint8:
        # Fast path: cv2.LUT is SIMD-optimized for uint8 → uint8.
        if image.shape[0] == 1:
            # Single band: cv2.LUT on 2-D array directly
            result = cv2.LUT(image[0], lut)[np.newaxis, :, :]
        else:
            # Multi-band with same LUT: transpose to HWC so cv2.LUT
            # processes all channels in a single C++ call, then
            # transpose back to CHW.
            hwc = np.ascontiguousarray(image.transpose(1, 2, 0))
            hwc_result = cv2.LUT(hwc, lut)
            result = hwc_result.transpose(2, 0, 1)
    else:
        # General path: numpy fancy indexing handles all bands in one shot.
        # Skip the clip when the LUT covers the full dtype range, since
        # every possible input value is a valid index.
        if np.issubdtype(image.dtype, np.integer) and lut_len > np.iinfo(image.dtype).max:
            result = lut[image]
        else:
            clipped = np.clip(image, 0, lut_len - 1)
            if not np.issubdtype(clipped.dtype, np.integer):
                clipped = clipped.astype(np.intp)
            result = lut[clipped]

    if squeeze:
        result = result[0]

    return result
