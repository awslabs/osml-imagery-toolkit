#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""OpenCV-based resampling functions for image pyramid generation.

This module defines the :data:`ResampleFunc` type alias, which describes
the common signature for every pluggable resampler used by the image
pyramid operations (``PyramidBuilder``, ``DownsampledImageProvider``,
``build_pyramid_levels``). It also provides four thin wrappers around
``cv2.resize`` — :func:`nearest_neighbor_resample`, :func:`bilinear_resample`,
:func:`area_resample`, and :func:`lanczos_resample` — covering OpenCV's
most common interpolation modes.

All resamplers in this module follow the same contract:

- Input and output are in CHW (bands, height, width) layout for 3-D arrays,
  or (H, W) for 2-D single-band arrays. The returned array preserves
  the input layout.
- The input dtype is preserved.
- The input array is never mutated — a new array is always returned
  (the identity case returns the input itself, which is still
  element-wise equal to the input).
- ``target_rows`` and ``target_cols`` must be strictly positive.
- The input must be 2-D (H, W) or 3-D (C, H, W); any other ``ndim``
  raises :class:`ValueError`.

For SIPS-compliant downsampling (NGA.STND.0014 v2.4 Section 2.2), see the
companion :mod:`aws.osml.image_processing.sips_resample` module.
"""

from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

#: Type alias for any resampler. A ``ResampleFunc`` takes a CHW (or HW)
#: array and target ``(rows, cols)`` dimensions and returns a new array
#: at the target size with the same dtype and layout.
ResampleFunc = Callable[[NDArray, int, int], NDArray]


def nearest_neighbor_resample(image: NDArray, target_rows: int, target_cols: int) -> NDArray:
    """Nearest-neighbor resampling via ``cv2.INTER_NEAREST``.

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout.
    :param target_rows: Target number of rows. Must be > 0.
    :param target_cols: Target number of columns. Must be > 0.
    :return: A new NDArray with the target dimensions, same dtype and
        layout as the input. Returns the input unchanged when target
        dimensions match the source dimensions.
    :raises ValueError: If the input is not 2-D or 3-D, or if either
        target dimension is <= 0.
    """
    return _resize(image, target_rows, target_cols, cv2.INTER_NEAREST)


def bilinear_resample(image: NDArray, target_rows: int, target_cols: int) -> NDArray:
    """Bilinear interpolation via ``cv2.INTER_LINEAR``.

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout.
    :param target_rows: Target number of rows. Must be > 0.
    :param target_cols: Target number of columns. Must be > 0.
    :return: A new NDArray with the target dimensions, same dtype and
        layout as the input. Returns the input unchanged when target
        dimensions match the source dimensions.
    :raises ValueError: If the input is not 2-D or 3-D, or if either
        target dimension is <= 0.
    """
    return _resize(image, target_rows, target_cols, cv2.INTER_LINEAR)


def area_resample(image: NDArray, target_rows: int, target_cols: int) -> NDArray:
    """Area-based resampling via ``cv2.INTER_AREA``.

    OpenCV's recommended interpolation for decimation — reduces aliasing
    compared to nearest-neighbor or bilinear while being cheaper than
    Lanczos.

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout.
    :param target_rows: Target number of rows. Must be > 0.
    :param target_cols: Target number of columns. Must be > 0.
    :return: A new NDArray with the target dimensions, same dtype and
        layout as the input. Returns the input unchanged when target
        dimensions match the source dimensions.
    :raises ValueError: If the input is not 2-D or 3-D, or if either
        target dimension is <= 0.
    """
    return _resize(image, target_rows, target_cols, cv2.INTER_AREA)


def lanczos_resample(image: NDArray, target_rows: int, target_cols: int) -> NDArray:
    """Lanczos interpolation via ``cv2.INTER_LANCZOS4`` (8x8 neighborhood).

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout.
    :param target_rows: Target number of rows. Must be > 0.
    :param target_cols: Target number of columns. Must be > 0.
    :return: A new NDArray with the target dimensions, same dtype and
        layout as the input. Returns the input unchanged when target
        dimensions match the source dimensions.
    :raises ValueError: If the input is not 2-D or 3-D, or if either
        target dimension is <= 0.
    """
    return _resize(image, target_rows, target_cols, cv2.INTER_LANCZOS4)


def _resize(image: NDArray, target_rows: int, target_cols: int, interpolation: int) -> NDArray:
    """Shared implementation for the OpenCV-based resamplers.

    Validates the input dimensions, handles the CHW ↔ HWC transpose
    expected by ``cv2.resize``, preserves the input dtype, and returns
    the input unchanged when the requested dimensions already match.

    :param image: Input array, 2-D (H, W) or 3-D (C, H, W).
    :param target_rows: Target number of rows. Must be > 0.
    :param target_cols: Target number of columns. Must be > 0.
    :param interpolation: One of the ``cv2.INTER_*`` constants.
    :return: Resampled array with the same dtype and layout as the input.
    :raises ValueError: If the input is not 2-D or 3-D, or if either
        target dimension is <= 0.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2-D (H, W) or 3-D (C, H, W) array, got {image.ndim}-D")
    if target_rows <= 0 or target_cols <= 0:
        raise ValueError(f"target_rows and target_cols must be > 0, got ({target_rows}, {target_cols})")

    if image.ndim == 2:
        src_rows, src_cols = image.shape
        if src_rows == target_rows and src_cols == target_cols:
            return image
        # cv2.resize takes dsize as (width, height) — column-major order.
        resized = cv2.resize(image, (target_cols, target_rows), interpolation=interpolation)
        # cv2.resize may promote the dtype internally; force it back.
        if resized.dtype != image.dtype:
            resized = resized.astype(image.dtype)
        return resized

    # 3-D CHW input
    src_rows, src_cols = image.shape[1], image.shape[2]
    if src_rows == target_rows and src_cols == target_cols:
        return image

    # cv2.resize operates on HWC; transpose in and back out.
    hwc = np.ascontiguousarray(image.transpose(1, 2, 0))
    resized_hwc = cv2.resize(hwc, (target_cols, target_rows), interpolation=interpolation)
    # cv2.resize squeezes a trailing size-1 channel dim; re-add it for
    # single-band 3-D input so the output shape stays (1, H', W').
    if resized_hwc.ndim == 2:
        resized_hwc = resized_hwc[:, :, np.newaxis]
    if resized_hwc.dtype != image.dtype:
        resized_hwc = resized_hwc.astype(image.dtype)
    return resized_hwc.transpose(2, 0, 1)
