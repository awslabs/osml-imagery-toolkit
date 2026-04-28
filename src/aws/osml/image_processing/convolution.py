#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""SIPS-compliant convolution and correlation for image processing.

This module provides :func:`sips_convolve` and :func:`sips_correlate`
functions that implement the convolution and correlation operations
defined in NGA.STND.0014 (SIPS) Section 2.4.

Both functions use the SIPS "Mirror Edge - Odd" boundary convention,
which corresponds to OpenCV's ``cv2.BORDER_REFLECT_101``.

Key difference between the two operations:

- **Convolution** flips the kernel in both dimensions before applying
  the weighted sum (Equation 2.9 in SIPS).
- **Correlation** applies the kernel directly without flipping
  (Equation 2.12 in SIPS).

For even-sized kernels, the two operations define different anchor
points (SIPS Equations 2.10 vs 2.13). This module handles the anchor
offset explicitly so that both operations produce SIPS-compliant
results for both odd and even kernel dimensions.

Since ``cv2.filter2D`` performs correlation by default, ``sips_convolve``
flips the kernel before calling ``filter2D``, while ``sips_correlate``
passes the kernel directly.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def sips_convolve(image: NDArray, kernel: NDArray) -> NDArray:
    """Apply SIPS-compliant convolution to an image.

    Convolves the image with the kernel using the SIPS Mirror Edge - Odd
    boundary convention (``cv2.BORDER_REFLECT_101``). The kernel is
    flipped in both dimensions per the convolution definition in SIPS
    Equation 2.9.

    For even-sized kernels, the anchor point follows SIPS Equation 2.10:
    the center is at ``(M//2, N//2)`` in the flipped kernel, which
    matches OpenCV's default anchor.

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout.
    :param kernel: 2-D convolution kernel.
    :return: A new NDArray with the same dimensions as the input.
    :raises ValueError: If the kernel is not 2-D, or if the input
        image is not 2-D or 3-D.
    """
    _validate_inputs(image, kernel)
    flipped_kernel = kernel[::-1, ::-1]
    # SIPS Eq 2.10 places the convolution anchor at index (M//2, N//2)
    # in the original kernel. After flipping, that position maps to
    # (M-1 - M//2, N-1 - N//2) in the flipped kernel. For odd kernels
    # this equals the default OpenCV anchor; for even kernels it differs.
    kh, kw = kernel.shape
    anchor = _convolution_anchor(kh, kw)
    return _apply_filter(image, flipped_kernel, anchor=anchor)


def sips_correlate(image: NDArray, kernel: NDArray) -> NDArray:
    """Apply SIPS-compliant correlation to an image.

    Correlates the image with the kernel using the SIPS Mirror Edge - Odd
    boundary convention (``cv2.BORDER_REFLECT_101``). The kernel is
    applied directly without flipping, per the correlation definition
    in SIPS Equation 2.12.

    For even-sized kernels, the anchor point follows SIPS Equation 2.13:
    the center is at ``(M//2 - 1, N//2 - 1)`` (0-indexed), which differs
    from OpenCV's default of ``(M//2, N//2)``. This function sets the
    anchor explicitly for even-sized kernels.

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout.
    :param kernel: 2-D correlation kernel.
    :return: A new NDArray with the same dimensions as the input.
    :raises ValueError: If the kernel is not 2-D, or if the input
        image is not 2-D or 3-D.
    """
    _validate_inputs(image, kernel)

    # For correlation with even-sized kernels, SIPS Equation 2.13 places
    # the anchor one position earlier than OpenCV's default. For odd
    # kernels, both conventions agree.
    kh, kw = kernel.shape
    anchor = _correlation_anchor(kh, kw)
    return _apply_filter(image, kernel, anchor=anchor)


def _validate_inputs(image: NDArray, kernel: NDArray) -> None:
    """Validate image and kernel dimensions.

    :param image: Input image array.
    :param kernel: Kernel array.
    :raises ValueError: If the kernel is not 2-D, or if the image
        is not 2-D or 3-D.
    """
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2-D, got {kernel.ndim}-D")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2-D (H, W) or 3-D (C, H, W) array, got {image.ndim}-D")


def _convolution_anchor(kh: int, kw: int) -> tuple:
    """Compute the SIPS convolution anchor in the **flipped** kernel.

    SIPS Equation 2.10 (even) places the convolution center at index
    ``M//2`` in the original kernel. After flipping, that position maps
    to ``M - 1 - M//2``. For odd kernels (Eq 2.11), the center is at
    ``(M-1)//2``, which after flipping is also ``(M-1)//2`` — matching
    OpenCV's default.

    :param kh: Kernel height.
    :param kw: Kernel width.
    :return: (anchor_y, anchor_x) tuple for cv2.filter2D on the
        flipped kernel.
    """
    ay = kh - 1 - kh // 2  # maps original center to flipped position
    ax = kw - 1 - kw // 2
    return (ay, ax)


def _correlation_anchor(kh: int, kw: int) -> tuple:
    """Compute the SIPS correlation anchor for a kernel of given size.

    For odd-sized kernels, the anchor is the center element, matching
    OpenCV's default. For even-sized kernels, SIPS Equation 2.13 places
    the anchor one position earlier in each even dimension.

    :param kh: Kernel height.
    :param kw: Kernel width.
    :return: (anchor_y, anchor_x) tuple for cv2.filter2D.
    """
    # SIPS Eq 2.13 (even): a = -M/2 + 1, so center index = M/2 - 1
    # SIPS Eq 2.14 (odd):  a = (1-M)/2, so center index = (M-1)/2
    # OpenCV default: center index = M//2
    # For odd M: (M-1)/2 == M//2, so they agree.
    # For even M: M/2 - 1 != M//2, so we must set anchor explicitly.
    ay = (kh - 1) // 2  # (kh-1)//2 gives M/2-1 for even, (M-1)/2 for odd
    ax = (kw - 1) // 2
    return (ay, ax)


def _apply_filter(image: NDArray, kernel: NDArray, anchor: tuple = None) -> NDArray:
    """Apply cv2.filter2D with BORDER_REFLECT_101 to each band.

    Handles CHW ↔ HWC transposition for OpenCV and preserves the
    input layout in the output.

    :param image: Input array, 2-D (H, W) or 3-D (C, H, W).
    :param kernel: 2-D kernel (already flipped for convolution, or
        direct for correlation).
    :param anchor: Optional (y, x) anchor point for cv2.filter2D.
        When None, OpenCV uses its default (kernel center).
    :return: Filtered array with the same dimensions as input.
    """
    kernel_f64 = kernel.astype(np.float64)
    filter_kwargs = dict(
        ddepth=-1,
        kernel=kernel_f64,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if anchor is not None:
        # cv2.filter2D expects anchor as (x, y) — column, row order
        filter_kwargs["anchor"] = (anchor[1], anchor[0])

    if image.ndim == 2:
        return cv2.filter2D(image, **filter_kwargs)

    # CHW → process per band or via HWC
    num_bands = image.shape[0]
    if num_bands == 1:
        filtered = cv2.filter2D(image[0], **filter_kwargs)
        return filtered[np.newaxis, :, :]

    hwc = np.ascontiguousarray(image.transpose(1, 2, 0))
    filtered_hwc = cv2.filter2D(hwc, **filter_kwargs)
    return filtered_hwc.transpose(2, 0, 1)
