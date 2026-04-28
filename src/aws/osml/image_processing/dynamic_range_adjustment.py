#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Dynamic range adjustment for image processing.

This module provides the :class:`DRAParameters` class for computing
percentile-based clipping boundaries from histograms, and the
:func:`dynamic_range_adjust` function for mapping wide-range pixel
values to a narrower output range suitable for display.

These components are migrated from ``aws.osml.gdal.dynamic_range_adjustment``
and extended with a stateless array-level DRA function that operates on
CHW NumPy arrays.

Typical usage::

    # 1. Compute statistics once from the full image
    stats = compute_image_statistics(provider)

    # 2. Build DRAParameters once per band
    params = [DRAParameters.from_counts(band.histogram, ...) for band in stats.bands]

    # 3. Build LUTs once (reusable across all blocks)
    luts = [p.build_lut(np.uint16, np.uint8) for p in params]

    # 4. Apply to each block — no recomputation
    for block in blocks:
        result = dynamic_range_adjust(block, luts)
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .lut import apply_lut


class DRAParameters:
    """Parameters for Dynamic Range Adjustment of image pixel values.

    Manages suggested and actual min/max values used to map pixel values
    from a wide source range (e.g. 11-bit panchromatic) to a narrower
    output range (e.g. 8-bit grayscale).

    :param suggested_min_value: Suggested minimum value of the relevant pixel range.
    :param suggested_max_value: Suggested maximum value of the relevant pixel range.
    :param actual_min_value: Actual minimum value of pixels in the image.
    :param actual_max_value: Actual maximum value of pixels in the image.
    """

    def __init__(
        self,
        suggested_min_value: float,
        suggested_max_value: float,
        actual_min_value: float,
        actual_max_value: float,
    ):
        self.suggested_min_value = suggested_min_value
        self.suggested_max_value = suggested_max_value
        self.actual_min_value = actual_min_value
        self.actual_max_value = actual_max_value

    @staticmethod
    def from_counts(
        counts,
        first_bucket_value: Optional[float] = None,
        last_bucket_value: Optional[float] = None,
        min_percentage: float = 0.02,
        max_percentage: float = 0.98,
        a: float = 0.2,
        b: float = 0.4,
    ) -> "DRAParameters":
        """Compute DRA parameters from a histogram of pixel values.

        :param counts: Histogram of pixel values. Accepts both
            ``list[float]`` and ``NDArray``.
        :param first_bucket_value: Pixel value of the first bucket.
            Defaults to 0.
        :param last_bucket_value: Pixel value of the last bucket.
            Defaults to the number of histogram bins.
        :param min_percentage: Set point for low intensity pixels that
            may be outliers.
        :param max_percentage: Set point for high intensity pixels that
            may be outliers.
        :param a: Weighting factor for the low intensity range.
        :param b: Weighting factor for the high intensity range.
        :return: DRA parameters containing recommended and actual ranges.
        """
        counts = np.asarray(counts, dtype=np.float64)

        num_histogram_bins = len(counts)
        if not first_bucket_value:
            first_bucket_value = 0
        if not last_bucket_value:
            last_bucket_value = num_histogram_bins

        # Find the first and last non-zero bins
        non_zero = np.nonzero(counts)[0]
        if len(non_zero) > 0:
            actual_min_value = int(non_zero[0])
            actual_max_value = int(non_zero[-1])
        else:
            actual_min_value = 0
            actual_max_value = num_histogram_bins - 1

        # Cumulative distribution (vectorized)
        cumulative_counts = np.cumsum(counts)

        # Find the values that exclude the lowest and highest percentages
        # of the counts. This identifies the range that contains most of
        # the pixels while excluding outliers.
        max_counts = cumulative_counts[-1]
        low_threshold = min_percentage * max_counts
        high_threshold = max_percentage * max_counts

        # e_min: first index where cumulative_counts >= low_threshold
        e_min = int(np.searchsorted(cumulative_counts, low_threshold, side="left"))

        # e_max: last index where cumulative_counts <= high_threshold.
        # searchsorted(..., side="right") gives the first index > threshold,
        # so subtract 1 to get the last index <= threshold.
        e_max_raw = int(np.searchsorted(cumulative_counts, high_threshold, side="right")) - 1
        e_max = max(e_max_raw, 0)

        min_value = max(actual_min_value, e_min - a * (e_max - e_min))
        max_value = min(actual_max_value, e_max + b * (e_max - e_min))

        value_step = (last_bucket_value - first_bucket_value) / num_histogram_bins
        return DRAParameters(
            suggested_min_value=min_value * value_step + first_bucket_value,
            suggested_max_value=max_value * value_step + first_bucket_value,
            actual_min_value=actual_min_value * value_step + first_bucket_value,
            actual_max_value=actual_max_value * value_step + first_bucket_value,
        )

    def build_lut(
        self,
        input_dtype: np.dtype,
        output_dtype: np.dtype = np.uint8,
        range_adjustment: str = "dra",
    ) -> NDArray:
        """Build a 1-D look-up table that performs the DRA linear mapping.

        Creates a table with one entry per possible input value. The
        linear mapping is computed once per table entry rather than once
        per pixel. The resulting LUT can be reused across all blocks
        that share the same DRA parameters.

        :param input_dtype: The integer dtype of the input image
            (e.g. ``numpy.uint8``, ``numpy.uint16``).
        :param output_dtype: Desired output dtype for the LUT entries.
            Default ``numpy.uint8``.
        :param range_adjustment: Which boundaries to use. ``"dra"``
            uses ``suggested_min_value`` / ``suggested_max_value``
            (percentile-based clipping). ``"minmax"`` uses
            ``actual_min_value`` / ``actual_max_value`` (full range).
        :return: A 1-D NDArray suitable for use with
            :func:`~aws.osml.image_processing.lut.apply_lut` or
            :func:`dynamic_range_adjust`.
        :raises ValueError: If ``range_adjustment`` is not ``"dra"``
            or ``"minmax"``, or if ``input_dtype`` is not an integer type.
        """
        if range_adjustment not in ("dra", "minmax"):
            raise ValueError(f"Unsupported range_adjustment: {range_adjustment!r}. Must be 'dra' or 'minmax'.")

        if not np.issubdtype(input_dtype, np.integer):
            raise ValueError(f"build_lut requires integer input_dtype, got {input_dtype}.")

        if range_adjustment == "dra":
            src_min = self.suggested_min_value
            src_max = self.suggested_max_value
        else:
            src_min = self.actual_min_value
            src_max = self.actual_max_value

        out_info = np.iinfo(output_dtype) if np.issubdtype(output_dtype, np.integer) else None
        out_max = float(out_info.max) if out_info is not None else 1.0

        info = np.iinfo(input_dtype)
        lut_input = np.arange(info.min, info.max + 1, dtype=np.float64)
        src_range = src_max - src_min
        if src_range == 0.0:
            lut_output = np.full_like(lut_input, out_max / 2.0)
        else:
            lut_output = (lut_input - src_min) / src_range * out_max
        lut_output = np.clip(lut_output, 0, out_max)
        return lut_output.astype(output_dtype)

    def __repr__(self):
        return (
            f"DRAParameters(min_value={self.suggested_min_value}, "
            f"max_value={self.suggested_max_value}, "
            f"e_first={self.actual_min_value}, "
            f"e_last={self.actual_max_value}, "
            f")"
        )


def dynamic_range_adjust(
    image: NDArray,
    luts: List[NDArray],
) -> NDArray:
    """Apply dynamic range adjustment to a CHW image array using pre-built LUTs.

    Each band is mapped through its corresponding LUT via
    :func:`~aws.osml.image_processing.lut.apply_lut`. The LUTs are
    built once from :meth:`DRAParameters.build_lut` and reused across
    every block — no per-block recomputation.

    :param image: Input array in CHW (bands, height, width) layout.
        A 2-D array (H, W) is treated as a single band.
    :param luts: Pre-built look-up tables, one per band. Typically
        produced by :meth:`DRAParameters.build_lut`. Length must match
        the number of bands in the input image.
    :return: A new CHW NDArray with the dtype of the LUT entries.
    :raises ValueError: If the length of ``luts`` does not match the
        number of bands in the input image, or if the input is not
        2-D or 3-D.
    """
    # Normalize to 3-D CHW
    squeeze = False
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
        squeeze = True
    elif image.ndim != 3:
        raise ValueError(f"Expected 2-D (H, W) or 3-D (C, H, W) array, got {image.ndim}-D")

    num_bands = image.shape[0]
    if len(luts) != num_bands:
        raise ValueError(f"Band count mismatch: image has {num_bands} bands but luts has {len(luts)} entries")

    output_dtype = luts[0].dtype
    result = np.empty_like(image, dtype=output_dtype)

    for b in range(num_bands):
        result[b] = apply_lut(image[b], luts[b])

    if squeeze:
        result = result[0]

    return result
