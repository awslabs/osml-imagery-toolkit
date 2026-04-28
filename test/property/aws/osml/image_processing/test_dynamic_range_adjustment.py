#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for the dynamic_range_adjustment module.

Tests correctness properties of dynamic_range_adjust().
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.dynamic_range_adjustment import DRAParameters, dynamic_range_adjust
from aws.osml.image_processing.statistics import compute_statistics
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_integer_dtypes = st.sampled_from([np.uint8, np.uint16])


@st.composite
def chw_image_with_luts(draw):
    """Generate a random CHW integer image and matching pre-built LUTs.

    Returns a tuple of (image, luts).
    """
    dtype = draw(_integer_dtypes)
    num_bands = draw(st.integers(min_value=1, max_value=4))
    height = draw(st.integers(min_value=2, max_value=32))
    width = draw(st.integers(min_value=2, max_value=32))
    mode = draw(st.sampled_from(["dra", "minmax"]))

    if dtype == np.uint8:
        max_val = 255
    else:
        max_val = 65535

    image = draw(
        hnp.arrays(
            dtype=dtype,
            shape=(num_bands, height, width),
            elements=st.integers(min_value=0, max_value=max_val),
        )
    )

    stats = compute_statistics(image, num_bins=256)
    luts = []
    for band in stats.bands:
        params = DRAParameters.from_counts(
            counts=band.histogram,
            first_bucket_value=float(band.bin_edges[0]),
            last_bucket_value=float(band.bin_edges[-1]),
        )
        luts.append(params.build_lut(image.dtype, np.uint8, range_adjustment=mode))

    return image, luts


# ---------------------------------------------------------------------------
# Property 8: DRA output invariants
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 8: DRA output invariants
@pytest.mark.property
@given(data=chw_image_with_luts())
@settings(pbt_settings)
def test_dra_output_invariants(data):
    """**Validates: Requirements 8.4, 8.5, 8.8**

    For any valid CHW NDArray and corresponding pre-built LUTs, the output
    of dynamic_range_adjust() SHALL have:
    (a) the same spatial dimensions (height, width) and number of bands as the input,
    (b) the dtype of the LUT entries, and
    (c) all values within the valid range of the output dtype (e.g., 0-255 for uint8).
    """
    image, luts = data

    result = dynamic_range_adjust(image, luts)

    # (a) Same spatial dimensions and band count
    assert result.shape == image.shape, f"Shape mismatch: input {image.shape}, output {result.shape}"

    # (b) Correct output dtype (matches LUT dtype)
    assert result.dtype == np.uint8, f"dtype mismatch: expected uint8, got {result.dtype}"

    # (c) All values within valid range of output dtype
    out_info = np.iinfo(np.uint8)
    assert np.all(result >= out_info.min), f"Output contains values below {out_info.min}: min={result.min()}"
    assert np.all(result <= out_info.max), f"Output contains values above {out_info.max}: max={result.max()}"
