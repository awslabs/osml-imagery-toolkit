#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for the lut module.

Tests correctness properties of apply_lut().
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.lut import apply_lut
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_integer_dtypes = st.sampled_from([np.uint8, np.uint16])


@st.composite
def chw_image_and_lut(draw):
    """Generate a random CHW integer image and a matching 1-D LUT.

    The image values are constrained to [0, len(lut)-1] so the property
    can verify exact element-wise equality without clipping effects.
    """
    dtype = draw(_integer_dtypes)
    num_bands = draw(st.integers(min_value=1, max_value=4))
    height = draw(st.integers(min_value=1, max_value=32))
    width = draw(st.integers(min_value=1, max_value=32))

    if dtype == np.uint8:
        lut_len = 256
        max_val = 255
    else:
        # Keep LUT size manageable for uint16 — use a subset
        lut_len = draw(st.integers(min_value=2, max_value=1024))
        max_val = lut_len - 1

    image = draw(
        hnp.arrays(
            dtype=dtype,
            shape=(num_bands, height, width),
            elements=st.integers(min_value=0, max_value=max_val),
        )
    )

    lut = draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=(lut_len,),
            elements=st.integers(min_value=0, max_value=255),
        )
    )

    return image, lut


# ---------------------------------------------------------------------------
# Property 13: LUT per-band application correctness
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 13: LUT per-band application correctness
@pytest.mark.property
@given(data=chw_image_and_lut())
@settings(pbt_settings)
def test_lut_per_band_application_correctness(data):
    """**Validates: Requirements 13.2**

    For any valid CHW NDArray with values in [0, len(lut)-1] and any 1-D
    LUT array, apply_lut(image, lut) SHALL produce an output where each
    pixel output[b, y, x] == lut[image[b, y, x]] for every band b, row y,
    and column x.
    """
    image, lut = data

    result = apply_lut(image, lut)

    # Verify element-wise: output[b, y, x] == lut[image[b, y, x]]
    expected = lut[image]
    np.testing.assert_array_equal(
        result,
        expected,
        err_msg="apply_lut output does not match element-wise LUT lookup",
    )
