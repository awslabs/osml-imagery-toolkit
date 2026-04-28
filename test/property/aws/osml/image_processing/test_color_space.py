#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for the color_space module.

Tests correctness properties of TRC encode/decode functions.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.color_space import (
    _adobe_rgb_trc_decode,
    _adobe_rgb_trc_encode,
    _prophoto_trc_decode,
    _prophoto_trc_encode,
    _srgb_trc_decode,
    _srgb_trc_encode,
)
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_trc_pairs = st.sampled_from(
    [
        ("srgb", _srgb_trc_decode, _srgb_trc_encode),
        ("prophoto_rgb", _prophoto_trc_decode, _prophoto_trc_encode),
        ("adobe_rgb", _adobe_rgb_trc_decode, _adobe_rgb_trc_encode),
    ]
)


@st.composite
def trc_round_trip_data(draw):
    """Generate a TRC pair and an array of values in [0, 1]."""
    name, decode, encode = draw(_trc_pairs)
    height = draw(st.integers(min_value=1, max_value=16))
    width = draw(st.integers(min_value=1, max_value=16))
    values = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(height, width),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    return name, decode, encode, values


# ---------------------------------------------------------------------------
# Property 10: TRC encode/decode round-trip
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 10: TRC encode/decode round-trip
@pytest.mark.property
@given(data=trc_round_trip_data())
@settings(pbt_settings)
def test_trc_encode_decode_round_trip(data):
    """**Validates: Requirements 10.4**

    For any supported RGB color space (sRGB, ProPhoto RGB, AdobeRGB) and
    any array of values in [0, 1], applying the TRC decode (linearize)
    followed by the TRC encode (gamma compress) SHALL produce values
    approximately equal to the original input (within floating-point
    tolerance).
    """
    name, decode, encode, values = data

    # decode (linearize) then encode (gamma compress)
    linearized = decode(values)
    round_tripped = encode(linearized)

    np.testing.assert_allclose(
        round_tripped,
        values,
        atol=1e-10,
        rtol=1e-7,
        err_msg=f"TRC round-trip failed for {name}",
    )
