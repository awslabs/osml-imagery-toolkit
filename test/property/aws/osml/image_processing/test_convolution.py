#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for the convolution module.

Tests correctness properties of sips_convolve() and sips_correlate().
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.convolution import sips_convolve, sips_correlate
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_float_dtypes = st.sampled_from([np.float32, np.float64])


def _floats_for_dtype(dtype, min_value=-1000.0, max_value=1000.0):
    """Return a float strategy compatible with the target numpy dtype."""
    width = 32 if np.dtype(dtype).itemsize <= 4 else 64
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=width
    )


@st.composite
def chw_image_and_kernel(draw):
    """Generate a random CHW float image and a 2-D kernel.

    Images are float to avoid integer overflow issues during filtering.
    Kernel values are kept small to avoid extreme output magnitudes.
    """
    dtype = draw(_float_dtypes)
    num_bands = draw(st.integers(min_value=1, max_value=4))
    height = draw(st.integers(min_value=1, max_value=32))
    width = draw(st.integers(min_value=1, max_value=32))

    image = draw(
        hnp.arrays(
            dtype=dtype,
            shape=(num_bands, height, width),
            elements=_floats_for_dtype(dtype),
        )
    )

    kernel_h = draw(st.integers(min_value=1, max_value=7))
    kernel_w = draw(st.integers(min_value=1, max_value=7))
    kernel = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(kernel_h, kernel_w),
            elements=_floats_for_dtype(np.float64, min_value=-1.0, max_value=1.0),
        )
    )

    return image, kernel


@st.composite
def hw_image_and_kernel(draw):
    """Generate a random single-band (H, W) float image and a 2-D kernel."""
    dtype = draw(_float_dtypes)
    height = draw(st.integers(min_value=1, max_value=32))
    width = draw(st.integers(min_value=1, max_value=32))

    image = draw(
        hnp.arrays(
            dtype=dtype,
            shape=(height, width),
            elements=_floats_for_dtype(dtype),
        )
    )

    kernel_h = draw(st.integers(min_value=1, max_value=7))
    kernel_w = draw(st.integers(min_value=1, max_value=7))
    kernel = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(kernel_h, kernel_w),
            elements=_floats_for_dtype(np.float64, min_value=-1.0, max_value=1.0),
        )
    )

    return image, kernel


# ---------------------------------------------------------------------------
# Property 11: Convolution and correlation shape preservation
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 11: Convolution and correlation shape preservation
@pytest.mark.property
@given(data=chw_image_and_kernel())
@settings(pbt_settings)
def test_convolution_shape_preservation_chw(data):
    """**Validates: Requirements 11.3, 12.3**

    For any valid CHW NDArray and any 2-D kernel, the output of both
    sips_convolve() and sips_correlate() SHALL have the same dimensions
    as the input array.
    """
    image, kernel = data

    conv_result = sips_convolve(image, kernel)
    corr_result = sips_correlate(image, kernel)

    assert conv_result.shape == image.shape, f"sips_convolve changed shape: input {image.shape}, output {conv_result.shape}"
    assert corr_result.shape == image.shape, f"sips_correlate changed shape: input {image.shape}, output {corr_result.shape}"


# Feature: image-processing-foundations, Property 11: Convolution and correlation shape preservation
@pytest.mark.property
@given(data=hw_image_and_kernel())
@settings(pbt_settings)
def test_convolution_shape_preservation_hw(data):
    """**Validates: Requirements 11.3, 12.3**

    For any valid (H, W) NDArray and any 2-D kernel, the output of both
    sips_convolve() and sips_correlate() SHALL have the same dimensions
    as the input array.
    """
    image, kernel = data

    conv_result = sips_convolve(image, kernel)
    corr_result = sips_correlate(image, kernel)

    assert conv_result.shape == image.shape, f"sips_convolve changed shape: input {image.shape}, output {conv_result.shape}"
    assert corr_result.shape == image.shape, f"sips_correlate changed shape: input {image.shape}, output {corr_result.shape}"


# ---------------------------------------------------------------------------
# Property 12: Correlation equals convolution with flipped kernel
# ---------------------------------------------------------------------------


@st.composite
def chw_image_and_odd_kernel(draw):
    """Generate a random CHW float image and an odd-sized 2-D kernel.

    The relationship correlate(image, k) == convolve(image, flip(k))
    holds exactly when both operations use the same anchor point, which
    is guaranteed for odd-sized kernels. Even-sized kernels have
    different SIPS-defined anchors for convolution vs correlation.
    """
    dtype = draw(_float_dtypes)
    num_bands = draw(st.integers(min_value=1, max_value=4))
    height = draw(st.integers(min_value=1, max_value=32))
    width = draw(st.integers(min_value=1, max_value=32))

    image = draw(
        hnp.arrays(
            dtype=dtype,
            shape=(num_bands, height, width),
            elements=_floats_for_dtype(dtype),
        )
    )

    # Odd-sized kernels only: 1, 3, 5, 7
    kernel_h = draw(st.integers(min_value=1, max_value=4)) * 2 - 1
    kernel_w = draw(st.integers(min_value=1, max_value=4)) * 2 - 1
    kernel = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(kernel_h, kernel_w),
            elements=_floats_for_dtype(np.float64, min_value=-1.0, max_value=1.0),
        )
    )

    return image, kernel


# Feature: image-processing-foundations, Property 12: Correlation equals convolution with flipped kernel
@pytest.mark.property
@given(data=chw_image_and_odd_kernel())
@settings(pbt_settings)
def test_correlate_equals_convolve_with_flipped_kernel(data):
    """**Validates: Requirements 12.2**

    For any valid CHW NDArray and any 2-D kernel,
    sips_correlate(image, kernel) SHALL produce the same result as
    sips_convolve(image, kernel[::-1, ::-1]).
    """
    image, kernel = data

    corr_result = sips_correlate(image, kernel)
    conv_flipped_result = sips_convolve(image, kernel[::-1, ::-1])

    np.testing.assert_allclose(
        corr_result,
        conv_flipped_result,
        rtol=1e-5,
        atol=1e-5,
        err_msg="sips_correlate(image, kernel) != sips_convolve(image, kernel[::-1, ::-1])",
    )
