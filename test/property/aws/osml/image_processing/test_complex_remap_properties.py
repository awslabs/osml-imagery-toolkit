#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for complex_remap module.

Tests mathematical invariants of the complex imagery remap functions:
- complex_to_power() output is always non-negative for random inputs
- AMP8I_PHS8I identity-table round-trip matches amplitude²
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.complex_remap import (
    ROLE_AMPLITUDE_INDEX,
    ROLE_IMAGINARY,
    ROLE_MAGNITUDE,
    ROLE_PHASE,
    ROLE_REAL,
    complex_to_power,
    decode_to_iq,
    magnitude_remap,
    quarter_power_remap,
)
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def complex_2hw_float32(draw):
    """Generate a random (2, H, W) float32 array representing I/Q data."""
    h = draw(st.integers(min_value=1, max_value=32))
    w = draw(st.integers(min_value=1, max_value=32))
    arr = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(2, h, w),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    return arr


@st.composite
def native_complex64_array(draw):
    """Generate a random (H, W) complex64 array."""
    h = draw(st.integers(min_value=1, max_value=32))
    w = draw(st.integers(min_value=1, max_value=32))
    real = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    imag = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=(h, w),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    return (real + 1j * imag).astype(np.complex64)


@st.composite
def amp8i_phs8i_data(draw):
    """Generate random AMP8I_PHS8I data with an identity amplitude table.

    The identity table maps index i → float(i), so amplitude = index value.
    """
    h = draw(st.integers(min_value=1, max_value=16))
    w = draw(st.integers(min_value=1, max_value=16))
    amp_indices = draw(hnp.arrays(dtype=np.uint8, shape=(h, w), elements=st.integers(min_value=0, max_value=255)))
    phase_indices = draw(hnp.arrays(dtype=np.uint8, shape=(h, w), elements=st.integers(min_value=0, max_value=255)))
    identity_table = np.arange(256, dtype=np.float32)
    block = np.stack([amp_indices, phase_indices], axis=0)
    return block, identity_table


# ---------------------------------------------------------------------------
# Property: complex_to_power output is always non-negative
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=complex_2hw_float32())
@settings(pbt_settings)
def test_complex_to_power_non_negative_2hw(data):
    """complex_to_power() output is always non-negative for (2, H, W) float32 input."""
    result = complex_to_power(data)
    assert np.all(result >= 0), f"Negative power value found: min={result.min()}"


@pytest.mark.property
@given(data=native_complex64_array())
@settings(pbt_settings)
def test_complex_to_power_non_negative_native(data):
    """complex_to_power() output is always non-negative for native complex64 input."""
    result = complex_to_power(data)
    assert np.all(result >= 0), f"Negative power value found: min={result.min()}"


@pytest.mark.property
@given(data=complex_2hw_float32())
@settings(pbt_settings)
def test_complex_to_power_agrees_with_abs_squared(data):
    """complex_to_power() on (2,H,W) agrees with |z|² computed via complex arithmetic."""
    z = data[0] + 1j * data[1]
    expected = np.abs(z.astype(np.complex64)) ** 2
    result = complex_to_power(data)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Property: AMP8I_PHS8I identity-table round-trip matches amplitude²
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=amp8i_phs8i_data())
@settings(pbt_settings)
def test_amp8i_phs8i_identity_table_roundtrip(data):
    """AMP8I_PHS8I with identity table: power after decode equals amplitude².

    When the amplitude table is the identity function (table[i] = i),
    decoding produces I/Q such that I² + Q² = amplitude², because:
        I = amplitude * cos(phase)
        Q = amplitude * sin(phase)
        I² + Q² = amplitude² * (cos²(phase) + sin²(phase)) = amplitude²
    """
    block, identity_table = data
    iq = decode_to_iq(block, band_interpretation=[ROLE_AMPLITUDE_INDEX, ROLE_PHASE], amplitude_table=identity_table)
    power = complex_to_power(iq)

    # Expected power = amplitude² where amplitude = table[amp_index] = amp_index (identity)
    amplitude = identity_table[block[0]].astype(np.float32)
    expected_power = amplitude**2

    np.testing.assert_allclose(power, expected_power, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Property: remap presets always produce finite non-negative output
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=complex_2hw_float32())
@settings(pbt_settings)
def test_quarter_power_remap_finite_non_negative(data):
    """quarter_power_remap() always produces finite, non-negative (1, H, W) float32."""
    result = quarter_power_remap(data)
    assert result.shape == (1, data.shape[1], data.shape[2])
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result)), (
        f"Non-finite values found: nan={np.isnan(result).sum()}, inf={np.isinf(result).sum()}"
    )
    assert np.all(result >= 0), f"Negative value found: min={result.min()}"


@pytest.mark.property
@given(data=complex_2hw_float32())
@settings(pbt_settings)
def test_magnitude_remap_finite_non_negative(data):
    """magnitude_remap() always produces finite, non-negative (1, H, W) float32."""
    result = magnitude_remap(data)
    assert result.shape == (1, data.shape[1], data.shape[2])
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result)), (
        f"Non-finite values found: nan={np.isnan(result).sum()}, inf={np.isinf(result).sum()}"
    )
    assert np.all(result >= 0), f"Negative value found: min={result.min()}"


# ---------------------------------------------------------------------------
# Property: decode_to_iq preserves magnitude for real/imaginary input
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=complex_2hw_float32())
@settings(pbt_settings)
def test_decode_real_imaginary_preserves_power(data):
    """decode_to_iq with real/imaginary interpretation preserves I² + Q²."""
    decoded = decode_to_iq(data, band_interpretation=[ROLE_REAL, ROLE_IMAGINARY])
    original_power = data[0] ** 2 + data[1] ** 2
    decoded_power = decoded[0] ** 2 + decoded[1] ** 2
    np.testing.assert_allclose(decoded_power, original_power, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Property: magnitude/phase round-trip preserves power
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=complex_2hw_float32())
@settings(pbt_settings)
def test_magnitude_phase_roundtrip_preserves_power(data):
    """Converting I/Q → M/P → I/Q preserves total power (I² + Q²).

    The magnitude/phase decode applies:
        I = M * cos(P)
        Q = M * sin(P)
    So I² + Q² = M² for float-radians input.
    """
    magnitude = np.sqrt(data[0] ** 2 + data[1] ** 2)
    phase = np.arctan2(data[1], data[0])
    mp_block = np.stack([magnitude, phase], axis=0)

    decoded = decode_to_iq(mp_block, band_interpretation=[ROLE_MAGNITUDE, ROLE_PHASE])
    decoded_power = decoded[0] ** 2 + decoded[1] ** 2
    original_power = data[0] ** 2 + data[1] ** 2

    np.testing.assert_allclose(decoded_power, original_power, rtol=1e-4, atol=1e-4)
