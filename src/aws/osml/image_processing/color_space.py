#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Color space transforms for image processing.

This module provides the :func:`color_space_transform` function for
converting images between well-known color spaces. It implements the
SIPS reference image chain's Color Space Transform step using hardcoded
published matrices and TRC (Tone Reproduction Curve) functions.

Supported color spaces:

- ``"srgb"`` — sRGB with piecewise linear+power TRC per IEC 61966-2-1
- ``"linear_srgb"`` — linear sRGB (no TRC, identity)
- ``"prophoto_rgb"`` — ProPhoto RGB with gamma 1.8 TRC, D50 white point
- ``"adobe_rgb"`` — AdobeRGB with gamma 563/256 ≈ 2.19921875 TRC
- ``"ciexyz"`` — CIE XYZ (delegated to cv2.cvtColor)
- ``"cielab"`` — CIE L*a*b* (delegated to cv2.cvtColor)
- ``"ycrcb"`` — YCrCb (delegated to cv2.cvtColor)

Transform pipeline for matrix/TRC-based RGB profiles:

1. Apply source TRC decode (linearize)
2. 3×3 matrix: source linear RGB → CIE XYZ
3. Bradford chromatic adaptation if white points differ (D65 ↔ D50)
4. 3×3 matrix: CIE XYZ → destination linear RGB
5. Apply destination TRC encode (gamma compress)
"""

from typing import Callable, Dict, NamedTuple, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Supported color space identifiers
# ---------------------------------------------------------------------------

SUPPORTED_SPACES = frozenset(["srgb", "linear_srgb", "prophoto_rgb", "adobe_rgb", "ciexyz", "cielab", "ycrcb"])

# Spaces that are delegated to cv2.cvtColor via linear_srgb intermediate
_CV2_DELEGATED_SPACES = frozenset(["ciexyz", "cielab", "ycrcb"])

# Spaces handled by the matrix/TRC pipeline
_MATRIX_TRC_SPACES = frozenset(["srgb", "linear_srgb", "prophoto_rgb", "adobe_rgb"])


# ---------------------------------------------------------------------------
# TRC (Tone Reproduction Curve) functions
# ---------------------------------------------------------------------------


def _srgb_trc_decode(v: NDArray) -> NDArray:
    """Linearize sRGB values using the IEC 61966-2-1 piecewise function.

    For values <= 0.04045: linear = v / 12.92
    For values >  0.04045: linear = ((v + 0.055) / 1.055) ^ 2.4

    :param v: Array of sRGB-encoded values in [0, 1].
    :return: Array of linear values.
    """
    return np.where(
        v <= 0.04045,
        v / 12.92,
        np.power((v + 0.055) / 1.055, 2.4),
    )


def _srgb_trc_encode(v: NDArray) -> NDArray:
    """Apply sRGB gamma encoding per IEC 61966-2-1.

    For values <= 0.04045/12.92: encoded = 12.92 * v
    For values >  0.04045/12.92: encoded = 1.055 * v^(1/2.4) - 0.055

    :param v: Array of linear values in [0, 1].
    :return: Array of sRGB-encoded values.
    """
    # Threshold derived from decode breakpoint (0.04045/12.92) for exact round-trip
    return np.where(
        v <= 0.04045 / 12.92,
        12.92 * v,
        1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055,
    )


def _prophoto_trc_decode(v: NDArray) -> NDArray:
    """Linearize ProPhoto RGB values using gamma 1.8.

    :param v: Array of ProPhoto RGB-encoded values in [0, 1].
    :return: Array of linear values.
    """
    return np.power(np.maximum(v, 0.0), 1.8)


def _prophoto_trc_encode(v: NDArray) -> NDArray:
    """Apply ProPhoto RGB gamma 1.8 encoding.

    :param v: Array of linear values in [0, 1].
    :return: Array of ProPhoto RGB-encoded values.
    """
    return np.power(np.maximum(v, 0.0), 1.0 / 1.8)


# AdobeRGB gamma is exactly 563/256 ≈ 2.19921875
_ADOBE_RGB_GAMMA = 563.0 / 256.0


def _adobe_rgb_trc_decode(v: NDArray) -> NDArray:
    """Linearize AdobeRGB values using gamma 563/256.

    :param v: Array of AdobeRGB-encoded values in [0, 1].
    :return: Array of linear values.
    """
    return np.power(np.maximum(v, 0.0), _ADOBE_RGB_GAMMA)


def _adobe_rgb_trc_encode(v: NDArray) -> NDArray:
    """Apply AdobeRGB gamma 563/256 encoding.

    :param v: Array of linear values in [0, 1].
    :return: Array of AdobeRGB-encoded values.
    """
    return np.power(np.maximum(v, 0.0), 1.0 / _ADOBE_RGB_GAMMA)


def _identity_trc(v: NDArray) -> NDArray:
    """Identity TRC (no-op) for linear color spaces.

    :param v: Array of values.
    :return: Same array unchanged.
    """
    return v


# ---------------------------------------------------------------------------
# Color space profile data
# ---------------------------------------------------------------------------


class _ColorSpaceProfile(NamedTuple):
    """Internal profile data for a matrix/TRC-based RGB color space."""

    to_xyz: NDArray  # 3×3 matrix: linear RGB → CIE XYZ
    from_xyz: NDArray  # 3×3 matrix: CIE XYZ → linear RGB
    trc_decode: Callable[[NDArray], NDArray]  # gamma decode (linearize)
    trc_encode: Callable[[NDArray], NDArray]  # gamma encode (compress)
    white_point: str  # "D65" or "D50"


# ---------------------------------------------------------------------------
# Published 3×3 matrices: linear RGB ↔ CIE XYZ
# ---------------------------------------------------------------------------

# sRGB to XYZ (D65), from IEC 61966-2-1
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

# ProPhoto RGB to XYZ (D50), from ROMM RGB / ISO 22028-2
_PROPHOTO_TO_XYZ = np.array(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0000000, 0.0000000, 0.8252100],
    ],
    dtype=np.float64,
)

_XYZ_TO_PROPHOTO = np.array(
    [
        [1.3459433, -0.2556075, -0.0511118],
        [-0.5445989, 1.5081673, 0.0205351],
        [0.0000000, 0.0000000, 1.2118128],
    ],
    dtype=np.float64,
)

# AdobeRGB (1998) to XYZ (D65)
_ADOBE_TO_XYZ = np.array(
    [
        [0.5767309, 0.1855540, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085],
    ],
    dtype=np.float64,
)

_XYZ_TO_ADOBE = np.array(
    [
        [2.0413690, -0.5649464, -0.3446944],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0134474, -0.1183897, 1.0154096],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Bradford chromatic adaptation: D65 ↔ D50
# ---------------------------------------------------------------------------

# Bradford cone response matrix
_BRADFORD_MA = np.array(
    [
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000],
    ],
    dtype=np.float64,
)

_BRADFORD_MA_INV = np.linalg.inv(_BRADFORD_MA)

# CIE standard illuminant white points (XYZ, Y=1)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
_D50_WHITE = np.array([0.96422, 1.00000, 0.82521], dtype=np.float64)


def _bradford_adaptation_matrix(src_white: NDArray, dst_white: NDArray) -> NDArray:
    """Compute the Bradford chromatic adaptation matrix.

    :param src_white: Source illuminant white point XYZ (Y=1).
    :param dst_white: Destination illuminant white point XYZ (Y=1).
    :return: 3×3 adaptation matrix.
    """
    src_cone = _BRADFORD_MA @ src_white
    dst_cone = _BRADFORD_MA @ dst_white
    scale = np.diag(dst_cone / src_cone)
    return _BRADFORD_MA_INV @ scale @ _BRADFORD_MA


# Precomputed adaptation matrices
_D65_TO_D50 = _bradford_adaptation_matrix(_D65_WHITE, _D50_WHITE)
_D50_TO_D65 = _bradford_adaptation_matrix(_D50_WHITE, _D65_WHITE)


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, _ColorSpaceProfile] = {
    "srgb": _ColorSpaceProfile(
        to_xyz=_SRGB_TO_XYZ,
        from_xyz=_XYZ_TO_SRGB,
        trc_decode=_srgb_trc_decode,
        trc_encode=_srgb_trc_encode,
        white_point="D65",
    ),
    "linear_srgb": _ColorSpaceProfile(
        to_xyz=_SRGB_TO_XYZ,
        from_xyz=_XYZ_TO_SRGB,
        trc_decode=_identity_trc,
        trc_encode=_identity_trc,
        white_point="D65",
    ),
    "prophoto_rgb": _ColorSpaceProfile(
        to_xyz=_PROPHOTO_TO_XYZ,
        from_xyz=_XYZ_TO_PROPHOTO,
        trc_decode=_prophoto_trc_decode,
        trc_encode=_prophoto_trc_encode,
        white_point="D50",
    ),
    "adobe_rgb": _ColorSpaceProfile(
        to_xyz=_ADOBE_TO_XYZ,
        from_xyz=_XYZ_TO_ADOBE,
        trc_decode=_adobe_rgb_trc_decode,
        trc_encode=_adobe_rgb_trc_encode,
        white_point="D65",
    ),
}


def _get_adaptation_matrix(src_wp: str, dst_wp: str) -> Optional[NDArray]:
    """Return the Bradford adaptation matrix for the given white point pair.

    :param src_wp: Source white point ("D65" or "D50").
    :param dst_wp: Destination white point ("D65" or "D50").
    :return: 3×3 adaptation matrix, or None if no adaptation is needed.
    """
    if src_wp == dst_wp:
        return None
    if src_wp == "D65" and dst_wp == "D50":
        return _D65_TO_D50
    if src_wp == "D50" and dst_wp == "D65":
        return _D50_TO_D65
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# cv2.cvtColor delegation mappings
# ---------------------------------------------------------------------------

# Mapping from (source, destination) to cv2 color conversion code.
# These conversions go through linear_srgb as the intermediate.
_CV2_FROM_LINEAR_SRGB = {
    "ciexyz": cv2.COLOR_RGB2XYZ,
    "cielab": cv2.COLOR_RGB2Lab,
    "ycrcb": cv2.COLOR_RGB2YCrCb,
}

_CV2_TO_LINEAR_SRGB = {
    "ciexyz": cv2.COLOR_XYZ2RGB,
    "cielab": cv2.COLOR_Lab2RGB,
    "ycrcb": cv2.COLOR_YCrCb2RGB,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def color_space_transform(
    image: NDArray,
    source: str,
    destination: str,
) -> NDArray:
    """Transform an image between well-known color spaces.

    Implements the SIPS Color Space Transform step. For matrix/TRC-based
    RGB profiles (sRGB, ProPhoto RGB, AdobeRGB, linear_srgb), the pipeline
    is: linearize → 3×3 to XYZ → Bradford adapt → 3×3 to dest → gamma
    encode. For cv2-delegated spaces (ciexyz, cielab, ycrcb), conversion
    goes through linear_srgb as the intermediate.

    :param image: Input 3-band CHW NDArray with float values in [0, 1].
    :param source: Source color space identifier.
    :param destination: Destination color space identifier.
    :return: A new 3-band CHW NDArray in the destination color space.
    :raises ValueError: If ``source`` or ``destination`` is not a
        supported color space, or if the input does not have exactly
        3 bands.
    """
    if source not in SUPPORTED_SPACES:
        raise ValueError(f"Unsupported source color space: {source!r}. Supported spaces: {sorted(SUPPORTED_SPACES)}")
    if destination not in SUPPORTED_SPACES:
        raise ValueError(
            f"Unsupported destination color space: {destination!r}. Supported spaces: {sorted(SUPPORTED_SPACES)}"
        )

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Input image must have exactly 3 bands (CHW layout), got shape {image.shape}")

    # No-op shortcut
    if source == destination:
        return image.copy()

    # Work in float64 for precision
    img = image.astype(np.float64)

    # Route through the appropriate pipeline
    if source in _MATRIX_TRC_SPACES and destination in _MATRIX_TRC_SPACES:
        return _matrix_trc_transform(img, source, destination)

    # One or both sides are cv2-delegated: route through linear_srgb
    return _cv2_delegated_transform(img, source, destination)


def _matrix_trc_transform(img: NDArray, source: str, destination: str) -> NDArray:
    """Transform between two matrix/TRC-based RGB color spaces.

    Pipeline: linearize → combined 3×3 matmul → gamma encode.

    The three matrix steps (source→XYZ, Bradford adapt, XYZ→dest) are
    composed into a single 3×3 matrix so only one matmul pass is made
    over the pixel buffer.

    :param img: 3-band CHW float64 array.
    :param source: Source color space identifier.
    :param destination: Destination color space identifier.
    :return: Transformed 3-band CHW float64 array.
    """
    src_profile = _PROFILES[source]
    dst_profile = _PROFILES[destination]

    _, h, w = img.shape

    # Step 1: Linearize (apply source TRC decode) — vectorized across all bands
    linear = src_profile.trc_decode(img)

    # Precompute combined matrix: from_xyz @ [adapt @] to_xyz
    # This collapses 2-3 matmuls into a single pass over the pixel buffer.
    combined = dst_profile.from_xyz
    adapt = _get_adaptation_matrix(src_profile.white_point, dst_profile.white_point)
    if adapt is not None:
        combined = combined @ adapt
    combined = combined @ src_profile.to_xyz

    # Reshape to (3, N), apply single matmul, reshape back to CHW
    dest_linear = (combined @ linear.reshape(3, -1)).reshape(3, h, w)

    # Step 5: Apply destination TRC encode (gamma compress) — vectorized across all bands
    return dst_profile.trc_encode(dest_linear)


def _cv2_delegated_transform(img: NDArray, source: str, destination: str) -> NDArray:
    """Transform involving cv2-delegated color spaces.

    Routes through linear_srgb as the intermediate. First converts
    source → linear_srgb (if needed), then linear_srgb → destination
    (if needed).

    :param img: 3-band CHW float64 array.
    :param source: Source color space identifier.
    :param destination: Destination color space identifier.
    :return: Transformed 3-band CHW float64 array.
    """
    # Step 1: Convert source → linear_srgb
    if source in _CV2_DELEGATED_SPACES:
        linear_srgb = _from_cv2_space(img, source)
    elif source in _MATRIX_TRC_SPACES:
        if source == "linear_srgb":
            linear_srgb = img
        else:
            linear_srgb = _matrix_trc_transform(img, source, "linear_srgb")
    else:
        linear_srgb = img  # pragma: no cover

    # Step 2: Convert linear_srgb → destination
    if destination in _CV2_DELEGATED_SPACES:
        return _to_cv2_space(linear_srgb, destination)
    elif destination in _MATRIX_TRC_SPACES:
        if destination == "linear_srgb":
            return linear_srgb.copy()
        return _matrix_trc_transform(linear_srgb, "linear_srgb", destination)

    return linear_srgb.copy()  # pragma: no cover


def _to_cv2_space(linear_srgb: NDArray, space: str) -> NDArray:
    """Convert from linear_srgb (CHW) to a cv2-delegated space.

    :param linear_srgb: 3-band CHW float64 array in linear sRGB.
    :param space: Target cv2-delegated space identifier.
    :return: 3-band CHW float64 array in the target space.
    """
    code = _CV2_FROM_LINEAR_SRGB[space]
    # CHW → HWC float32 in one step (astype produces a contiguous copy)
    hwc = linear_srgb.transpose(1, 2, 0).astype(np.float32)
    converted = cv2.cvtColor(hwc, code)
    # HWC → CHW, upcast back to float64
    return converted.transpose(2, 0, 1).astype(np.float64)


def _from_cv2_space(img: NDArray, space: str) -> NDArray:
    """Convert from a cv2-delegated space (CHW) to linear_srgb.

    :param img: 3-band CHW float64 array in the source cv2 space.
    :param space: Source cv2-delegated space identifier.
    :return: 3-band CHW float64 array in linear sRGB.
    """
    code = _CV2_TO_LINEAR_SRGB[space]
    # CHW → HWC float32 in one step (astype produces a contiguous copy)
    hwc = img.transpose(1, 2, 0).astype(np.float32)
    converted = cv2.cvtColor(hwc, code)
    # HWC → CHW, upcast back to float64
    return converted.transpose(2, 0, 1).astype(np.float64)
