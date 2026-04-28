#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""SIPS-compliant Reduced Resolution Dataset (RRDS) resampling.

This module implements the SIPS RRDS 2x downsampling algorithm defined
in NGA.STND.0014 v2.4 Section 2.2. The pipeline is:

1. **Anti-alias filter** — convolve with the 7x7 kernel in Table 2.2
   using the SIPS "Mirror Edge - Odd" boundary convention.
2. **LaGrange interpolation** — correlate with the 4x4 separable
   kernel formed from Table 2.7 LaGrange coefficients at sub-pixel
   offset ``(0.5, 0.5)``.
3. **2x subsample** — keep every other pixel in both dimensions.

The public API:

- :data:`SIPS_ANTIALIAS_KERNEL_7x7` — the 7x7 anti-alias kernel constant.
- :func:`compute_lagrange_coefficients` — 4-element LaGrange polynomial
  coefficients for any sub-pixel spacing in ``[0, 1)``.
- :func:`compute_compromise_coefficients` — 4-element "compromise"
  coefficients from SIPS Table 2.8 (LaGrange variant with reduced
  ringing near high-contrast edges).
- :func:`build_lagrange_kernel_2d` — 4x4 separable kernel as the
  outer product of two coefficient vectors.
- :func:`sips_rrds_resample` — ``ResampleFunc``-compatible 2x
  downsampler used as the default resampler for pyramid generation.
  Accepts an optional keyword-only ``bit_depth`` argument that
  switches to the integer-intermediate pipeline used by the
  published SIPS reference values.

All array-level functions follow the toolkit conventions: CHW (or HW)
layout, no input mutation, and dtype preserved through the operation.
"""

import numpy as np
from numpy.typing import NDArray

from .convolution import sips_convolve, sips_correlate

#: SIPS Table 2.2 — 7x7 anti-alias kernel (NGA.STND.0014 v2.4 Section 2.2).
#:
#: The kernel is 4-fold symmetric and sums to 1.0 (within the rounding
#: precision of the published table). Rows/columns 1 and 5 are zero by
#: design (the kernel is supported on a 5x5 lattice inside the 7x7
#: footprint).
SIPS_ANTIALIAS_KERNEL_7x7: NDArray = np.array(
    [
        [0.00694389, 0.0, -0.02777640, -0.04166500, -0.02777640, 0.0, 0.00694389],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.02777640, 0.0, 0.11110900, 0.16666500, 0.11110900, 0.0, -0.02777640],
        [-0.04166500, 0.0, 0.16666500, 0.25000000, 0.16666500, 0.0, -0.04166500],
        [-0.02777640, 0.0, 0.11110900, 0.16666500, 0.11110900, 0.0, -0.02777640],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.00694389, 0.0, -0.02777640, -0.04166500, -0.02777640, 0.0, 0.00694389],
    ],
    dtype=np.float64,
)


#: SIPS Table 2.8 — Sample compromise coefficients at 1/64 pixel spacing.
#:
#: Row ``i`` corresponds to ``spacing = i / 64``. Values are transcribed
#: directly from NGA.STND.0014 v2.4 Table 2.8. The published rows are
#: rounded, so individual row sums may differ from 1.0 by ~1e-4.
_SIPS_COMPROMISE_COEFFICIENTS_TABLE: NDArray = np.array(
    [
        [0.087, 0.731, 0.277, -0.095],
        [0.081688, 0.7285, 0.282313, -0.0925],
        [0.076375, 0.726, 0.287625, -0.09],
        [0.071063, 0.7235, 0.292938, -0.0875],
        [0.06575, 0.721, 0.29825, -0.085],
        [0.060438, 0.7185, 0.303563, -0.0825],
        [0.055125, 0.716, 0.308875, -0.08],
        [0.049813, 0.7135, 0.314188, -0.0775],
        [0.0445, 0.711, 0.3195, -0.075],
        [0.039188, 0.7085, 0.324813, -0.0725],
        [0.033875, 0.706, 0.330125, -0.07],
        [0.028563, 0.7035, 0.335438, -0.0675],
        [0.02325, 0.701, 0.34075, -0.065],
        [0.017938, 0.6985, 0.346063, -0.0625],
        [0.012625, 0.696, 0.351375, -0.06],
        [0.007312, 0.6935, 0.356688, -0.0575],
        [0.002, 0.691, 0.362, -0.055],
        [-0.00331, 0.6885, 0.367313, -0.0525],
        [-0.00863, 0.686, 0.372625, -0.05],
        [-0.01394, 0.6835, 0.377938, -0.0475],
        [-0.01925, 0.681, 0.38325, -0.045],
        [-0.02456, 0.6785, 0.388563, -0.0425],
        [-0.02988, 0.676, 0.393875, -0.04],
        [-0.03519, 0.6735, 0.399188, -0.0375],
        [-0.0405, 0.671, 0.4045, -0.035],
        [-0.04581, 0.6685, 0.409813, -0.0325],
        [-0.05113, 0.666, 0.415125, -0.03],
        [-0.05644, 0.6635, 0.420438, -0.0275],
        [-0.06175, 0.661, 0.42575, -0.025],
        [-0.06706, 0.6585, 0.431063, -0.0225],
        [-0.07238, 0.656, 0.436375, -0.02],
        [-0.07769, 0.6535, 0.441688, -0.0175],
        [-0.01625, 0.444344, 0.65225, -0.08034],
        [-0.0175, 0.441688, 0.6535, -0.07769],
        [-0.02, 0.436375, 0.656, -0.07238],
        [-0.0225, 0.431063, 0.6585, -0.06706],
        [-0.025, 0.42575, 0.661, -0.06175],
        [-0.0275, 0.420438, 0.6635, -0.05644],
        [-0.03, 0.415125, 0.666, -0.05113],
        [-0.0325, 0.409813, 0.6685, -0.04581],
        [-0.035, 0.4045, 0.671, -0.0405],
        [-0.0375, 0.399188, 0.6735, -0.03519],
        [-0.04, 0.393875, 0.676, -0.02988],
        [-0.0425, 0.388563, 0.6785, -0.02456],
        [-0.045, 0.38325, 0.681, -0.01925],
        [-0.0475, 0.377938, 0.6835, -0.01394],
        [-0.05, 0.372625, 0.686, -0.00863],
        [-0.0525, 0.367313, 0.6885, -0.00331],
        [-0.055, 0.362, 0.691, 0.002],
        [-0.0575, 0.356688, 0.6935, 0.007312],
        [-0.06, 0.351375, 0.696, 0.012625],
        [-0.0625, 0.346063, 0.6985, 0.017938],
        [-0.065, 0.34075, 0.701, 0.02325],
        [-0.0675, 0.335438, 0.7035, 0.028563],
        [-0.07, 0.330125, 0.706, 0.033875],
        [-0.0725, 0.324813, 0.7085, 0.039188],
        [-0.075, 0.3195, 0.711, 0.0445],
        [-0.0775, 0.314188, 0.7135, 0.049813],
        [-0.08, 0.308875, 0.716, 0.055125],
        [-0.0825, 0.303563, 0.7185, 0.060438],
        [-0.085, 0.29825, 0.721, 0.06575],
        [-0.0875, 0.292938, 0.7235, 0.071063],
        [-0.09, 0.287625, 0.726, 0.076375],
        [-0.0925, 0.282313, 0.7285, 0.081688],
    ],
    dtype=np.float64,
)


def _validate_spacing(spacing: float) -> None:
    """Raise ``ValueError`` if ``spacing`` is not in ``[0, 1)``.

    :param spacing: Sub-pixel spacing to validate.
    :raises ValueError: If ``spacing < 0`` or ``spacing >= 1``.
    """
    if not (0.0 <= spacing < 1.0):
        raise ValueError(f"spacing must be in [0, 1), got {spacing}")


def compute_lagrange_coefficients(spacing: float) -> NDArray:
    """Compute the 4-element LaGrange interpolation coefficients.

    These are the cubic LaGrange polynomial weights from SIPS Section
    2.3.5 (equations 2.3-2.6), parameterized by the sub-pixel offset
    ``d = spacing``. The four coefficients apply to a 4-sample window
    centered one sample to the left of the fractional-offset sample
    being interpolated.

    The coefficients sum to 1.0 for any ``spacing`` in ``[0, 1)``
    (exact identity for the LaGrange polynomial form).

    :param spacing: Sub-pixel offset in ``[0, 1)``.
    :return: A shape-``(4,)`` float64 array ``[C1, C2, C3, C4]``.
    :raises ValueError: If ``spacing`` is outside ``[0, 1)``.
    """
    _validate_spacing(spacing)
    d = float(spacing)
    c1 = -d * (d - 1.0) * (d - 2.0) / 6.0
    c2 = (d + 1.0) * (d - 1.0) * (d - 2.0) / 2.0
    c3 = -d * (d + 1.0) * (d - 2.0) / 2.0
    c4 = d * (d + 1.0) * (d - 1.0) / 6.0
    return np.array([c1, c2, c3, c4], dtype=np.float64)


def compute_compromise_coefficients(spacing: float) -> NDArray:
    """Compute the 4-element compromise interpolation coefficients.

    The compromise coefficients are SIPS Table 2.8 — a LaGrange variant
    that trades a small amount of interpolation accuracy for reduced
    ringing near high-contrast edges. This implementation returns the
    tabulated row at the quantized index ``int(spacing * 64)``.

    :param spacing: Sub-pixel offset in ``[0, 1)``.
    :return: A new shape-``(4,)`` float64 array ``[C1, C2, C3, C4]``.
    :raises ValueError: If ``spacing`` is outside ``[0, 1)``.
    """
    _validate_spacing(spacing)
    index = int(spacing * 64.0)
    # Defensive guard: floating-point edge-case where spacing is almost 1.
    if index >= _SIPS_COMPROMISE_COEFFICIENTS_TABLE.shape[0]:
        index = _SIPS_COMPROMISE_COEFFICIENTS_TABLE.shape[0] - 1
    # Return a copy so callers cannot mutate the module-level table.
    return _SIPS_COMPROMISE_COEFFICIENTS_TABLE[index].copy()


def build_lagrange_kernel_2d(row_spacing: float, col_spacing: float) -> NDArray:
    """Build a separable 4x4 LaGrange interpolation kernel.

    The kernel is the outer product of the row- and column-direction
    LaGrange coefficient vectors. For ``(0.5, 0.5)`` this matches SIPS
    Table 2.3.

    :param row_spacing: Row sub-pixel offset in ``[0, 1)``.
    :param col_spacing: Column sub-pixel offset in ``[0, 1)``.
    :return: A shape-``(4, 4)`` float64 kernel.
    :raises ValueError: If either spacing is outside ``[0, 1)``.
    """
    row_coeffs = compute_lagrange_coefficients(row_spacing)
    col_coeffs = compute_lagrange_coefficients(col_spacing)
    return np.outer(row_coeffs, col_coeffs)


def sips_rrds_resample(
    image: NDArray,
    target_rows: int,
    target_cols: int,
    *,
    bit_depth: int | None = None,
) -> NDArray:
    """SIPS-compliant Reduced Resolution Dataset (RRDS) 2x downsampler.

    Conforms to the ``ResampleFunc`` type alias in
    :mod:`aws.osml.image_processing.resample`. Implements the three-step
    pipeline defined in NGA.STND.0014 v2.4 Section 2.2:

    1. Anti-alias filter via ``sips_convolve`` with
       :data:`SIPS_ANTIALIAS_KERNEL_7x7`.
    2. LaGrange interpolation via ``sips_correlate`` with the 4x4
       separable kernel at sub-pixel offset ``(0.5, 0.5)``.
    3. 2x subsample (drop odd rows and columns).

    Only exact 2x downsampling is supported. ``target_rows`` and
    ``target_cols`` must equal ``(src_rows + 1) // 2`` and
    ``(src_cols + 1) // 2`` respectively (the SIPS even/odd rounding
    rule: odd-sized sources round up by 1).

    Precision options:

    - By default (``bit_depth=None``), the entire pipeline runs in the
      input dtype (typically float32 or float64) and returns a float
      array. This is the fastest path because ``cv2.filter2D`` has
      SIMD-optimized float kernels; ``float32`` is 3-4x faster than
      ``uint16`` for the same operation. Use this mode when feeding
      the output into further float pipelines or when approximate
      SIPS compliance is acceptable.
    - When ``bit_depth`` is an integer ``n``, the intermediate
      produced by the anti-alias convolution is rounded and clipped
      to ``[0, 2**n - 1]`` before the LaGrange correlation. This
      matches the SIPS reference exactly (as published in Tables 2.5
      and 2.6), which assumed integer-valued raster imagery at each
      stage. Typical values: ``bit_depth=8`` for 8-bit panchromatic
      or RGB, ``bit_depth=11`` for 11-bit EO, ``bit_depth=16`` for
      16-bit multispectral.

    SIPS reference compliance notes:

    Passing ``bit_depth`` matching the source's dynamic range is
    required to reproduce the Table 2.5 and Table 2.6 verification
    matrices in SIPS Section 2.2.7. The reference values were computed
    on an 11-bit source (R0) with the anti-alias output stored as
    integers before the LaGrange step; the published standard is
    scoped to "integer-valued raster image data" (Section 1.2). A
    pure-float pipeline propagates unclipped values through the
    LaGrange correlation and can differ from the reference by up to
    ~50 digital counts at cells whose neighborhood contains pixels
    clipped to the source's max value (e.g. 2047 for 11-bit).

    Even with ``bit_depth=11`` the second cascade stage (R1 -> R2)
    retains a small residual discrepancy of <=2 digital counts at a
    handful of boundary cells, due to how the 513 -> 257 odd-dimension
    subsample interacts with the 7-pixel-wide anti-alias kernel at
    the image edge. The underlying primitives (``sips_convolve``,
    ``sips_correlate``) match their own SIPS verification tables
    (2.15, 2.25, 2.26) exactly.

    :param image: Input array in CHW (bands, height, width) layout or
        single-band (H, W) layout. The input is not mutated.
    :param target_rows: Target number of output rows. Must equal
        ``(src_rows + 1) // 2``.
    :param target_cols: Target number of output columns. Must equal
        ``(src_cols + 1) // 2``.
    :param bit_depth: If provided, round and clip the intermediate
        post-anti-alias result to ``[0, 2**bit_depth - 1]`` before the
        LaGrange correlation. Reproduces the SIPS reference values in
        Tables 2.5 and 2.6 when set to the source's bit depth. Must
        be a positive integer when specified.
    :return: A new NDArray with the SIPS-rounded half dimensions and
        the same dtype and layout as the input.
    :raises ValueError: If the input is not 2-D or 3-D, if the target
        dimensions do not match the SIPS-rounded half of the source
        dimensions, or if ``bit_depth`` is not a positive integer.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2-D (H, W) or 3-D (C, H, W) array, got {image.ndim}-D")

    if image.ndim == 2:
        src_rows, src_cols = image.shape
    else:
        src_rows, src_cols = image.shape[1], image.shape[2]

    expected_rows = (src_rows + 1) // 2
    expected_cols = (src_cols + 1) // 2
    if target_rows != expected_rows or target_cols != expected_cols:
        raise ValueError(
            "sips_rrds_resample supports 2x downsampling only: expected "
            f"target_rows={expected_rows} and target_cols={expected_cols} for a "
            f"{src_rows}x{src_cols} source, got ({target_rows}, {target_cols})"
        )

    if bit_depth is not None and (not isinstance(bit_depth, int) or bit_depth <= 0):
        raise ValueError(f"bit_depth must be a positive integer or None, got {bit_depth!r}")

    # Step 1: anti-alias filter (Mirror Edge - Odd boundary handling)
    filtered = sips_convolve(image, SIPS_ANTIALIAS_KERNEL_7x7)

    # SIPS reference values (Tables 2.5, 2.6) were computed with an
    # integer intermediate between the anti-alias filter and the
    # LaGrange correlation — consistent with SIPS Section 1.2 scoping
    # the standard to "integer-valued raster image data". Opt into
    # that behaviour by specifying ``bit_depth``; by default we stay
    # in float for performance (cv2.filter2D SIMD float path is
    # 3-4x faster than the integer path on typical kernels). See the
    # function docstring for the empirical measurement notes.
    if bit_depth is not None:
        max_value = (1 << bit_depth) - 1
        filtered = np.clip(np.round(filtered), 0, max_value)

    # Step 2: LaGrange interpolation at sub-pixel offset (0.5, 0.5)
    lagrange_kernel = build_lagrange_kernel_2d(0.5, 0.5)
    interpolated = sips_correlate(filtered, lagrange_kernel)

    # Step 3: 2x subsample — drop odd rows and columns. Per SIPS
    # Section 2.2.7.1.2, "sub-sampling is achieved simply by removing
    # the odd rows and odd columns."
    if interpolated.ndim == 2:
        output = interpolated[::2, ::2]
    else:
        output = interpolated[:, ::2, ::2]

    # ``.copy()`` so the returned array owns its data (not a view over
    # the intermediate correlation output). Cast back to the input
    # dtype — cv2.filter2D with ddepth=-1 preserves dtype already, but
    # be explicit to keep the ResampleFunc contract obvious.
    output = np.ascontiguousarray(output)
    if output.dtype != image.dtype:
        output = output.astype(image.dtype)
    return output
