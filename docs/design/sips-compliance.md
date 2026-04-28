# SIPS Compliance Matrix

Tracking implementation status of algorithms defined in the
NGA Softcopy Image Processing Standard (NGA.STND.0014 v2.4, 2019-08-21).

This is a living document. It is updated as operators are implemented and
validated. The goal is incremental progress toward full compliance, with
clear documentation of what is and isn't covered at each release.

## Reference

- **Standard**: NGA.STND.0014_2.4 — Softcopy Image Processing Standard
- **Version**: 2.4 (2019-08-21)
- **Source**: `osml-imagery-io/reference-materials/SIPS/SIPS_v24_21Aug2019.pdf`

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented and validated against SIPS verification matrices |
| 🔨 | In progress |
| 📋 | Planned for current release |
| 📅 | Planned for future release |
| ➖ | Not planned (out of scope or deferred indefinitely) |

## Algorithm Compliance Matrix

| # | SIPS Algorithm | Section | Status | Toolkit Module | Notes |
|---|---------------|---------|--------|---------------|-------|
| 1 | **RRDS Generation** | 2.2 | 📋 v2.0 | `sips_resample.sips_rrds_resample()` | 7×7 anti-alias kernel + 4×4 LaGrange interpolation + 2× subsample. Validates against Tables 2.4–2.6. |
| 2 | **Four-Point Interpolation** | 2.3 | 📋 v2.0 | `sips_resample` (internal) | LaGrange 4×4 kernel with 1/32 pixel quantization. Compromise coefficients (Table 2.8). Used by RRDS and Scaling. Validates against Tables 2.9–2.15. |
| 3 | **Convolution and Correlation** | 2.4 | 📋 v2.0 | `convolution.sips_convolve()`, `convolution.sips_correlate()` | Mirror Edge - Odd boundary handling via `cv2.BORDER_REFLECT_101`. Validates against Tables 2.25–2.26. |
| 4 | **1-D Look-Up Table** | 2.5 | 📋 v2.0 | `lut.apply_lut()` | Per-band LUT application. Supports 8, 11, 12, 16-bit. Validates against Table 2.29. |
| 5 | **Dynamic Range Adjustment** | 2.6 | 📋 v2.0 | `dynamic_range_adjustment.dynamic_range_adjust()` | Histogram percentile-based clipping (Equations 2.16–2.20). `DRAParameters.from_counts()` implements the A/B modifier logic. Validates against Tables 2.32–2.34. |
| 6 | **Scaling** | 2.7 | 📅 v2.1 | `sips_resample.sips_scale_resample()` | Intra-octave scaling for non-power-of-2 factors. Uses 1-D anti-aliasing kernels derived from RRDS filter LSF + LaGrange interpolation. Validates against Table 2.43. |
| 7 | **Asymmetric Pixel Correction** | 2.8 | 📅 Future | — | Resamples non-square pixels to square pixel space via spatially-variant LaGrange correlation. Uses the same interpolation engine as RRDS. Niche use case. |
| 8 | **Color Management Module** | 2.9 | 📋 v2.0 (partial) | `color_space.color_space_transform()` | **v2.0 delta**: Hardcoded transforms for well-known color spaces (sRGB, ProPhoto RGB, AdobeRGB, CIE XYZ, CIELab). No ICC profile file parsing. Validates against Table 2.44 for sRGB→ProPhoto RGB. **Full ICC support**: post-v2.0 optional extra. |
| 9 | **Max-Pixel Decimation** | 2.10 | 📅 v2.1 | — | Per-channel max from 2×2 blocks for colorized change composites. ~20 lines. Low priority — specialized for change detection products. |
| 10 | **Band Dependent Linearization** | 2.11 | 📅 v2.1 | — | Per-band black/white point normalization for multiband imagery. Related to DRA but multiband-aware. ~50 lines. |
| 11 | **LUT from Piecewise Math Formulae** | 2.12 | ➖ | — | Utility for generating TTC tables from piecewise functions. Not needed for image display. |

## v2.0 Compliance Summary

### What v2.0 Covers

- **RRDS Generation** (the core pyramid quality algorithm) — full compliance
  with SIPS Section 2.2, validated against reference matrices
- **Four-Point Interpolation** — full compliance, used internally by RRDS
- **Convolution and Correlation** — full compliance with boundary handling
- **1-D LUT Application** — full compliance
- **Dynamic Range Adjustment** — full compliance with auto-DRA algorithm
- **Color Management** — partial compliance (well-known color spaces only)

### Known v2.0 Deltas from Full SIPS Compliance

1. **CMM does not parse ICC profile files.** Transforms are limited to
   hardcoded well-known color spaces. Users with custom sensor ICC profiles
   cannot use them through our CMM. Workaround: use Pillow's `ImageCms`
   or `pylcms2` externally, then pass the transformed array to our
   functions.

2. **Intra-octave scaling not implemented.** Only power-of-2 downsampling
   is supported via `sips_rrds_resample()`. Smooth zoom between pyramid
   levels uses OpenCV resamplers as an interim. Planned for v2.1.

3. **No asymmetric pixel correction.** Images with non-square pixels are
   not automatically corrected. The LaGrange interpolation engine needed
   for this is implemented (shared with RRDS), so adding this is
   straightforward when needed.

4. **No max-pixel decimation.** Colorized change composite products use
   standard RRDS instead of the max-pixel algorithm. Planned for v2.1.

5. **No band dependent linearization.** Multiband imagery uses per-band
   DRA instead of the BDL algorithm. Planned for v2.1.

6. **SIPS databases not bundled.** The SIPS spec is accompanied by
   databases of kernels (MTFR, MTFE, anti-aliasing, TTC families). We
   implement the reference kernels from the spec text but do not bundle
   the full SIPS database files. Users with access to the SIPS databases
   can load them via `apply_lut()` and `sips_convolve()`.

## Verification Approach

Each implemented algorithm is validated against the SIPS verification
matrices provided in the specification:

| Algorithm | SIPS Verification Data | Our Test |
|-----------|----------------------|----------|
| RRDS Generation | Tables 2.4 (R₀), 2.5 (R₁), 2.6 (R₂) | Exact matrix comparison |
| Four-Point Interpolation | Tables 2.10–2.15 | Single-point and 10×10 matrix verification |
| Convolution | Tables 2.17–2.26 | 5×5 and 4×4 kernel examples, both odd and even |
| 1-D LUT | Tables 2.27–2.29 | TTC Family 0 Member 0 verification |
| DRA | Tables 2.31–2.34 | 10×10 11-bit matrix with known parameters |
| Scaling | Tables 2.35–2.43 | 0.4× absolute magnification example |
| CMM | Table 2.44 | sRGB→ProPhoto RGB conversion |

Error tolerances follow SIPS Section 4, Table 4.1:

| Bit Depth | Max Absolute Error |
|-----------|-------------------|
| 8-bit | ±1 digital count |
| 11-bit | ±1 digital count |
| 12-bit | ±1 digital count |
| 16-bit | ±1 digital count |

## Reference Image Chain Coverage

The SIPS reference image chain (Section 3) defines the processing order
for single-band and multiband imagery. Here is our coverage of each
element:

### Single-Band Reference Chain (Figure 3.8)

```
Sharpness Enhancement → DRA → 1-D LUT (TTC)
        ↓                 ↓         ↓
   MTFC/MTFE d/b     Histogram   TTC d/b
```

| Element | v2.0 Status | Notes |
|---------|-------------|-------|
| Sharpness Enhancement | ✅ via `sips_convolve()` | User supplies MTFC/MTFE kernel |
| Histogram | ✅ via `compute_statistics()` | |
| DRA | ✅ via `dynamic_range_adjust()` | |
| 1-D LUT (TTC) | ✅ via `apply_lut()` | |

### Multiband Reference Chain (Figure 3.9)

```
DRA → CST → [ACS Operations] → CST → Output Preparation
 ↓     ↓                              ↓
DRA  ICC Profile                    ICC Profile
d/b    d/b                            d/b
```

| Element | v2.0 Status | Notes |
|---------|-------------|-------|
| DRA | ✅ via `dynamic_range_adjust()` | Per-band |
| CST (Color Space Transform) | 📋 partial | Well-known spaces only |
| ACS Operations (sharpening, LUT) | ✅ via `sips_convolve()`, `apply_lut()` | |
| Output Preparation | 📋 partial | Quantization to output bit-depth; CMM limited to well-known spaces |

### Non-Interactive Processing (Figure 3.3)

| Element | v2.0 Status | Notes |
|---------|-------------|-------|
| Band Equalization | ➖ | Sensor-specific; not in scope |
| MTF Restoration | ✅ via `sips_convolve()` | User supplies MTFR kernel |
| Atmospheric Correction | ➖ | Sensor-specific; not in scope |
| Data Remapping (PEDF/LinLog) | 📅 v2.1 | Equations 3.1–3.4 in spec |
| Asymmetric Pixel Correction | 📅 Future | |
| RRDS Generation | 📋 v2.0 | |
