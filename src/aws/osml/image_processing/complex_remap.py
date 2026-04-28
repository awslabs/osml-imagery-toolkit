#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Complex imagery remap — domain transform from complex I/Q to scalar magnitude.

This module provides building-block functions for converting complex-valued
SAR image data (I/Q pairs, magnitude/phase, or native complex) into
real-valued scalar magnitudes suitable for display processing via DRA.

Band interpretation constants define the semantic role of each band in a
multi-band complex image asset. The :func:`decode_to_iq` function normalizes
any supported encoding into canonical ``(2, H, W)`` float32 I/Q form.

The :func:`is_complex` utility detects whether an image asset contains
complex-valued pixel data using image subheader indicators only (no false
positives on SAR-derived display products).

The :func:`quarter_power_remap` and :func:`magnitude_remap` presets convert
normalized ``(2, H, W)`` float32 I/Q to ``(1, H, W)`` float32 scalars.

The :class:`ComplexRemapFactory` builds a :class:`MappedImageProvider` that
wraps a complex-valued source with a decode+remap closure.

The :func:`load_complex_remap` convenience function handles reader interaction
(DES XML parsing, amplitude table extraction) and delegates to the factory.
"""

import logging
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TWO_PI = np.float32(np.pi * 2.0)

# --- Band interpretation constants ---

ROLE_REAL = "real"
ROLE_IMAGINARY = "imaginary"
ROLE_MAGNITUDE = "magnitude"
ROLE_PHASE = "phase"
ROLE_AMPLITUDE_INDEX = "amplitude_index"

_VALID_ROLES = frozenset({ROLE_REAL, ROLE_IMAGINARY, ROLE_MAGNITUDE, ROLE_PHASE, ROLE_AMPLITUDE_INDEX})


# --- Building blocks ---


def decode_to_iq(
    data: NDArray,
    band_interpretation: Optional[Sequence[str]] = None,
    amplitude_table: Optional[NDArray] = None,
) -> NDArray:
    """Normalize raw pixel data to canonical (2, H, W) float32 I/Q form.

    Handles all supported complex encodings: real/imaginary pairs,
    amplitude-index/phase (with lookup table), magnitude/phase (polar),
    and native complex dtypes.

    :param data: Raw pixel block. Shape must be ``(2, H, W)`` for
        two-band interpretations or ``(1, H, W)`` / ``(H, W)`` for
        native complex dtypes.
    :param band_interpretation: Per-band semantic role list. Valid roles:
        "real", "imaginary", "magnitude", "phase", "amplitude_index".
        When None, inferred from dtype (complex → native complex path;
        2-band numeric → defaults to ["real", "imaginary"]).
    :param amplitude_table: Lookup table for "amplitude_index" bands.
        Required when "amplitude_index" appears in band_interpretation.
    :return: Canonical (2, H, W) float32 array with [0]=real, [1]=imaginary.
    :raises ValueError: If parameters are inconsistent or data shape is invalid.
    """
    if band_interpretation is not None:
        for role in band_interpretation:
            if role not in _VALID_ROLES:
                raise ValueError(f"Unrecognized band role: {role!r}. Valid roles: {sorted(_VALID_ROLES)}")

    if np.iscomplexobj(data):
        if data.ndim == 2:
            real = data.real.astype(np.float32)
            imag = data.imag.astype(np.float32)
            return np.stack([real, imag], axis=0)
        elif data.ndim == 3 and data.shape[0] == 1:
            real = data[0].real.astype(np.float32)
            imag = data[0].imag.astype(np.float32)
            return np.stack([real, imag], axis=0)
        raise ValueError(f"Native complex data must be (1, H, W) or (H, W), got shape {data.shape}")

    if band_interpretation is None:
        if data.ndim == 3 and data.shape[0] >= 2:
            band_interpretation = [ROLE_REAL, ROLE_IMAGINARY]
        else:
            raise ValueError(
                f"Cannot infer band_interpretation: data is not complex dtype and shape {data.shape} "
                f"does not have >= 2 bands in CHW layout"
            )

    if data.ndim != 3:
        raise ValueError(f"Expected (C, H, W) array, got shape {data.shape}")

    if data.shape[0] < len(band_interpretation):
        raise ValueError(f"Data has {data.shape[0]} bands but band_interpretation has {len(band_interpretation)} roles")

    roles = list(band_interpretation)

    if ROLE_AMPLITUDE_INDEX in roles and amplitude_table is None:
        raise ValueError("amplitude_table is required when 'amplitude_index' appears in band_interpretation")

    if amplitude_table is not None and ROLE_AMPLITUDE_INDEX not in roles:
        raise ValueError("amplitude_table provided but 'amplitude_index' not in band_interpretation")

    if ROLE_REAL in roles and ROLE_IMAGINARY in roles:
        i_idx = roles.index(ROLE_REAL)
        q_idx = roles.index(ROLE_IMAGINARY)
        real = data[i_idx].astype(np.float32)
        imag = data[q_idx].astype(np.float32)
        return np.stack([real, imag], axis=0)

    if ROLE_AMPLITUDE_INDEX in roles and ROLE_PHASE in roles:
        amp_idx = roles.index(ROLE_AMPLITUDE_INDEX)
        phs_idx = roles.index(ROLE_PHASE)
        amp_table = np.asarray(amplitude_table, dtype=np.float32)
        amplitude = amp_table[data[amp_idx].astype(np.intp)]
        phase_raw = data[phs_idx].astype(np.float32)
        nbits = data.dtype.itemsize * 8
        phase = phase_raw / np.float32(2**nbits) * TWO_PI
        real = amplitude * np.cos(phase)
        imag = amplitude * np.sin(phase)
        return np.stack([real, imag], axis=0)

    if ROLE_MAGNITUDE in roles and ROLE_PHASE in roles:
        mag_idx = roles.index(ROLE_MAGNITUDE)
        phs_idx = roles.index(ROLE_PHASE)
        magnitude = data[mag_idx].astype(np.float32)
        if np.issubdtype(data.dtype, np.integer):
            nbits = data.dtype.itemsize * 8
            phase = data[phs_idx].astype(np.float32) / np.float32(2**nbits) * TWO_PI
        else:
            phase = data[phs_idx].astype(np.float32)
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)
        return np.stack([real, imag], axis=0)

    raise ValueError(f"Cannot decode band_interpretation {roles} to I/Q")


def complex_to_power(data: NDArray) -> NDArray:
    """Compute pixel power (intensity) from complex or I/Q data.

    For native complex dtypes: ``real² + imag²``
    For (2, H, W) I/Q layout: ``I² + Q²``

    :param data: Either a native complex array or (2, H, W) float/int array.
    :return: Real-valued power array. Shape is (H, W) for complex input,
        (H, W) for (2, H, W) input.
    """
    if np.iscomplexobj(data):
        if data.ndim == 3 and data.shape[0] == 1:
            arr = data[0]
        elif data.ndim == 2:
            arr = data
        else:
            raise ValueError(f"Expected complex array with shape (H, W) or (1, H, W), got shape {data.shape}")
        return arr.real.astype(np.float32) ** 2 + arr.imag.astype(np.float32) ** 2

    if data.ndim == 3 and data.shape[0] >= 2:
        band0 = data[0].astype(np.float32)
        band1 = data[1].astype(np.float32)
        return band0**2 + band1**2

    raise ValueError(f"Expected complex array or (2+, H, W) I/Q array, got shape {data.shape}, dtype {data.dtype}")


def power_to_decibels(power: NDArray) -> NDArray:
    """Convert power values to decibels: 10 * log10(power).

    Zeros and negative values produce -inf; use with caution.

    :param power: Real-valued power array.
    :return: Power in decibels (float32).
    """
    return np.float32(10.0) * np.log10(power.astype(np.float32))


# --- Remap presets ---


def quarter_power_remap(block: NDArray) -> NDArray:
    """Normalized float I/Q → fourth root of power (sqrt(sqrt(I² + Q²))).

    Produces a roughly Gaussian distribution suitable for DRA.

    :param block: Canonical (2, H, W) float32 I/Q array.
    :return: (1, H, W) float32 scalar magnitude.
    """
    power = block[0] ** 2 + block[1] ** 2
    result = np.sqrt(np.sqrt(power))
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, np.float32(0.0))
    return result[np.newaxis, :, :]


def magnitude_remap(block: NDArray) -> NDArray:
    """Normalized float I/Q → sqrt(I² + Q²).

    Linear magnitude — preserves relative intensity relationships.

    :param block: Canonical (2, H, W) float32 I/Q array.
    :return: (1, H, W) float32 scalar magnitude.
    """
    power = block[0] ** 2 + block[1] ** 2
    result = np.sqrt(power)
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, np.float32(0.0))
    return result[np.newaxis, :, :]


# --- Detection utility ---

_SAR_IQ_ICATS = frozenset({"SAR", "SARIQ", "ISAR"})


def is_complex(source: Any) -> bool:
    """Check if an ImageAssetProvider contains complex-valued pixel data.

    Uses definitive indicators from the image subheader only — no false
    positives on SAR-derived display products (SIDD, detected ISAR).
    Checks are ordered first-match-wins by reliability.

    Checks:
        1. pixel_value_type contains "COMPLEX" → True
        2. ICAT == "SARIQ" → True
        3. Band subcategories contain both "I" and "Q" → True
        4. Band subcategories contain "M" and "P" with ICAT in
           ("SAR", "SARIQ", "ISAR") → True
        5. IREP == "POLAR" and ICAT == "SAR" → True

    Does NOT trigger on:
        - ICAT=SAR alone (SIDD and other SAR-derived products)
        - IREP=POLAR with ICAT in (CCD, WIND, CURRENT)

    :param source: An ImageAssetProvider with ``pixel_value_type``,
        ``num_bands``, and ``metadata`` attributes.
    :return: True if the source contains complex-valued pixel data.
    """
    # 1. Complex pixel type
    pixel_type = _safe_pixel_type(source)
    if pixel_type is not None and "COMPLEX" in pixel_type.upper():
        return True

    # Extract metadata for remaining checks
    metadata = _safe_metadata(source)
    icat = _meta_field(metadata, "ICAT")
    irep = _meta_field(metadata, "IREP")

    # 2. ICAT == "SARIQ" is definitionally I/Q complex
    if icat == "SARIQ":
        return True

    # 3. Band subcategories contain both "I" and "Q"
    num_bands = _safe_num_bands(source)
    subcategories = _meta_subcategories(metadata)
    if num_bands >= 2 and subcategories is not None:
        subcat_upper = {s.upper() for s in subcategories if s}
        if "I" in subcat_upper and "Q" in subcat_upper:
            return True

        # 4. M and P subcategories with SAR-family ICAT
        if "M" in subcat_upper and "P" in subcat_upper:
            if icat in _SAR_IQ_ICATS:
                return True

    # 5. IREP == "POLAR" and ICAT == "SAR"
    if irep == "POLAR" and icat == "SAR":
        return True

    return False


# --- Internal helpers for is_complex() ---


def _safe_pixel_type(source: Any) -> Optional[str]:
    """Extract pixel_value_type as a string, stripping enum prefixes."""
    try:
        pvt = source.pixel_value_type
        if pvt is None:
            return None
        pvt_str = str(pvt)
        if "." in pvt_str:
            pvt_str = pvt_str.rsplit(".", 1)[-1]
        return pvt_str
    except Exception:
        return None


def _safe_num_bands(source: Any) -> int:
    """Extract num_bands from source, defaulting to 1."""
    try:
        return source.num_bands
    except Exception:
        return 1


def _safe_metadata(source: Any) -> Optional[dict]:
    """Extract the metadata dictionary from source."""
    try:
        meta = source.metadata
        if meta is not None:
            return dict(meta)
    except Exception:
        pass
    return None


def _meta_field(metadata: Optional[dict], field: str) -> Optional[str]:
    """Safely extract a NITF field from metadata as an uppercase stripped string."""
    if metadata is None:
        return None
    value = metadata.get(field)
    if value is None:
        return None
    return str(value).strip().upper()


def _meta_subcategories(metadata: Optional[dict]) -> Optional[List[str]]:
    """Extract per-band ISUBCAT values from metadata."""
    if metadata is None:
        return None
    value = metadata.get("ISUBCAT")
    if value is not None:
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value]
        value_str = str(value).strip()
        if "," in value_str:
            return [v.strip() for v in value_str.split(",")]
        return [value_str]

    band_info = metadata.get("BAND_INFO")
    if isinstance(band_info, (list, tuple)) and len(band_info) > 0:
        result = [str(b.get("ISUBCAT", "")).strip() for b in band_info if isinstance(b, dict)]
        if result:
            return result

    return None


# --- Remap preset registry ---

_REMAP_PRESETS = {
    "quarter_power": quarter_power_remap,
    "magnitude": magnitude_remap,
}


# --- Factory ---


class ComplexRemapFactory:
    """Builds MappedImageProviders that convert complex pixels to scalar magnitude."""

    @staticmethod
    def build(
        source: Any,
        band_interpretation: Optional[Sequence[str]] = None,
        amplitude_table: Optional[NDArray] = None,
        remap: Union[str, Callable[[NDArray], NDArray]] = "quarter_power",
        cache: Optional[Any] = None,
    ) -> Any:
        """Wrap a complex-valued asset with a domain transform.

        :param source: ImageAssetProvider with complex pixel data.
        :param band_interpretation: Per-band semantic role list. Length must
            equal source.num_bands. Valid roles: "real", "imaginary",
            "magnitude", "phase", "amplitude_index". When None, inferred
            from source: complex dtype → native complex path; 2-band
            numeric → defaults to ["real", "imaginary"].
        :param amplitude_table: Lookup table for "amplitude_index" bands.
            Required when "amplitude_index" appears in band_interpretation;
            must be None otherwise.
        :param remap: Remap preset name ("quarter_power", "magnitude") or
            a custom callable. Custom callables receive normalized float
            I/Q as (2, H, W) float32 and must return (1, H, W) float32.
        :param cache: Optional shared TileCache for caching remapped blocks.
        :return: A MappedImageProvider reporting num_bands=1,
            pixel_value_type="FLOAT32".
        :raises ValueError: If parameters are internally inconsistent.
        """
        from .mapped_provider import MappedImageProvider

        num_bands = _safe_num_bands(source)

        if band_interpretation is not None:
            interp_list = list(band_interpretation)
            if len(interp_list) != num_bands:
                raise ValueError(f"band_interpretation has {len(interp_list)} roles but source has {num_bands} bands")
            for role in interp_list:
                if role not in _VALID_ROLES:
                    raise ValueError(f"Unrecognized band role: {role!r}. Valid roles: {sorted(_VALID_ROLES)}")
            if ROLE_AMPLITUDE_INDEX in interp_list and amplitude_table is None:
                raise ValueError("amplitude_table is required when 'amplitude_index' appears in band_interpretation")
            if amplitude_table is not None and ROLE_AMPLITUDE_INDEX not in interp_list:
                raise ValueError("amplitude_table provided but 'amplitude_index' not in band_interpretation")
        else:
            if amplitude_table is not None:
                raise ValueError("amplitude_table provided but band_interpretation is None and no 'amplitude_index' role")
            pixel_type = _safe_pixel_type(source)
            is_native_complex = pixel_type is not None and "COMPLEX" in pixel_type.upper()
            if not is_native_complex and num_bands < 2:
                raise ValueError(
                    f"band_interpretation is None, source is not complex dtype, "
                    f"and num_bands={num_bands} < 2 — cannot infer interpretation"
                )

        remap_fn: Callable[[NDArray], NDArray]
        if isinstance(remap, str):
            if remap not in _REMAP_PRESETS:
                raise ValueError(f"Unknown remap preset: {remap!r}. Available: {sorted(_REMAP_PRESETS)}")
            remap_fn = _REMAP_PRESETS[remap]
        else:
            remap_fn = remap

        frozen_interp: Optional[List[str]] = list(band_interpretation) if band_interpretation is not None else None
        frozen_amp_table: Optional[NDArray] = (
            np.asarray(amplitude_table, dtype=np.float32) if amplitude_table is not None else None
        )

        def _decode_and_remap(block: NDArray) -> NDArray:
            iq = decode_to_iq(block, band_interpretation=frozen_interp, amplitude_table=frozen_amp_table)
            return remap_fn(iq)

        return MappedImageProvider(
            source=source,
            func=_decode_and_remap,
            cache=cache,
            name="complex_remap",
            num_bands=1,
            pixel_value_type="FLOAT32",
        )


# --- Convenience function ---


def load_complex_remap(
    reader: Any,
    asset_key: Optional[str] = None,
    remap: Union[str, Callable[[NDArray], NDArray]] = "quarter_power",
    cache: Optional[Any] = None,
) -> Any:
    """Extract metadata from reader and build a complex remap provider.

    Iterates all DES assets, identifies SICD XML by DESSHTN namespace
    ("urn:SICD:*"), extracts PixelType and AmpTable. Falls back to NITF
    image subheader fields (ICAT, IREP, ISUBCAT, PVTYPE) when no SICD
    DES is present. Maps all metadata to a band_interpretation list and
    delegates to ComplexRemapFactory.build().

    :param reader: An osml-imagery-io DatasetReader (from IO.open).
    :param asset_key: Specific image asset key to use (default: first
        "image:" asset).
    :param remap: Remap preset name or custom callable.
    :param cache: Optional shared TileCache.
    :return: A MappedImageProvider reporting num_bands=1, pixel_value_type="FLOAT32".
    :raises ValueError: If a valid band_interpretation cannot be determined.
    """
    if asset_key is None:
        all_keys = reader.get_asset_keys()
        image_keys = [k for k in all_keys if k.startswith("image:")]
        if not image_keys:
            raise ValueError("No image assets found in dataset")
        asset_key = image_keys[0]

    image_asset = reader.get_asset(asset_key)

    band_interpretation, amplitude_table = _extract_complex_metadata(reader, image_asset)

    return ComplexRemapFactory.build(
        source=image_asset,
        band_interpretation=band_interpretation,
        amplitude_table=amplitude_table,
        remap=remap,
        cache=cache,
    )


def _extract_complex_metadata(reader: Any, image_asset: Any) -> tuple:
    """Extract band_interpretation and amplitude_table from reader metadata.

    Strategy:
    1. Iterate DES assets, find SICD DES by DESSHTN="urn:SICD:*"
    2. If found, parse SICD XML for PixelType and AmpTable
    3. If no SICD DES, fall back to NITF subheader fields

    :return: Tuple of (band_interpretation, amplitude_table).
    :raises ValueError: If no valid interpretation can be determined.
    """
    sicd_result = _try_sicd_des(reader)
    if sicd_result is not None:
        return sicd_result

    subheader_result = _try_nitf_subheader(image_asset)
    if subheader_result is not None:
        return subheader_result

    raise ValueError(
        "Cannot determine complex band interpretation: no SICD DES found and "
        "NITF subheader fields do not indicate complex data"
    )


def _try_sicd_des(reader: Any) -> Optional[tuple]:
    """Iterate DES assets looking for SICD XML; extract PixelType and AmpTable."""
    try:
        all_keys = reader.get_asset_keys()
    except Exception:
        return None

    des_keys = [k for k in all_keys if k.startswith("des:")]
    if not des_keys:
        return None

    for key in des_keys:
        try:
            des_asset = reader.get_asset(key)
            des_meta = dict(des_asset.metadata) if des_asset.metadata else {}
            desid = des_meta.get("DESID", "").strip()
            if desid != "XML_DATA_CONTENT":
                continue
            desshtn = des_meta.get("DESSHTN", "")
            if not isinstance(desshtn, str) or not desshtn.startswith("urn:SICD:"):
                continue

            xml_str = des_asset.raw_asset.read().decode("utf-8")
            if not xml_str:
                continue

            return _parse_sicd_xml_metadata(xml_str)
        except Exception as e:
            logger.debug("Failed to process DES asset %s: %s", key, e)
            continue

    return None


def _parse_sicd_xml_metadata(xml_str: str) -> tuple:
    """Parse SICD XML and extract PixelType → band_interpretation and AmpTable."""
    import aws.osml.formats.sicd.models.sicd_v1_2_1 as sicd121  # noqa: F401
    import aws.osml.formats.sicd.models.sicd_v1_3_0 as sicd130  # noqa: F401
    from aws.osml.formats.model_utils import sicd_parser

    sicd = sicd_parser.from_string(xml_str)

    pixel_type = sicd.image_data.pixel_type if sicd.image_data else None
    if pixel_type is None:
        raise ValueError("SICD XML lacks ImageData/PixelType")

    pixel_type_value = pixel_type.value if hasattr(pixel_type, "value") else str(pixel_type)

    if pixel_type_value == "RE32F_IM32F":
        return [ROLE_REAL, ROLE_IMAGINARY], None
    elif pixel_type_value == "RE16I_IM16I":
        return [ROLE_REAL, ROLE_IMAGINARY], None
    elif pixel_type_value == "AMP8I_PHS8I":
        amp_table_data = _extract_amp_table(sicd)
        return [ROLE_AMPLITUDE_INDEX, ROLE_PHASE], amp_table_data
    else:
        raise ValueError(f"Unrecognized SICD PixelType: {pixel_type_value}")


def _extract_amp_table(sicd: Any) -> NDArray:
    """Extract the AmpTable from a parsed SICD dataclass as a float32 array."""
    if sicd.image_data.amp_table is None:
        raise ValueError("SICD PixelType is AMP8I_PHS8I but AmpTable is missing")

    amplitudes = sicd.image_data.amp_table.amplitude
    if not amplitudes:
        raise ValueError("SICD AmpTable has no Amplitude entries")

    table = np.zeros(len(amplitudes), dtype=np.float32)
    for entry in amplitudes:
        idx = entry.index
        val = entry.value
        if idx is not None and val is not None and 0 <= idx < len(table):
            table[idx] = np.float32(val)

    return table


def _try_nitf_subheader(image_asset: Any) -> Optional[tuple]:
    """Extract band_interpretation from NITF subheader fields.

    Checks ISUBCAT for I/Q or M/P band labels. Only returns a result if
    the metadata definitively indicates complex data.
    """
    metadata = _safe_metadata(image_asset)
    if metadata is None:
        return None

    subcategories = _meta_subcategories(metadata)
    if subcategories is not None:
        subcat_upper = [s.upper() for s in subcategories if s]

        if "I" in subcat_upper and "Q" in subcat_upper:
            interp = []
            for s in subcat_upper:
                if s == "I":
                    interp.append(ROLE_REAL)
                elif s == "Q":
                    interp.append(ROLE_IMAGINARY)
                else:
                    interp.append(ROLE_REAL)
            return interp, None

        icat = _meta_field(metadata, "ICAT")
        if "M" in subcat_upper and "P" in subcat_upper and icat in _SAR_IQ_ICATS:
            interp = []
            for s in subcat_upper:
                if s == "M":
                    interp.append(ROLE_MAGNITUDE)
                elif s == "P":
                    interp.append(ROLE_PHASE)
                else:
                    interp.append(ROLE_MAGNITUDE)
            return interp, None

    pixel_type = _safe_pixel_type(image_asset)
    if pixel_type is not None and "COMPLEX" in pixel_type.upper():
        return None, None

    return None
