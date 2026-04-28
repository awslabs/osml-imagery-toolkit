#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""DisplayChainFactory — metadata-aware display chain builder.

This module provides :class:`DisplayChainFactory`, a metadata-aware builder
that inspects an image source's NITF subheader fields and GeoTIFF tags,
classifies the image modality, and constructs the appropriate
:class:`~aws.osml.image_processing.processing_chain.ProcessingChain` for
interactive display.

The factory follows the SIPS (NGA.STND.0014) reference image chain
architecture: a composable sequence of operators whose order matters but
whose individual elements are interchangeable.  It encapsulates all
format-specific detection logic; the chain and individual step functions
remain format-agnostic.

Typical usage::

    from aws.osml.image_processing.display_chain_factory import DisplayChainFactory

    chain = DisplayChainFactory.build(source, stats=my_stats)
    result = chain(raw_chw_array)
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from defusedxml import ElementTree as SafeET
from numpy.typing import NDArray

from .dynamic_range_adjustment import DRAParameters, dynamic_range_adjust
from .processing_chain import ProcessingChain
from .statistics import ImageStatistics, SamplingStrategy

logger = logging.getLogger(__name__)

# Valid range adjustment preset names
_RANGE_ADJUSTMENT_PRESETS = ("dra", "minmax")


class DisplayChainFactory:
    """Builds processing chains for interactive display.

    Examines image metadata to construct an appropriate sequence of
    operations that converts raw sensor pixels into display-ready output.
    """

    @staticmethod
    def build(
        source: Any,
        stats: Optional[ImageStatistics] = None,
        range_adjustment: str = "dra",
        band_selection: Optional[Tuple[int, ...]] = None,
        output_dtype: np.dtype = np.dtype(np.uint8),
        **kwargs,
    ) -> ProcessingChain:
        """Build a display processing chain for the given image source.

        Inspects the source's metadata to classify its modality and
        constructs the appropriate chain of processing steps.

        Complex-valued sources must be remapped to scalar magnitude
        before being passed to this factory (see
        :func:`~aws.osml.image_processing.complex_remap.is_complex` and
        :class:`~aws.osml.image_processing.complex_remap.ComplexRemapFactory`).

        Args:
            source: A duck-typed ImageAssetProvider with ``pixel_value_type``,
                ``num_bands``, ``metadata``, and optionally
                ``get_data_extensions()`` methods.
            stats: Optional pre-computed image statistics.  When None,
                statistics will be computed from the source if needed.
            range_adjustment: Range adjustment method for EO imagery.
                Must be ``"dra"`` or ``"minmax"``.
            band_selection: Optional explicit tuple of band indices to
                select from the source.  When provided, overrides
                automatic band detection.
            output_dtype: Desired output pixel dtype.  Default uint8.
            **kwargs: Additional keyword arguments forwarded to chain
                builders (e.g., ``scale_factor``).

        Returns:
            A configured ProcessingChain instance.

        Raises:
            ValueError: If ``range_adjustment`` is not a recognized value.
        """
        # Validate range_adjustment
        if range_adjustment not in _RANGE_ADJUSTMENT_PRESETS:
            raise ValueError(f"Unknown range_adjustment: {range_adjustment!r}. Must be one of {_RANGE_ADJUSTMENT_PRESETS}.")

        # Extract source properties
        pixel_type = _safe_get_pixel_type(source)
        num_bands = _safe_get_num_bands(source)

        # Attempt to get metadata
        metadata = _safe_get_metadata(source)

        # --- Modality detection (first-match-wins priority) ---

        # 1. PhotometricInterpretation=3 (palette)
        if metadata is not None:
            photometric = _get_geotiff_photometric(metadata)
            if photometric == 3:
                # Palette chain — for now return empty chain as placeholder
                return ProcessingChain(
                    steps=[],
                    output_bands=3,
                    output_dtype=np.dtype(np.uint8),
                )

            # 2. PhotometricInterpretation=6 (YCbCr — display-ready)
            if photometric == 6:
                return ProcessingChain(
                    steps=[],
                    output_bands=num_bands,
                    output_dtype=np.dtype(np.uint8),
                )

        # 3. uint8 with 1 or 3 bands → display-ready
        if _is_uint8_pixel_type(pixel_type) and num_bands in (1, 3):
            return ProcessingChain(
                steps=[],
                output_bands=num_bands,
                output_dtype=np.dtype(np.uint8),
            )

        # 4. Fall through to EO chain
        # Resolve band selection: explicit overrides detection
        effective_band_selection = band_selection
        if effective_band_selection is None and num_bands > 3:
            effective_band_selection = _detect_rgb_bands(metadata, num_bands)

        return _build_eo_chain(
            stats=stats,
            source=source,
            num_bands=num_bands,
            band_selection=effective_band_selection,
            range_adjustment=range_adjustment,
            output_dtype=output_dtype,
            input_dtype=_pixel_type_to_numpy(pixel_type),
            metadata=metadata,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Modality detection helpers
# ---------------------------------------------------------------------------


def _pixel_type_to_numpy(pixel_type: Optional[str]) -> np.dtype:
    """Convert a pixel type string to a numpy dtype.

    Args:
        pixel_type: Pixel type string from the source, or None.

    Returns:
        The corresponding numpy dtype, defaulting to uint16 when
        the string is unrecognized or None.
    """
    if pixel_type is None:
        return np.dtype(np.uint16)
    mapping = {
        "UINT8": np.uint8,
        "BYTE": np.uint8,
        "B": np.uint8,
        "INT8": np.int8,
        "UINT16": np.uint16,
        "INT16": np.int16,
        "UINT32": np.uint32,
        "INT32": np.int32,
        "FLOAT32": np.float32,
        "FLOAT64": np.float64,
    }
    np_type = mapping.get(pixel_type.upper())
    if np_type is None:
        return np.dtype(np.uint16)
    return np.dtype(np_type)


def _is_uint8_pixel_type(pixel_type: Optional[str]) -> bool:
    """Check if the pixel type string indicates uint8 data.

    Args:
        pixel_type: Pixel type string from the source, or None.

    Returns:
        True if the pixel type indicates uint8 data.
    """
    if pixel_type is None:
        return False
    upper = pixel_type.upper()
    return upper in ("UINT8", "INT8", "BYTE", "B")


def _get_nitf_field(metadata: dict, field_name: str) -> Optional[str]:
    """Safely extract a NITF field from a metadata dictionary.

    Looks for the field name directly and also with common prefixes.

    Args:
        metadata: Metadata dictionary from the source.
        field_name: NITF field name to look up (e.g., "IREP", "ICAT").

    Returns:
        The field value as a stripped string, or None if not found.
    """
    value = metadata.get(field_name)
    if value is not None:
        return str(value).strip()
    return None


def _get_geotiff_photometric(metadata: dict) -> Optional[int]:
    """Extract the PhotometricInterpretation tag value from metadata.

    Looks for the tag by name and by TIFF tag number (262).

    Args:
        metadata: Metadata dictionary from the source.

    Returns:
        The PhotometricInterpretation integer value, or None if not found.
    """
    value = metadata.get("PhotometricInterpretation")
    if value is None:
        value = metadata.get(262)
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _get_nitf_band_subcategories(metadata: dict) -> Optional[List[str]]:
    """Get per-band ISUBCAT values from metadata.

    Checks for a top-level ``ISUBCAT`` key first, then falls back to
    extracting per-band values from a ``BAND_INFO`` list of dicts.

    Args:
        metadata: Metadata dictionary from the source.

    Returns:
        A list of ISUBCAT strings (one per band), or None if not found.
    """
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


# ---------------------------------------------------------------------------
# Safe source property accessors
# ---------------------------------------------------------------------------


def _get_effective_bits(metadata: Optional[dict], input_dtype: np.dtype) -> int:
    """Determine the effective bits per pixel from metadata or dtype.

    Checks NITF ABPP (Actual Bits Per Pixel) first, then GeoTIFF
    BitsPerSample (tag 258) for non-standard values, and falls back
    to the storage dtype's bit width.

    Args:
        metadata: Metadata dictionary from the source, or None.
        input_dtype: The numpy dtype of the pixel data.

    Returns:
        Effective bits per pixel (e.g., 11 for an 11-bit sensor
        stored in uint16).
    """
    dtype_bits = input_dtype.itemsize * 8

    if metadata is not None:
        # NITF: ABPP is the actual bits per pixel
        abpp = metadata.get("ABPP")
        if abpp is not None:
            try:
                bits = int(abpp)
                if 1 <= bits <= dtype_bits:
                    return bits
            except (ValueError, TypeError):
                pass

        # GeoTIFF: tag 258 (BitsPerSample) may indicate sub-dtype bit depth
        bps = metadata.get("258") or metadata.get("BitsPerSample")
        if bps is not None:
            try:
                if isinstance(bps, (list, tuple)):
                    bits = int(bps[0])
                else:
                    bits = int(bps)
                if 1 <= bits < dtype_bits:
                    return bits
            except (ValueError, TypeError, IndexError):
                pass

    return dtype_bits


def _safe_get_pixel_type(source: Any) -> Optional[str]:
    """Safely get the pixel_value_type from a source as a string.

    Handles both string pixel types and enum types (e.g.,
    ``PixelType.UInt16``) by converting to string and stripping
    any enum prefix.

    Args:
        source: A duck-typed ImageAssetProvider.

    Returns:
        The pixel type string, or None on failure.
    """
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


def _safe_get_num_bands(source: Any) -> int:
    """Safely get the num_bands from a source.

    Args:
        source: A duck-typed ImageAssetProvider.

    Returns:
        The number of bands, defaulting to 1 on failure.
    """
    try:
        return source.num_bands
    except Exception:
        return 1


def _safe_get_metadata(source: Any) -> Optional[dict]:
    """Safely get the metadata dictionary from a source.

    Args:
        source: A duck-typed ImageAssetProvider.

    Returns:
        The metadata dictionary, or None on failure.
    """
    try:
        meta_provider = source.metadata
        if meta_provider is not None:
            return dict(meta_provider)
    except Exception:
        logger.debug("Failed to read metadata from source, will use dtype/band-count fallback")
    return None


# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------


def _build_eo_chain(
    stats: Optional[ImageStatistics],
    source: Any,
    num_bands: int,
    band_selection: Optional[Tuple[int, ...]],
    range_adjustment: str,
    output_dtype: np.dtype,
    input_dtype: np.dtype = np.dtype(np.uint16),
    metadata: Optional[dict] = None,
    **kwargs,
) -> ProcessingChain:
    """Build an EO display chain with dynamic range adjustment.

    Constructs a chain that maps high-bit-depth EO pixels to
    display-ready output using LUT-based range adjustment.

    Args:
        stats: Optional pre-computed image statistics.
        source: The image source (used for stats computation if needed).
        num_bands: Number of bands in the source.
        band_selection: Optional explicit band selection tuple.
        range_adjustment: Range adjustment method ("dra" or "minmax").
        output_dtype: Desired output dtype.
        metadata: Metadata dictionary from the source, used to determine
            effective bit depth for histogram bin selection.
        **kwargs: Additional arguments.

    Returns:
        A ProcessingChain with appropriate output_bands and input_bands.
    """
    # Determine effective band count and input_bands
    if band_selection is not None:
        effective_bands = len(band_selection)
        input_bands = band_selection
    else:
        effective_bands = num_bands
        input_bands = None

    # Compute statistics if not provided
    if stats is None:
        try:
            from .statistics import compute_image_statistics

            effective_bits = _get_effective_bits(metadata, input_dtype)
            num_bins = 2**effective_bits

            sampling, sample_rate, num_workers = _select_statistics_strategy(source)
            stats = compute_image_statistics(
                source,
                num_bins=num_bins,
                sampling=sampling,
                sample_rate=sample_rate,
                num_workers=num_workers,
            )
        except Exception:
            logger.warning("Failed to compute statistics for EO chain; using identity chain")
            return ProcessingChain(
                steps=[],
                output_bands=effective_bands,
                output_dtype=np.dtype(output_dtype),
                input_bands=input_bands,
            )

    if stats is None:
        logger.warning("No statistics available for EO chain; using identity chain")
        return ProcessingChain(
            steps=[],
            output_bands=effective_bands,
            output_dtype=np.dtype(output_dtype),
            input_bands=input_bands,
        )

    # Build LUTs from statistics for the selected bands. When
    # band_selection indices are valid within the stats, use those
    # specific band stats. Otherwise (e.g., caller pre-computed stats
    # on the already-selected subset), take the first N bands.
    if band_selection is not None and all(i < len(stats.bands) for i in band_selection):
        selected_stats = [stats.bands[i] for i in band_selection]
    else:
        selected_stats = stats.bands[:effective_bands]

    # Float inputs cannot use LUT-based DRA (LUTs require integer indexing).
    # Instead, build a direct linear-stretch step from the DRA parameters.
    if np.issubdtype(input_dtype, np.floating):
        dra_params_list = []
        for band_stat in selected_stats:
            params = DRAParameters.from_counts(
                band_stat.histogram,
                first_bucket_value=float(band_stat.bin_edges[0]),
                last_bucket_value=float(band_stat.bin_edges[-1]),
            )
            dra_params_list.append(params)

        target_dtype = np.dtype(output_dtype)
        out_max = float(np.iinfo(target_dtype).max) if np.issubdtype(target_dtype, np.integer) else 1.0

        def float_dra_step(image: NDArray) -> NDArray:
            result = np.empty_like(image, dtype=target_dtype)
            for i, params in enumerate(dra_params_list):
                band_idx = i if i < image.shape[0] else 0
                low = params.suggested_min_value if range_adjustment == "dra" else params.actual_min_value
                high = params.suggested_max_value if range_adjustment == "dra" else params.actual_max_value
                if high <= low:
                    result[band_idx] = target_dtype.type(0)
                else:
                    scaled = (
                        (image[band_idx].astype(np.float32) - np.float32(low)) / np.float32(high - low) * np.float32(out_max)
                    )
                    result[band_idx] = np.clip(scaled, 0, out_max).astype(target_dtype)
            return result

        return ProcessingChain(
            steps=[float_dra_step],
            output_bands=effective_bands,
            output_dtype=target_dtype,
            input_bands=input_bands,
        )

    luts: List[NDArray] = []
    for band_stat in selected_stats:
        params = DRAParameters.from_counts(
            band_stat.histogram,
            first_bucket_value=float(band_stat.bin_edges[0]),
            last_bucket_value=float(band_stat.bin_edges[-1]),
        )
        lut = params.build_lut(
            input_dtype=np.dtype(input_dtype),
            output_dtype=np.dtype(output_dtype),
            range_adjustment=range_adjustment,
        )
        luts.append(lut)

    def dra_step(image: NDArray) -> NDArray:
        return dynamic_range_adjust(image, luts)

    return ProcessingChain(
        steps=[dra_step],
        output_bands=effective_bands,
        output_dtype=np.dtype(output_dtype),
        input_bands=input_bands,
    )


# ---------------------------------------------------------------------------
# Adaptive statistics strategy
# ---------------------------------------------------------------------------

# Threshold in total blocks above which auto statistics switches to block
# sampling. A 10k×10k 16-bit image with 512×512 tiles has ~400 blocks;
# 100 blocks covers most "small" images where full scan is acceptable.
_ADAPTIVE_BLOCK_THRESHOLD = 100

# Default workers for threaded block reads when adaptive strategy triggers.
_ADAPTIVE_NUM_WORKERS = 4

# Target number of blocks to sample when the image exceeds the threshold.
# 64 blocks of 512×512 = ~16M pixels — more than enough for histogram-based
# DRA while limiting I/O to a small fraction of a large image.
_ADAPTIVE_TARGET_BLOCKS = 64


def _select_statistics_strategy(source: Any) -> Tuple[SamplingStrategy, float, int]:
    """Choose sampling parameters based on image size.

    For small images (≤ _ADAPTIVE_BLOCK_THRESHOLD blocks), returns full
    scan with no threading. For larger images, returns block sampling at
    a rate that targets _ADAPTIVE_TARGET_BLOCKS blocks, with threaded
    reads.

    Args:
        source: A duck-typed ImageAssetProvider with ``block_grid_size``.

    Returns:
        A tuple of (SamplingStrategy, sample_rate, num_workers).
    """
    try:
        grid_rows, grid_cols = source.block_grid_size
        total_blocks = grid_rows * grid_cols
    except Exception:
        return (SamplingStrategy.ALL, 1.0, 0)

    if total_blocks <= _ADAPTIVE_BLOCK_THRESHOLD:
        return (SamplingStrategy.ALL, 1.0, 0)

    sample_rate = min(1.0, _ADAPTIVE_TARGET_BLOCKS / total_blocks)
    return (SamplingStrategy.BLOCK, sample_rate, _ADAPTIVE_NUM_WORKERS)


# ---------------------------------------------------------------------------
# Band selection detection helpers
# ---------------------------------------------------------------------------


def _detect_rgb_bands(metadata: Optional[dict], num_bands: int) -> Tuple[int, ...]:
    """Detect RGB band indices from metadata.

    Applies a priority-ordered set of heuristics to determine which
    bands in a multiband image correspond to red, green, and blue.

    Detection priority:
        1. NITF per-band IREPBAND containing R, G, B
        2. GeoTIFF PhotometricInterpretation=2 (RGB)
        3. GDAL_METADATA ColorInterp (Red, Green, Blue)
        4. BANDSB TRE center wavelengths (visible R/G/B ranges)
        5. NITF per-band ISUBCAT containing R, G, B color codes
        6. Fallback to (0, 1, 2) with warning

    Args:
        metadata: Metadata dictionary from the source, or None.
        num_bands: Total number of bands in the source.

    Returns:
        A tuple of 3 band indices (red, green, blue).
        Falls back to (0, 1, 2) with a warning if no metadata signal
        is found.
    """
    if metadata is not None:
        # 1. NITF IREPBAND
        irepband = _get_nitf_band_representations(metadata)
        if irepband is not None:
            r_idx = _find_band(irepband, "R")
            g_idx = _find_band(irepband, "G")
            b_idx = _find_band(irepband, "B")
            if r_idx is not None and g_idx is not None and b_idx is not None:
                return (r_idx, g_idx, b_idx)

        # 2. GeoTIFF PhotometricInterpretation=2
        photometric = _get_geotiff_photometric(metadata)
        if photometric == 2:
            return (0, 1, 2)

        # 3. GDAL_METADATA ColorInterp
        color_interps = _get_gdal_color_interps(metadata)
        if color_interps is not None:
            r_idx = _find_band(color_interps, "Red")
            g_idx = _find_band(color_interps, "Green")
            b_idx = _find_band(color_interps, "Blue")
            if r_idx is not None and g_idx is not None and b_idx is not None:
                return (r_idx, g_idx, b_idx)

        # 4. BANDSB TRE wavelengths
        wavelengths = _get_bandsb_tre(metadata)
        if wavelengths is not None and len(wavelengths) > 3:
            result = _match_bands_by_wavelength(wavelengths)
            if result is not None:
                return result

        # 5. NITF ISUBCAT color codes
        subcategories = _get_nitf_band_subcategories(metadata)
        if subcategories is not None:
            r_idx = _find_band(subcategories, "R")
            g_idx = _find_band(subcategories, "G")
            b_idx = _find_band(subcategories, "B")
            if r_idx is not None and g_idx is not None and b_idx is not None:
                return (r_idx, g_idx, b_idx)

    # 6. Fallback
    logger.warning(
        "No metadata signal identified RGB bands for multiband source with %d bands; falling back to bands (0, 1, 2)",
        num_bands,
    )
    return (0, 1, 2)


def _get_nitf_band_representations(metadata: dict) -> Optional[List[str]]:
    """Get per-band IREPBAND values from metadata.

    Checks for a top-level ``IREPBAND`` key first, then falls back to
    extracting per-band values from a ``BAND_INFO`` list of dicts.

    Args:
        metadata: Metadata dictionary from the source.

    Returns:
        A list of IREPBAND strings (one per band), or None if not found.
    """
    value = metadata.get("IREPBAND")
    if value is not None:
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value]
        value_str = str(value).strip()
        if "," in value_str:
            return [v.strip() for v in value_str.split(",")]
        return [value_str]

    band_info = metadata.get("BAND_INFO")
    if isinstance(band_info, (list, tuple)) and len(band_info) > 0:
        result = [str(b.get("IREPBAND", "")).strip() for b in band_info if isinstance(b, dict)]
        if result:
            return result

    return None


def _get_gdal_color_interps(metadata: dict) -> Optional[List[str]]:
    """Extract per-band ColorInterp values from GDAL_METADATA XML.

    Parses the GDAL_METADATA XML string and extracts ColorInterp
    Item elements, returning them ordered by sample index.

    Expected XML format::

        <GDALMetadata>
          <Item name="ColorInterp" sample="0">Red</Item>
          <Item name="ColorInterp" sample="1">Green</Item>
          <Item name="ColorInterp" sample="2">Blue</Item>
        </GDALMetadata>

    Args:
        metadata: Metadata dictionary from the source.

    Returns:
        A list of ColorInterp strings ordered by sample index,
        or None if GDAL_METADATA is not present or cannot be parsed.
    """
    xml_str = metadata.get("GDAL_METADATA")
    if xml_str is None:
        return None
    if not isinstance(xml_str, str):
        return None

    try:
        root = SafeET.fromstring(xml_str)
    except Exception:
        logger.debug("Failed to parse GDAL_METADATA XML")
        return None

    # Collect ColorInterp items with their sample indices
    interps: dict = {}
    for item in root.iter("Item"):
        name = item.get("name", "")
        if name == "ColorInterp":
            sample_str = item.get("sample")
            if sample_str is not None and item.text:
                try:
                    sample_idx = int(sample_str)
                    interps[sample_idx] = item.text.strip()
                except (ValueError, TypeError):
                    continue

    if not interps:
        return None

    # Build ordered list
    max_idx = max(interps.keys())
    result = []
    for i in range(max_idx + 1):
        result.append(interps.get(i, ""))

    return result


def _get_bandsb_tre(metadata: dict) -> Optional[List[float]]:
    """Get per-band center wavelengths from BANDSB TRE metadata.

    Looks for wavelength data in two possible locations:
        - ``metadata["BANDSB"]["center_wavelengths"]`` — a dict with
          a list of floats
        - ``metadata["BANDSB_CENTER_WAVELENGTHS"]`` — a list of floats
          directly

    Args:
        metadata: Metadata dictionary from the source.

    Returns:
        A list of center wavelengths in nanometers (one per band),
        or None if not found.
    """
    # Try dict form: metadata["BANDSB"]["center_wavelengths"]
    bandsb = metadata.get("BANDSB")
    if bandsb is not None:
        if isinstance(bandsb, dict):
            wavelengths = bandsb.get("center_wavelengths")
            if wavelengths is not None and isinstance(wavelengths, (list, tuple)):
                try:
                    return [float(w) for w in wavelengths]
                except (ValueError, TypeError):
                    pass

    # Try flat form: metadata["BANDSB_CENTER_WAVELENGTHS"]
    wavelengths = metadata.get("BANDSB_CENTER_WAVELENGTHS")
    if wavelengths is not None and isinstance(wavelengths, (list, tuple)):
        try:
            return [float(w) for w in wavelengths]
        except (ValueError, TypeError):
            pass

    return None


def _match_bands_by_wavelength(wavelengths: List[float]) -> Optional[Tuple[int, ...]]:
    """Match bands to visible R/G/B wavelength ranges.

    Finds the bands whose center wavelengths are closest to the
    center of each visible color range:
        - Red: 620-700nm (center 660nm)
        - Green: 495-570nm (center 532.5nm)
        - Blue: 450-495nm (center 472.5nm)

    For each color, the band with the minimum distance to the range
    center is selected. If no band falls within a reasonable distance
    of any range, returns None.

    Args:
        wavelengths: List of center wavelengths in nanometers.

    Returns:
        A tuple of 3 band indices (red, green, blue), or None if
        matching fails (e.g., no bands near visible ranges).
    """
    # Visible range centers (nm)
    red_center = 660.0
    green_center = 532.5
    blue_center = 472.5

    def closest_band(target: float) -> int:
        """Find the band index with wavelength closest to target."""
        min_dist = float("inf")
        best_idx = 0
        for i, wl in enumerate(wavelengths):
            dist = abs(wl - target)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    r_idx = closest_band(red_center)
    g_idx = closest_band(green_center)
    b_idx = closest_band(blue_center)

    # Sanity check: all three indices should be distinct
    # If they're not, the wavelength data likely doesn't cover visible range
    if r_idx == g_idx or r_idx == b_idx or g_idx == b_idx:
        return None

    return (r_idx, g_idx, b_idx)


def _find_band(values: List[str], target: str) -> Optional[int]:
    """Find the index of a target value in a list (case-insensitive).

    Args:
        values: List of string values to search.
        target: Target value to find.

    Returns:
        The index of the first matching value, or None if not found.
    """
    target_upper = target.upper()
    for i, v in enumerate(values):
        if v.upper() == target_upper:
            return i
    return None
