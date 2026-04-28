#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""
Image statistics data model and computation functions.

This module provides data classes for per-band image statistics and functions
to compute, merge, and serialize/deserialize statistics. These are the
foundational building blocks for dynamic range adjustment and other
image processing operations.
"""

import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from defusedxml import ElementTree as SafeET
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Controls how pixels are sampled during statistics computation.

    :cvar ALL: Use every pixel in the array.
    :cvar RANDOM: Use a random subset of pixels controlled by sample_rate.
    :cvar BLOCK: Include/exclude entire blocks stochastically at sample_rate.
        Included blocks use ALL pixels internally.
    """

    ALL = "all"
    RANDOM = "random"
    BLOCK = "block"


@dataclass
class BandStatistics:
    """Per-band statistical measures for an image band.

    :param min: Minimum pixel value in the band.
    :param max: Maximum pixel value in the band.
    :param mean: Mean pixel value in the band.
    :param stddev: Standard deviation of pixel values in the band.
    :param count: Number of pixels contributing to this statistic.
    :param m2: Sum of squared deviations from the mean (variance * count).
        Uses population variance semantics.
    :param histogram: 1-D array of histogram counts, shape (num_bins,).
    :param bin_edges: 1-D array of histogram bin edges, shape (num_bins + 1,).
        Follows numpy.histogram conventions.
    """

    min: float
    max: float
    mean: float
    stddev: float
    count: int
    m2: float
    histogram: NDArray
    bin_edges: NDArray

    def __post_init__(self):
        if len(self.bin_edges) != len(self.histogram) + 1:
            raise ValueError(
                f"bin_edges length ({len(self.bin_edges)}) must be len(histogram) + 1 ({len(self.histogram) + 1})"
            )


@dataclass
class ImageStatistics:
    """Aggregate statistics for a multi-band image.

    :param bands: List of BandStatistics, one per band.
    :param sample_rate: Fraction of pixels sampled (0.0 to 1.0). Default 1.0
        indicates all pixels were used.
    """

    bands: List[BandStatistics]
    sample_rate: float = 1.0


def _dtype_range(dtype: np.dtype):
    """Return the (min, max) value range for a given numpy dtype.

    For integer types, returns the full representable range.
    For floating-point types, returns (0.0, 1.0) as a conventional range.

    :param dtype: The numpy dtype to query.
    :return: Tuple of (min_value, max_value).
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    else:
        return 0.0, 1.0


def _num_bins_for_dtype(dtype: np.dtype) -> int:
    """Determine an appropriate histogram bin count for a given dtype.

    For integer types, uses one bin per possible value capped at 65536.
    For floating-point types, defaults to 256.

    :param dtype: The numpy dtype to query.
    :return: Number of bins.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return min(int(info.max) - int(info.min) + 1, 65536)
    return 256


def compute_statistics(
    image: NDArray,
    num_bins: int = 0,
    bin_edges: Optional[Union[NDArray, List[NDArray]]] = None,
    sampling: SamplingStrategy = SamplingStrategy.ALL,
    sample_rate: float = 1.0,
) -> ImageStatistics:
    """Compute per-band statistics from a CHW NumPy array.

    :param image: Input array in CHW (bands, height, width) layout. A 2-D
        array (H, W) is treated as a single band.
    :param num_bins: Number of histogram bins when bin_edges is not provided.
        When 0 (the default), the bin count is derived from the image's
        dtype: one bin per possible value for integer types (capped at
        65536), or 256 for floating-point types.
    :param bin_edges: Explicit bin edges for histogram computation. Accepts
        a single NDArray (applied to all bands) or a List[NDArray] for
        per-band edges. When None, edges are derived automatically: per-band
        data-derived ranges for floating-point dtypes, or shared dtype-range
        edges for integer dtypes.
    :param sampling: Pixel sampling strategy. Default SamplingStrategy.ALL.
    :param sample_rate: Fraction of pixels to sample when using
        SamplingStrategy.RANDOM. Ignored for SamplingStrategy.ALL.
        Default 1.0.
    :return: ImageStatistics with one BandStatistics per band.
    """
    # Normalize to 3-D CHW
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim != 3:
        raise ValueError(f"Expected 2-D (H, W) or 3-D (C, H, W) array, got {image.ndim}-D")

    num_bands, height, width = image.shape
    num_pixels = height * width

    # Resolve bin_edges into per-band list or shared edges
    effective_bins = _num_bins_for_dtype(image.dtype) if num_bins == 0 else num_bins

    if bin_edges is not None:
        if isinstance(bin_edges, list):
            per_band_edges = [np.asarray(e, dtype=np.float64) for e in bin_edges]
        else:
            per_band_edges = None
            shared_edges = np.asarray(bin_edges, dtype=np.float64)
    elif np.issubdtype(image.dtype, np.floating):
        # Float dtype: derive per-band edges from each band's actual data range
        per_band_edges = []
        for b in range(num_bands):
            band_min = float(np.min(image[b]))
            band_max = float(np.max(image[b]))
            if band_min == band_max:
                band_max = band_min + 1.0
            per_band_edges.append(np.linspace(band_min, band_max, effective_bins + 1))
    else:
        # Integer dtype: shared edges from dtype range
        per_band_edges = None
        dtype_min, dtype_max = _dtype_range(image.dtype)
        shared_edges = np.linspace(dtype_min, dtype_max, effective_bins + 1)

    # Hoist random sampling above the band loop: all bands use the same indices
    rng = np.random.default_rng()
    if sampling == SamplingStrategy.RANDOM and sample_rate < 1.0:
        n_samples = max(1, int(num_pixels * sample_rate))
        indices = rng.integers(0, num_pixels, size=n_samples)
        effective_sample_rate = sample_rate
    else:
        indices = None
        effective_sample_rate = 1.0

    band_stats = []
    for b in range(num_bands):
        band_pixels = image[b].ravel()

        if indices is not None:
            band_pixels = band_pixels[indices]

        band_min = float(np.min(band_pixels))
        band_max = float(np.max(band_pixels))
        band_mean = float(np.mean(band_pixels))
        band_stddev = float(np.std(band_pixels))
        count = len(band_pixels)
        variance = float(np.var(band_pixels))
        m2 = variance * count

        edges_for_band = per_band_edges[b] if per_band_edges is not None else shared_edges
        hist_counts, hist_edges = np.histogram(band_pixels, bins=edges_for_band)

        band_stats.append(
            BandStatistics(
                min=band_min,
                max=band_max,
                mean=band_mean,
                stddev=band_stddev,
                count=count,
                m2=m2,
                histogram=hist_counts,
                bin_edges=hist_edges,
            )
        )

    return ImageStatistics(bands=band_stats, sample_rate=effective_sample_rate)


def merge_statistics(stats: Iterable[ImageStatistics]) -> ImageStatistics:
    """Combine multiple ImageStatistics into a single merged result.

    Merges per-block or per-tile statistics into aggregate statistics
    without re-scanning pixels. All inputs must have the same band count
    and identical bin_edges for corresponding bands.

    For performance, only the first two inputs are validated for band
    count and bin_edges consistency. Callers are responsible for ensuring
    all inputs were computed with the same parameters (e.g., same
    ``bin_edges`` argument to :func:`compute_statistics`).

    :param stats: An iterable of ImageStatistics to merge. Must not be
        empty, and all elements must have the same number of bands with
        identical bin_edges per band.
    :return: A single merged ImageStatistics.
    :raises ValueError: If the iterable is empty, band counts differ,
        or corresponding bands have different bin_edges (checked on
        first two inputs).
    """
    stats_list = list(stats)
    if len(stats_list) == 0:
        raise ValueError("Cannot merge an empty iterable of ImageStatistics")

    num_bands = len(stats_list[0].bands)

    # Validate only the first element against the reference to catch
    # programming errors without O(N × num_bins) overhead. In practice,
    # all inputs come from the same compute_statistics() call with
    # identical parameters, so checking the first is sufficient.
    if len(stats_list) > 1:
        s = stats_list[1]
        if len(s.bands) != num_bands:
            raise ValueError(f"Band count mismatch: input 0 has {num_bands} bands, but input 1 has {len(s.bands)} bands")
        for b in range(num_bands):
            if not np.array_equal(stats_list[0].bands[b].bin_edges, s.bands[b].bin_edges):
                raise ValueError(f"bin_edges mismatch for band {b}: input 0 and input 1 have different bin_edges")

    merged_bands = []
    for b in range(num_bands):
        bin_edges_b = stats_list[0].bands[b].bin_edges

        # Merged histogram: element-wise sum across inputs
        merged_hist = np.sum(
            np.array([s.bands[b].histogram for s in stats_list], dtype=np.float64),
            axis=0,
        )

        # Chan/Welford parallel variance merge
        merged_count = 0
        merged_mean = 0.0
        merged_m2 = 0.0

        for s in stats_list:
            band = s.bands[b]
            n_b = band.count
            if n_b == 0:
                continue
            n_ab = merged_count + n_b
            delta = band.mean - merged_mean
            merged_mean += delta * n_b / n_ab
            merged_m2 += band.m2 + delta**2 * merged_count * n_b / n_ab
            merged_count = n_ab

        merged_stddev = sqrt(merged_m2 / merged_count) if merged_count > 0 else 0.0

        # Min/max: exact from per-block values, filtered to non-zero-count blocks
        non_empty_bands = [s.bands[b] for s in stats_list if s.bands[b].count > 0]
        if non_empty_bands:
            merged_min = min(band.min for band in non_empty_bands)
            merged_max = max(band.max for band in non_empty_bands)
        else:
            merged_min = 0.0
            merged_max = 0.0

        merged_bands.append(
            BandStatistics(
                min=merged_min,
                max=merged_max,
                mean=merged_mean,
                stddev=merged_stddev,
                count=merged_count,
                m2=merged_m2,
                histogram=merged_hist,
                bin_edges=bin_edges_b.copy(),
            )
        )

    # Merged sample_rate: sum of input sample_rates, capped at 1.0
    merged_sample_rate = min(sum(s.sample_rate for s in stats_list), 1.0)

    return ImageStatistics(bands=merged_bands, sample_rate=merged_sample_rate)


def statistics_from_gdal_metadata(metadata: dict) -> Optional[ImageStatistics]:
    """Parse pre-computed statistics from a GDAL_METADATA XML blob.

    Extracts per-band statistics (min, max, mean, stddev) and optional
    histogram data from the GDAL_METADATA XML string stored in TIFF
    tag 42112. Returns None on any parse failure without raising.

    :param metadata: A metadata dict that may contain a ``"GDAL_METADATA"``
        key with an XML string value.
    :return: An ImageStatistics instance if parsing succeeds, or None if
        the key is missing, the XML is malformed, required entries are
        absent, or band counts are inconsistent.
    """
    xml_str = metadata.get("GDAL_METADATA")
    if xml_str is None:
        return None

    try:
        root = SafeET.fromstring(xml_str)
    except Exception:
        # Catches ET.ParseError for malformed XML and defusedxml
        # exceptions (DTDForbidden, EntitiesForbidden, etc.) for
        # malicious XML payloads.
        return None

    # Collect items grouped by sample (band index)
    # Each item has name="STATISTICS_*" sample="N" and text content
    band_items: Dict[int, Dict[str, str]] = {}
    for item in root.findall("Item"):
        name = item.get("name", "")
        sample_str = item.get("sample")
        if sample_str is None or not name.startswith("STATISTICS_"):
            continue
        try:
            sample = int(sample_str)
        except ValueError:
            continue
        if sample not in band_items:
            band_items[sample] = {}
        band_items[sample][name] = (item.text or "").strip()

    if not band_items:
        return None

    # Ensure contiguous band indices starting from 0
    num_bands = max(band_items.keys()) + 1
    if set(band_items.keys()) != set(range(num_bands)):
        return None

    required_keys = {"STATISTICS_MINIMUM", "STATISTICS_MAXIMUM", "STATISTICS_MEAN", "STATISTICS_STDDEV"}
    histogram_keys = {"STATISTICS_HISTOMIN", "STATISTICS_HISTOMAX", "STATISTICS_HISTONUMBINS", "STATISTICS_HISTOBINVALUES"}

    bands: List[BandStatistics] = []
    for sample in range(num_bands):
        items = band_items[sample]

        # Check required stat entries
        if not required_keys.issubset(items.keys()):
            return None

        try:
            band_min = float(items["STATISTICS_MINIMUM"])
            band_max = float(items["STATISTICS_MAXIMUM"])
            band_mean = float(items["STATISTICS_MEAN"])
            band_stddev = float(items["STATISTICS_STDDEV"])
        except (ValueError, TypeError):
            return None

        # Parse histogram if all histogram entries are present
        if histogram_keys.issubset(items.keys()):
            try:
                histomin = float(items["STATISTICS_HISTOMIN"])
                histomax = float(items["STATISTICS_HISTOMAX"])
                numbins = int(items["STATISTICS_HISTONUMBINS"])
                bin_values_str = items["STATISTICS_HISTOBINVALUES"]

                # Parse pipe-delimited histogram counts
                counts = [int(v) for v in bin_values_str.split("|") if v.strip()]
                if len(counts) != numbins:
                    return None

                histogram = np.array(counts, dtype=np.float64)
                bin_edges = np.linspace(histomin, histomax, numbins + 1)
            except (ValueError, TypeError):
                return None
        else:
            # Create a minimal single-bin histogram with count 0
            histogram = np.array([0], dtype=np.float64)
            bin_edges = np.array([band_min, band_max if band_max > band_min else band_min + 1.0])

        count = int(histogram.sum())
        m2 = band_stddev**2 * count

        bands.append(
            BandStatistics(
                min=band_min,
                max=band_max,
                mean=band_mean,
                stddev=band_stddev,
                count=count,
                m2=m2,
                histogram=histogram,
                bin_edges=bin_edges,
            )
        )

    return ImageStatistics(bands=bands)


def statistics_to_gdal_metadata(stats: ImageStatistics) -> str:
    """Serialize an ImageStatistics instance to a GDAL_METADATA XML string.

    Produces an XML string compatible with TIFF tag 42112 containing
    per-band STATISTICS_MINIMUM, STATISTICS_MAXIMUM, STATISTICS_MEAN,
    and STATISTICS_STDDEV entries. When histogram data is present,
    also includes STATISTICS_HISTOMIN, STATISTICS_HISTOMAX,
    STATISTICS_HISTONUMBINS, and STATISTICS_HISTOBINVALUES entries.

    :param stats: The ImageStatistics instance to serialize.
    :return: A GDAL_METADATA XML string.
    """
    root = ET.Element("GDALMetadata")

    for sample, band in enumerate(stats.bands):
        sample_str = str(sample)

        # Required stat entries (locale-safe formatting with full double precision)
        for stat_name, stat_value in [
            ("STATISTICS_MINIMUM", band.min),
            ("STATISTICS_MAXIMUM", band.max),
            ("STATISTICS_MEAN", band.mean),
            ("STATISTICS_STDDEV", band.stddev),
        ]:
            item = ET.SubElement(root, "Item")
            item.set("name", stat_name)
            item.set("sample", sample_str)
            item.text = format(stat_value, ".17g")

        # Histogram entries when histogram data is present
        if len(band.histogram) > 0:
            histomin = float(band.bin_edges[0])
            histomax = float(band.bin_edges[-1])
            numbins = len(band.histogram)
            bin_values = "|".join(str(int(c)) for c in band.histogram)

            for stat_name, stat_value in [
                ("STATISTICS_HISTOMIN", format(histomin, ".17g")),
                ("STATISTICS_HISTOMAX", format(histomax, ".17g")),
                ("STATISTICS_HISTONUMBINS", str(numbins)),
                ("STATISTICS_HISTOBINVALUES", bin_values),
            ]:
                item = ET.SubElement(root, "Item")
                item.set("name", stat_name)
                item.set("sample", sample_str)
                item.text = stat_value

    return ET.tostring(root, encoding="unicode")


def compute_image_statistics(
    image_asset,
    num_bins: int = 0,
    bin_edges: Optional[Union[NDArray, List[NDArray]]] = None,
    sampling: SamplingStrategy = SamplingStrategy.ALL,
    sample_rate: float = 1.0,
    compute_if_missing: bool = True,
    force_recompute: bool = False,
    num_workers: int = 0,
) -> Optional[ImageStatistics]:
    """Compute statistics for an ImageAssetProvider.

    Provider-level convenience that orchestrates metadata lookup, block
    reads, and merging. When possible, returns pre-computed statistics
    from GDAL metadata without reading any pixel data.

    When ``sampling=SamplingStrategy.BLOCK`` and ``sample_rate < 1.0``,
    blocks are included or excluded stochastically. Included blocks use
    ``SamplingStrategy.ALL`` internally for full pixel coverage within
    each block. Results computed with block sampling are not guaranteed
    to have merge-compatible ``bin_edges`` across separate calls because
    the first included block (which pins per-band edges) is stochastic.
    Do not pass block-sampled results to :func:`merge_statistics` with
    results from other calls.

    :param image_asset: A duck-typed ImageAssetProvider with
        ``metadata``, ``block_grid_size``, ``has_block()``, and
        ``get_block()`` methods. When ``num_workers >= 2``, the
        provider's ``get_block()`` must be safe for concurrent calls
        from multiple threads. Rust-based osml-imagery-io providers
        satisfy this. Python providers with internal mutable state
        (e.g. DownsampledImageProvider's LRU cache) must not be used with
        threading unless externally synchronized.
    :param num_bins: Number of histogram bins when bin_edges is not
        provided. When 0 (the default), the bin count is derived from
        the first block's dtype: one bin per possible value for integer
        types (capped at 65536), or 256 for floating-point types.
    :param bin_edges: Explicit bin edges for histogram computation.
        Accepts a single NDArray (applied to all bands) or a
        List[NDArray] for per-band edges. When None, edges are derived
        from the array's dtype range.
    :param sampling: Pixel sampling strategy. Default SamplingStrategy.ALL.
        When SamplingStrategy.BLOCK, blocks are included stochastically
        at the given sample_rate.
    :param sample_rate: Fraction of pixels to sample when using
        SamplingStrategy.RANDOM, or fraction of blocks to include when
        using SamplingStrategy.BLOCK. Default 1.0.
    :param compute_if_missing: When True (default), compute statistics
        from pixel data if metadata is absent. When False, return None
        if metadata is absent.
    :param force_recompute: When True, skip the metadata check and
        always compute statistics from pixel data. Default False.
    :param num_workers: Number of threads for concurrent block
        processing. When < 2 (default 0), blocks are processed serially.
        When >= 2, a ThreadPoolExecutor is used with ``max_workers=num_workers``.
    :return: An ImageStatistics instance, or None if metadata is absent
        and ``compute_if_missing`` is False.
    """
    # Fast path: try metadata first (unless forced to recompute)
    if not force_recompute:
        try:
            metadata_dict = dict(image_asset.metadata)
            cached = statistics_from_gdal_metadata(metadata_dict)
            if cached is not None:
                return cached
        except Exception:
            logger.debug("Failed to read metadata from provider, falling back to pixel scan")

    # If metadata was absent and caller doesn't want computation, bail out
    if not force_recompute and not compute_if_missing:
        return None

    # Compute from blocks
    grid_rows, grid_cols = image_asset.block_grid_size

    # For block sampling, use RNG to stochastically include/exclude blocks
    rng = np.random.default_rng()
    block_sampling = sampling == SamplingStrategy.BLOCK and sample_rate < 1.0

    # Build list of block positions to process
    block_positions = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if not image_asset.has_block(row, col):
                continue
            block_positions.append((row, col))

    total_blocks = len(block_positions)

    # Apply block sampling to determine which blocks to include
    if block_sampling:
        included_positions = [(r, c) for r, c in block_positions if rng.random() < sample_rate]
    else:
        included_positions = block_positions

    blocks_processed = len(included_positions)

    if not included_positions:
        return None

    # Determine per-block compute parameters
    block_sampling_strategy = SamplingStrategy.ALL if block_sampling else sampling
    block_sample_rate = 1.0 if block_sampling else sample_rate

    # First block is always read serially to pin bin_edges
    first_row, first_col = included_positions[0]
    first_block = image_asset.get_block(first_row, first_col)
    first_stats = compute_statistics(
        first_block,
        num_bins=num_bins,
        bin_edges=bin_edges,
        sampling=block_sampling_strategy,
        sample_rate=block_sample_rate,
    )

    # Pin per-band bin_edges from the first block
    if bin_edges is None:
        pinned_edges: List[NDArray] = [band.bin_edges for band in first_stats.bands]
    elif isinstance(bin_edges, list):
        pinned_edges = bin_edges
    else:
        pinned_edges = [bin_edges] * len(first_stats.bands)

    remaining_positions = included_positions[1:]
    block_stats: List[ImageStatistics] = [first_stats]

    if not remaining_positions:
        merged = first_stats
    elif num_workers >= 2:
        # Threaded execution for remaining blocks
        def _process_block(row_col):
            row, col = row_col
            block = image_asset.get_block(row, col)
            return compute_statistics(
                block,
                num_bins=num_bins,
                bin_edges=pinned_edges,
                sampling=block_sampling_strategy,
                sample_rate=block_sample_rate,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_block, pos): pos for pos in remaining_positions}
            for future in as_completed(futures):
                block_stats.append(future.result())

        merged = merge_statistics(block_stats)
    else:
        # Serial execution for remaining blocks
        for row, col in remaining_positions:
            block = image_asset.get_block(row, col)
            stats = compute_statistics(
                block,
                num_bins=num_bins,
                bin_edges=pinned_edges,
                sampling=block_sampling_strategy,
                sample_rate=block_sample_rate,
            )
            block_stats.append(stats)

        merged = merge_statistics(block_stats)

    # Override sample_rate with actual coverage fraction for block sampling
    if block_sampling and total_blocks > 0:
        merged.sample_rate = blocks_processed / total_blocks

    return merged
