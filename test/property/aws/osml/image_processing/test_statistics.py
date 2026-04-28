#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Property-based tests for the statistics module.

Tests correctness properties of BandStatistics, ImageStatistics,
compute_statistics(), and merge_statistics().
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aws.osml.image_processing.statistics import (
    BandStatistics,
    ImageStatistics,
    compute_statistics,
    merge_statistics,
    statistics_from_gdal_metadata,
    statistics_to_gdal_metadata,
)
from property.conftest import pbt_settings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def compatible_image_statistics_list(draw):
    """Generate a list of ImageStatistics with identical band counts and bin_edges.

    Each ImageStatistics in the list has the same number of bands and the same
    bin_edges per band, but different histogram counts, min, max, mean, stddev.
    """
    num_bands = draw(st.integers(min_value=1, max_value=4))
    num_stats = draw(st.integers(min_value=2, max_value=5))
    num_bins = 256
    bin_edges = np.linspace(0, 255, num_bins + 1)

    stats_list = []
    for _ in range(num_stats):
        bands = []
        for _ in range(num_bands):
            histogram = draw(
                hnp.arrays(
                    dtype=np.int64,
                    shape=(num_bins,),
                    elements=st.integers(min_value=0, max_value=1000),
                )
            )
            band_min = draw(st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False))
            band_max = draw(st.floats(min_value=float(band_min), max_value=255.0, allow_nan=False, allow_infinity=False))
            band_mean = draw(
                st.floats(min_value=float(band_min), max_value=float(band_max), allow_nan=False, allow_infinity=False)
            )
            band_stddev = draw(st.floats(min_value=0.0, max_value=128.0, allow_nan=False, allow_infinity=False))
            count = int(np.sum(histogram))
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
                    bin_edges=bin_edges.copy(),
                )
            )
        sample_rate = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
        stats_list.append(ImageStatistics(bands=bands, sample_rate=sample_rate))

    return stats_list


# ---------------------------------------------------------------------------
# Property: Chan merge correctness (stddev matches full-array stddev)
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    num_bands=st.integers(min_value=1, max_value=3),
    height=st.integers(min_value=4, max_value=32),
    width=st.integers(min_value=4, max_value=32),
    num_partitions=st.integers(min_value=2, max_value=6),
    data=st.data(),
)
@settings(pbt_settings)
def test_chan_merge_stddev_matches_full_array(num_bands, height, width, num_partitions, data):
    """For any CHW array partitioned into non-overlapping blocks, computing
    statistics per block and merging via merge_statistics() SHALL produce
    a merged stddev that matches np.std(full_array) within 1e-10 for each band.
    """
    image = data.draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=(num_bands, height, width),
        )
    )

    bin_edges = np.linspace(0, 255, 257)
    strips = [s for s in np.array_split(image, num_partitions, axis=1) if s.shape[1] > 0]
    per_block_stats = [compute_statistics(strip, bin_edges=bin_edges) for strip in strips]

    merged = merge_statistics(per_block_stats)

    for b in range(num_bands):
        expected_stddev = float(np.std(image[b]))
        assert merged.bands[b].stddev == pytest.approx(expected_stddev, abs=1e-10), (
            f"Band {b}: merged stddev {merged.bands[b].stddev} != expected {expected_stddev}"
        )
        expected_mean = float(np.mean(image[b]))
        assert merged.bands[b].mean == pytest.approx(expected_mean, abs=1e-10), (
            f"Band {b}: merged mean {merged.bands[b].mean} != expected {expected_mean}"
        )


# ---------------------------------------------------------------------------
# Property 5: Merge histogram is element-wise sum
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 5: Merge histogram is element-wise sum
@pytest.mark.property
@given(stats_list=compatible_image_statistics_list())
@settings(pbt_settings)
def test_merge_histogram_is_element_wise_sum(stats_list):
    """**Validates: Requirements 3.2**

    For any collection of ImageStatistics with identical band counts and
    identical bin_edges, the merged histogram produced by merge_statistics()
    SHALL equal the element-wise sum of the input histograms for each band.
    """
    merged = merge_statistics(stats_list)

    num_bands = len(stats_list[0].bands)
    assert len(merged.bands) == num_bands

    for b in range(num_bands):
        expected_histogram = np.sum(
            np.array([s.bands[b].histogram for s in stats_list], dtype=np.float64),
            axis=0,
        )
        np.testing.assert_array_equal(
            merged.bands[b].histogram,
            expected_histogram,
            err_msg=f"Merged histogram for band {b} does not equal element-wise sum of inputs",
        )


# ---------------------------------------------------------------------------
# Property 6: Merge count preservation across partitions
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 6: Merge count preservation across partitions
@pytest.mark.property
@given(
    num_bands=st.integers(min_value=1, max_value=3),
    height=st.integers(min_value=4, max_value=32),
    width=st.integers(min_value=4, max_value=32),
    num_partitions=st.integers(min_value=2, max_value=4),
    data=st.data(),
)
@settings(pbt_settings)
def test_merge_count_preservation_across_partitions(num_bands, height, width, num_partitions, data):
    """**Validates: Requirements 3.9**

    For any CHW array partitioned into non-overlapping blocks, computing
    statistics per block with identical bin_edges and then merging via
    merge_statistics() SHALL produce a merged histogram whose total count
    equals the total pixel count of the original array.
    """
    # Generate a random CHW uint8 array
    image = data.draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=(num_bands, height, width),
        )
    )

    # Use fixed bin_edges for all partitions (uint8 range, 256 bins)
    bin_edges = np.linspace(0, 255, 257)

    # Partition the image into non-overlapping horizontal strips along the height dimension.
    # Compute split points that divide the height into num_partitions parts.
    # np.array_split handles uneven divisions gracefully.
    strips = np.array_split(image, num_partitions, axis=1)

    # Compute statistics on each strip with the same bin_edges
    per_block_stats = [compute_statistics(strip, bin_edges=bin_edges) for strip in strips]

    # Merge the per-block statistics
    merged = merge_statistics(per_block_stats)

    # Assert that for each band, merged histogram total count equals total pixel count
    total_pixel_count = height * width
    for b in range(num_bands):
        merged_count = int(merged.bands[b].histogram.sum())
        assert merged_count == total_pixel_count, (
            f"Band {b}: merged histogram count {merged_count} != total pixel count {total_pixel_count}"
        )
        # Also verify the count field
        assert merged.bands[b].count == total_pixel_count, (
            f"Band {b}: merged count field {merged.bands[b].count} != total pixel count {total_pixel_count}"
        )


# ---------------------------------------------------------------------------
# Strategy for Property 7
# ---------------------------------------------------------------------------


@st.composite
def valid_image_statistics_with_histogram(draw):
    """Generate a valid ImageStatistics instance with histogram data for each band.

    Produces 1-3 bands, each with:
    - 256 bins of non-negative integer histogram counts
    - bin_edges from np.linspace(histomin, histomax, 257) where histomin < histomax
    - Finite min, max, mean, stddev with min <= mean <= max and min <= max
    """
    num_bands = draw(st.integers(min_value=1, max_value=3))

    bands = []
    for _ in range(num_bands):
        # Generate histogram: 256 bins with non-negative integer counts
        histogram = draw(
            hnp.arrays(
                dtype=np.int64,
                shape=(256,),
                elements=st.integers(min_value=0, max_value=500),
            )
        )

        # Generate bin edge boundaries where histomin < histomax
        histomin = draw(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False))
        histomax = draw(
            st.floats(min_value=histomin + 0.01, max_value=histomin + 2e4, allow_nan=False, allow_infinity=False)
        )
        bin_edges = np.linspace(histomin, histomax, 257)

        # Generate valid min, max, mean, stddev (finite, min <= max, min <= mean <= max)
        band_min = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
        band_max = draw(
            st.floats(min_value=float(band_min), max_value=float(band_min) + 2e6, allow_nan=False, allow_infinity=False)
        )
        band_mean = draw(
            st.floats(min_value=float(band_min), max_value=float(band_max), allow_nan=False, allow_infinity=False)
        )
        band_stddev = draw(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
        count = int(np.sum(histogram))
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

    sample_rate = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    return ImageStatistics(bands=bands, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Property 7: GDAL metadata serialization round-trip
# ---------------------------------------------------------------------------


# Feature: image-processing-foundations, Property 7: GDAL metadata serialization round-trip
@pytest.mark.property
@given(stats=valid_image_statistics_with_histogram())
@settings(pbt_settings)
def test_gdal_metadata_serialization_round_trip(stats):
    """**Validates: Requirements 5.4**

    For any valid ImageStatistics instance, serializing via
    statistics_to_gdal_metadata() and then parsing via
    statistics_from_gdal_metadata() SHALL produce an ImageStatistics
    with equivalent min, max, mean, and stddev values (within
    floating-point tolerance) for each band.
    """
    # Step 1: Serialize
    xml_str = statistics_to_gdal_metadata(stats)

    # Step 2: Parse back
    parsed = statistics_from_gdal_metadata({"GDAL_METADATA": xml_str})

    # Step 3: Assert parsed result is not None
    assert parsed is not None, "Round-trip parsing returned None"

    # Step 4: Assert same number of bands
    assert len(parsed.bands) == len(stats.bands), (
        f"Band count mismatch: original {len(stats.bands)}, parsed {len(parsed.bands)}"
    )

    # Step 5: Assert each band's min, max, mean, stddev are approximately equal
    for b in range(len(stats.bands)):
        orig = stats.bands[b]
        rt = parsed.bands[b]

        assert rt.min == pytest.approx(orig.min, abs=1e-6), f"Band {b} min mismatch: original {orig.min}, parsed {rt.min}"
        assert rt.max == pytest.approx(orig.max, abs=1e-6), f"Band {b} max mismatch: original {orig.max}, parsed {rt.max}"
        assert rt.mean == pytest.approx(orig.mean, abs=1e-6), (
            f"Band {b} mean mismatch: original {orig.mean}, parsed {rt.mean}"
        )
        assert rt.stddev == pytest.approx(orig.stddev, abs=1e-6), (
            f"Band {b} stddev mismatch: original {orig.stddev}, parsed {rt.stddev}"
        )

        # Step 6: Assert histogram counts are preserved
        # The serializer converts counts to int, so compare after int conversion
        expected_counts = np.array([int(c) for c in orig.histogram], dtype=np.float64)
        np.testing.assert_array_equal(
            parsed.bands[b].histogram,
            expected_counts,
            err_msg=f"Band {b} histogram counts not preserved after round-trip",
        )

        # Step 7: Assert count and m2 are correctly derived
        expected_count = int(expected_counts.sum())
        assert parsed.bands[b].count == expected_count, (
            f"Band {b} count mismatch: expected {expected_count}, got {parsed.bands[b].count}"
        )

        # Step 8: Assert m2 is derived from stddev^2 * count
        expected_m2 = rt.stddev**2 * expected_count
        assert rt.m2 == pytest.approx(expected_m2, abs=1e-6), f"Band {b} m2 mismatch: expected {expected_m2}, got {rt.m2}"
