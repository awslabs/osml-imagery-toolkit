#  Copyright 2024 Amazon.com, Inc. or its affiliates.

"""Build a COG or NITF R-Set pyramid from a full-resolution source image.

Given a full-resolution image (TIFF or NITF), this script opens it via
``aws.osml.io``, constructs a :class:`PyramidBuilder`, and writes the
result as either a Cloud Optimized GeoTIFF (single file with internal
overviews) or a multi-file NITF R-Set (base ``.ntf`` plus ``.r1``,
``.r2``, ... sidecars per overview level).

The pyramid is generated using a single-pass incremental accumulation
algorithm: R0 tiles are read exactly once in raster order, and each
decoded tile's contribution propagates up through every pyramid level
immediately. The default resampler is the SIPS-compliant RRDS algorithm
(NGA.STND.0014 v2.4 Section 2.2), but faster OpenCV-based resamplers
are available for non-compliance workflows.

Example usage::

    # Write a COG with default SIPS resampling
    python examples/build_pyramid.py test/data/small.tif --output output.tif --format cog

    # Write a NITF R-Set (produces input.ntf.r1, input.ntf.r2, ...)
    python examples/build_pyramid.py input.ntf --format rset

    # Use a faster OpenCV resampler instead of SIPS
    python examples/build_pyramid.py input.tif --resampler area

    # Auto-pick the format from the source's extension
    python examples/build_pyramid.py input.tif

    # Single-threaded mode (useful for debugging)
    python examples/build_pyramid.py input.tif --num-workers 0
"""

import argparse
import os
import sys
import time
from typing import Callable, List, Optional

from aws.osml.image_processing import (
    PyramidBuilder,
    area_resample,
    bilinear_resample,
    lanczos_resample,
    nearest_neighbor_resample,
    sips_rrds_resample,
)
from aws.osml.io import IO, BufferedMetadataProvider

# Map of CLI resampler names to their callables.
_RESAMPLERS = {
    "sips": sips_rrds_resample,
    "area": area_resample,
    "bilinear": bilinear_resample,
    "lanczos": lanczos_resample,
    "nearest": nearest_neighbor_resample,
}


def _infer_format(source_path: str) -> str:
    """Pick an output format from the source's extension.

    Returns ``"cog"`` for ``.tif``/``.tiff`` sources and ``"rset"`` for
    ``.ntf``/``.nitf`` sources. Raises :class:`SystemExit` if the
    extension isn't recognized — the caller should then pass
    ``--format`` explicitly.
    """
    ext = os.path.splitext(source_path)[1].lower()
    if ext in (".tif", ".tiff"):
        return "cog"
    if ext in (".ntf", ".nitf"):
        return "rset"
    sys.exit(
        f"Cannot infer output format from extension {ext!r}; "
        f"pass --format cog|rset explicitly."
    )


def _default_output_path(source_path: str, output_format: str) -> str:
    """Derive a default output path next to the source.

    For COG output the default is ``<source_stem>.tif``; for R-Set
    output the base path is the source itself (sidecars ``.r1``,
    ``.r2``, ... are written alongside).
    """
    if output_format == "rset":
        return source_path
    stem, _ = os.path.splitext(source_path)
    return f"{stem}.tif"


def _nitf_image_metadata_fn(level_index: int) -> BufferedMetadataProvider:
    """Per-level NITF image subheader metadata.

    The NITF writer needs at minimum a compression code (``IC``) and
    an interleave mode (``IMODE``) on every image subheader. We use
    uncompressed (``IC=NC``) with band-interleaved-by-block
    (``IMODE=B``) — the simplest combination that doesn't depend on
    host codec availability.

    Args:
        level_index: The pyramid level index (0 = base, 1+ = overviews).

    Returns:
        A :class:`BufferedMetadataProvider` with ``IC`` and ``IMODE`` set.
    """
    md = BufferedMetadataProvider()
    md.set("IC", "NC")
    md.set("IMODE", "B")
    return md


def _print_level_plan(builder: PyramidBuilder) -> None:
    """Print a table summarizing the planned pyramid levels."""
    levels = builder._levels
    print(f"\nPyramid plan: {len(levels)} level(s) "
          f"(1 base + {len(levels) - 1} overview(s))")
    print(f"  {'Level':<8} {'Dimensions':<24} {'Tile Grid':<16} {'Tiles':<8}")
    print(f"  {'-----':<8} {'----------':<24} {'---------':<16} {'-----':<8}")
    for lvl in levels:
        dims = f"{lvl.num_rows} x {lvl.num_columns}"
        grid_rows, grid_cols = lvl.block_grid_size
        grid = f"{grid_rows} x {grid_cols}"
        total_tiles = grid_rows * grid_cols
        label = "R0 (source)" if lvl.level_index == 0 else f"R{lvl.level_index}"
        print(f"  {label:<8} {dims:<24} {grid:<16} {total_tiles:<8}")
    print()


def _write_cog(
    source,
    output_path: str,
    min_size: int,
    num_workers: int,
    resample_func: Optional[Callable] = None,
) -> List[str]:
    """Write a Cloud Optimized GeoTIFF and return the list of files produced.

    Args:
        source: An ``ImageAssetProvider`` for the full-resolution image.
        output_path: Destination file path for the COG.
        min_size: The pyramid includes levels until either dimension
            drops below this value; that level is included.
        num_workers: Background threads for prefetch + writeback.
        resample_func: Resampling function to use. ``None`` defaults to
            SIPS RRDS.

    Returns:
        A list containing the single output file path.
    """
    builder = PyramidBuilder(
        source,
        min_size=min_size,
        num_workers=num_workers,
        resample_func=resample_func,
    )
    _print_level_plan(builder)

    print("Building pyramid and writing COG...")
    t_start = time.perf_counter()

    with IO.open(output_path, "w", "geotiff") as writer:
        builder.build_and_write(writer, base_key="image:0")

    t_elapsed = time.perf_counter() - t_start
    print(f"  Build + write completed in {t_elapsed:.2f}s")

    return [output_path]


def _write_rset(
    source,
    output_path: str,
    min_size: int,
    num_workers: int,
    resample_func: Optional[Callable] = None,
) -> List[str]:
    """Write a multi-file NITF R-Set and return the list of files produced.

    The R-Set convention uses one path per overview level: ``.r1``,
    ``.r2``, ... sidecars alongside the original source file. The
    source file itself is R0 and is not copied — only the overview
    sidecars are written. The writer is opened with the full path list
    (base + sidecars) because the format requires it, but the base
    asset is the original source passed through unchanged.

    Args:
        source: An ``ImageAssetProvider`` for the full-resolution image.
        output_path: Base file path for the R-Set. Sidecars are derived
            by appending ``.r1``, ``.r2``, etc. to this path.
        min_size: The pyramid includes levels until either dimension
            drops below this value; that level is included.
        num_workers: Background threads for prefetch + writeback.
        resample_func: Resampling function to use. ``None`` defaults to
            SIPS RRDS.

    Returns:
        A list of the sidecar file paths produced (excludes the base).
    """
    builder = PyramidBuilder(
        source,
        min_size=min_size,
        num_workers=num_workers,
        resample_func=resample_func,
    )
    _print_level_plan(builder)

    num_levels = len(builder._levels)
    # The writer needs the full path list (base + sidecars) to set up
    # the multi-file structure, but we only care about the sidecars as
    # output — the base file is the original source.
    all_paths = [output_path] + [
        f"{output_path}.r{i}" for i in range(1, num_levels)
    ]
    sidecar_paths = all_paths[1:]

    print(
        f"Building pyramid and writing {len(sidecar_paths)} "
        f"NITF R-Set sidecar(s)..."
    )
    t_start = time.perf_counter()

    with IO.open(all_paths, "w", "nitf") as writer:
        builder.build_and_write(
            writer,
            base_key="image:0",
            image_metadata_fn=_nitf_image_metadata_fn,
        )

    t_elapsed = time.perf_counter() - t_start
    print(f"  Build + write completed in {t_elapsed:.2f}s")

    return sidecar_paths


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 ** 3):.2f} GiB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 ** 2):.1f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes} B"


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a COG or NITF R-Set pyramid from a full-resolution image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Resamplers:\n"
            "  sips      SIPS RRDS (NGA.STND.0014 v2.4 §2.2) — default, highest quality\n"
            "  area      OpenCV area-based downsampling — good quality, fast\n"
            "  bilinear  OpenCV bilinear interpolation\n"
            "  lanczos   OpenCV Lanczos (8x8 neighborhood) — high quality, slower\n"
            "  nearest   OpenCV nearest-neighbor — fastest, lowest quality\n"
        ),
    )
    parser.add_argument(
        "source",
        help="Path to the full-resolution input image (TIFF or NITF).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output file path. For COG, defaults to '<source_stem>.tif'. "
            "For R-Set, defaults to the source path (sidecars .r1, .r2, ... "
            "are written alongside the source)."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("cog", "rset"),
        default=None,
        help="Output format. When omitted, inferred from the source extension.",
    )
    parser.add_argument(
        "-r",
        "--resampler",
        choices=sorted(_RESAMPLERS.keys()),
        default="sips",
        help="Resampling algorithm (default: sips).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help=(
            "Target size for the smallest overview level. The pyramid "
            "keeps generating levels until either dimension drops "
            "below this value; that level is included as the final "
            "overview. SIPS specifies that the last level's shortest "
            "dimension should be in [256, 512) (default: 512)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Background threads for prefetch + writeback; 0 runs single-threaded (default: 2).",
    )
    args = parser.parse_args(argv)

    # -- Validate inputs --------------------------------------------------
    if not os.path.isfile(args.source):
        sys.exit(f"Source file not found: {args.source}")

    output_format = args.format or _infer_format(args.source)
    output_path = args.output or _default_output_path(args.source, output_format)
    resample_func = _RESAMPLERS[args.resampler]

    # -- Print configuration ----------------------------------------------
    source_size = os.path.getsize(args.source)
    print(f"Source:     {args.source} ({_format_size(source_size)})")
    print(f"Output:     {output_path} ({output_format})")
    print(f"Resampler:  {args.resampler}")
    print(f"Workers:    {args.num_workers}")

    # -- Open source and build pyramid ------------------------------------
    total_start = time.perf_counter()

    with IO.open(args.source, "r") as reader:
        source = reader.get_asset("image:0")
        pixel_count = source.num_rows * source.num_columns
        print(
            f"Image:      {source.num_rows} x {source.num_columns} x "
            f"{source.num_bands} band(s), dtype={source.pixel_value_type}"
        )
        if source.num_resolution_levels > 1:
            print(f"Native levels: {source.num_resolution_levels} "
                  f"(builder will use native reads where possible)")

        if output_format == "cog":
            produced = _write_cog(
                source, output_path, args.min_size,
                args.num_workers, resample_func,
            )
        else:
            produced = _write_rset(
                source, output_path, args.min_size,
                args.num_workers, resample_func,
            )

    total_elapsed = time.perf_counter() - total_start

    # -- Summary ----------------------------------------------------------
    total_output_bytes = sum(os.path.getsize(p) for p in produced)
    megapixels = pixel_count / 1_000_000

    print(f"\nCompleted in {total_elapsed:.2f}s "
          f"({megapixels / total_elapsed:.1f} MP/s)")
    print(f"Output: {len(produced)} file(s), {_format_size(total_output_bytes)} total")
    for path in produced:
        size = os.path.getsize(path)
        print(f"  {path}  ({_format_size(size)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
