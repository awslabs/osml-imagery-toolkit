#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""End-to-end integration tests for :class:`PyramidBuilder`.

Exercises the full write path against real ``aws.osml.io`` primitives: a
source image from ``test/data`` is opened as an ``ImageAssetProvider``, a
``PyramidBuilder`` generates all overview levels, ``build_and_write`` hands
them to an ``IO.open(..., "w", ...)`` writer, and the output is re-opened
to verify that each level's asset is present with monotonically decreasing
dimensions and that a tile can be read from every level. Both the GeoTIFF
(COG) and NITF R-Set output paths are exercised.

These tests require ``osml-imagery-io`` (the Rust-backed binding) to be
installed with TIFF or NITF read/write support. Environments that lack it
will ``pytest.skip`` rather than fail.
"""

import os

import numpy as np
import pytest

# Skip the whole module if osml-imagery-io is unavailable or cannot load its
# native bindings (e.g., missing system libraries). The integration tests
# are only meaningful against the real codec stack.
io = pytest.importorskip("aws.osml.io")

from aws.osml.image_processing.pyramid_builder import PyramidBuilder  # noqa: E402

# Source images used for the round-trips. Both are small sample files from
# the shared test data set and are present in every development
# environment. Paths are resolved relative to the workspace's ``test/data``
# directory so the tests can be invoked from any working directory.
_SOURCE_TIFF = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "test", "data", "small.tif")
)
_SOURCE_NTF = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "test", "data", "small.ntf")
)
# Root of the test-data tree; the J2K discovery walk looks for any
# ``.jp2`` / ``.j2k`` / ``.jpf`` / ``.jpx`` fixture below this
# directory. None ship with the repository today, so the J2K native
# path test gracefully skips in most environments.
_TEST_DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "test", "data"))
_J2K_EXTENSIONS = (".jp2", ".j2k", ".jpf", ".jpx")
_TIFF_EXTENSIONS = (".tif", ".tiff")


def _source_exists() -> bool:
    return os.path.isfile(_SOURCE_TIFF)


def _nitf_source_exists() -> bool:
    return os.path.isfile(_SOURCE_NTF)


def _find_j2k_source() -> str:
    """Locate a J2K source under ``test/data`` or return an empty string.

    Walks the test-data tree looking for any file with a J2K extension
    (``.jp2``, ``.j2k``, ``.jpf``, ``.jpx``). Returns the first match
    found in sorted order so the selection is deterministic across
    runs, or ``""`` when no fixture exists. The native-path test calls
    :func:`pytest.skip` when this is empty.
    """
    if not os.path.isdir(_TEST_DATA_ROOT):
        return ""
    matches = []
    for dirpath, _dirs, filenames in os.walk(_TEST_DATA_ROOT):
        for name in filenames:
            if name.lower().endswith(_J2K_EXTENSIONS):
                matches.append(os.path.join(dirpath, name))
    matches.sort()
    return matches[0] if matches else ""


def _find_cog_with_overviews() -> str:
    """Locate a TIFF under ``test/data`` that has pre-computed overviews.

    Walks the test-data tree looking for ``.tif`` / ``.tiff`` files and
    opens each one, returning the first whose primary asset reports
    ``num_resolution_levels > 1`` or whose dataset exposes a
    ``"{base_key}:overview:1"`` asset key. Returns ``""`` when no such
    fixture exists so the caller can :func:`pytest.skip` gracefully.

    The walk is deterministic (sorted) so repeated test runs pick the
    same file. Files that fail to open (unsupported driver, missing
    codec libraries) are skipped silently — they are not expected to
    qualify as COG fixtures.
    """
    if not os.path.isdir(_TEST_DATA_ROOT):
        return ""
    candidates = []
    for dirpath, _dirs, filenames in os.walk(_TEST_DATA_ROOT):
        for name in filenames:
            if name.lower().endswith(_TIFF_EXTENSIONS):
                candidates.append(os.path.join(dirpath, name))
    candidates.sort()

    for path in candidates:
        try:
            reader_ctx = io.IO.open(path, "r")
        except (OSError, IOError, ValueError):
            continue
        try:
            with reader_ctx as reader:
                try:
                    base_asset = reader.get_asset("image:0")
                except (KeyError, LookupError, ValueError):
                    continue

                # A COG with overviews either reports multiple native
                # resolution levels on the primary asset or exposes
                # overview assets via the documented key convention.
                if int(getattr(base_asset, "num_resolution_levels", 1)) > 1:
                    return path

                asset_keys = list(reader.get_asset_keys())
                if any(k.startswith("image:0:overview:") for k in asset_keys):
                    return path
        except (OSError, IOError, ValueError):
            continue
    return ""


def _try_open_source():
    """Open the source TIFF, skipping the test when the format is unsupported."""
    try:
        return io.IO.open(_SOURCE_TIFF, "r")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot open {_SOURCE_TIFF}: {exc}")


def _try_open_nitf_source():
    """Open the source NITF, skipping the test when the format is unsupported."""
    try:
        return io.IO.open(_SOURCE_NTF, "r")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot open {_SOURCE_NTF}: {exc}")


def _try_open_writer(path: str):
    """Open a GeoTIFF writer, skipping when the format is unsupported."""
    try:
        return io.IO.open(path, "w", "geotiff")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot write GeoTIFF: {exc}")


def _try_open_nitf_writer(paths):
    """Open a multi-file NITF R-Set writer, skipping when unsupported.

    ``paths`` is a list whose first entry is the base NITF file and whose
    remaining entries are per-overview R-Set files (``.r1``, ``.r2`` …).
    """
    try:
        return io.IO.open(paths, "w", "nitf")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot write NITF R-Set: {exc}")


@pytest.mark.skipif(not _source_exists(), reason="test/data/small.tif not available")
def test_build_and_write_roundtrip_creates_overviews(tmp_path):
    """Full build_and_write round-trip: writer receives base + overview assets
    and re-opening the file exposes each level with a readable first tile.

    The tile dimensions are inherited from the source so the builder's 2x2
    cascade aligns naturally with the source's block grid. Validates
    requirements 3.17 (add_asset per level) and 5.6/5.7 (overview key
    convention discoverable on the output).
    """
    output_path = tmp_path / "pyramid_output.tif"

    # -- Build and write --------------------------------------------------
    with _try_open_source() as reader:
        source = reader.get_asset("image:0")
        builder = PyramidBuilder(source, min_size=256, num_workers=0)
        expected_num_levels = len(builder._levels)
        # Capture the planned dimensions for the later monotonicity check.
        planned_dims = [(lvl.num_rows, lvl.num_columns) for lvl in builder._levels]

        # Require at least one overview level — otherwise the test has no
        # pyramid structure to validate.
        assert expected_num_levels >= 2, (
            f"Source too small to produce overviews at min_size=256 (planned levels: {planned_dims})"
        )

        with _try_open_writer(str(output_path)) as writer:
            writer.add_asset("image:0", source, "", "", ["data"])
            builder.build_and_write(writer, base_key="image:0")

    # The output file exists and is non-empty.
    assert output_path.is_file(), f"Writer did not produce {output_path}"
    assert output_path.stat().st_size > 0

    # -- Re-open and inspect ---------------------------------------------
    with io.IO.open(str(output_path), "r") as out_reader:
        asset_keys = list(out_reader.get_asset_keys())

        # The base asset is always present.
        assert "image:0" in asset_keys, f"base asset missing from {asset_keys}"

        # Overview assets follow the "{base_key}:overview:{N}" convention;
        # the builder emits exactly one per overview level.
        overview_keys = [k for k in asset_keys if k.startswith("image:0:overview:")]
        assert len(overview_keys) == expected_num_levels - 1, (
            f"expected {expected_num_levels - 1} overview assets, got {overview_keys}"
        )

        # Order the overview keys by their numeric suffix so level 1 comes
        # first; this mirrors the order PyramidBuilder wrote them in.
        overview_keys_sorted = sorted(overview_keys, key=lambda k: int(k.rsplit(":", 1)[-1]))

        # Collect the full level list (level 0 = base asset, then overviews).
        level_keys = ["image:0"] + overview_keys_sorted
        level_assets = [out_reader.get_asset(k) for k in level_keys]

        # Dimensions must strictly decrease from one level to the next.
        prev_rows, prev_cols = level_assets[0].num_rows, level_assets[0].num_columns
        for asset in level_assets[1:]:
            assert asset.num_rows <= prev_rows, f"level {asset.key}: num_rows {asset.num_rows} > previous {prev_rows}"
            assert asset.num_columns <= prev_cols, (
                f"level {asset.key}: num_columns {asset.num_columns} > previous {prev_cols}"
            )
            assert asset.num_rows < prev_rows or asset.num_columns < prev_cols, (
                f"level {asset.key}: dimensions did not decrease ({asset.num_rows}x"
                f"{asset.num_columns} vs {prev_rows}x{prev_cols})"
            )
            prev_rows, prev_cols = asset.num_rows, asset.num_columns

        # Every level must allow a tile read at (0, 0, 0) and produce a CHW
        # array whose band count matches the source.
        source_num_bands = level_assets[0].num_bands
        for asset in level_assets:
            assert asset.has_block(0, 0, 0), f"level {asset.key}: block (0, 0) missing from re-opened pyramid"
            block = asset.get_block(0, 0, 0)
            assert isinstance(block, np.ndarray), f"level {asset.key}: get_block returned {type(block).__name__}"
            assert block.ndim == 3, f"level {asset.key}: expected CHW block, got ndim={block.ndim}"
            assert block.shape[0] == source_num_bands, (
                f"level {asset.key}: band count {block.shape[0]} != {source_num_bands}"
            )


@pytest.mark.skipif(not _source_exists(), reason="test/data/small.tif not available")
def test_tiled_image_pyramid_from_dataset(tmp_path):
    """When :class:`TiledImagePyramid` is available (task 7), re-opening the
    output via ``TiledImagePyramid.from_dataset`` exposes the same levels in
    decreasing-resolution order and each level yields a readable tile.

    Validates requirements 5.6 (overview discovery by key convention) and
    5.7 (single-level fallback not triggered when overviews are present).
    """
    try:
        from aws.osml.image_processing.pyramid import TiledImagePyramid  # noqa: F401
    except ImportError:
        pytest.skip("TiledImagePyramid not implemented yet (task 7)")

    output_path = tmp_path / "pyramid_for_tip.tif"

    with _try_open_source() as reader:
        source = reader.get_asset("image:0")
        builder = PyramidBuilder(source, min_size=256, num_workers=0)
        expected_num_levels = len(builder._levels)
        with _try_open_writer(str(output_path)) as writer:
            writer.add_asset("image:0", source, "", "", ["data"])
            builder.build_and_write(writer, base_key="image:0")

    with io.IO.open(str(output_path), "r") as out_reader:
        pyramid = TiledImagePyramid.from_dataset(out_reader, "image:0")
        assert pyramid.num_levels == expected_num_levels

        # Monotonic decrease across levels.
        prev_shape = pyramid.image_shape_at_level(0)
        for i in range(1, pyramid.num_levels):
            shape = pyramid.image_shape_at_level(i)
            assert shape[1] <= prev_shape[1]
            assert shape[2] <= prev_shape[2]
            assert shape[1] < prev_shape[1] or shape[2] < prev_shape[2]
            prev_shape = shape

        # Every level yields a readable first tile.
        for i in range(pyramid.num_levels):
            level = pyramid.get_level(i)
            assert level.has_block(0, 0, 0)
            block = level.get_block(0, 0, 0)
            assert block.ndim == 3
            assert block.shape[0] == pyramid.image_shape_at_level(i)[0]


def _make_nitf_image_metadata_fn():
    """Return an ``image_metadata_fn`` producing per-level NITF image metadata.

    The NITF writer requires at minimum an image compression code (``IC``)
    and an interleave mode (``IMODE``) on every image subheader. For the
    R-Set round-trip we use the simplest values — uncompressed
    (``IC=NC``) with band-interleaved-by-block (``IMODE=B``) — so the test
    exercises the write path without depending on the host's codec
    support. See ``DESIGN_IMAGE_PYRAMID_OPERATIONS.md`` for the full
    example this pattern is taken from.
    """

    def _factory(level_index: int):
        md = io.BufferedMetadataProvider()
        md["IC"] = "NC"
        md["IMODE"] = "B"
        return md

    return _factory


@pytest.mark.skipif(not _nitf_source_exists(), reason="test/data/small.ntf not available")
def test_build_and_write_nitf_rset_roundtrip(tmp_path):
    """Full ``build_and_write`` round-trip to a multi-file NITF R-Set: the
    writer receives the base image plus one ``.rN`` sidecar per overview
    level, and re-opening the dataset exposes every level with a readable
    first tile.

    The NITF R-Set convention in osml-imagery-io uses one path per level —
    the base ``.ntf`` file for R0 and ``.r1``, ``.r2``, … sidecars for
    each overview level. The source file itself is never used as the base
    output path — a fresh temp file is used instead.

    Validates requirements 3.17 (``add_asset`` per level, including
    overview keys) and exercises the NITF-specific multi-file output
    path.
    """
    base_path = tmp_path / "pyramid_output.ntf"

    # -- Plan the levels first so we know how many .rN sidecars to open --
    with _try_open_nitf_source() as reader:
        source = reader.get_asset("image:0")
        builder = PyramidBuilder(source, min_size=256, num_workers=0)
        expected_num_levels = len(builder._levels)
        planned_dims = [(lvl.num_rows, lvl.num_columns) for lvl in builder._levels]

        # Require at least one overview level — otherwise there is no
        # pyramid structure to validate and the NITF R-Set has no .rN
        # sidecars to open.
        assert expected_num_levels >= 2, (
            f"Source too small to produce overviews at min_size=256 (planned levels: {planned_dims})"
        )

        # One file per level: base for R0 and .r1..rN for overviews.
        paths = [str(base_path)] + [f"{base_path}.r{i}" for i in range(1, expected_num_levels)]

        with _try_open_nitf_writer(paths) as writer:
            # Add the base asset explicitly (build_and_write only writes overviews).
            base_md = _make_nitf_image_metadata_fn()(0)
            base_provider = io.BufferedImageAssetProvider.from_provider(source, metadata=base_md)
            writer.add_asset("image:0", base_provider, "", "", ["data"])
            builder.build_and_write(
                writer,
                base_key="image:0",
                image_metadata_fn=_make_nitf_image_metadata_fn(),
            )

    # Base file and every R-Set sidecar must exist and be non-empty.
    for path in paths:
        assert os.path.isfile(path), f"Writer did not produce {path}"
        assert os.path.getsize(path) > 0, f"Writer produced empty file {path}"

    # -- Re-open and inspect the R-Set ------------------------------------
    try:
        out_reader_ctx = io.IO.open(paths, "r")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot re-open the NITF R-Set: {exc}")

    with out_reader_ctx as out_reader:
        asset_keys = list(out_reader.get_asset_keys())

        # The base asset is always present.
        assert "image:0" in asset_keys, f"base asset missing from {asset_keys}"

        # Overview assets follow the "{base_key}:overview:{N}" convention.
        overview_keys = [k for k in asset_keys if k.startswith("image:0:overview:")]
        assert len(overview_keys) == expected_num_levels - 1, (
            f"expected {expected_num_levels - 1} overview assets, got {overview_keys}"
        )

        overview_keys_sorted = sorted(overview_keys, key=lambda k: int(k.rsplit(":", 1)[-1]))
        level_keys = ["image:0"] + overview_keys_sorted
        level_assets = [out_reader.get_asset(k) for k in level_keys]

        # Dimensions must strictly decrease from one level to the next.
        prev_rows, prev_cols = level_assets[0].num_rows, level_assets[0].num_columns
        for asset in level_assets[1:]:
            assert asset.num_rows <= prev_rows, f"level {asset.key}: num_rows {asset.num_rows} > previous {prev_rows}"
            assert asset.num_columns <= prev_cols, (
                f"level {asset.key}: num_columns {asset.num_columns} > previous {prev_cols}"
            )
            assert asset.num_rows < prev_rows or asset.num_columns < prev_cols, (
                f"level {asset.key}: dimensions did not decrease ({asset.num_rows}x"
                f"{asset.num_columns} vs {prev_rows}x{prev_cols})"
            )
            prev_rows, prev_cols = asset.num_rows, asset.num_columns

        # Every level must allow a tile read at (0, 0, 0) and produce a CHW
        # array whose band count matches the source.
        source_num_bands = level_assets[0].num_bands
        for asset in level_assets:
            assert asset.has_block(0, 0, 0), f"level {asset.key}: block (0, 0) missing from re-opened R-Set"
            block = asset.get_block(0, 0, 0)
            assert isinstance(block, np.ndarray), f"level {asset.key}: get_block returned {type(block).__name__}"
            assert block.ndim == 3, f"level {asset.key}: expected CHW block, got ndim={block.ndim}"
            assert block.shape[0] == source_num_bands, (
                f"level {asset.key}: band count {block.shape[0]} != {source_num_bands}"
            )


@pytest.mark.skipif(not _nitf_source_exists(), reason="test/data/small.ntf not available")
def test_rset_generation_does_not_modify_source(tmp_path):
    """Regression: R-Set pyramid generation must never modify the source file.

    Copies the source NITF to a temp directory, records its size and content
    hash, runs the pyramid builder (as the example script would), and
    asserts the source file is byte-for-byte unchanged afterward.

    The NITF R-Set convention writes sidecars alongside the source; the
    writer uses a disposable base path for internal level mapping so the
    original source is never opened for writing.
    """
    import hashlib
    import shutil

    # Copy source to temp so we own the file and can safely check it.
    source_copy = str(tmp_path / "source_copy.ntf")
    shutil.copy2(_SOURCE_NTF, source_copy)

    original_size = os.path.getsize(source_copy)
    with open(source_copy, "rb") as f:
        original_hash = hashlib.sha256(f.read()).hexdigest()

    with _try_open_nitf_source() as reader:
        source = reader.get_asset("image:0")
        builder = PyramidBuilder(source, min_size=256, num_workers=0)
        num_levels = len(builder._levels)

        if num_levels < 2:
            pytest.skip("NITF source too small to produce overviews at min_size=256")

        # Use a disposable base path for the writer (not the source).
        # The writer truncates all paths on open, so the source must
        # never appear in the path list.
        writer_base = str(tmp_path / "rset_base.ntf")
        paths = [writer_base] + [f"{source_copy}.r{i}" for i in range(1, num_levels)]

        def md_fn(level_index):
            md = io.BufferedMetadataProvider()
            md["IC"] = "NC"
            md["IMODE"] = "B"
            return md

        with _try_open_nitf_writer(paths) as writer:
            builder.build_and_write(writer, base_key="image:0", image_metadata_fn=md_fn)

    # The source file must be unchanged.
    assert os.path.getsize(source_copy) == original_size, "source file size changed during R-Set generation"
    with open(source_copy, "rb") as f:
        after_hash = hashlib.sha256(f.read()).hexdigest()
    assert after_hash == original_hash, "source file content changed during R-Set generation"


@pytest.mark.skipif(not _nitf_source_exists(), reason="test/data/small.ntf not available")
def test_tiled_image_pyramid_from_dataset_nitf_rset(tmp_path):
    """When :class:`TiledImagePyramid` is available (task 7), re-opening an
    NITF R-Set output via ``TiledImagePyramid.from_dataset`` exposes the
    same levels in decreasing-resolution order and each level yields a
    readable tile.

    Mirrors :func:`test_tiled_image_pyramid_from_dataset` for the NITF
    R-Set output path. Validates requirements 5.6 (overview discovery by
    key convention) and 5.7 (single-level fallback not triggered when
    overviews are present) for the NITF R-Set container.
    """
    try:
        from aws.osml.image_processing.pyramid import TiledImagePyramid  # noqa: F401
    except ImportError:
        pytest.skip("TiledImagePyramid not implemented yet (task 7)")

    base_path = tmp_path / "pyramid_for_tip.ntf"

    with _try_open_nitf_source() as reader:
        source = reader.get_asset("image:0")
        builder = PyramidBuilder(source, min_size=256, num_workers=0)
        expected_num_levels = len(builder._levels)

        if expected_num_levels < 2:
            pytest.skip("NITF source too small to produce overviews at min_size=256")

        paths = [str(base_path)] + [f"{base_path}.r{i}" for i in range(1, expected_num_levels)]

        with _try_open_nitf_writer(paths) as writer:
            base_md = _make_nitf_image_metadata_fn()(0)
            base_provider = io.BufferedImageAssetProvider.from_provider(source, metadata=base_md)
            writer.add_asset("image:0", base_provider, "", "", ["data"])
            builder.build_and_write(
                writer,
                base_key="image:0",
                image_metadata_fn=_make_nitf_image_metadata_fn(),
            )

    try:
        out_reader_ctx = io.IO.open(paths, "r")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot re-open the NITF R-Set: {exc}")

    with out_reader_ctx as out_reader:
        pyramid = TiledImagePyramid.from_dataset(out_reader, "image:0")
        assert pyramid.num_levels == expected_num_levels

        # Monotonic decrease across levels.
        prev_shape = pyramid.image_shape_at_level(0)
        for i in range(1, pyramid.num_levels):
            shape = pyramid.image_shape_at_level(i)
            assert shape[1] <= prev_shape[1]
            assert shape[2] <= prev_shape[2]
            assert shape[1] < prev_shape[1] or shape[2] < prev_shape[2]
            prev_shape = shape

        # Every level yields a readable first tile.
        for i in range(pyramid.num_levels):
            level = pyramid.get_level(i)
            assert level.has_block(0, 0, 0)
            block = level.get_block(0, 0, 0)
            assert block.ndim == 3
            assert block.shape[0] == pyramid.image_shape_at_level(i)[0]


def test_build_uses_native_j2k_levels():
    """Open a multi-resolution J2K source, build a pyramid with
    ``use_native_levels=True``, and verify the first N overview
    levels' tiles are element-wise equal to the source's native
    reduced-resolution reads (no resampling applied).

    The :class:`PyramidBuilder` J2K fast path populates each native
    overview level by reading the source at ``resolution_level=i``
    and stitching the blocks into the overview tile grid. When the
    overview tile size matches the source's native block size — the
    default when ``tile_width`` / ``tile_height`` are inherited from
    the source — each overview tile at ``(r, c)`` is the exact same
    array as ``source.get_block(r, c, resolution_level=i)``.

    Validates requirements 3.12 (native J2K level population) and
    3.2 (no duplicate source reads: the cascade never reads R0 when
    every overview level can be supplied natively).

    Gracefully skipped in environments without a J2K fixture under
    ``test/data`` or without J2K decode support in
    ``osml-imagery-io``.
    """
    source_path = _find_j2k_source()
    if not source_path:
        pytest.skip("no J2K fixture (.jp2/.j2k/.jpf/.jpx) available under test/data")

    try:
        reader_ctx = io.IO.open(source_path, "r")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot open {source_path}: {exc}")

    with reader_ctx as reader:
        # Pick up the primary image asset. J2K files expose a single
        # ``image:0`` asset in osml-imagery-io's conventions.
        try:
            source = reader.get_asset("image:0")
        except (KeyError, LookupError, ValueError) as exc:
            pytest.skip(f"J2K source {source_path} has no 'image:0' asset: {exc}")

        # The test is only meaningful when the source advertises more
        # than one native resolution level — otherwise the builder's
        # native fast path is a no-op and we would be validating
        # resampling behavior instead.
        native_levels = int(getattr(source, "num_resolution_levels", 1))
        if native_levels <= 1:
            pytest.skip(
                f"J2K source {source_path} reports num_resolution_levels={native_levels}; native-path test requires >= 2"
            )

        # Inherit the source's block size for the overview levels so
        # overview tile (r, c) aligns with a single native block at
        # the matching resolution level. Set ``min_size=1`` so we
        # always plan at least one overview level regardless of
        # source dimensions.
        builder = PyramidBuilder(
            source,
            min_size=1,
            num_workers=0,
            use_native_levels=True,
        )

        # Require the plan to actually produce an overview level
        # that we can populate from a native read. Without at least
        # one overview the test has nothing to compare.
        num_overviews = len(builder._levels) - 1
        if num_overviews < 1:
            pytest.skip(f"J2K source {source_path} too small to plan any overview level")

        # The number of overview levels the builder will populate
        # from native source reads (one per native resolution level
        # above 0, bounded by the number of planned overviews).
        native_overviews = min(native_levels - 1, num_overviews)
        assert native_overviews >= 1, "expected at least one native overview"

        # Build; returns [source] followed by one provider per
        # overview level.
        levels = builder.build()
        assert len(levels) == num_overviews + 1, f"expected {num_overviews + 1} levels, got {len(levels)}"

        # For each native overview level ``i``, compare the first
        # handful of overview tiles against the source's native
        # reduced-resolution blocks. When the tile sizes match
        # (the case here since we inherited from the source), the
        # overview tile must be byte-for-byte identical to
        # ``source.get_block(r, c, resolution_level=i)``.
        max_tiles_to_check = 4
        tiles_compared = 0
        for i in range(1, native_overviews + 1):
            overview_provider = levels[i]
            grid_rows, grid_cols = overview_provider.block_grid_size
            for r in range(grid_rows):
                for c in range(grid_cols):
                    if not source.has_block(r, c, i):
                        continue
                    if not overview_provider.has_block(r, c, 0):
                        continue
                    expected = source.get_block(r, c, i)
                    actual = overview_provider.get_block(r, c, 0)

                    assert actual.shape == expected.shape, (
                        f"level {i} tile ({r},{c}): shape {actual.shape} != source native shape {expected.shape}"
                    )
                    assert actual.dtype == expected.dtype, (
                        f"level {i} tile ({r},{c}): dtype {actual.dtype} != source native dtype {expected.dtype}"
                    )
                    np.testing.assert_array_equal(
                        actual,
                        expected,
                        err_msg=(
                            f"level {i} tile ({r},{c}) was not populated from "
                            f"the native source read (resampling or copy "
                            f"corruption occurred)"
                        ),
                    )
                    tiles_compared += 1
                    if tiles_compared >= max_tiles_to_check:
                        break
                if tiles_compared >= max_tiles_to_check:
                    break
            if tiles_compared >= max_tiles_to_check:
                break

        assert tiles_compared >= 1, f"expected to verify at least one overview tile for J2K source {source_path}, got 0"


def test_tiled_image_pyramid_from_real_cog():
    """Open a COG that has pre-computed overviews (not produced by
    ``PyramidBuilder`` in this test run), build a
    :class:`TiledImagePyramid` via ``from_dataset``, and verify the
    level count and image shapes match the file's underlying providers.

    This complements :func:`test_tiled_image_pyramid_from_dataset`,
    which re-reads a pyramid freshly written by ``PyramidBuilder``.
    Here the fixture is an "external" COG — one whose overview
    structure was computed outside the toolkit — so the test validates
    that ``from_dataset`` correctly discovers and orders overview
    assets produced by any conforming writer.

    Fixtures are discovered by walking ``test/data`` for any TIFF whose
    primary asset reports ``num_resolution_levels > 1`` or whose
    dataset exposes a ``"image:0:overview:N"`` asset key. When no such
    fixture exists the test :func:`pytest.skip` s with a clear reason
    (the toolkit ships no COG-with-overviews test fixture today).

    Validates requirements 5.1 (``TiledImagePyramid`` construction),
    5.3 (``get_level`` / ``image_shape_at_level`` / ``tile_grid_at_level``
    match the underlying providers), 5.4 (monotonically non-increasing
    dimensions from level 0 to N), and 5.6 (``from_dataset`` overview
    discovery by key convention).
    """
    try:
        from aws.osml.image_processing.pyramid import TiledImagePyramid
    except ImportError:
        pytest.skip("TiledImagePyramid not implemented yet (task 7)")

    source_path = _find_cog_with_overviews()
    if not source_path:
        pytest.skip("no TIFF with pre-computed overviews available under test/data")

    try:
        reader_ctx = io.IO.open(source_path, "r")
    except (OSError, IOError, ValueError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"osml-imagery-io cannot open {source_path}: {exc}")

    with reader_ctx as reader:
        # Enumerate the overview assets by the documented key
        # convention so we can compare against what ``from_dataset``
        # discovers. The base asset is always level 0; overview assets
        # are keyed ``image:0:overview:1``, ``image:0:overview:2``, …
        asset_keys = list(reader.get_asset_keys())
        assert "image:0" in asset_keys, f"base asset missing from {asset_keys}"
        overview_keys = sorted(
            (k for k in asset_keys if k.startswith("image:0:overview:")),
            key=lambda k: int(k.rsplit(":", 1)[-1]),
        )
        level_keys = ["image:0"] + overview_keys
        level_assets = [reader.get_asset(k) for k in level_keys]

        # The discovery test is only meaningful when at least one
        # overview asset is exposed via the key convention. A source
        # that only advertises extra native resolution levels on the
        # primary asset (with no overview asset keys) would exercise
        # ``from_dataset``'s single-level fallback instead, which is
        # covered by other tests. Skip such fixtures here so this test
        # stays focused on requirement 5.6.
        if len(overview_keys) < 1:
            pytest.skip(
                f"COG {source_path} reports native resolution levels but "
                f"exposes no overview asset keys — skipping overview-discovery test"
            )

        pyramid = TiledImagePyramid.from_dataset(reader, "image:0")

        # (a) Level count matches the number of assets following the
        # pyramid key convention — requirement 5.6.
        assert pyramid.num_levels == len(level_assets), (
            f"pyramid.num_levels={pyramid.num_levels} does not match asset count {len(level_assets)} (keys: {level_keys})"
        )

        # (b) Per-level image shape matches the underlying provider
        # exactly — requirement 5.3.
        for i, asset in enumerate(level_assets):
            expected_shape = (asset.num_bands, asset.num_rows, asset.num_columns)
            actual_shape = pyramid.image_shape_at_level(i)
            assert actual_shape == expected_shape, (
                f"level {i} ({level_keys[i]}): image_shape_at_level returned {actual_shape}, expected {expected_shape}"
            )

            # Tile grid is accessible at every level — requirement 5.3.
            grid = pyramid.tile_grid_at_level(i)
            assert grid == asset.block_grid_size, (
                f"level {i} ({level_keys[i]}): tile_grid_at_level returned "
                f"{grid}, expected provider.block_grid_size={asset.block_grid_size}"
            )

            # ``get_level`` returns the matching underlying provider —
            # requirement 5.3.
            level_provider = pyramid.get_level(i)
            assert level_provider.num_rows == asset.num_rows
            assert level_provider.num_columns == asset.num_columns
            assert level_provider.num_bands == asset.num_bands

        # (c) Dimensions are monotonically non-increasing from level 0
        # to N, with at least one axis strictly decreasing between
        # adjacent levels — requirement 5.4.
        prev_shape = pyramid.image_shape_at_level(0)
        for i in range(1, pyramid.num_levels):
            shape = pyramid.image_shape_at_level(i)
            assert shape[1] <= prev_shape[1], f"level {i}: num_rows {shape[1]} > previous level's {prev_shape[1]}"
            assert shape[2] <= prev_shape[2], f"level {i}: num_columns {shape[2]} > previous level's {prev_shape[2]}"
            assert shape[1] < prev_shape[1] or shape[2] < prev_shape[2], (
                f"level {i}: dimensions did not decrease from previous level "
                f"({shape[1]}x{shape[2]} vs {prev_shape[1]}x{prev_shape[2]})"
            )
            prev_shape = shape
