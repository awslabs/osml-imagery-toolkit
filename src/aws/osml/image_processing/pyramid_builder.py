#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Single-pass R-Set pyramid generation via incremental accumulation.

This module provides :class:`PyramidBuilder`, the write-optimized
entry point for generating image pyramids (COG overviews, NITF
R-Sets) from a full-resolution source in a single raster-order sweep.
The builder reads every R0 tile exactly once and incrementally
propagates each decoded tile's contribution up every pyramid level,
so peak memory stays at roughly one partial tile per level rather
than one full level at a time.

See :mod:`aws.osml.image_processing.resample` for the ``ResampleFunc``
protocol and OpenCV-based resamplers, and
:mod:`aws.osml.image_processing.sips_resample` for the default
``sips_rrds_resample`` used when no resampler is supplied.
"""

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from aws.osml.io import BufferedImageAssetProvider

from .block_utils import read_block_or_pad, stitch_source_blocks
from .resample import ResampleFunc
from .retiled_provider import RetiledImageProvider
from .sips_resample import sips_rrds_resample

# Default tile dimensions used when the source's block size cannot be
# inherited and the caller does not override. 1024 is the size used by
# the v2.0 design examples and aligns with COG/NITF conventions.
_DEFAULT_TILE_SIZE = 1024

# Maximum tile dimension inherited from a source provider. NITF NPPBH/NPPBV
# fields are limited to 8192; untiled imagery (NPPBH=0) reports full image
# width as block size which can exceed this. Clamping avoids producing
# output that cannot be encoded.
_MAX_TILE_SIZE = 8192

# Sentinel value pushed onto the prefetch queue to signal end-of-stream
# and onto the writeback queue to signal clean shutdown.
_SENTINEL = object()


class ProgressCallback(Protocol):
    """Protocol for pyramid build progress reporting.

    Implementations receive periodic updates as tiles are processed.
    The callback is invoked once per source tile consumed by the
    cascade — the natural unit of progress for the single-pass
    algorithm.

    Args:
        completed: Number of source tiles processed so far.
        total: Total number of source tiles to process.
        level: The pyramid level currently being fed (1-based overview
            index). For native-level population this is the native
            level index; for the resampling cascade it is the first
            resampled level.

    Example::

        def my_progress(completed: int, total: int, level: int) -> None:
            pct = 100.0 * completed / total
            print(f"\\r  [{pct:5.1f}%] {completed}/{total} tiles", end="", flush=True)


        builder = PyramidBuilder(source, progress=my_progress)
        builder.build()
    """

    def __call__(self, completed: int, total: int, level: int) -> None: ...


@dataclass
class _LevelPlan:
    """Per-level build schedule computed by :meth:`PyramidBuilder._plan_levels`.

    Attributes:
        level_index: 0-based index. Level 0 is the source (R0); level
            1+ are overview levels produced by the builder.
        num_rows: Pixel rows at this level.
        num_columns: Pixel columns at this level.
        num_pixels_per_block_horizontal: Tile width in pixels.
        num_pixels_per_block_vertical: Tile height in pixels.
        block_grid_size: ``(grid_rows, grid_cols)`` — the tile grid
            dimensions, derived from the pixel dimensions and tile size.
        provider: The pre-allocated ``BufferedImageAssetProvider`` for
            this level, or ``None`` for level 0 (R0 is the source itself
            and is not copied).
    """

    level_index: int
    num_rows: int
    num_columns: int
    num_pixels_per_block_horizontal: int
    num_pixels_per_block_vertical: int
    block_grid_size: Tuple[int, int]
    provider: Optional[Any] = None


class PyramidBuilder:
    """Single-pass R-Set generation via incremental accumulation.

    Reads R0 tiles in raster order and incrementally builds every
    reduced-resolution level. Completed overview tiles are stored in
    ``BufferedImageAssetProvider`` instances allocated at construction
    time and drained by the caller's :class:`DatasetWriter` via
    :meth:`build_and_write`.

    Args:
        source: A duck-typed ``ImageAssetProvider`` representing the
            full-resolution image (R0).
        min_size: Overview generation stops once either image dimension
            of the current level falls strictly below this threshold;
            that level is included as the final overview.
            Must be greater than zero. Defaults to ``256``.
        scale_factor: Per-level reduction factor. v2.0 supports only
            ``scale_factor=2``; any other value raises
            :class:`ValueError`. Defaults to ``2``.
        tile_width: Tile width for overview levels. When ``None``,
            inherits from the source's
            ``num_pixels_per_block_horizontal``. Defaults to ``None``.
        tile_height: Tile height for overview levels. When ``None``,
            inherits from the source's
            ``num_pixels_per_block_vertical``. Defaults to ``None``.
        resample_func: A ``ResampleFunc`` callable used to downsample
            each 2x2 tile group. Defaults to
            :func:`~aws.osml.image_processing.sips_resample.sips_rrds_resample`
            when ``None``.
        num_workers: Number of background threads used for the
            prefetch + writeback pipeline. ``0`` runs entirely in the
            calling thread (no executors). Must be non-negative.
            Defaults to ``2``.
        use_native_levels: When ``True`` and the source exposes
            ``num_resolution_levels > 1``, populate the first N
            overview levels from native reduced-resolution reads
            instead of resampling. Defaults to ``True``.
        progress: Optional callback invoked after each source tile is
            processed. Receives ``(completed, total, level)`` — see
            :class:`ProgressCallback` for the full protocol. Defaults
            to ``None`` (no reporting).

    Raises:
        ValueError: If ``scale_factor != 2``, ``min_size <= 0``, or
            ``num_workers < 0``.
    """

    def __init__(
        self,
        source: Any,
        min_size: int = 256,
        scale_factor: int = 2,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        resample_func: Optional[ResampleFunc] = None,
        num_workers: int = 2,
        use_native_levels: bool = True,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        # Validate scale_factor first — the accumulator cascade is
        # inherently 2x2-grouped and other factors are not supported
        # in v2.0.
        if scale_factor != 2:
            raise ValueError(f"v2.0 supports scale_factor=2 only, got {scale_factor}")
        if min_size <= 0:
            raise ValueError(f"min_size must be positive, got {min_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")

        self._source = source
        self._min_size = int(min_size)
        self._scale_factor = int(scale_factor)
        self._resample_func: ResampleFunc = resample_func or sips_rrds_resample
        self._num_workers = int(num_workers)
        self._use_native_levels = bool(use_native_levels)
        self._progress = progress

        # Resolve tile dimensions: explicit args win; otherwise inherit
        # from the source (clamped to _MAX_TILE_SIZE); otherwise fall back
        # to the module default.
        resolved_tile_width = tile_width
        if resolved_tile_width is None:
            resolved_tile_width = min(
                getattr(source, "num_pixels_per_block_horizontal", _DEFAULT_TILE_SIZE),
                _MAX_TILE_SIZE,
            )
        resolved_tile_height = tile_height
        if resolved_tile_height is None:
            resolved_tile_height = min(
                getattr(source, "num_pixels_per_block_vertical", _DEFAULT_TILE_SIZE),
                _MAX_TILE_SIZE,
            )
        self._tile_width = int(resolved_tile_width)
        self._tile_height = int(resolved_tile_height)

        # When the source's block dimensions don't match the target tile
        # size, wrap it in a RetiledImageProvider so the cascade's 2:1
        # grid invariant holds. This handles untiled imagery (e.g. NITF
        # with NPPBH=0) where the source reports a single oversized block.
        src_bw = int(getattr(source, "num_pixels_per_block_horizontal", self._tile_width))
        src_bh = int(getattr(source, "num_pixels_per_block_vertical", self._tile_height))
        if src_bw != self._tile_width or src_bh != self._tile_height:
            source = RetiledImageProvider(source, tile_width=self._tile_width, tile_height=self._tile_height)
            self._source = source

        # Plan the level schedule and pre-allocate BufferedImageAssetProvider
        # instances for every overview level.
        self._levels: List[_LevelPlan] = self._plan_levels(
            source,
            min_size=self._min_size,
            scale_factor=self._scale_factor,
            tile_width=self._tile_width,
            tile_height=self._tile_height,
        )

    # ------------------------------------------------------------------
    # Public properties (read-only introspection)
    # ------------------------------------------------------------------

    @property
    def source(self) -> Any:
        """The full-resolution source provider (R0)."""
        return self._source

    @property
    def min_size(self) -> int:
        """Threshold below which overview generation stops."""
        return self._min_size

    @property
    def scale_factor(self) -> int:
        """Per-level reduction factor (always 2 in v2.0)."""
        return self._scale_factor

    @property
    def tile_width(self) -> int:
        """Tile width used for every overview level."""
        return self._tile_width

    @property
    def tile_height(self) -> int:
        """Tile height used for every overview level."""
        return self._tile_height

    @property
    def resample_func(self) -> ResampleFunc:
        """The resampling callable applied to each 2x2 tile group."""
        return self._resample_func

    @property
    def num_workers(self) -> int:
        """Background thread count for prefetch and writeback."""
        return self._num_workers

    @property
    def use_native_levels(self) -> bool:
        """Whether to prefer native reduced-resolution source reads."""
        return self._use_native_levels

    @property
    def progress(self) -> Optional[ProgressCallback]:
        """The progress callback, or ``None`` if not set."""
        return self._progress

    @progress.setter
    def progress(self, value: Optional[ProgressCallback]) -> None:
        """Set or clear the progress callback."""
        self._progress = value

    # ------------------------------------------------------------------
    # Level planning
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_levels(
        source: Any,
        min_size: int,
        scale_factor: int,
        tile_width: int,
        tile_height: int,
    ) -> List[_LevelPlan]:
        """Compute the per-level build schedule.

        Level 0 captures the source dimensions and tile grid but does
        not allocate a ``BufferedImageAssetProvider`` (R0 is returned as
        the source itself by :meth:`build`). Levels 1..N are overview
        levels with dimensions derived from the SIPS even/odd rounding
        rule and a freshly-allocated ``BufferedImageAssetProvider``
        each.

        Args:
            source: Duck-typed ``ImageAssetProvider``. Must expose
                ``num_rows``, ``num_columns``, ``num_bands``, and
                ``pixel_value_type``.
            min_size: Stop adding overview levels once either dimension
                of the current level falls strictly below this value.
                That level is included as the final overview.
            scale_factor: Per-level reduction factor. Must be ``2``;
                other values are rejected by
                :meth:`PyramidBuilder.__init__`.
            tile_width: Overview tile width in pixels.
            tile_height: Overview tile height in pixels.

        Returns:
            A list of :class:`_LevelPlan` entries, ordered highest
            resolution (``level_index=0``) to lowest.
        """
        num_bands = int(source.num_bands)
        pixel_value_type = source.pixel_value_type

        levels: List[_LevelPlan] = []

        # Level 0 (R0) — captured for bookkeeping. No provider is
        # allocated; the source is returned directly by build().
        r0_tile_w = int(getattr(source, "num_pixels_per_block_horizontal", tile_width))
        r0_tile_h = int(getattr(source, "num_pixels_per_block_vertical", tile_height))
        r0_rows = int(source.num_rows)
        r0_cols = int(source.num_columns)
        levels.append(
            _LevelPlan(
                level_index=0,
                num_rows=r0_rows,
                num_columns=r0_cols,
                num_pixels_per_block_horizontal=r0_tile_w,
                num_pixels_per_block_vertical=r0_tile_h,
                block_grid_size=_grid_size(r0_rows, r0_cols, r0_tile_h, r0_tile_w),
                provider=None,
            )
        )

        # Overview levels — apply SIPS even/odd rounding:
        #   next_rows = (prev_rows + 1) // 2
        #   next_cols = (prev_cols + 1) // 2
        # Keep generating until either dimension drops below min_size;
        # that level is included as the final overview.
        prev_rows = r0_rows
        prev_cols = r0_cols
        level_index = 1
        while True:
            next_rows = (prev_rows + 1) // scale_factor
            next_cols = (prev_cols + 1) // scale_factor
            # Guard against degenerate cases: if dimensions stop
            # shrinking (e.g. 1 // 2 == 0), stop before creating a
            # zero-sized level.
            if next_rows <= 0 or next_cols <= 0:
                break

            # Check whether this level crosses the min_size threshold
            # on either axis. If so, include it as the final level.
            is_last = next_rows < min_size or next_cols < min_size

            provider = BufferedImageAssetProvider.create(
                key="image:0:overview:{}".format(level_index),
                num_columns=next_cols,
                num_rows=next_rows,
                num_bands=num_bands,
                block_width=tile_width,
                block_height=tile_height,
                pixel_type=pixel_value_type,
            )
            levels.append(
                _LevelPlan(
                    level_index=level_index,
                    num_rows=next_rows,
                    num_columns=next_cols,
                    num_pixels_per_block_horizontal=tile_width,
                    num_pixels_per_block_vertical=tile_height,
                    block_grid_size=_grid_size(next_rows, next_cols, tile_height, tile_width),
                    provider=provider,
                )
            )

            if is_last:
                break

            prev_rows = next_rows
            prev_cols = next_cols
            level_index += 1

        return levels

    # ------------------------------------------------------------------
    # Build entry points
    # ------------------------------------------------------------------

    def build(self) -> List[Any]:
        """Generate every pyramid level in a single R0 pass.

        Reads each R0 tile exactly once (raster order), incrementally
        cascades each tile's contribution up every overview level, and
        populates the pre-allocated ``BufferedImageAssetProvider``
        instances in place.

        Returns a list whose first element is the source (R0) itself
        and whose remaining elements are the populated
        ``BufferedImageAssetProvider`` instances for each overview
        level, ordered highest to lowest resolution.

        When ``use_native_levels=True`` and the source exposes
        ``num_resolution_levels > 1``, the first ``N`` overview levels
        are filled from native reduced-resolution reads; the accumulator
        cascade starts at the first level beyond native support.

        When ``num_workers == 0``, execution is entirely single-threaded
        — no ``ThreadPoolExecutor`` instances are created. Otherwise a
        prefetch thread decodes R0 tiles ahead of the main thread and a
        writeback thread drains ``set_block`` calls; exceptions in
        either background thread are propagated on the main thread and
        both executors are shut down before re-raising.

        Returns:
            ``[source, level_1_provider, ..., level_N_provider]``.
        """
        # Short-circuit: no overview levels to produce.
        if len(self._levels) <= 1:
            return [self._source]

        # Optional J2K native fast path — populate as many overview
        # levels as the source exposes natively, bypassing the
        # accumulator cascade for those.
        first_resample_level = self._populate_native_levels()

        # If native reads already covered every overview level there's
        # nothing left for the cascade to do.
        if first_resample_level >= len(self._levels):
            return [self._source] + [lvl.provider for lvl in self._levels[1:]]

        # Initialise per-level accumulators for every level from
        # ``first_resample_level`` upward. Keys are tile coordinates at
        # that level; values are ``(buffer, quadrants_mask)`` tuples.
        self._accumulators: Dict[int, Dict[Tuple[int, int], Tuple[NDArray, int]]] = {
            lvl: {} for lvl in range(first_resample_level, len(self._levels))
        }

        # The R0-equivalent input for the cascade: when the source has
        # native levels we've already consumed, read the source at
        # ``first_resample_level - 1`` (which is the highest level we
        # have not yet produced an overview for).
        source_resolution_level = first_resample_level - 1
        parent_level_index = first_resample_level - 1
        parent_plan = self._levels[parent_level_index]

        if self._num_workers == 0:
            self._run_single_threaded(parent_plan, source_resolution_level, first_resample_level)
        else:
            self._run_threaded(parent_plan, source_resolution_level, first_resample_level)

        return [self._source] + [lvl.provider for lvl in self._levels[1:]]

    def build_and_write(
        self,
        writer: Any,
        base_key: str = "image:0",
        file_metadata: Optional[Any] = None,
        image_metadata_fn: Optional[Callable[[int], Any]] = None,
    ) -> None:
        """Build overview levels and write them to a ``DatasetWriter``.

        Invokes :meth:`build` to produce the full level list, then
        writes only the overview levels (level 1+) as assets on
        ``writer``. The source image (level 0) is never written —
        callers that need the base in the output (e.g. for COG) should
        add it to the writer themselves.

        Args:
            writer: A duck-typed ``DatasetWriter`` exposing
                ``add_asset(key, provider, title, description, roles)``
                and a ``metadata`` setter.
            base_key: Base asset key prefix. Overview levels are keyed
                as ``f"{base_key}:overview:{i}"``. Defaults to
                ``"image:0"``.
            file_metadata: Optional dataset-level metadata. When
                provided, it is applied to ``writer`` (via the
                ``metadata`` setter) before any asset is added.
            image_metadata_fn: Optional callable taking a level index
                (1-based for overviews) and returning a
                ``MetadataProvider`` (or compatible) to attach to that
                level's output asset. Metadata is attached by wrapping
                the overview provider with
                :meth:`BufferedImageAssetProvider.from_provider`.
        """
        levels = self.build()

        if file_metadata is not None:
            _apply_writer_metadata(writer, file_metadata)

        for level_index, provider in enumerate(levels):
            if level_index == 0:
                continue

            metadata = image_metadata_fn(level_index) if image_metadata_fn is not None else None
            asset_provider = provider
            if metadata is not None:
                asset_provider = BufferedImageAssetProvider.from_provider(provider, metadata=metadata)

            key = "{}:overview:{}".format(base_key, level_index)
            writer.add_asset(key, asset_provider, "", "", ["overview"])

    # ------------------------------------------------------------------
    # Single-threaded execution path
    # ------------------------------------------------------------------

    def _run_single_threaded(
        self,
        parent_plan: "_LevelPlan",
        source_resolution_level: int,
        first_resample_level: int,
    ) -> None:
        """Iterate the parent level's tile grid in raster order, feeding
        each tile into the cascade synchronously."""
        parent_rows, parent_cols = parent_plan.block_grid_size
        total_tiles = parent_rows * parent_cols
        completed = 0
        for r in range(parent_rows):
            for c in range(parent_cols):
                tile = self._read_parent_tile(parent_plan, r, c, source_resolution_level)
                self._cascade_into_level(
                    level=first_resample_level,
                    tile_row=r // 2,
                    tile_col=c // 2,
                    quadrant=(r % 2, c % 2),
                    quadrant_data=tile,
                    set_block_fn=_direct_set_block,
                )
                completed += 1
                if self._progress is not None:
                    self._progress(completed, total_tiles, first_resample_level)

    # ------------------------------------------------------------------
    # Threaded execution path
    # ------------------------------------------------------------------

    def _run_threaded(
        self,
        parent_plan: "_LevelPlan",
        source_resolution_level: int,
        first_resample_level: int,
    ) -> None:
        """Prefetch R0 tiles on a background thread, cascade on the
        main thread, and offload ``set_block`` calls to a writeback
        thread.

        Exceptions raised in either background thread are captured via
        ``Future.result()`` after the main loop finishes; both
        executors are shut down before any captured exception is
        re-raised.
        """
        parent_rows, parent_cols = parent_plan.block_grid_size
        prefetch_queue: "queue.Queue[Any]" = queue.Queue(maxsize=self._num_workers)
        writeback_queue: "queue.Queue[Any]" = queue.Queue(maxsize=self._num_workers)
        error_flag = threading.Event()

        prefetch_executor = ThreadPoolExecutor(max_workers=1)
        writeback_executor = ThreadPoolExecutor(max_workers=1)

        prefetch_future: Future = prefetch_executor.submit(
            _prefetch_worker,
            parent_plan,
            parent_rows,
            parent_cols,
            source_resolution_level,
            self._source,
            self._read_parent_tile,
            prefetch_queue,
            error_flag,
        )
        writeback_future: Future = writeback_executor.submit(
            _writeback_worker,
            writeback_queue,
            error_flag,
        )

        def queued_set_block(provider: Any, row: int, col: int, data: NDArray) -> None:
            # Ensure the array passed to the writeback thread is a
            # contiguous copy — cv2 / numpy may return views that the
            # Rust binding cannot accept.
            writeback_queue.put((provider, row, col, np.ascontiguousarray(data)))

        total_tiles = parent_rows * parent_cols
        completed = 0
        captured_exc: Optional[BaseException] = None
        try:
            while True:
                item = prefetch_queue.get()
                if item is _SENTINEL:
                    break
                r, c, tile = item
                self._cascade_into_level(
                    level=first_resample_level,
                    tile_row=r // 2,
                    tile_col=c // 2,
                    quadrant=(r % 2, c % 2),
                    quadrant_data=tile,
                    set_block_fn=queued_set_block,
                )
                completed += 1
                if self._progress is not None:
                    self._progress(completed, total_tiles, first_resample_level)
        except BaseException as exc:  # noqa: BLE001 — propagate any failure
            captured_exc = exc
            error_flag.set()
            # Drain the prefetch queue so the prefetch worker can exit.
            _drain_queue(prefetch_queue)

        # Signal the writeback worker to shut down and wait for it.
        writeback_queue.put(_SENTINEL)

        # Surface any exception from the background threads via
        # Future.result() — this is the canonical propagation path.
        try:
            prefetch_future.result()
        except BaseException as exc:  # noqa: BLE001 — see above
            if captured_exc is None:
                captured_exc = exc
        try:
            writeback_future.result()
        except BaseException as exc:  # noqa: BLE001 — see above
            if captured_exc is None:
                captured_exc = exc

        prefetch_executor.shutdown(wait=True)
        writeback_executor.shutdown(wait=True)

        if captured_exc is not None:
            raise captured_exc

    # ------------------------------------------------------------------
    # Parent-level tile reading (handles sparse sources and native levels)
    # ------------------------------------------------------------------

    def _read_parent_tile(
        self,
        parent_plan: "_LevelPlan",
        tile_row: int,
        tile_col: int,
        source_resolution_level: int,
    ) -> NDArray:
        """Return the CHW block at ``(tile_row, tile_col)`` of the
        parent level, substituting a pad-filled tile when the block is
        sparse (``has_block`` is False).

        For the very first cascade step (when no native levels have
        been consumed), the "parent level" is the source and
        ``source_resolution_level == 0``. When native levels have been
        consumed, the parent is the last native level's
        ``BufferedImageAssetProvider`` and
        ``source_resolution_level == 0`` for that provider.
        """
        if parent_plan.level_index == 0:
            provider = self._source
        else:
            provider = parent_plan.provider
        return read_block_or_pad(
            provider,
            tile_row,
            tile_col,
            resolution_level=source_resolution_level,
            tile_height=parent_plan.num_pixels_per_block_vertical,
            tile_width=parent_plan.num_pixels_per_block_horizontal,
            num_bands=int(self._source.num_bands),
            pad_value=_source_pad_value(self._source),
            dtype=_source_dtype(self._source),
        )

    # ------------------------------------------------------------------
    # Cascade core
    # ------------------------------------------------------------------

    def _cascade_into_level(
        self,
        level: int,
        tile_row: int,
        tile_col: int,
        quadrant: Tuple[int, int],
        quadrant_data: NDArray,
        set_block_fn: Callable[[Any, int, int, NDArray], None],
    ) -> None:
        """Place ``quadrant_data`` into level ``level``'s accumulator
        entry at ``(tile_row, tile_col)`` and, if that entry becomes
        complete, resample it, write it back, and recursively cascade
        into the next level.

        ``quadrant_data`` is never mutated — it is copied into the
        accumulator buffer. The cascade terminates when either the
        resulting tile's level is beyond the highest overview level or
        the parent tile remains incomplete.

        Args:
            level: 1-based overview level index this quadrant belongs
                to.
            tile_row: Row in the level's tile grid for the target tile.
            tile_col: Column in the level's tile grid for the target
                tile.
            quadrant: ``(quadrant_row, quadrant_col)`` in ``{0, 1}``.
            quadrant_data: The CHW block forming one of the four
                quadrants of the target tile.
            set_block_fn: Callable used to write the resampled tile to
                its ``BufferedImageAssetProvider``. Accepts
                ``(provider, tile_row, tile_col, data)``.
        """
        plan = self._levels[level]
        entry_key = (tile_row, tile_col)
        accumulator = self._accumulators[level]

        parent_plan = self._levels[level - 1]
        expected_mask = _expected_quadrant_mask(parent_plan, tile_row, tile_col)

        if entry_key not in accumulator:
            buffer = self._allocate_tile_buffer(plan)
            accumulator[entry_key] = (buffer, 0)

        buffer, mask = accumulator[entry_key]
        _place_quadrant(buffer, quadrant, quadrant_data, plan)
        mask |= _quadrant_bit(quadrant)
        accumulator[entry_key] = (buffer, mask)

        if mask != expected_mask:
            return

        # Tile complete: optionally extend with a halo, resample, write
        # back, and propagate upward.
        src_image = self._maybe_apply_halo(buffer, parent_plan, tile_row, tile_col)

        target_rows = min(
            plan.num_pixels_per_block_vertical,
            plan.num_rows - tile_row * plan.num_pixels_per_block_vertical,
        )
        target_cols = min(
            plan.num_pixels_per_block_horizontal,
            plan.num_columns - tile_col * plan.num_pixels_per_block_horizontal,
        )

        resampled = self._resample_with_optional_trim(
            src_image=src_image,
            buffer=buffer,
            plan=plan,
            parent_plan=parent_plan,
            tile_row=tile_row,
            tile_col=tile_col,
            target_rows=target_rows,
            target_cols=target_cols,
        )

        # The set_block Rust binding expects the block-sized array for
        # interior tiles but the edge-clipped array for boundary tiles
        # (it stores the raw bytes and reshapes on read based on the
        # computed in-bounds dimensions). We therefore trim the
        # resampled tile to its in-bounds footprint — no padding.
        provider_tile = _trim_to_image_bounds(
            resampled,
            plan=plan,
            tile_row=tile_row,
            tile_col=tile_col,
        )

        set_block_fn(plan.provider, tile_row, tile_col, provider_tile)

        # Free the accumulator entry now that the tile is written back.
        del accumulator[entry_key]

        # Propagate upward if there's a higher level to feed.
        if level + 1 < len(self._levels):
            self._cascade_into_level(
                level=level + 1,
                tile_row=tile_row // 2,
                tile_col=tile_col // 2,
                quadrant=(tile_row % 2, tile_col % 2),
                # Use the untrimmed resampled tile as the quadrant data
                # for the next level; the parent-block trim will happen
                # naturally as expected_mask accounts for edge tiles.
                quadrant_data=resampled,
                set_block_fn=set_block_fn,
            )

    def _resample_with_optional_trim(
        self,
        src_image: NDArray,
        buffer: NDArray,
        plan: "_LevelPlan",
        parent_plan: "_LevelPlan",
        tile_row: int,
        tile_col: int,
        target_rows: int,
        target_cols: int,
    ) -> NDArray:
        """Run ``resample_func`` on ``src_image`` and trim the result to
        the tile's expected output size.

        ``src_image`` may be the raw buffer (no halo) or a halo-extended
        patch returned by :meth:`_maybe_apply_halo`. When a halo was
        applied the resampler output is trimmed by ``halo // 2`` on each
        edge (the resampler halves the input dimensions).
        """
        halo = _halo_pixels(self._resample_func)
        if halo == 0 or src_image is buffer:
            # No halo path — resample the buffer directly. The buffer
            # may itself represent a smaller edge tile when the parent
            # tile's footprint is incomplete; in that case we have
            # already populated only the filled region, and resample
            # directly to the target output size.
            src_rows = buffer.shape[1]
            src_cols = buffer.shape[2]
            expected_rows = (src_rows + 1) // 2
            expected_cols = (src_cols + 1) // 2
            if target_rows != expected_rows or target_cols != expected_cols:
                # Edge tile whose footprint inside the parent doesn't
                # line up with the buffer size (e.g. parent tile is
                # smaller than the full block at the image boundary).
                # Trim the buffer to the filled region before
                # resampling.
                filled_rows = (
                    min(
                        buffer.shape[1],
                        parent_plan.num_rows - tile_row * buffer.shape[1],
                    )
                    if parent_plan.level_index == 0
                    else buffer.shape[1]
                )
                filled_cols = (
                    min(
                        buffer.shape[2],
                        parent_plan.num_columns - tile_col * buffer.shape[2],
                    )
                    if parent_plan.level_index == 0
                    else buffer.shape[2]
                )
                trimmed_buffer = buffer[:, :filled_rows, :filled_cols]
                return self._resample_func(trimmed_buffer, target_rows, target_cols)
            return self._resample_func(buffer, expected_rows, expected_cols)

        # Halo path — resample the halo-extended patch, then strip the
        # halo region from the output.
        halo_out = halo // 2
        patch_rows = src_image.shape[1]
        patch_cols = src_image.shape[2]
        resample_rows = (patch_rows + 1) // 2
        resample_cols = (patch_cols + 1) // 2
        resampled = self._resample_func(src_image, resample_rows, resample_cols)
        trimmed = resampled[
            :,
            halo_out : halo_out + target_rows,
            halo_out : halo_out + target_cols,
        ]
        return np.ascontiguousarray(trimmed)

    def _allocate_tile_buffer(self, plan: "_LevelPlan") -> NDArray:
        """Allocate a fresh CHW tile buffer sized for ``plan``'s parent
        tile (i.e. ``2 * tile_height`` x ``2 * tile_width``).

        The buffer size is twice the output tile's size on each axis
        because four R0 quadrants fill it. For edge tiles the unused
        portion stays at zero (we trim before resampling).
        """
        num_bands = int(self._source.num_bands)
        tile_h = plan.num_pixels_per_block_vertical
        tile_w = plan.num_pixels_per_block_horizontal
        dtype = _source_dtype(self._source)
        buffer_rows = tile_h * 2
        buffer_cols = tile_w * 2
        return np.zeros((num_bands, buffer_rows, buffer_cols), dtype=dtype)

    # ------------------------------------------------------------------
    # SIPS halo handling
    # ------------------------------------------------------------------

    def _maybe_apply_halo(
        self,
        buffer: NDArray,
        parent_plan: "_LevelPlan",
        tile_row: int,
        tile_col: int,
    ) -> NDArray:
        """Return a halo-extended patch when ``resample_func`` advertises
        a non-zero halo, otherwise return the buffer unchanged.

        For SIPS RRDS (halo = 5), the returned patch extends 5 pixels on
        every edge beyond the 2x2 group footprint at the parent level.
        Out-of-image pixels are filled via reflection
        (``numpy.pad(mode='reflect')``), matching SIPS Mirror Edge - Odd
        semantics (OpenCV ``BORDER_REFLECT_101``).
        """
        halo = _halo_pixels(self._resample_func)
        if halo == 0:
            return buffer

        # Parent-level image coordinates for this group (in-bounds window).
        parent_tile_h = parent_plan.num_pixels_per_block_vertical
        parent_tile_w = parent_plan.num_pixels_per_block_horizontal
        y0 = tile_row * parent_tile_h * 2
        x0 = tile_col * parent_tile_w * 2
        y1 = y0 + parent_tile_h * 2
        x1 = x0 + parent_tile_w * 2
        # Clip to image bounds.
        img_rows = parent_plan.num_rows
        img_cols = parent_plan.num_columns
        y1 = min(y1, img_rows)
        x1 = min(x1, img_cols)

        # Extended window including halo (may extend beyond image bounds).
        ext_y0 = y0 - halo
        ext_x0 = x0 - halo
        ext_y1 = y1 + halo
        ext_x1 = x1 + halo

        # In-bounds read window.
        read_y0 = max(ext_y0, 0)
        read_x0 = max(ext_x0, 0)
        read_y1 = min(ext_y1, img_rows)
        read_x1 = min(ext_x1, img_cols)

        provider = self._source if parent_plan.level_index == 0 else parent_plan.provider

        patch = stitch_source_blocks(
            provider,
            row_range=(read_y0, read_y1),
            col_range=(read_x0, read_x1),
            tile_height=parent_tile_h,
            tile_width=parent_tile_w,
            num_bands=int(self._source.num_bands),
            pad_value=_source_pad_value(self._source),
            dtype=_source_dtype(self._source),
            resolution_level=0,
        )

        # Reflect-pad the patch to fill halo regions that extend beyond
        # the image boundary.
        pad_top = read_y0 - ext_y0
        pad_bottom = ext_y1 - read_y1
        pad_left = read_x0 - ext_x0
        pad_right = ext_x1 - read_x1
        if pad_top or pad_bottom or pad_left or pad_right:
            patch = np.pad(
                patch,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )
        return patch

    # ------------------------------------------------------------------
    # J2K native fast path
    # ------------------------------------------------------------------

    def _populate_native_levels(self) -> int:
        """Fill overview levels from native source resolution levels.

        Returns the index of the first overview level that must still
        be produced via the accumulator cascade. When no native levels
        are available (or ``use_native_levels`` is False), returns
        ``1`` — i.e. the cascade starts from the first overview level.
        """
        if not self._use_native_levels:
            return 1
        native_levels = int(getattr(self._source, "num_resolution_levels", 1))
        if native_levels <= 1:
            return 1

        # We can populate at most ``native_levels - 1`` overview levels
        # from the source's own resolution levels. We are further
        # bounded by the number of overview levels we actually plan to
        # produce.
        num_overviews = len(self._levels) - 1
        native_overviews = min(native_levels - 1, num_overviews)

        for i in range(1, native_overviews + 1):
            self._populate_native_overview(i)

        return native_overviews + 1

    def _populate_native_overview(self, level_index: int) -> None:
        """Fill overview level ``level_index`` by reading the source at
        native resolution level ``level_index`` and composing the tiles
        via :func:`~aws.osml.image_processing.block_utils.stitch_source_blocks`.

        JPEG 2000 resolution levels return progressively smaller blocks
        while keeping the same block grid as R0. At resolution level N
        each block is ``source_block_size // 2**N`` pixels. To assemble
        a full-sized pyramid tile (e.g. 1024×1024) we must stitch
        ``2**N × 2**N`` source blocks together per output tile.
        """
        plan = self._levels[level_index]
        grid_rows, grid_cols = plan.block_grid_size
        num_bands = int(self._source.num_bands)
        dtype = _source_dtype(self._source)
        pad_value = _source_pad_value(self._source)

        source_block_h = int(getattr(self._source, "num_pixels_per_block_vertical", plan.num_pixels_per_block_vertical))
        source_block_w = int(getattr(self._source, "num_pixels_per_block_horizontal", plan.num_pixels_per_block_horizontal))
        native_block_h = source_block_h // (2**level_index)
        native_block_w = source_block_w // (2**level_index)

        total_tiles = grid_rows * grid_cols
        completed = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                y0 = r * plan.num_pixels_per_block_vertical
                x0 = c * plan.num_pixels_per_block_horizontal
                y1 = min(y0 + plan.num_pixels_per_block_vertical, plan.num_rows)
                x1 = min(x0 + plan.num_pixels_per_block_horizontal, plan.num_columns)
                tile = stitch_source_blocks(
                    self._source,
                    row_range=(y0, y1),
                    col_range=(x0, x1),
                    tile_height=native_block_h,
                    tile_width=native_block_w,
                    num_bands=num_bands,
                    pad_value=pad_value,
                    dtype=dtype,
                    resolution_level=level_index,
                )
                plan.provider.set_block(r, c, np.ascontiguousarray(tile))
                completed += 1
                if self._progress is not None:
                    self._progress(completed, total_tiles, level_index)


# ======================================================================
# Module-level helpers
# ======================================================================


def _grid_size(num_rows: int, num_columns: int, tile_height: int, tile_width: int) -> Tuple[int, int]:
    """Compute tile grid dimensions via ceiling division.

    Args:
        num_rows: Image rows.
        num_columns: Image columns.
        tile_height: Tile height in pixels.
        tile_width: Tile width in pixels.

    Returns:
        ``(grid_rows, grid_cols)`` such that the grid covers every
        pixel of the image.
    """
    grid_rows = (num_rows + tile_height - 1) // tile_height
    grid_cols = (num_columns + tile_width - 1) // tile_width
    return (grid_rows, grid_cols)


def _quadrant_bit(quadrant: Tuple[int, int]) -> int:
    """Return the bitmask bit for a 2x2 quadrant coordinate."""
    qr, qc = quadrant
    return 1 << (qr * 2 + qc)


def _expected_quadrant_mask(parent_plan: "_LevelPlan", tile_row: int, tile_col: int) -> int:
    """Return the bitmask of quadrants expected at ``(tile_row, tile_col)``.

    Interior tiles have all four quadrants (``0b1111``). Edge tiles
    whose 2x2 parent footprint extends beyond the parent level's image
    bounds have a reduced mask.
    """
    parent_rows, parent_cols = parent_plan.block_grid_size
    mask = 0
    for qr in (0, 1):
        for qc in (0, 1):
            parent_r = tile_row * 2 + qr
            parent_c = tile_col * 2 + qc
            if parent_r < parent_rows and parent_c < parent_cols:
                mask |= 1 << (qr * 2 + qc)
    return mask


def _place_quadrant(buffer: NDArray, quadrant: Tuple[int, int], data: NDArray, plan: "_LevelPlan") -> None:
    """Copy ``data`` into the correct quadrant of ``buffer``.

    The buffer is ``(bands, 2*tile_h, 2*tile_w)``. Each quadrant is
    ``tile_h x tile_w`` at its expected position. Edge tiles may be
    smaller than a full tile; only the filled region is copied.
    """
    qr, qc = quadrant
    tile_h = plan.num_pixels_per_block_vertical
    tile_w = plan.num_pixels_per_block_horizontal
    row_off = qr * tile_h
    col_off = qc * tile_w
    data_rows = data.shape[1]
    data_cols = data.shape[2]
    # Guard against oversized data (can happen when source block size
    # doesn't match the plan tile size for some reason).
    copy_rows = min(data_rows, tile_h)
    copy_cols = min(data_cols, tile_w)
    buffer[
        :,
        row_off : row_off + copy_rows,
        col_off : col_off + copy_cols,
    ] = data[:, :copy_rows, :copy_cols]


def _pad_to_block(tile: NDArray, block_height: int, block_width: int, pad_value: float) -> NDArray:
    """Pad a CHW tile up to the full block size, right/bottom only.

    Retained for callers that genuinely need a block-sized array; the
    osml-imagery-io ``BufferedImageAssetProvider`` itself does NOT
    want padding at edge tiles (it returns the in-bounds region on
    read), so most callers should use :func:`_trim_to_image_bounds`
    instead.
    """
    if tile.shape[1] == block_height and tile.shape[2] == block_width:
        return tile
    num_bands = tile.shape[0]
    out = np.full((num_bands, block_height, block_width), pad_value, dtype=tile.dtype)
    out[:, : tile.shape[1], : tile.shape[2]] = tile
    return out


def _trim_to_image_bounds(
    tile: NDArray,
    plan: "_LevelPlan",
    tile_row: int,
    tile_col: int,
) -> NDArray:
    """Trim a CHW tile to its in-bounds footprint at the given level.

    ``BufferedImageAssetProvider.set_block`` stores the raw bytes it
    receives; on read it returns an array of shape
    ``(bands, min(block_h, img_h - row*block_h), min(block_w, img_w - col*block_w))``.
    Passing a larger-than-in-bounds block therefore causes the get
    path to fail with a reshape error. We trim here to match the
    expected in-bounds size exactly.
    """
    in_bounds_h = min(
        plan.num_pixels_per_block_vertical,
        plan.num_rows - tile_row * plan.num_pixels_per_block_vertical,
    )
    in_bounds_w = min(
        plan.num_pixels_per_block_horizontal,
        plan.num_columns - tile_col * plan.num_pixels_per_block_horizontal,
    )
    if tile.shape[1] == in_bounds_h and tile.shape[2] == in_bounds_w:
        return tile
    return tile[:, :in_bounds_h, :in_bounds_w]


def _halo_pixels(resample_func: ResampleFunc) -> int:
    """Halo width required by ``resample_func`` (0 if not required)."""
    if resample_func is sips_rrds_resample:
        return 5
    return int(getattr(resample_func, "halo_pixels", 0))


def _source_dtype(provider: Any) -> np.dtype:
    """Return the numpy dtype associated with ``provider``'s pixel type.

    Supports osml-imagery-io's ``PixelType`` (exposes
    ``to_numpy_dtype()`` returning a string like ``"uint8"``) and plain
    string / dtype descriptors used by duck-typed mocks.
    """
    from .block_utils import _source_dtype as _shared_source_dtype

    return _shared_source_dtype(provider)


def _source_pad_value(provider: Any) -> float:
    """Return ``provider.pad_pixel_value`` or ``0.0`` if unavailable."""
    return float(getattr(provider, "pad_pixel_value", 0.0))


def _direct_set_block(provider: Any, row: int, col: int, data: NDArray) -> None:
    """Single-threaded writeback shim — calls ``set_block`` inline."""
    provider.set_block(row, col, np.ascontiguousarray(data))


def _prefetch_worker(
    parent_plan: "_LevelPlan",
    grid_rows: int,
    grid_cols: int,
    source_resolution_level: int,
    source: Any,  # noqa: ARG001 — kept for symmetry / debugging
    read_fn: Callable[["_LevelPlan", int, int, int], NDArray],
    out_queue: "queue.Queue[Any]",
    error_flag: threading.Event,
) -> None:
    """Background thread: iterate the parent-level tile grid in raster
    order, decode each tile, and push it onto ``out_queue``.

    Always terminates by pushing a single :data:`_SENTINEL`. If
    ``error_flag`` is set partway through (e.g. by the main thread
    capturing a cascade exception), the worker exits early without
    decoding any remaining tiles."""
    try:
        for r in range(grid_rows):
            for c in range(grid_cols):
                if error_flag.is_set():
                    return
                tile = read_fn(parent_plan, r, c, source_resolution_level)
                out_queue.put((r, c, tile))
    finally:
        out_queue.put(_SENTINEL)


def _writeback_worker(
    in_queue: "queue.Queue[Any]",
    error_flag: threading.Event,
) -> None:
    """Background thread: drain ``in_queue`` and call ``set_block`` on
    each pending ``(provider, row, col, data)`` tuple.

    Exits cleanly on :data:`_SENTINEL`. Any raised exception is
    re-raised on the main thread via ``Future.result()`` after the main
    loop returns."""
    while True:
        item = in_queue.get()
        if item is _SENTINEL:
            return
        provider, row, col, data = item
        try:
            provider.set_block(row, col, data)
        except BaseException:
            # Set the error flag so the prefetch worker can exit
            # promptly, then re-raise on this worker's Future.
            error_flag.set()
            raise


def _drain_queue(q: "queue.Queue[Any]") -> None:
    """Discard every pending item in ``q`` (non-blocking)."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _apply_writer_metadata(writer: Any, file_metadata: Any) -> None:
    """Attach dataset-level metadata to the writer.

    Tries the attribute setter ``writer.metadata = file_metadata``
    first (the osml-imagery-io Rust binding uses a Python setter); if
    that fails, falls back to a ``set_metadata`` method when present.
    """
    try:
        writer.metadata = file_metadata
        return
    except AttributeError:
        pass
    setter = getattr(writer, "set_metadata", None)
    if setter is not None:
        setter(file_metadata)
