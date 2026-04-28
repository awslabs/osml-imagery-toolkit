# Tiling and Caching

Processing pipelines need a consistent tile grid — the pyramid builder
expects power-of-two aligned blocks, the warp engine needs manageable
chunks, and display chains want to decode only what the viewport
requires. Source images rarely cooperate: a TIFF may use 256x256 tiles
or full-width strips, a JPEG 2000 file may encode the entire image as a
single codestream block, and a NITF may tile at dimensions chosen for
compression efficiency rather than processing convenience.

The toolkit provides a retiling adapter that presents any source as a
uniform virtual tile grid, and a shared byte-budget cache that
eliminates redundant decoding across operators.

## TileCache

`TileCache` is a thread-safe LRU (least-recently-used) cache with a
byte budget. When the total size of cached arrays exceeds `max_bytes`,
the least-recently-accessed entries are evicted until the budget is
satisfied.

```python
from aws.osml.image_processing import TileCache

cache = TileCache(max_bytes=512 * 1024**2)  # 512 MiB budget
```

### Eviction and Sizing

The cache is backed by `cachetools.LRUCache` configured with a
`getsizeof` function that reads the `.nbytes` property on each cached
numpy array. This means eviction decisions are based on the actual
memory footprint of each tile — a 3-band uint8 tile at 1024x1024
occupies 3 MiB, while a 4-band float32 tile at the same dimensions
occupies 16 MiB. Tiles whose `nbytes` exceeds `max_bytes` are silently
not cached (the cache cannot hold them regardless of eviction) — caching is an optimization, not a
correctness requirement.

### Frozen Arrays

Arrays are marked read-only (`writeable=False`) on insertion. Cache
hits return the same array object (zero-copy), so multiple consumers
reading the same tile share one allocation. Consumers that need to
mutate a tile must call `.copy()` explicitly.

### Shared Budget Across Operators

A single `TileCache` instance is designed to be shared across every
operator in a processing chain — the retiled source, the display chain,
the pyramid builder, and the warp engine can all use one cache. This
gives you a single knob (`max_bytes`) to bound the total decoded-tile
memory for an entire pipeline. You can choose to create new tile caches and manage them independently. This is sometimes useful if you want to protect the initial image download/decode of source tiles independent of the intermediate tiles produced by a processing chain.

Sharing works because each operator's cache entries are keyed by a
tuple of `(provider_key, row, col, resolution_level, bands)`. The
`provider_key` component is a string that each operator constructs to
be chain-unique:

- A source asset from `osml-imagery-io` uses its dataset key
  (e.g. `"image:0"`).
- `RetiledImageProvider` appends `:retiled:<width>x<height>`.
- `MappedImageProvider` appends `:mapped:<name>`.
- `DownsampledImageProvider` appends `:downsample:<scale-factor>`.

Because keys are hierarchical and include the operator's identity, two
operators sharing the same `TileCache` will never collide — each sees
only its own entries while the LRU policy manages the combined budget.

### Why a Toolkit-Level Cache?

The codec layer (`osml-imagery-io`) is a Rust library with Python
bindings. Each call to `get_block()` decodes compressed data on the
native side and returns a new numpy array to the Python process.
Without caching, every repeated access to the same tile — whether from
overlapping retiling, the pyramid builder reading parent blocks, or a
warp engine hitting the same source region from adjacent output tiles —
pays the full cost of decompression and a Rust-to-Python memory copy.

`TileCache` stores decoded numpy arrays that live directly in the
Python process's heap. Subsequent reads are a dict lookup and a
pointer return — no decompression, no cross-language data transfer.
This is especially significant for JPEG 2000 sources where
decompression is computationally expensive.

## CachedImageProvider

`CachedImageProvider` is a transparent wrapper that interposes a
`TileCache` in front of any source's `get_block()` method. It
delegates all properties (dimensions, bands, metadata) unchanged — the
rest of the pipeline cannot distinguish it from the unwrapped source.

When `cache=None` is passed, the wrapper becomes a no-op pass-through
with negligible overhead.

## RetiledImageProvider

`RetiledImageProvider` presents a virtual tile grid of configurable
dimensions over any source. When the source's physical blocks are
larger than the requested tile size, virtual tiles are sliced from
source blocks. When they are smaller, multiple source blocks are
stitched together to fill the virtual tile.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_width` | `1024` | Virtual tile width in pixels |
| `tile_height` | `1024` | Virtual tile height in pixels |
| `pad_edges` | `False` | Pad edge tiles to full dimensions using the source's pad pixel value |
| `cache` | `None` | Optional shared `TileCache` for output caching |

## Example: Retiled Source with Shared Cache

```python
from aws.osml.io import IO
from aws.osml.image_processing import (
    CachedImageProvider,
    DisplayChainFactory,
    MappedImageProvider,
    RetiledImageProvider,
    TileCache,
)

# One cache for the entire pipeline — 512 MiB total budget
cache = TileCache(max_bytes=512 * 1024**2)

with IO.open("large_image.ntf", "r") as reader:
    source = reader.get_asset("image:0")

    # Cache decoded source blocks to avoid redundant decompression
    cached = CachedImageProvider(source, cache=cache)

    # Present a uniform 1024x1024 grid regardless of native block layout
    retiled = RetiledImageProvider(cached, tile_width=1024, tile_height=1024, cache=cache)

    # Build a display chain — its output tiles also share the cache
    chain = DisplayChainFactory.build(retiled)
    display = MappedImageProvider(
        retiled, chain,
        source_bands=chain.input_bands,
        num_bands=chain.output_bands,
        cache=cache,
    )

    # First access decodes and caches; subsequent reads are instant
    tile_a = display.get_block(0, 0)
    tile_b = display.get_block(0, 0)  # cache hit — same object returned
```

In this example, `cache` holds entries from three layers:

- Source blocks keyed as `("image:0", row, col, ...)`
- Retiled blocks keyed as `("image:0:retiled:1024x1024", row, col, ...)`
- Display-mapped blocks keyed as `("image:0:retiled:1024x1024:mapped:...", row, col, ...)`

All share a single 512 MiB budget, and the LRU policy automatically
evicts the coldest entries regardless of which layer produced them.

## Related Pages

```{seealso}
- [Image Pyramids](image-pyramids.md) — the pyramid builder uses
  `RetiledImageProvider` internally to normalize source tile grids before
  building overviews.
- [Display Processing](display-processing.md) — `MappedImageProvider`
  accepts a `TileCache` to avoid redundant processing of repeated block
  requests.
- [Image Warping](image-warping.md) — `WarpedImageProvider` accepts a
  `TileCache` to avoid recomputing warp grids for repeated tile accesses.
```
