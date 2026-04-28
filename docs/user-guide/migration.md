# Migrating from v1 (GDAL)

Version 2.0 replaces GDAL with `osml-imagery-io` — a Rust-based image
codec library with Python bindings (PyO3). This eliminates the most
painful operational burden of v1: the system-level GDAL dependency and
its cascading native library requirements (libgdal, libproj, libtiff,
openjpeg, HDF5, etc.).

The motivations behind this rewrite:

- **Pure pip install** — no conda, no system packages, no Docker build
  gymnastics. `pip install osml-imagery-toolkit` works everywhere.
- **Performance** — Rust-native NITF/GeoTIFF/JPEG2000 decoding with
  zero-copy handoff to NumPy via PyO3.
- **Reproducibility** — self-contained wheels with no version-sensitive
  native library coupling.

If you have existing code built against the v1 `aws.osml.gdal` package
or the GDAL-based `GDALTileFactory`, this guide maps every v1 entry
point to its v2 equivalent.

## Key Conceptual Changes

Before diving into the API tables, understand the four paradigm shifts
that affect nearly every call site:

### 1. Array Axis Order: HWB to CHW

v1 used GDAL's `ReadAsArray()` which returns **(height, width, bands)**
or **(height, width)** for single-band. v2 uses the deep-learning
convention **(bands, height, width)** — also called CHW (channels,
height, width).

```python
# v1 — HWB
pixels = band.ReadAsArray()        # shape: (H, W)
pixels = ds.ReadAsArray()          # shape: (H, W, B) or (B, H, W) depending on interleave

# v2 — always CHW
tile = asset.get_block(0, 0)       # shape: (B, H, W)
full = imread("image.ntf")        # shape: (B, H, W)
```

### 2. GDAL Dataset to IO Reader + Assets

v1 exposed a monolithic `gdal.Dataset` object containing bands,
metadata, and overviews. v2 separates concerns: an `IO.open()` call
returns a `DatasetReader` that provides typed *assets* (image segments,
DES segments) and structured metadata.

```python
# v1
ds = gdal.Open("image.ntf")
width = ds.RasterXSize
meta = ds.GetMetadata("xml:DES")

# v2
with IO.open("image.ntf", "r") as reader:
    asset = reader.get_asset("image:0")
    width = asset.num_columns
    meta = asset.metadata  # dict keyed by TRE name or TIFF tag ID
```

### 3. Block-Oriented Access (No Arbitrary Windowed Reads)

GDAL allows arbitrary `ReadAsArray(x, y, w, h)` on any region. The
Rust codec layer is block-oriented — you call `get_block(row, col)` to
read a physical tile. For arbitrary windows, use the toolkit's
`read_window()` helper which stitches blocks internally.

```python
# v1 — arbitrary window
pixels = band.ReadAsArray(x_off=100, y_off=200, xsize=512, ysize=512)

# v2 — block-based with helper for arbitrary windows
from aws.osml.image_processing import read_window
pixels = read_window(asset, x=100, y=200, width=512, height=512)
```

### 4. Metadata as Dicts (Not GDAL Domains)

GDAL organizes metadata into string-valued "domains"
(`ds.GetMetadata("TRE")`, `ds.GetMetadata("xml:DES")`). v2 provides
metadata as Python dicts keyed by TRE name (NITF) or numeric tag ID
(TIFF). Values are already parsed — no manual string splitting needed.

```python
# v1
igeolo = ds.GetMetadata("TRE")["IGEOLO"]  # raw string

# v2
meta = asset.metadata
ichipb = meta.get("ICHIPB")  # already a dict with named fields
```

## Before / After Code Snippets

### Open and Read Pixels

```python
# === v1 (GDAL) ===
from osgeo import gdal
from aws.osml.gdal import load_gdal_dataset

ds, sensor_model = load_gdal_dataset("image.ntf")
band = ds.GetRasterBand(1)
pixels = band.ReadAsArray()  # shape (H, W)
```

```python
# === v2 (osml-imagery-io) ===
from aws.osml.io import IO, imread
from aws.osml.metadata import load_sensor_model

# Quick full-image read
pixels = imread("image.ntf")  # shape (B, H, W)

# Structured access with sensor model
with IO.open("image.ntf", "r") as reader:
    sensor_model = load_sensor_model(reader)
    asset = reader.get_asset("image:0")
    tile = asset.get_block(0, 0)  # one physical tile, shape (B, H, W)
```

See [Getting Started](getting-started.md) for a full guided tour.

### Chip Extraction

```python
# === v1 (GDAL) ===
from aws.osml.gdal import GDALImageFormats, load_gdal_dataset
from aws.osml.image_processing import GDALTileFactory

ds, sensor_model = load_gdal_dataset("image.ntf")
factory = GDALTileFactory(ds, sensor_model, tile_format=GDALImageFormats.NITF)
chip_bytes = factory.create_encoded_tile([0, 0, 1024, 1024])
```

```python
# === v2 ===
from aws.osml.io import IO
from aws.osml.image_processing import ChipFactory, TiledImagePyramid, PixelWindow
from aws.osml.metadata import load_sensor_model

with IO.open("image.ntf", "r") as reader:
    sensor_model = load_sensor_model(reader)
    pyramid = TiledImagePyramid.from_dataset(reader)
    factory = ChipFactory(source=pyramid, sensor_model=sensor_model, output_format="nitf")
    chip_bytes = factory.create_chip(PixelWindow(0, 0, 1024, 1024))
```

See [Image Chipping](chip-factory.md) for display chips, scaling, and
format options.

### Orthorectification

```python
# === v1 (GDAL) ===
from aws.osml.gdal import load_gdal_dataset
from aws.osml.image_processing import GDALTileFactory

ds, sensor_model = load_gdal_dataset("image.ntf")
factory = GDALTileFactory(ds, sensor_model)
ortho_bytes = factory.create_orthophoto_tile(geo_bbox, output_size=(256, 256))
```

```python
# === v2 ===
from aws.osml.io import IO
from aws.osml.image_processing import (
    ChipFactory,
    MapTileSetFactory,
    MapTileId,
    OrthoGridBuilder,
    TiledImagePyramid,
    WarpedImageProvider,
)
from aws.osml.metadata import load_sensor_model

with IO.open("image.ntf", "r") as reader:
    sensor_model = load_sensor_model(reader)
    pyramid = TiledImagePyramid.from_dataset(reader)

    tile_set = MapTileSetFactory.get_for_id("WebMercatorQuad")
    grid_builder = OrthoGridBuilder(
        tile_set=tile_set,
        tile_matrix=16,
        sensor_model=sensor_model,
        source_width=pyramid.get_level(0).num_columns,
        source_height=pyramid.get_level(0).num_rows,
    )
    warped = WarpedImageProvider(pyramid, grid_builder)
    ortho_tile = warped.get_block(0, 0)  # CHW ndarray
```

See [Image Warping](image-warping.md) for the full warping pipeline
including elevation correction and occlusion handling.

## API Mapping Tables

### Image I/O

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| `load_gdal_dataset(path)` | `IO.open(path, "r")` | See [Getting Started](getting-started.md) |
| `gdal.Open(path)` | `IO.open(path, "r")` | Returns a context-managed `DatasetReader` |
| `ds.GetRasterBand(n).ReadAsArray()` | `asset.get_block(row, col)` or `imread(path)` | See [Image Pyramids](image-pyramids.md) |
| `gdal.Translate(dst, src, ...)` | `imsave(path, array)` | See [Getting Started](getting-started.md) |
| `ds.GetMetadata("TRE")` | `asset.metadata` (dict) | Keys are TRE names or TIFF tag IDs |
| `ds.GetMetadata("xml:DES")` | `reader.get_asset("des:0")` | DES segments are separate assets |

### Sensor Models

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| `aws.osml.gdal.SensorModelFactory(ds)` | `load_sensor_model(reader)` | See [Sensor Models](photogrammetry.md) |
| `sensor_model_factory.build()` | `load_sensor_model(reader)` | One-call convenience function |
| `aws.osml.gdal.SensorModelFactory` (class) | `aws.osml.metadata.SensorModelFactory` | Same builder pattern, takes dicts instead of GDAL dataset |

### Chipping

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| `GDALTileFactory(ds, format, compression)` | `ChipFactory(source=pyramid, output_format="nitf")` | See [Image Chipping](chip-factory.md) |
| `factory.create_encoded_tile([x, y, w, h])` | `factory.create_chip(PixelWindow(x, y, w, h))` | Named tuple replaces list |
| `factory.create_encoded_tile(..., output_size=(w,h))` | `factory.create_chip(..., output_size=ImageSize(w, h))` | Explicit width/height ordering |
| `GDALImageFormats.NITF` | `"nitf"` | String-based format selection |
| `GDALImageFormats.GTIFF` | `"geotiff"` | String-based format selection |
| `GDALImageFormats.PNG` | `"png"` | String-based format selection |
| `range_adjustment=RangeAdjustmentType.DRA` | `processing_chain=DisplayChainFactory.build(source)` | See [Display Processing](display-processing.md) |

### Orthorectification

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| `factory.create_orthophoto_tile(...)` | `OrthoGridBuilder` + `WarpedImageProvider` + `ChipFactory` | See [Image Warping](image-warping.md) |

### Array Convention

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| HWB (height, width, bands) | **CHW** (bands, height, width) | All providers, all functions |

### Statistics

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| `statistics_from_gdal_metadata(ds)` | `compute_image_statistics(asset)` | See [Display Processing](display-processing.md) |
| `statistics_to_gdal_metadata(ds, stats)` | Store in `ImageStatistics` dataclass | Serialize with `statistics_to_gdal_metadata()` if needed |

### DEM / Elevation

| v1 (GDAL) | v2 | Details |
|-----------|-----|---------|
| `GDALDigitalElevationModelTileFactory(path)` | `StoredDEMTileFactory(path)` with `ElevationModelBuilder` | See [Elevation Models](elevation.md) |
| `GDALOffsetProvider(path)` | `RasterOffsetProvider(path)` | See [Elevation Models](elevation.md) |
| `GDALShapefileQuery(path)` | `ShapefileQuery(path)` | Uses pyshp + Shapely instead of OGR |

## Removed APIs

The following v1 classes and functions have been removed entirely. They
have no direct v2 equivalent because their functionality was either
absorbed into `osml-imagery-io`, replaced by a different architecture,
or eliminated as unnecessary.

| Removed Class/Function | What To Do Instead |
|------------------------|-------------------|
| `GDALConfigEnv` / `set_gdal_default_configuration()` | Not needed. No GDAL configuration to manage. |
| `GDALCompressionOptions` enum | Pass compression as a string to osml-imagery-io's writer options. |
| `GDALImageFormats` enum | Use format strings: `"nitf"`, `"geotiff"`, `"png"`, `"jpeg"`. |
| `RangeAdjustmentType` enum | Use `DisplayChainFactory.build()` with `range_adjustment="dra"` or `"minmax"`. See [Display Processing](display-processing.md). |
| `NITFDESAccessor` | DES segments are exposed as typed assets via `reader.get_asset("des:N")`. XML content is parsed automatically. |
| `ChippedImageInfoFacade` | Chip metadata is now derived by `ChipMetadataBuilder` internally. See [Image Chipping](chip-factory.md). |
| `get_type_and_scales()` | Pixel type detection is handled internally by `DisplayChainFactory`. |
| `get_image_extension()` | Use osml-imagery-io format detection or simply pass the desired output format string. |
| `GDALTileFactory` | Replaced by `ChipFactory` + `TiledImagePyramid`. See [Image Chipping](chip-factory.md). |
| `quarter_power_image()` (standalone) | Use `ComplexRemapFactory.build(source, remap="quarter_power")`. See [Display Processing](display-processing.md). |

```{note}
This list covers the public API of `aws.osml.gdal` as it existed at removal. If you used
internal/private helpers (prefixed with `_`), check the source of the corresponding v2
module for equivalent functionality.
```

## Common Pitfalls

### Array Axis Ordering (HWB to CHW)

The most frequent migration bug. Every function in v2 expects and
returns **(bands, height, width)**. If you have downstream code that
indexes pixels as `image[row, col, band]`, transpose on ingestion:

```python
# Convert CHW to HWB for legacy code (not recommended long-term)
hwb = chw_array.transpose(1, 2, 0)

# Convert HWB to CHW when ingesting external data
chw = hwb_array.transpose(2, 0, 1)
```

```{tip}
Prefer updating downstream code to use CHW indexing
(`image[band, row, col]`) rather than transposing on every read. The
CHW convention is standard in PyTorch, TensorFlow, and most computer
vision pipelines.
```

### No Arbitrary Windowed Reads

GDAL's `ReadAsArray(xoff, yoff, xsize, ysize)` reads any rectangular
region. The Rust codec layer reads physical blocks only. To read an
arbitrary window, use the toolkit's `read_window()` helper:

```python
from aws.osml.image_processing import read_window

# Reads and stitches the necessary blocks internally
pixels = read_window(asset, x=100, y=200, width=512, height=512)
```

For bulk random-access patterns, wrap the asset in a
`CachedImageProvider` to avoid redundant block decodes:

```python
from aws.osml.image_processing import CachedImageProvider, TileCache

cache = TileCache(max_bytes=512 * 1024 * 1024)  # 512 MB budget
cached = CachedImageProvider(asset, cache=cache)
pixels = read_window(cached, x=100, y=200, width=512, height=512)
```

See [Image Pyramids](image-pyramids.md) for details on tiling and
caching strategies.

### Metadata Access Patterns

v1 used GDAL's string-based metadata domains. v2 provides structured
Python dicts directly:

```python
# v1 — parse strings yourself
tre_dict = ds.GetMetadata("TRE")
rpc_str = tre_dict.get("RPC00B")  # raw string needing manual parsing

# v2 — already parsed
meta = asset.metadata
rpc = meta.get("RPC00B")  # dict with named numeric fields
```

Key differences:
- NITF TREs are dicts keyed by TRE name, values are dicts of parsed fields
- GeoTIFF tags are keyed by numeric tag ID (e.g., `33550` for ModelPixelScale)
- DES segments are separate assets, not metadata domains
- No need for `NITFDESAccessor` — XML DES content is accessible via
  `reader.get_asset("des:N")`

### Sensor Model Construction

The v1 pattern of `load_gdal_dataset()` returning both a dataset and
sensor model in one call is replaced by two explicit steps:

```python
# v2 — explicit separation of I/O and model construction
with IO.open("image.ntf", "r") as reader:
    sensor_model = load_sensor_model(reader)  # from aws.osml.metadata
    asset = reader.get_asset("image:0")
    # ... use both independently
```

See [Sensor Models](photogrammetry.md) for the full model hierarchy.

```{seealso}
- [Getting Started](getting-started.md) — installation and guided tour
- [Display Processing](display-processing.md) — DRA, band selection, SAR remap
- [Image Chipping](chip-factory.md) — chip extraction with metadata
- [Image Warping](image-warping.md) — orthorectification pipeline
- [Image Pyramids](image-pyramids.md) — multi-resolution access and caching
```
