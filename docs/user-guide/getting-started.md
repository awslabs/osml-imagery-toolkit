# Getting Started

The osml-imagery-toolkit is a Python library for processing satellite and
aerial imagery. It provides sensor models, display normalization, image
pyramids, tiling, orthorectification, and utilities for workign with features.

## What This Library Does

Remote sensing imagery arrives as raw sensor measurements — high
bit-depth photon counts, complex-valued radar returns, or tiled mosaics
in vendor-specific formats. Working with this data requires common
utilities that bridge the gap between raw sensor output and usable
products:

- **Sensor models** that relate every pixel to a geographic position on
  the Earth's surface
- **Display processing** that maps high-dynamic-range measurements to
  viewable 8-bit images
- **Multi-resolution pyramids** for efficient zoom-level access to
  gigapixel imagery
- **Chip extraction** for cutting self-contained subsets with correct
  metadata
- **Orthorectification** for projecting imagery onto map-accurate grids
- **Feature geolocation and projection** for converting between pixel
  coordinates and geographic coordinates on detections and annotations

The toolkit is organized into packages that can be used independently or
composed together:

- **`metadata`** — Extracts TREs, GeoKeys, and DES XML from imagery
  files and constructs sensor models automatically.
- **`photogrammetry`** — Implements sensor models (RPC, RSM, SICD,
  SIDD, projective, affine) for pixel-to-world coordinate conversion.
- **`elevation`** — Loads DEM tiles, computes raster offsets, and
  builds elevation models for terrain-corrected geolocation.
- **`image_processing`** — Display chains, chipping, pyramids,
  orthorectification, resampling, and SAR complex-to-display conversion.
- **`formats`** — Auto-generated Python dataclasses for SICD and SIDD
  XML schemas, used internally by metadata parsers.
- **`features`** — Bridges pixel-space and geographic-space features:
  geolocates ML detections to map coordinates and projects known
  geographic annotations into image pixel space for overlay.

## Prerequisites

```{note}
The toolkit requires **Python 3.10 or later** and installs entirely via
pip. No system-level native libraries are needed — the `osml-imagery-io`
dependency ships self-contained binary wheels.
```

## Installation

Install from PyPI:

```bash
pip install osml-imagery-toolkit
```

For development, clone the repository and use
[Hatch](https://hatch.pypa.io/) to manage the environment:

```bash
git clone https://github.com/awslabs/osml-imagery-toolkit.git
cd osml-imagery-toolkit
pip install hatch
hatch env create        # creates the default virtualenv with dev deps
hatch run test          # run the full test suite
hatch run lint:check    # run linting
```

```{tip}
Hatch manages isolated environments per task. Use `hatch run test` for
testing, `hatch run lint:check` for linting, and `hatch run docs:build`
for documentation. Run `hatch env show` to see all available
environments.
```

## Guided Tour

The following snippets illustrate the major capabilities. Each links to
its full documentation page.


### Convert to a Displayable Image

Satellite sensors capture at bit depths (11-16 bits) far exceeding what
a monitor can render, and SAR sensors produce complex-valued I/Q data.
The [display chain](display-processing.md) automatically classifies the
image modality and builds an appropriate pixel processing pipeline that
maps raw measurements to 8-bit RGB output:

```python
from aws.osml.io import IO
from aws.osml.image_processing import DisplayChainFactory, MappedImageProvider

with IO.open("image.ntf", "r") as reader:
    source = reader.get_asset("image:0")
    chain = DisplayChainFactory.build(source)

    display = MappedImageProvider(
        source, chain,
        source_bands=chain.input_bands,
        num_bands=chain.output_bands,
    )
    tile = display.get_block(0, 0)  # uint8 RGB output
```

### Build an Image Pyramid

Large satellite images can exceed 100,000 pixels per side. Serving them
at multiple zoom levels requires pre-computed reduced-resolution
overviews (R-Sets). The [pyramid builder](image-pyramids.md) generates
these in a single pass over the source tiles:

```python
from aws.osml.io import IO
from aws.osml.image_processing import PyramidBuilder

with IO.open("source.tif", "r") as reader:
    source = reader.get_asset("image:0")
    builder = PyramidBuilder(source, min_size=256)

    with IO.open("output.tif", "w", "geotiff") as writer:
        builder.build_and_write(writer, base_key="image:0")
```

### Extract Chips

ML inference pipelines and human review tools need small, encoded image
tiles with correct geospatial metadata. The [chip factory](chip-factory.md)
reads from the pyramid, applies an optional display chain, and encodes
the result with derived metadata:

```python
from aws.osml.io import IO
from aws.osml.image_processing import ChipFactory, TiledImagePyramid, PixelWindow

with IO.open("image.ntf", "r") as reader:
    pyramid = TiledImagePyramid.from_dataset(reader)

    factory = ChipFactory(source=pyramid, output_format="png")
    chip_bytes = factory.create_chip(PixelWindow(0, 0, 512, 512))
```

### Geolocate Pixels

Every pixel in a remote sensing image corresponds to a specific point on
the Earth. The [photogrammetry](photogrammetry.md) package supports RPC,
RSM, SICD, SIDD, projective, and affine sensor models:

```python
from aws.osml.io import IO
from aws.osml.metadata import load_sensor_model
from aws.osml.photogrammetry import ImageCoordinate
from math import degrees

with IO.open("image.ntf", "r") as dataset:
    sensor_model = load_sensor_model(dataset)

    world = sensor_model.image_to_world(ImageCoordinate([512, 384]))
    print(f"{degrees(world.latitude):.6f}N, {degrees(world.longitude):.6f}E")
```

### Orthorectify

Raw satellite imagery contains perspective distortion and terrain
displacement. The [warping engine](image-warping.md) removes these
effects, producing north-up, map-aligned tiles:

```python
from aws.osml.image_processing import (
    MapTileSetFactory, OrthoGridBuilder, WarpedImageProvider, WarpGridOptions,
)

tile_set = MapTileSetFactory.get_for_id("WebMercatorQuad")

grid_builder = OrthoGridBuilder(
    tile_set=tile_set,
    tile_matrix=16,
    sensor_model=sensor_model,
    source_width=source.num_columns,
    source_height=source.num_rows,
    options=WarpGridOptions.TERRAIN_CORRECTED,
    num_source_levels=source_pyramid.num_levels,
)

warped = WarpedImageProvider(source_pyramid, grid_builder)

min_row, min_col, max_row, max_col = grid_builder.tile_limits
for r in range(min_row, max_row + 1):
    for c in range(min_col, max_col + 1):
        ortho_block = warped.get_block(r, c)
```

### Work with Features

The [features](features.md) package converts between pixel-space
annotations and geographic features. Geolocate ML detections to map
coordinates:

```python
from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor

geolocator = Geolocator(
    property_accessor=ImagedFeaturePropertyAccessor(),
    sensor_model=sensor_model,
)
geolocator.geolocate_features(detections)
# detections now have GeoJSON "geometry" with lon/lat coordinates
```

Or project known geographic features into image pixel space for overlay:

```python
from aws.osml.features import Projector, ImagedFeaturePropertyAccessor

projector = Projector(
    property_accessor=ImagedFeaturePropertyAccessor(),
    sensor_model=sensor_model,
    image_bounds=(0.0, 0.0, float(width), float(height)),
)
visible = projector.project_features(reference_features)
# visible features now have "imageGeometry" with pixel coordinates
```

## What's Next

| Page | Capability |
|------|-----------|
| [Sensor Models & Geolocation](photogrammetry.md) | Pixel ↔ geographic coordinate conversion |
| [Elevation Models](elevation.md) | Terrain-aware geolocation using DEMs |
| [Display Processing](display-processing.md) | Raw sensor data → viewable images |
| [Image Pyramids](image-pyramids.md) | Multi-resolution overviews |
| [Image Chipping](chip-factory.md) | Encoded chip extraction with metadata |
| [Image Warping](image-warping.md) | Orthorectification and reprojection |
| [Features & Annotations](features.md) | Geolocation, projection, and spatial indexing of vector data |
