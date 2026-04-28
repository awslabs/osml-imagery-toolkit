# OversightML Imagery Toolkit

[![Build](https://github.com/awslabs/osml-imagery-toolkit/actions/workflows/build.yml/badge.svg)](https://github.com/awslabs/osml-imagery-toolkit/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/osml-imagery-toolkit)](https://pypi.org/project/osml-imagery-toolkit/)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/awslabs/osml-imagery-toolkit?color=blue)](LICENSE)

Image processing and photogrammetry for satellite and UAV imagery. Sensor models,
orthorectification, tiling, dynamic range adjustment, and map tile generation — built
on [osml-imagery-io](https://github.com/awslabs/osml-imagery-io) for fast, dependency-light
access to NITF, GeoTIFF, JPEG 2000, and DTED formats.

## What's New in v2

This is version 2 of the OversightML Imagery Toolkit. The major change is the
replacement of GDAL with [osml-imagery-io](https://github.com/awslabs/osml-imagery-io),
a Rust-based image codec library that ships as self-contained Python wheels. This
simplifies the deployment environment — `pip install` is all you need, no Conda managed system dependencies are required.

The GDAL removal required a rewrite of the image processing layer, which previously
relied heavily on `gdal.Translate`. We used that opportunity to:

- **Build new operations from the Softcopy Image Processing Standard (SIPS)** —
  standardized pipelines for dynamic range adjustments and provided integration points to libraries like [SarPy](https://github.com/ngageoint/sarpy) to deliver pixel operators backed by 30+ years of overhead imagery research.
- **Expanded the map tile builders into robust warping operators** — proper
  sensor model–driven back projection using elevation data allows us to create orthophotos and reproject imagery into a target image plane to support change detection pipelines.
- **Modernize the build tooling** — replaced Tox v3 with
  [Hatch](https://hatch.pypa.io/) and adopted [Ruff](https://docs.astral.sh/ruff/)
  for all-in-one formatting, linting, and import sorting.

The sensor and elevation models remain largely unchanged, though their factories were
updated to read metadata through osml-imagery-io and
[PyShp](https://github.com/GeospatialPython/pyshp).

Both the v1 and v2 baselines of this library will be maintained indefinitely; new
feature requests should target v2.

## Packages

Six packages under the `aws.osml` namespace:

- **photogrammetry** — sensor models converting between image (x, y) and geodetic
  (lon, lat, elev) coordinates. Implements RPC, RSM polynomial, SICD, SIDD,
  projective, and affine models.
- **elevation** — loading and composing digital elevation models (DTED, GeoTIFF)
  with support for multi-source priority, spatial conditions, and geoid offset
  correction.
- **metadata** — sensor model builders. `SensorModelFactory` reads metadata from
  dict-based TRE structures via osml-imagery-io.
- **image_processing** — tiling, map tile generation, orthorectification, dynamic
  range adjustment, resampling, pyramid building, SAR complex-to-display conversion,
  and processing chains.
- **formats** — xsdata-generated Python models for SICD (v1.2.1, v1.3.0) and SIDD
  (v1.0, v2.0, v3.0) XML schemas.
- **features** — geospatial feature indexing and property accessors for
  imagery-derived detections.

## Example Usage

```python
from aws.osml.io import IO
from aws.osml.image_processing import (
    DisplayChainFactory, TiledImagePyramid, ChipFactory, PixelWindow, ImageSize
)
from aws.osml.metadata import load_sensor_model

# Open a multi-file NITF R-Set (list of paths, one per resolution level)
with IO.open(["sample.ntf", "sample.ntf.r1", "sample.ntf.r2"], "r") as dataset:
    image = dataset.get_asset("image:0")
    sensor_model = load_sensor_model(dataset)

    # Build a tiled pyramid from the dataset's overview assets
    pyramid = TiledImagePyramid.from_dataset(dataset)
    stats = pyramid.compute_statistics()

    # Build a display processing chain using the statistics
    display_chain = DisplayChainFactory.build(image, stats=stats)

    # Create a chip factory that encodes display-ready PNG chips
    chip_factory = ChipFactory(
        source=pyramid, sensor_model=sensor_model,
        output_format="png", processing_chain=display_chain,
    )

    # Extract a 512x512 PNG-encoded chip from the upper-left corner
    png_bytes = chip_factory.create_chip(PixelWindow(0, 0, 1024, 1024), output_size=ImageSize(512, 512))
```

## Documentation

- **API Reference**: [awslabs.github.io/osml-imagery-toolkit](https://awslabs.github.io/osml-imagery-toolkit/)
- **Examples**: Scripts and notebooks in the [`examples/`](examples/) directory
- **Building docs from source**: `hatch run docs:build`

## Installation

Install from PyPI:

```bash
pip install osml-imagery-toolkit
```

Or install from source:

```bash
pip install .
```

No system libraries, no Conda — just pip. Requires Python 3.10+.

## Development

```bash
# Run tests
hatch run test

# Lint and format
hatch run lint:check
hatch run lint:format

# Build documentation
hatch run docs:build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Security

Please do not open a public GitHub issue to report security concerns. Follow the
reporting mechanisms described in [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## License

This library is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file.
