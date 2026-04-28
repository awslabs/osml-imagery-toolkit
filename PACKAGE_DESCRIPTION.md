The OversightML Imagery Toolkit is a Python package for image processing and photogrammetry
on satellite and UAV imagery. It provides sensor models, orthorectification, tiling, dynamic
range adjustment, and map tile generation — built on
[osml-imagery-io](https://github.com/awslabs/osml-imagery-io) for fast, dependency-light
access to NITF, GeoTIFF, JPEG 2000, and DTED formats.

This library contains six packages under the `aws.osml` namespace:

* **photogrammetry**: sensor models converting between image (x, y) and geodetic (lon, lat, elev) coordinates
* **elevation**: loading and composing digital elevation models (DTED, GeoTIFF) with multi-source priority, spatial conditions, and geoid offset correction
* **metadata**: sensor model builders reading metadata through osml-imagery-io
* **image_processing**: tiling, map tile generation, orthorectification, dynamic range adjustment, and processing chains
* **formats**: xsdata-generated Python models for SICD and SIDD XML schemas
* **features**: geospatial feature indexing and property accessors for imagery-derived detections

## Installation

```shell
pip install osml-imagery-toolkit
```

No system libraries, no Conda — just pip. Requires Python 3.10+.

## Documentation

* **API Reference**: [awslabs.github.io/osml-imagery-toolkit](https://awslabs.github.io/osml-imagery-toolkit/)
* **Examples**: Scripts and notebooks in the [`examples/`](https://github.com/awslabs/osml-imagery-toolkit/tree/main/examples) directory

## Contributing

We welcome contributions and suggestions. Please see
[CONTRIBUTING.md](https://github.com/awslabs/osml-imagery-toolkit/blob/main/CONTRIBUTING.md)
for details.
