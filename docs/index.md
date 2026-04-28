# osml-imagery-toolkit

Image processing and photogrammetry routines for satellite and UAV imagery analysis.

osml-imagery-toolkit provides sensor model implementations (RPC, RSM, SICD, SIDD,
projective, affine), tile generation with metadata preservation, map tile
orthorectification, dynamic range adjustment, complex SAR visualization, and
spatial feature geolocation. It supports NITF, GeoTIFF, SICD, and SIDD imagery.

## Key Features

- Sensor models for image-to-world and world-to-image coordinate transforms
- Tile factory with format encoding, compression, and metadata updates
- Map tile generation with orthorectification via sensor models
- Dynamic range adjustment and SAR complex data visualization
- Spatial feature indexing and geolocation from image coordinates
- SICD/SIDD XML metadata parsing via xsdata-generated models

```{toctree}
:maxdepth: 2
:caption: Contents

user-guide/index
api/index
design/index
```
