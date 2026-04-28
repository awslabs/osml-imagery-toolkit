# Sensor Models & Geolocation

## What Is a Sensor Model?

A satellite or airborne sensor captures a 2D image of the 3D world.
The relationship between a pixel in the image and a point on the
Earth's surface depends on the geometry of the collection — the
sensor's position and orientation in space, the optics projecting
ground points onto the focal plane, and the shape of the Earth beneath.

A **sensor model** encodes this relationship mathematically. It
exposes two complementary operations:

- **`image_to_world`** — given a pixel location, determine the
  geographic position on the ground.
- **`world_to_image`** — given a geographic position, determine where
  it falls in the image.

These transforms are the foundation of all geospatial analysis:
geolocating the results from ML models, projecting map overlays
onto imagery, cutting chips around features of interest, and
orthorectifying images.

The toolkit represents these concepts with three core classes:

- **`SensorModel`** — the abstract interface that all sensor model
  implementations share. Provides `image_to_world` and
  `world_to_image` methods.
- **`ImageCoordinate`** — a pixel position in the image, constructed
  as `ImageCoordinate([column, row])`. The upper-left corner of the
  upper-left pixel is `(0.0, 0.0)`.
- **`GeodeticWorldCoordinate`** — a geographic position on the Earth,
  constructed as `GeodeticWorldCoordinate([longitude, latitude,
  elevation])`. 
  
```{important}
Longitude and latitude are in **radians** (not
degrees), and elevation is meters above the WGS84 ellipsoid. Note
the (lon, lat) order — this matches the mathematical (x, y, z)
convention and standards like [RFC 7946 (GeoJSON)](https://datatracker.ietf.org/doc/html/rfc7946) but differs from the "lat/lon" order used by many tools. See [Elevation Models](elevation.md) for more details on the height above ellipsoid and geoid corrections.
```

## Quick Start

```python
from math import radians

from aws.osml.io import IO
from aws.osml.metadata import load_sensor_model
from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate

with IO.open("image.ntf", "r") as dataset:
    sensor_model = load_sensor_model(dataset)

    # Geolocate the image center
    image = dataset.get_asset("image:0")
    center = ImageCoordinate([image.num_columns / 2, image.num_rows / 2])
    world = sensor_model.image_to_world(center)

    print(f"Center: {world:%l %o}")                      # decimal degrees
    print(f"Center: {world:%ld%lm%ls%lH %od%om%os%oH}")  # DMS

    # Project a known location back into the image
    point = GeodeticWorldCoordinate([
        radians(-77.404453),  # longitude
        radians(38.954831),   # latitude
        100.0                 # elevation (meters above WGS84 ellipsoid)
    ])
    pixel = sensor_model.world_to_image(point)
    print(f"Pixel: ({pixel.x:.1f}, {pixel.y:.1f})")
```

`load_sensor_model` examines the dataset's TREs, DES XML segments,
and GeoTIFF tags, then constructs the best available model
automatically — including ICHIPB chip correction when present.

## Building a Sensor Model

### From a Dataset (Recommended)

The simplest path uses `load_sensor_model`, which extracts metadata
from a dataset reader and builds the best available model:

```python
from aws.osml.io import IO
from aws.osml.metadata import load_sensor_model

with IO.open("image.ntf", "r") as dataset:
    sensor_model = load_sensor_model(dataset)
    print(type(sensor_model).__name__)
```

This handles TRE extraction, DES XML parsing, GeoTIFF transforms, and
ICHIPB chip correction automatically. When the image is a chip
(sub-region of a larger image), the NITF `ICHIPB` TRE records the
relationship between chip pixels and the full image. If this metadata
is present, the factory wraps the sensor model in a
`ChippedImageSensorModel` that maps chip coordinates to full-image
coordinates before applying the underlying sensor model — no extra
handling is needed on your part.

### From Metadata Dicts

If you already have parsed metadata (e.g., from a custom pipeline),
construct the factory directly:

```python
from aws.osml.metadata import SensorModelFactory

sensor_model = SensorModelFactory(
    actual_image_width=width,
    actual_image_height=height,
    tre_dicts=tre_dicts,           # dict of TRE name -> field dict
    des_xml_strings=des_xml_list,  # list of SICD/SIDD XML strings
    geo_transform=geo_transform,   # 6-coefficient affine [a, b, c, d, e, f]
    proj_wkt=proj_wkt,             # coordinate system WKT string
).build()
```

The factory tries models in priority order (RSM > RPC > SICD/SIDD >
Projective > Affine) and returns the best available, or `None` if no
model can be constructed.

### Restricting Model Types

You can limit which model types the factory attempts:

```python
from aws.osml.metadata import SensorModelFactory, SensorModelTypes

# Only attempt RPC — skip RSM even if metadata is present
sensor_model = SensorModelFactory(
    actual_image_width=width,
    actual_image_height=height,
    tre_dicts=tre_dicts,
    selected_sensor_model_types=[SensorModelTypes.RPC],
).build()
```

## Using Elevation Data

A sensor model maps 3D world points (longitude, latitude, elevation)
to 2D pixel coordinates, but `image_to_world` must invert that
mapping — recovering three unknowns from only two measurements. This
is underdetermined without an additional constraint: the elevation of
the ground point. By default, `image_to_world` assumes the point lies
on the WGS84 ellipsoid (elevation = 0). For terrain with significant
relief, providing an `ElevationModel` supplies the missing constraint
and reduces horizontal error — especially at high off-nadir angles:

```python
from aws.osml.photogrammetry import (
    DefaultedSensorModel,
    DigitalElevationModel,
    GenericDEMTileSet,
    ImageCoordinate,
)

elevation_model = DigitalElevationModel(
    GenericDEMTileSet(format_spec="dted/%oh%od/%lh%ld.dt2"),
    tile_factory,
)

# Pass elevation_model on each call
world = sensor_model.image_to_world(
    ImageCoordinate([col, row]),
    elevation_model=elevation_model,
)

# Or wrap the sensor model so elevation is applied automatically
defaulted = DefaultedSensorModel(
    inner_sensor_model=sensor_model,
    elevation_model=elevation_model,
)
world = defaulted.image_to_world(ImageCoordinate([col, row]))
```

```{seealso}
[Elevation Models](elevation.md) for details on configuring DEM-based
terrain correction, geoid offsets, and multi-resolution elevation
strategies.
```

## Sensor Model Types

The models implemented in this toolkit are *replacement sensor models*
— mathematical approximations that reproduce the mapping of a full
physical sensor model without requiring knowledge of the sensor's
internal geometry. They are fitted to the physical model (or ground
control points) and distributed as metadata alongside the image,
allowing geolocation without access to the original sensor
calibration. The toolkit supports several types, selected
automatically based on available metadata:

| Model Type | Class | Metadata Source | Typical Use |
|-----------|-------|----------------|-------------|
| RSM | `RSMPolynomialSensorModel` | NITF RSM TREs | Modern high-resolution satellite sensors |
| RPC | `RPCSensorModel` | NITF RPC00B TRE | Most commercial satellite imagery |
| SICD | `SICDSensorModel` | NITF XML DES | SAR complex imagery |
| SIDD | (via SIDD builder) | NITF XML DES | SAR derived products |

The toolkit also provides approximate coordinate transforms for
imagery that has already been orthorectified or that lacks the
metadata needed for a rigorous sensor model:

| Model Type | Class | Metadata Source | Typical Use |
|-----------|-------|----------------|-------------|
| Projective | `ProjectiveSensorModel` | NITF CSCRNA TRE or GCPs | Corner-based approximation |
| Affine | `AffineSensorModel` | GeoTIFF geo transform | Orthorectified (map-projected) imagery |

The factory tries models in priority order: RSM > RPC > SICD/SIDD >
Projective > Affine. It returns the best available model, or `None`
if no model can be constructed.

### The Composite Sensor Model

When multiple models are available — typically a precision model (RSM
or RPC) and an approximate model (Projective from corner coordinates)
— the factory returns a `CompositeSensorModel` that combines both.

**Why combine models?** The precision models (RSM, RPC) use iterative
numerical optimization for `image_to_world`. The optimizer needs a
good initial guess and a bounded search region to converge quickly and
correctly. The approximate model provides that initial guess with
minimal computation.

The composite works as follows:

1. **`image_to_world`** — calls the approximate model first to get an
   initial geographic estimate, then passes that estimate to the
   precision model as `initial_guess` in the options dict. The
   precision model refines it to sub-pixel accuracy.
2. **`world_to_image`** — delegates directly to the precision model.
   This direction is a direct evaluation (no iteration needed for RPC,
   and fast for RSM), so the approximate model adds no value.

```python
from aws.osml.photogrammetry import CompositeSensorModel

# The factory builds this automatically, but you can construct it manually:
composite = CompositeSensorModel(
    approximate_sensor_model=projective_model,  # fast initial guess
    precision_sensor_model=rpc_model,           # accurate refinement
)

# Usage is identical to any other SensorModel
world = composite.image_to_world(ImageCoordinate([500, 300]))
pixel = composite.world_to_image(world)
```

```{tip}
You rarely need to construct a `CompositeSensorModel` yourself —
`load_sensor_model()` and `SensorModelFactory` do this automatically
when both an approximate and precision model can be built from the
available metadata.
```



