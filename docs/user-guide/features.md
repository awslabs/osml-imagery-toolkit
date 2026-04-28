# Features & Annotations

Vector data associated with overhead imagery — bounding boxes, polygons,
points, and other geometries — can exist in two coordinate spaces.
**Image annotations** live in pixel coordinates relative to the image
grid: an ML model outputs a bounding box at pixel `(512, 384)`, or an
analyst draws a polygon around a building in row/column space.
**Geospatial features** live in geographic coordinates (longitude,
latitude, elevation) and can be plotted on a map, stored in a GIS
database, or fused across multiple collects.

This package provides utilities for converting between the two:

- **`Geolocator`** converts image annotations into geospatial features
  — assigning WGS-84 coordinates to pixel-space detections using the
  image's sensor model.
- **`Projector`** converts geospatial features into image annotations
  — computing pixel coordinates for known geographic locations so they
  can be rendered as overlays on the imagery.
- **`STRFeature2DSpatialIndex`** provides efficient region queries over
  features in either coordinate space.

Both transforms delegate the actual coordinate math to a
[sensor model](photogrammetry.md). This chapter focuses on the
higher-level workflow: encoding detections, transforming batches of
features, and querying the results spatially.

All input and output uses standard [GeoJSON](https://geojson.org/)
([RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946)), extended
with image coordinate properties described in the
[encoding convention](encoding-image-coordinates-in-geojson) at the
end of this chapter.

## Geolocation (Image → World)

The `Geolocator` reads `imageGeometry` and `imageBBox` from each
feature, transforms the pixel values through the sensor model, and
populates the standard GeoJSON `"geometry"` and `"bbox"` members with
WGS-84 coordinates. It also adds `center_longitude` and
`center_latitude` properties (in degrees).

```python
import geojson
from aws.osml.io import IO
from aws.osml.metadata import load_sensor_model
from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor

detections = [
    geojson.Feature(properties={
        "imageGeometry": {"type": "Point", "coordinates": [512, 384]},
        "imageBBox": [480, 352, 544, 416],
    }),
]

with IO.open("image.ntf", "r") as reader:
    sensor_model = load_sensor_model(reader)

geolocator = Geolocator(
    property_accessor=ImagedFeaturePropertyAccessor(),
    sensor_model=sensor_model,
)
geolocator.geolocate_features(detections)

# Each feature now has standard GeoJSON geometry with WGS-84 coordinates
print(detections[0]["geometry"])
# {"type": "Point", "coordinates": [-77.0365, 38.8977, 0.0]}
```

By default, the `Geolocator` skips features that already have a
non-None `geometry` — avoiding overwrites from a previous geolocation
pass or an external source. Pass `force=True` to re-geolocate all
features regardless.

For batch geolocation the `Geolocator` builds a regular grid of
sensor-model evaluations across the feature extent and uses bivariate
spline interpolation for individual coordinates. This provides sub-pixel
accuracy at a fraction of the cost of per-point evaluation. Increase
`approximation_grid_size` (default 11) for large extents where higher
interpolation fidelity is needed.

Without an elevation model, the geolocator assumes all detections are at
sea level — introducing lateral error proportional to terrain height and
sensor look angle. Supply an elevation model for improved accuracy:

```python
from aws.osml.photogrammetry import ConstantElevationModel

geolocator = Geolocator(
    property_accessor=ImagedFeaturePropertyAccessor(),
    sensor_model=sensor_model,
    elevation_model=ConstantElevationModel(500.0),
)
```

```{seealso}
[Elevation Models](elevation.md) for DTED-based and other terrain
sources. [Sensor Models](photogrammetry.md) for details on the
image-to-world coordinate transform.
```

## Projection (World → Image)

The `Projector` is the inverse operation. Given features with geographic
coordinates — from a GIS database, a reference layer, or a prior
geolocation pass — it computes where they appear in a specific image and
populates `imageGeometry` and `imageBBox` with pixel coordinates. Only
features whose projected geometry intersects the provided image bounds
are included in the result. Use cases include:

- Rendering known annotations as overlays on a new collect
- Pre-filtering a feature database to only features visible in a scene
- Generating training labels by projecting ground truth into image space

```python
import geojson
from aws.osml.io import IO
from aws.osml.metadata import load_sensor_model
from aws.osml.features import Projector, ImagedFeaturePropertyAccessor

features = [
    geojson.Feature(
        geometry=geojson.Point((-77.0365, 38.8977)),
        properties={"label": "building-A"},
    ),
]

with IO.open("image.ntf", "r") as reader:
    sensor_model = load_sensor_model(reader)
    width, height = reader.segments[0].width, reader.segments[0].height

projector = Projector(
    property_accessor=ImagedFeaturePropertyAccessor(),
    sensor_model=sensor_model,
    image_bounds=(0.0, 0.0, float(width), float(height)),
)
visible_features = projector.project_features(features)

for f in visible_features:
    print(f["properties"]["imageGeometry"])
    # {"type": "Point", "coordinates": [512.0, 384.0]}
```

| Parameter | Description |
|-----------|-------------|
| `property_accessor` | Facade for reading/writing image coordinate properties. |
| `sensor_model` | Sensor model used for `world_to_image()` conversion. |
| `image_bounds` | Pixel-space bounding box `(min_x, min_y, max_x, max_y)`. For full images use `(0, 0, width, height)`. Add a buffer (e.g. `(-50, -50, w+50, h+50)`) to include features in a margin around the image. |
| `elevation_model` | Optional; queried for terrain height when a coordinate lacks an explicit Z value. |
| `force` | If `False` (default), features with existing `imageGeometry` are included without re-projection. |

The `Projector` resolves vertex elevation by precedence: an explicit Z
coordinate in the GeoJSON is used first; if absent and an elevation
model is provided, it is queried; otherwise elevation defaults to 0.0 m.

## Spatial Indexing

`STRFeature2DSpatialIndex` provides efficient spatial queries using
Shapely's Sort-Tile-Recursive tree. By default it indexes features by
their `imageGeometry`, so query coordinates are in **pixels**:

```python
import shapely
from aws.osml.features import STRFeature2DSpatialIndex

feature_collection = geojson.FeatureCollection(features)
index = STRFeature2DSpatialIndex(feature_collection)

query_box = shapely.box(400, 300, 600, 500)
results = index.find_intersects(query_box)

nearest = index.find_nearest(shapely.Point(500, 400), max_distance=100)
```

To index by geographic geometries instead (after geolocation), pass
`use_image_geometries=False`. The STR tree construction is O(n log n)
but subsequent queries are O(log n) — build the index once and reuse it
for multiple queries.

(encoding-image-coordinates-in-geojson)=
## Encoding Image Coordinates in GeoJSON

[GeoJSON](https://geojson.org/) does not define a way to represent image
pixel coordinates. This library adopts a convention that stores
image-space locations alongside standard geographic geometry, allowing a
single feature to carry both representations as it moves through the
geolocation or projection workflow.

Features that have not yet been assigned geographic coordinates use a
`null` `"geometry"` (section 3.2 of RFC 7946) and store image-space
locations in `"properties"`:

**`imageGeometry`** — A GeoJSON-like Geometry Object where coordinate
values are in **pixels**. Origin `(0, 0)` is the top-left corner, `x`
increases right, `y` increases down. Coordinates are ordered `(x, y)`.

**`imageBBox`** — An axis-aligned bounding box in pixels:
`[min x, min y, max x, max y]`.

### Supported Geometry Types

| Type | imageGeometry coordinates | Geographic result after geolocation |
|------|---------------------------|-------------------------------------|
| Point | Single `(x, y)` | Single `(lon, lat, elev)` |
| LineString | List of `(x, y)` | List of `(lon, lat, elev)` |
| Polygon | Exterior + optional interior rings | Geographic polygon |
| Multi* / GeometryCollection | Collection of the above | Matching collection type |

### Examples

**Point** — a single detected object with bounding box:

```json
{
    "type": "Feature",
    "geometry": null,
    "properties": {
        "imageGeometry": {"type": "Point", "coordinates": [105.0, 5.0]},
        "imageBBox": [100, 0, 110, 10]
    }
}
```

**LineString** — a road segment traced through the image:

```json
{
    "type": "Feature",
    "geometry": null,
    "properties": {
        "imageGeometry": {
            "type": "LineString",
            "coordinates": [[170.0, 45.0], [180.0, 47.0], [182.0, 49.0]]
        }
    }
}
```

**Polygon** — a building footprint (first coordinate repeated to close
the ring):

```json
{
    "type": "Feature",
    "geometry": null,
    "properties": {
        "imageGeometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 2], [10, 12], [0, 12], [0, 0]]]
        },
        "imageBBox": [0, 0, 10, 12]
    }
}
```

### Deprecated Property Formats

Earlier versions used different property names. The library still reads
these for backwards compatibility, but new code should use
`imageGeometry` and `imageBBox` exclusively.

| Property | Format | Status |
|----------|--------|--------|
| `imageGeometry` | GeoJSON geometry object | Preferred |
| `imageBBox` | `[minx, miny, maxx, maxy]` | Preferred |
| `geom_imcoords` | Coordinate list | Deprecated |
| `bounds_imcoords` | `[minx, miny, maxx, maxy]` | Deprecated |
| `detection.pixelCoordinates` | GeoJSON-like | Deprecated |
