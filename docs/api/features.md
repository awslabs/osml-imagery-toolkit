# aws.osml.features

Working with geospatial features (detections, annotations) derived from imagery. This package
covers converting image-coordinate features to geographic coordinates (geolocation), the inverse
projection (geographic to image), spatial indexing for efficient region queries, and a property
accessor facade for the imaged-feature GeoJSON convention. It does **not** include the sensor
model mathematics themselves (`photogrammetry`) or pixel extraction and display logic
(`image_processing`).

**Dependencies:** Imports from `photogrammetry` (`SensorModel`, `ElevationModel`,
`GeodeticWorldCoordinate`, `ImageCoordinate`). Uses `geojson`, `shapely`, and `scipy`
(for bilinear interpolation). Does not import `image_processing` or `metadata`.

### Design abstractions

The package is organized around a small set of collaborating abstractions:

**Imaged-feature GeoJSON convention** — Features carry image-coordinate geometry in
`properties.imageGeometry` and `properties.imageBBox`. The `Geolocator` populates the standard
GeoJSON `geometry` field by projecting these through a sensor model and elevation model. The
`Projector` performs the inverse operation (geographic to image).

**ImagedFeaturePropertyAccessor** — A facade that encapsulates how image coordinates are encoded
in feature properties, isolating the rest of the package from the specific JSON schema.

**LocationGridInterpolator** — Pre-computes a bilinear interpolation grid of sensor model results
over a tile extent, amortizing the cost of sensor model evaluation and providing O(1) per-feature
geolocation within that tile.

**Feature2DSpatialIndex** (ABC) and **STRFeature2DSpatialIndex** — A query-only spatial index
backed by Shapely's STR-tree. Operates in either image-pixel or geographic coordinate space,
enabling efficient region-based feature retrieval without scanning the full collection.

### Geolocation pipeline

```{mermaid}
flowchart LR
    A[pixel detections<br/>imageGeometry] --> B[Geolocator<br/>sensor model + elevation]
    B --> C[GeoJSON features<br/>geometry]
    C --> D[SpatialIndex]
    D --> E[query results]
```

### Contributor rules

- New query or filter operations belong on the spatial index or in a new class — not inlined at call sites.
- Coordinate transforms must go through `Geolocator` or `Projector`, not raw sensor model calls.
- No pixel-reading or display logic belongs in this package.

## Geolocation

```{eval-rst}
.. autoclass:: aws.osml.features.Geolocator
   :members:
   :undoc-members:
   :show-inheritance:
```

## Spatial Indexing

```{eval-rst}
.. autoclass:: aws.osml.features.Feature2DSpatialIndex
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.features.STRFeature2DSpatialIndex
   :members:
   :undoc-members:
   :show-inheritance:
```

## Property Accessors

```{eval-rst}
.. autoclass:: aws.osml.features.ImagedFeaturePropertyAccessor
   :members:
   :undoc-members:
   :show-inheritance:
```
