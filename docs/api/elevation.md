# aws.osml.elevation

I/O-aware implementations of the elevation model abstractions defined in
{mod}`aws.osml.photogrammetry`. This package reads DEM raster tiles from disk
(StoredDEMTileFactory), derives geo-transforms from raster metadata, applies
geoid offsets from raster grids (RasterOffsetProvider), evaluates spatial
conditions backed by shapefiles (GeometryCondition, ShapefileQuery), and
composes all of these into a ready-to-use elevation model via the fluent
ElevationModelBuilder.

This package does **not** contain the abstract interfaces themselves (those
live in `photogrammetry`) nor any pixel display operations
(`image_processing`).

## Dependencies

Imports `photogrammetry` for all elevation ABCs (`ElevationModel`,
`DigitalElevationModelTileFactory`, `DigitalElevationModelTileSet`,
`ElevationModelCondition`, `ElevationOffsetProvider`). Uses `osml-imagery-io`
for raster reads and `shapely` / `fiona` for geometry queries. Does not import
`image_processing`, `features`, or `metadata`.

## Design

**Interface/implementation split** --- `photogrammetry` defines the ABCs; this
package provides the concrete implementations that perform actual file I/O.

**ElevationModelBuilder** (fluent builder) --- composes DEM sources,
conditions, geoid offsets, and normalization into a single `ElevationModel`
instance suitable for use with any sensor model.

**StoredDEMTileFactory** --- implements `DigitalElevationModelTileFactory` by
reading DTED or GeoTIFF raster files from a local directory.

**RasterOffsetProvider** --- implements `ElevationOffsetProvider` by
interpolating values from a geoid raster grid (e.g. EGM96/EGM2008).

**GeometryCondition** --- implements `ElevationModelCondition` by testing
point-in-polygon against a `GeometryQuery`.

```{mermaid}
flowchart LR
    B[ElevationModelBuilder]
    B -->|add_source| F[StoredDEMTileFactory]
    B -->|with_geoid| R[RasterOffsetProvider]
    B -->|build| EM[Composed ElevationModel]
    EM --- M[Multi]
    EM --- O[Offset]
    EM --- N[Normalized]
```

## Contributor rules

- New DEM data sources --- implement `DigitalElevationModelTileFactory`.
- New spatial conditions --- implement `ElevationModelCondition` or wrap a
  `GeometryQuery`.
- Keep ABCs in `photogrammetry`; I/O implementations belong here.

## Builder

```{eval-rst}
.. autoclass:: aws.osml.elevation.ElevationModelBuilder
   :members:
   :undoc-members:
   :show-inheritance:
```

## Tile Factory

```{eval-rst}
.. autoclass:: aws.osml.elevation.StoredDEMTileFactory
   :members:
   :undoc-members:
   :show-inheritance:
```

## Offset Provider

```{eval-rst}
.. autoclass:: aws.osml.elevation.RasterOffsetProvider
   :members:
   :undoc-members:
   :show-inheritance:
```

## Geometry Conditions

```{eval-rst}
.. autoclass:: aws.osml.elevation.GeometryCondition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.elevation.ShapefileQuery
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utilities

```{eval-rst}
.. autofunction:: aws.osml.elevation.derive_geo_transform

.. autofunction:: aws.osml.elevation.read_single_band
```
