# aws.osml.metadata

Sensor model construction from raw image metadata. This package is the bridge
between `osml-imagery-io`'s metadata dictionaries and the `photogrammetry`
package's sensor model instances. It provides the `SensorModelFactory`
orchestrator, individual builder classes for each supported model type, and the
convenience function `load_sensor_model(reader)` that handles the common
end-to-end path from a DatasetReader to a ready-to-use SensorModel.

This package does **not** contain sensor model implementations (those live in
`photogrammetry`) or any pixel-level operations (`image_processing`).

## Dependencies

The metadata package imports from `photogrammetry` (SensorModel, ImageCoordinate,
ChippedImageSensorModel, CompositeSensorModel) and uses the `formats` package
parsers for SICD/SIDD XML deserialization. It uses `defusedxml` for safe XML
parsing and optionally `pyproj` for CRS resolution from EPSG codes. It does not
import `image_processing` or `features`.

## Design

The package follows the **builder pattern** with a factory orchestrator:

- **SensorModelBuilder** -- abstract base class. Each metadata format has a
  concrete subclass whose `build()` method returns `Optional[SensorModel]`.
  Builders return `None` when the required metadata fields are absent or
  malformed; they never raise on missing data.

- **SensorModelFactory** -- accepts all available metadata sources (TRE dicts,
  DES XML strings, GeoTIFF affine transform, CRS WKT, ground control points)
  and tries builders in priority order: RSM > RPC > SICD/SIDD > projective >
  affine. When both a precision model (RSM/RPC/SICD) and an approximate model
  (projective/affine) are available, the factory wraps them in a
  `CompositeSensorModel`. If the image is a chip (ICHIPB TRE present), the
  factory wraps the result in a `ChippedImageSensorModel`.

- **load_sensor_model(reader)** -- top-level convenience that extracts metadata
  from an `osml-imagery-io` DatasetReader, normalizes format-specific metadata
  layouts, and delegates to `SensorModelFactory`. This is the primary public
  entry point for most callers.

Metadata is represented as flat Python dicts keyed by TRE name (NITF) or numeric
TIFF tag ID (GeoTIFF).

```{mermaid}
flowchart LR
    R[DatasetReader] --> L[load_sensor_model]
    L --> F[SensorModelFactory]
    F --> B1[RSMBuilder]
    F --> B2[RPCBuilder]
    F --> B3[SICD/SIDDBuilder]
    F --> B4[ProjectiveBuilder]
    F --> B5[AffineBuilder]
    B1 & B2 & B3 & B4 & B5 --> SM[SensorModel]
```

## Contributing a new builder

To support a new metadata format:

1. Create a `SensorModelBuilder` subclass in a new module under
   `src/aws/osml/metadata/`.
2. Implement `build() -> Optional[SensorModel]`. Return `None` when required
   fields are missing -- do not raise.
3. Keep the builder stateless after construction (all state set in `__init__`).
4. Register the new builder in `SensorModelFactory.build()` at the appropriate
   priority level relative to existing builders.
5. Export any public types from `__init__.py`.

## Convenience Functions

```{eval-rst}
.. autofunction:: aws.osml.metadata.load_sensor_model

.. autofunction:: aws.osml.metadata.derive_geotiff_georeference
```

## Factory

```{eval-rst}
.. autoclass:: aws.osml.metadata.SensorModelFactory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.metadata.SensorModelTypes
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.metadata.sensor_model_factory.ChippedImageInfoFacade
   :members:
   :undoc-members:
   :show-inheritance:
```

## Builder Interface

```{eval-rst}
.. autoclass:: aws.osml.metadata.SensorModelBuilder
   :members:
   :undoc-members:
   :show-inheritance:
```

## Builder Implementations

```{eval-rst}
.. automodule:: aws.osml.metadata.rpc_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.metadata.rsm_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.metadata.sicd_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.metadata.sidd_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.metadata.projective_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.metadata.affine_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aws.osml.metadata.gcp_sensor_model_builder
   :members:
   :undoc-members:
   :show-inheritance:
```
