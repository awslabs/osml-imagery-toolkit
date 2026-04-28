# aws.osml.photogrammetry

This package contains sensor model implementations (image-to-world and world-to-image
transforms), coordinate types, elevation model abstractions, and supporting math. It does
**not** include metadata parsing (that belongs to the `metadata` package) or display/pixel
operations (that belongs to `image_processing`).

`photogrammetry` is the leaf dependency of the toolkit. The `metadata`,
`image_processing`, and `features` packages all depend on it; it has no
sibling-package imports.

### Design abstractions

**`SensorModel` ABC** is the central interface. Every model implements `image_to_world`
and `world_to_image`. Concrete implementations range from rational polynomial cameras
(`RPCSensorModel`) to SAR-specific models (`SICDSensorModel`) and grid-based RSM models.
Two structural wrappers compose models: `CompositeSensorModel` pairs an approximate model
with a precision model, and `ChippedImageSensorModel` adapts a full-image model to a
chip's local pixel coordinate system.

**`ElevationModel` ABC** represents a terrain surface used to constrain the image-to-world
calculation. The package provides composable decorators: `MultiElevationModel` merges
multiple sources by priority, `OffsetElevationModel` applies a provider-supplied bias,
`ConditionalElevationModel` selects a model based on a runtime condition, and
`NormalizedElevationModel` clamps outputs to a valid range.

**Coordinate value objects** (`ImageCoordinate`, `GeodeticWorldCoordinate`,
`WorldCoordinate`) are numpy-backed vectors. Angular quantities are in radians.

```{mermaid}
classDiagram
    class SensorModel {
        <<abstract>>
        +image_to_world(ImageCoordinate) GeodeticWorldCoordinate
        +world_to_image(GeodeticWorldCoordinate) ImageCoordinate
    }
    class RPCSensorModel
    class SICDSensorModel
    class RSMPolynomialSensorModel
    class CompositeSensorModel
    class ChippedImageSensorModel

    SensorModel <|-- RPCSensorModel
    SensorModel <|-- SICDSensorModel
    SensorModel <|-- RSMPolynomialSensorModel
    SensorModel <|-- CompositeSensorModel
    SensorModel <|-- ChippedImageSensorModel
    CompositeSensorModel o-- "2" SensorModel : approximate + precision
    ChippedImageSensorModel o-- SensorModel : delegates

    SensorModel ..> ImageCoordinate
    SensorModel ..> GeodeticWorldCoordinate

    note for SensorModel "Additional implementations (Affine, Projective,\nRSMSectioned, Defaulted) omitted for clarity"
```

```{mermaid}
classDiagram
    class ElevationModel {
        <<abstract>>
        +set_elevation(GeodeticWorldCoordinate) GeodeticWorldCoordinate
    }
    class DigitalElevationModel
    class MultiElevationModel
    class OffsetElevationModel
    class ConditionalElevationModel
    class ElevationOffsetProvider {
        <<abstract>>
    }
    class ElevationModelCondition {
        <<abstract>>
    }
    class DigitalElevationModelTileSet {
        <<abstract>>
    }
    class DigitalElevationModelTileFactory {
        <<abstract>>
    }

    ElevationModel <|-- DigitalElevationModel
    ElevationModel <|-- MultiElevationModel
    ElevationModel <|-- OffsetElevationModel
    ElevationModel <|-- ConditionalElevationModel
    DigitalElevationModel o-- DigitalElevationModelTileSet
    DigitalElevationModel o-- DigitalElevationModelTileFactory
    OffsetElevationModel o-- ElevationOffsetProvider
    ConditionalElevationModel o-- ElevationModelCondition
    MultiElevationModel o-- "1..*" ElevationModel
```

### Contributor guidance

New sensor model types go here as a new file implementing the `SensorModel` ABC. The
corresponding builder that constructs the model from raw metadata belongs in the `metadata`
package. Keep models pure-math — no I/O, no format parsing.

## Coordinates

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.WorldCoordinate
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.GeodeticWorldCoordinate
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ImageCoordinate
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: aws.osml.photogrammetry.geocentric_to_geodetic

.. autofunction:: aws.osml.photogrammetry.geodetic_to_geocentric
```

## Sensor Models

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.SensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.SensorModelOptions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RPCSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RPCPolynomial
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ProjectiveSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.AffineSensorModel
   :members:
   :undoc-members:
   :show-inheritance:
```

## Replacement Sensor Model (RSM)

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.RSMPolynomialSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMSectionedPolynomialSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMPolynomial
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMLowOrderPolynomial
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMContext
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMGroundDomain
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMGroundDomainForm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RSMImageDomain
   :members:
   :undoc-members:
   :show-inheritance:
```

## SICD Sensor Model

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.SICDSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.SARImageCoordConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.COAProjectionSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.PFAProjectionSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.INCAProjectionSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.RGAZCOMPProjectionSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.PlaneProjectionSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.Polynomial2D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.PolynomialXYZ
   :members:
   :undoc-members:
   :show-inheritance:
```

## Composite and Chipped Models

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.CompositeSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ChippedImageSensorModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.DefaultedSensorModel
   :members:
   :undoc-members:
   :show-inheritance:
```

## Elevation Models

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.ElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ConstantElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.DigitalElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.DigitalElevationModelTileFactory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.DigitalElevationModelTileSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.MultiElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.NormalizedElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.OffsetElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ConditionalElevationModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ElevationRegionSummary
   :members:
   :undoc-members:
   :show-inheritance:
```

## DEM Tile Sets

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.GenericDEMTileSet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.SRTMTileSet
   :members:
   :undoc-members:
   :show-inheritance:
```

## Elevation Conditions and Offsets

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.ElevationModelCondition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.EMConditionTrue
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.EMConditionFalse
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ElevationOffsetProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aws.osml.photogrammetry.ConstantOffsetProvider
   :members:
   :undoc-members:
   :show-inheritance:
```

## Geometry Utilities

```{eval-rst}
.. autoclass:: aws.osml.photogrammetry.GeometryQuery
   :members:
   :undoc-members:
   :show-inheritance:
```
