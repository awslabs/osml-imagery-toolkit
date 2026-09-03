#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2026 General Atomics Integrated Intelligence, Inc.

# Telling flake8 to not flag errors in this file. It is normal that these classes are imported but not used in an
# __init__.py file.
# flake8: noqa
"""
Many users need to estimate the geographic position of an object found in a georeferenced image. The osml-imagery-toolkit
provides open source implementations of the image-to-world and world-to-image equations for some common replacement
sensor models. These sensor models work with many georeferenced imagery types and do not require orthorectification of
the image. In the current implementation support is provided for:

* **Rational Polynomials**: Models based on rational polynomials specified using RSM and RPC metadata found in NITF TREs
* **SAR Sensor Independent Models**: Models as defined by the SICD and SIDD standards with metadata found in the NITF XML data segment.
* **Perspective and Affine Projections**: Simple matrix based projections that can be computed from geolocations of the 4 image corners or `tags found in GeoTIFF images <https://docs.ogc.org/is/19-008r4/19-008r4.html#_geotiff_tags_for_coordinate_transformations>`_.

*Note that the current implementation does not support the RSM Grid based sensor models or the adjustable parameter
options. These features will be added in a future release.*

In addition to the core sensor model implementations the library offers multiple wrappers that facilitate the use of
sensor models:

* **Composite**: A combination of a fast approximate model that can be used to calculate an initial guess for a more rigorus model.
* **Chipped Image**: A wrapper that converts image coordinates to or from a chip before calling a sensor model defined for the entire image.
* **Defaulted**: A wrapper that provides default values for the elevation model and other parameters in the image to world and world to image calculations.

.. mermaid::

   classDiagram
       class SensorModel {
           <<abstract>>
           +image_to_world() GeodeticWorldCoordinate
           +world_to_image() ImageCoordinate
       }

       namespace CoreSensorModels {
           class RSMSensorModel {
               <<abstract>>
           }
           class RSMPolynomialSensorModel
           class RSMSectionedPolynomialSensorModel
           class RPCSensorModel
           class SICDSensorModel
           class ProjectiveSensorModel
           class GDALAffineSensorModel
       }

       namespace Wrappers {
           class CompositeSensorModel
           class ChippedImageSensorModel
           class DefaultedSensorModel
       }

       namespace EarthIntersectionSolvers {
           class EarthIntersectionMinimizer {
               <<abstract>>
               +solve() GeodeticWorldCoordinate
           }
           class EIMNelderMead
           class EIMRayMarch
       }

       SensorModel <|-- RSMSensorModel
       RSMSensorModel <|-- RSMPolynomialSensorModel
       RSMSensorModel <|-- RSMSectionedPolynomialSensorModel

       SensorModel <|-- RPCSensorModel
       SensorModel <|-- SICDSensorModel
       SensorModel <|-- ProjectiveSensorModel
       SensorModel <|-- GDALAffineSensorModel

       SensorModel <|-- CompositeSensorModel
       SensorModel <|-- ChippedImageSensorModel
       SensorModel <|-- DefaultedSensorModel

       CompositeSensorModel o-- SensorModel : approximate, precision
       ChippedImageSensorModel o-- SensorModel : full image
       DefaultedSensorModel o-- SensorModel : inner
       DefaultedSensorModel o-- ElevationModel : default
       RSMSectionedPolynomialSensorModel o-- SensorModel : per section

       EarthIntersectionMinimizer <|-- EIMNelderMead
       EarthIntersectionMinimizer <|-- EIMRayMarch
       EIMRayMarch o-- EIMNelderMead : refinement
       RPCSensorModel o-- EarthIntersectionMinimizer : default solver
       RSMPolynomialSensorModel o-- EarthIntersectionMinimizer : default solver


The world coordinates used by sensor models are 3D latitude, longitude, height above ellipsoid. The
ElevationModel abstraction provides a way to provide the surface elevations needed by image to world
calculations or to assign an elevation to world coordinates that only have latitude and longitude
specified. Two implementations source elevations directly:

* **Constant**: A fixed elevation for every location, typically taken from the image metadata when no terrain data is available.
* **Digital**: Elevations read from a DEM product, locating the file that covers a coordinate with a tile set (SRTM naming or a generic format specification) and loading it through a tile factory.

The remaining implementations wrap another elevation model to adapt it:

* **Multi**: Tries several models in order and uses the first that resolves an elevation.
* **Offset**: Applies a correction from an ElevationOffsetProvider, most often the geoid separation needed to convert DEM heights to heights above the WGS84 ellipsoid.
* **Normalized**: Normalizes the coordinate before delegating to the inner model.
* **Conditional**: Consults the inner model only where an ElevationModelCondition holds, such as inside a region described by a GeometryQuery.

.. mermaid::
   :caption: Elevation models. ElevationModel implementations that compose to describe a terrain surface, along with
             the tile set, offset provider, and condition abstractions they delegate to.

   classDiagram
       direction LR

       class ElevationModel {
           <<abstract>>
           +set_elevation() bool
           +describe_region() ElevationRegionSummary
       }
       class DigitalElevationModelTileSet {
           <<abstract>>
           +find_tile_id() str
       }
       class DigitalElevationModelTileFactory {
           <<abstract>>
           +get_tile() tuple
       }
       class ElevationOffsetProvider {
           <<abstract>>
           +get_offset() float
       }
       class ElevationModelCondition {
           <<abstract>>
           +is_true() bool
       }
       class GeometryQuery {
           <<abstract>>
           +get_geometry()
       }

       ElevationModel <|-- ConstantElevationModel
       ElevationModel <|-- DigitalElevationModel
       ElevationModel <|-- MultiElevationModel
       ElevationModel <|-- OffsetElevationModel
       ElevationModel <|-- NormalizedElevationModel
       ElevationModel <|-- ConditionalElevationModel

       MultiElevationModel o-- ElevationModel : ordered list
       OffsetElevationModel o-- ElevationModel : inner
       NormalizedElevationModel o-- ElevationModel : inner
       ConditionalElevationModel o-- ElevationModel : inner

       DigitalElevationModel o-- DigitalElevationModelTileSet
       DigitalElevationModel o-- DigitalElevationModelTileFactory
       DigitalElevationModelTileSet <|-- GenericDEMTileSet
       DigitalElevationModelTileSet <|-- SRTMTileSet

       OffsetElevationModel o-- ElevationOffsetProvider
       ElevationOffsetProvider <|-- ConstantOffsetProvider

       ConditionalElevationModel o-- ElevationModelCondition
       ElevationModelCondition <|-- EMConditionTrue
       ElevationModelCondition <|-- EMConditionFalse
       ElevationModelCondition <|-- EMConditionIntersects
       EMConditionIntersects o-- GeometryQuery

Geolocating Image Pixels: Basic Examples
****************************************

Applications do not typically interact with a specific sensor model implementation directly. Instead, they let the
SensorModel abstraction encapsulate the details and rely on the image IO utilities to construct the appropriate
type given the available metadata.

.. code-block:: python
    :caption: Example showing calculation of an image location for a geodetic location

    dataset, sensor_model = load_gdal_dataset("./imagery/sample.nitf")

    lon_degrees = -77.404453
    lat_degrees = 38.954831
    meters_above_ellipsoid = 100.0

    # Note the GeodeticWorldCoordinate is (longitude, latitude, elevation) with longitude and latitude in **radians**
    # and elevation in meters above the WGS84 ellipsoid. The resulting ImageCoordinate is returned in (x, y) pixels.
    image_location = sensor_model.world_to_image(
        GeodeticWorldCoordinate([radians(lon_degrees),
                                 radians(lat_degrees),
                                 meters_above_ellipsoid]))

.. code-block:: python
    :caption: Example showing use of a SensorModel to geolocate 4 image corners

    dataset, sensor_model = load_gdal_dataset("./imagery/sample.nitf")
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    image_corners = [[0, 0], [width, 0], [width, height], [0, height]]
    geo_image_corners = [sensor_model.image_to_world(ImageCoordinate(corner))
                         for corner in image_corners]

    # GeodeticWorldCoordinates have custom formatting defined that supports a variety of common output formats.
    # The example shown below will produce a ddmmssXdddmmssY formatted coordinate (e.g. 295737N0314003E)
    for geodetic_corner in geo_image_corners:
        print(f"{geodetic_corner:%ld%lm%ls%lH%od%om%os%oH}")

Tuning the Image-to-World Solve
*******************************

Sensor models that cannot invert their equations analytically solve image-to-world by iteratively minimizing
reprojection error, so the result depends on where the search starts and when it is allowed to stop.
`SensorModelOptions` names the hints these implementations accept in the ``options`` dictionary. Support varies by
implementation.

* ``INITIAL_GUESS`` and ``INITIAL_SEARCH_DISTANCE`` seed the search. The default guess comes from the camera's
  normalization offsets, which sit near the image center or a corner.
* ``MIN_SUCCESS_DISTANCE_PIXELS`` is the reprojection residual, in pixels, that counts as converged. Default is 1.0.
* ``FALLBACK_INITIAL_GUESS`` and ``EXCEPTION_ON_FAILURE`` decide what a failed solve does. With neither set, the
  non-converged estimate is returned as-is.
* ``FORCE_INITIAL_GUESS`` skips iteration and returns the seed, useful when an approximation is enough.
* ``EARTH_INTERSECTION_MINIMIZER`` selects a solver by name.

An `EarthIntersectionMinimizer` performs the search. Solvers are looked up in the ``eim_registry`` module by name and
the implementations shipped with this library register themselves when the package is imported, so no setup is needed
to select one.

* ``"neldermead"``, implemented by `EIMNelderMead`, runs a single Nelder-Mead minimization from the initial guess. It
  is the default used by `RPCSensorModel` and `RSMPolynomialSensorModel`.
* ``"raymarch"``, implemented by `EIMRayMarch`, estimates the line of sight from two constant-height solves, steps
  along it until it crosses the terrain surface, then refines that approximate intersection with Nelder-Mead.

.. code-block:: python
    :caption: Example showing selection of an alternate earth intersection solver

    world_location = sensor_model.image_to_world(
        ImageCoordinate([512, 512]),
        elevation_model=elevation_model,
        options={SensorModelOptions.EARTH_INTERSECTION_MINIMIZER: "raymarch",
                 SensorModelOptions.EXCEPTION_ON_FAILURE: True})

Which solver to prefer depends on the terrain. A single Nelder-Mead search assumes the reprojection residual falls off
smoothly from the initial guess toward the answer. That holds over gentle terrain, and it holds trivially when
elevations come from a `ConstantElevationModel`, where a flat surface leaves only one solution to find. It degrades
where relief is steep enough that the residual surface develops local minima, so the search settles into the wrong
basin or stops short of the target. Marching along the line of sight sidesteps the problem by locating the terrain
crossing geometrically before any refinement, which is what makes ``"raymarch"`` the better choice over mountainous
terrain or for oblique collections with long shadows and layover.

That robustness is not free. `EIMRayMarch` performs two Nelder-Mead solves to establish the ray direction and a third
to refine the result, and the stepping passes add on the order of a hundred elevation lookups per call. Against a
`ConstantElevationModel` the extra work buys nothing, because there is no relief for the march to discover. Reserve it
for calls that supply real terrain data and where accuracy matters more than throughput.

Note that a solver reports whether it converged, and the two implementations behave differently when they do not.
`EIMNelderMead` returns its best iterate, which is typically close to the answer. `EIMRayMarch` returns an unrefined
position when the pre-solves or the stepping passes fail, which may be far from the answer. Pair ``"raymarch"`` with
``EXCEPTION_ON_FAILURE`` or ``FALLBACK_INITIAL_GUESS`` rather than relying on the returned coordinate being usable.

Applications needing a different search can implement `EarthIntersectionMinimizer` and register it under a name of
their own. Names must be unique; registering one that is already taken raises a ``ValueError``, so do not re-register
``"neldermead"`` or ``"raymarch"``.

.. code-block:: python
    :caption: Example showing registration of a custom earth intersection solver

    class MyMinimizer(EarthIntersectionMinimizer):
        def solve(self, minimization_function, elevation_model, initial_guess, search_distance,
                  lon_bounds=None, lat_bounds=None, height_bounds=None):
            ...
            return world_coordinate, success

    eim_registry.register("mysolver", MyMinimizer())

    world_location = sensor_model.image_to_world(
        ImageCoordinate([512, 512]),
        elevation_model=elevation_model,
        options={SensorModelOptions.EARTH_INTERSECTION_MINIMIZER: "mysolver"})

The ``lon_bounds``, ``lat_bounds``, and ``height_bounds`` arguments are hints describing the region the calling sensor
model considers valid, and a solver is free to ignore the ones it cannot use. `RSMPolynomialSensorModel` supplies all
three, taken from its ground domain and normalization ranges, while `RPCSensorModel` supplies only ``height_bounds``.
`EIMNelderMead` constrains the search to ``lon_bounds`` and ``lat_bounds`` when both are given and ignores
``height_bounds``; `EIMRayMarch` uses ``height_bounds`` to decide the altitude its march starts from.

Passing the same elevation model and options on every call gets repetitive. `DefaultedSensorModel` binds them to the
model once, and options supplied at call time are merged over the bound defaults so an individual call can still
override a single setting.

.. code-block:: python
    :caption: Example showing an elevation model and options bound to a sensor model once

    dataset, sensor_model = load_gdal_dataset("./imagery/sample.nitf")

    tuned_sensor_model = DefaultedSensorModel(
        sensor_model,
        elevation_model=elevation_model,
        options={SensorModelOptions.EARTH_INTERSECTION_MINIMIZER: "raymarch",
                 SensorModelOptions.EXCEPTION_ON_FAILURE: True})

    # This call uses the bound elevation model and both bound options.
    image_center = tuned_sensor_model.image_to_world(
        ImageCoordinate([dataset.RasterXSize / 2, dataset.RasterYSize / 2]))

    # Default options can still be overriden if needed.
    image_corner = tuned_sensor_model.image_to_world(
        ImageCoordinate([0, 0]),
        options={SensorModelOptions.EXCEPTION_ON_FAILURE: False,
                 SensorModelOptions.IGNORE_DEFAULT_ELEVATION_MODEL: True,
                 SensorModelOptions.FALLBACK_INITIAL_GUESS: True})

Geolocating Image Pixels: Addition of an External Elevation Model
*****************************************************************

The image-to-world calculation can optionally use an external digital elevation model (DEM) when geolocating points
on an image. How the elevation model will be used varies by sensor model but examples include:

* Using DEM elevations as a constraint during iterations of a rational polynomial camera's image-to-world calculation.
* Computing the intersection of a R/Rdot contour with a DEM for sensor independent SAR models.

All of these approaches make the fundamental assumption that the pixel lies on the terrain surface. If a DEM is not
available we assume a constant surface with elevation provided in the image metadata.

.. code-block:: python
    :caption: Example showing use of an external SRTM DEM to provide elevation data for image center

    ds, sensor_model = load_gdal_dataset("./imagery/sample.nitf")

    # This sets up an external elevation model assuming terrain data is named something like:
    # ./SRTM/dted/w044/s23.dt2.
    elevation_model = DigitalElevationModel(
        GenericDEMTileSet(format_spec="dted/%oh%od/%lh%ld.dt2"),
        GDALDigitalElevationModelTileFactory("./SRTM"))

    # Note the order of ImageCoordinate is (x, y) and the resulting geodetic coordinate is
    # (longitude, latitude, elevation) with longitude and latitude in **radians** and elevation in meters
    # above the ellipsoid.
    geodetic_location_of_image_center = sensor_model.image_to_world(
        ImageCoordinate([ds.RasterXSize/2, ds.RasterYSize/2]),
        elevation_model=elevation_model)

DEM products such as SRTM and DTED report heights above a geoid model, while a GeodeticWorldCoordinate carries height
above the WGS84 ellipsoid. Converting between the two means adding the geoid separation at that location.
`OffsetElevationModel` layers that correction over any inner model, taking the value from an
`ElevationOffsetProvider`.

.. code-block:: python
    :caption: Example showing a geoid separation applied over a DEM

    dem = DigitalElevationModel(
        GenericDEMTileSet(format_spec="dted/%oh%od/%lh%ld.dt2"),
        GDALDigitalElevationModelTileFactory("./SRTM"))

    # Elevations from this DEM are relative to EGM96. Adding the geoid separation, which is roughly -32.5 meters
    # near the sample coordinate used above, converts them to heights above the WGS84 ellipsoid. A single constant
    # is adequate across a small area; a collection spanning a wide area wants an ElevationOffsetProvider that
    # varies the offset with location.
    elevation_model = OffsetElevationModel(dem, ConstantOffsetProvider(-32.5))

    geodetic_location_of_image_center = sensor_model.image_to_world(
        ImageCoordinate([ds.RasterXSize/2, ds.RasterYSize/2]),
        elevation_model=elevation_model)

Note that `OffsetElevationModel` does not implement ``describe_region``. Sensor models that use the region summary to
bound their search, such as `SICDSensorModel`, fall back to their unbounded defaults when one is in use.

The elevation models also compose. `MultiElevationModel` tries several sources in order, `OffsetElevationModel`
applies a per-location correction from an `ElevationOffsetProvider`, `NormalizedElevationModel` normalizes input
coordinates before delegating, and `ConditionalElevationModel` consults its inner model only where an
`ElevationModelCondition` holds. `EMConditionIntersects` restricts it to a region described by a `GeometryQuery`.


External References
*******************

* Manual of Photogrammetry: https://www.amazon.com/Manual-Photogrammetry-PhD-Chris-McGlone/dp/1570830991
* NITF Compendium of Controlled Support Data Extensions: https://nsgreg.nga.mil/doc/view?i=5417
* The Replacement Sensor Model (RSM): Overview, Status, and Performance Summary: https://citeseerx.ist.psu.edu/doc_view/pid/c25de8176fe790c28cf6e1aff98ecdea8c726c96
* RPC Whitepaper: https://users.cecs.anu.edu.au/~hartley/Papers/cubic/cubic.pdf
* SICD Volume 3, Image Projections Description Document: https://nsgreg.nga.mil/doc/view?i=5383
* WGS84 Standard: https://nsgreg.nga.mil/doc/view?i=4085

-------------------------

APIs
****

"""

from .chipped_image_sensor_model import ChippedImageSensorModel
from .composite_sensor_model import CompositeSensorModel
from .conditional_elevation_model import ConditionalElevationModel
from .coordinates import (
    GeodeticWorldCoordinate,
    ImageCoordinate,
    WorldCoordinate,
    geocentric_to_geodetic,
    geodetic_to_geocentric,
)
from .defaulted_sensor_model import DefaultedSensorModel
from .digital_elevation_model import DigitalElevationModel, DigitalElevationModelTileFactory, DigitalElevationModelTileSet
from .earth_intersection_minimizer import EarthIntersectionMinimizer
from .eim_neldermead import EIMNelderMead
from .eim_raymarch import EIMRayMarch
from .elevation_model import ConstantElevationModel, ElevationModel, ElevationRegionSummary
from .elevation_offset_provider import ConstantOffsetProvider, ElevationOffsetProvider
from .em_condition import ElevationModelCondition, EMConditionFalse, EMConditionTrue
from .em_condition_intersects import EMConditionIntersects
from .gdal_sensor_model import GDALAffineSensorModel
from .generic_dem_tile_set import GenericDEMTileSet
from .geometry_query import GeometryQuery
from .multi_elevation_model import MultiElevationModel
from .normalized_elevation_model import NormalizedElevationModel
from .offset_elevation_model import OffsetElevationModel
from .projective_sensor_model import ProjectiveSensorModel
from .replacement_sensor_model import (
    RSMContext,
    RSMGroundDomain,
    RSMGroundDomainForm,
    RSMImageDomain,
    RSMLowOrderPolynomial,
    RSMPolynomial,
    RSMPolynomialSensorModel,
    RSMSectionedPolynomialSensorModel,
)
from .rpc_sensor_model import RPCPolynomial, RPCSensorModel
from .sensor_model import SensorModel, SensorModelOptions
from .sicd_sensor_model import (
    COAProjectionSet,
    INCAProjectionSet,
    PFAProjectionSet,
    PlaneProjectionSet,
    Polynomial2D,
    PolynomialXYZ,
    RGAZCOMPProjectionSet,
    SARImageCoordConverter,
    SICDSensorModel,
)
from .srtm_dem_tile_set import SRTMTileSet

__all__ = [
    "ChippedImageSensorModel",
    "CompositeSensorModel",
    "ConditionalElevationModel",
    "ConstantElevationModel",
    "DigitalElevationModel",
    "DigitalElevationModelTileFactory",
    "DigitalElevationModelTileSet",
    "EIMNelderMead",
    "EIMRayMarch",
    "EMConditionFalse",
    "EMConditionTrue",
    "EarthIntersectionMinimizer",
    "ElevationModel",
    "ElevationModelCondition",
    "ElevationRegionSummary",
    "GDALAffineSensorModel",
    "GenericDEMTileSet",
    "GeodeticWorldCoordinate",
    "GeometryQuery",
    "INCAProjectionSet",
    "ImageCoordinate",
    "MultiElevationModel",
    "PFAProjectionSet",
    "PlaneProjectionSet",
    "Polynomial2D",
    "PolynomialXYZ",
    "ProjectiveSensorModel",
    "RGAZCOMPProjectionSet",
    "RPCPolynomial",
    "RPCSensorModel",
    "RSMContext",
    "RSMGroundDomain",
    "RSMGroundDomainForm",
    "RSMImageDomain",
    "RSMLowOrderPolynomial",
    "RSMPolynomial",
    "RSMPolynomialSensorModel",
    "RSMSectionedPolynomialSensorModel",
    "SARImageCoordConverter",
    "SICDSensorModel",
    "SRTMTileSet",
    "SensorModel",
    "SensorModelOptions",
    "WorldCoordinate",
    "geocentric_to_geodetic",
    "geodetic_to_geocentric",
]
