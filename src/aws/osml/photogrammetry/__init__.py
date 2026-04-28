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

Geolocating Image Pixels: Basic Examples
****************************************

Applications do not typically interact with a specific sensor model implementation directly. Instead, they let the
SensorModel abstraction encapsulate the details and rely on the image IO utilities to construct the appropriate
type given the available metadata.

.. code-block:: python
    :caption: Example showing calculation of an image location for a geodetic location

    from math import radians
    from aws.osml.io import IO
    from aws.osml.metadata import load_sensor_model
    from aws.osml.photogrammetry import GeodeticWorldCoordinate

    with IO.open("./imagery/sample.nitf", "r") as reader:
        sensor_model = load_sensor_model(reader)

    lon_degrees = -77.404453
    lat_degrees = 38.954831
    meters_above_ellipsoid = 100.0

    # Note the GeodeticWorldCoordinate is (longitude, latitude, elevation) with longitude and latitude in **radians**
    # and elevation in meters above the WGS84 ellipsoid. The resulting ImageCoordinate is returned in (x, y) pixels.
    image_location = sensor_model.world_to_image(
        GeodeticWorldCoordinate([radians(lon_degrees), radians(lat_degrees), meters_above_ellipsoid])
    )

.. code-block:: python
    :caption: Example showing use of a SensorModel to geolocate 4 image corners

    from aws.osml.io import IO
    from aws.osml.metadata import load_sensor_model
    from aws.osml.photogrammetry import ImageCoordinate

    with IO.open("./imagery/sample.nitf", "r") as reader:
        sensor_model = load_sensor_model(reader)
        image = reader.get_asset("image:0")
        width, height = image.num_columns, image.num_rows

    image_corners = [[0, 0], [width, 0], [width, height], [0, height]]
    geo_image_corners = [sensor_model.image_to_world(ImageCoordinate(corner)) for corner in image_corners]

    # GeodeticWorldCoordinates have custom formatting defined that supports a variety of common output formats.
    # The example shown below will produce a ddmmssXdddmmssY formatted coordinate (e.g. 295737N0314003E)
    for geodetic_corner in geo_image_corners:
        print(f"{geodetic_corner:%ld%lm%ls%lH%od%om%os%oH}")

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

    from aws.osml.io import IO
    from aws.osml.metadata import load_sensor_model
    from aws.osml.photogrammetry import DigitalElevationModel, GenericDEMTileSet, ImageCoordinate
    from aws.osml.elevation import StoredDEMTileFactory

    with IO.open("./imagery/sample.nitf", "r") as reader:
        sensor_model = load_sensor_model(reader)
        image = reader.get_asset("image:0")
        width, height = image.num_columns, image.num_rows

    # This sets up an external elevation model assuming terrain data is named something like:
    # ./SRTM/dted/w044/s23.dt2.
    elevation_model = DigitalElevationModel(
        GenericDEMTileSet(format_spec="dted/%oh%od/%lh%ld.dt2"), StoredDEMTileFactory("./SRTM")
    )

    # Note the order of ImageCoordinate is (x, y) and the resulting geodetic coordinate is
    # (longitude, latitude, elevation) with longitude and latitude in **radians** and elevation in meters.
    geodetic_location_of_image_center = sensor_model.image_to_world(
        ImageCoordinate([width / 2, height / 2]), elevation_model=elevation_model
    )


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

from .affine_sensor_model import AffineSensorModel
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
from .elevation_model import ConstantElevationModel, ElevationModel, ElevationRegionSummary
from .elevation_offset_provider import ConstantOffsetProvider, ElevationOffsetProvider
from .em_condition import ElevationModelCondition, EMConditionFalse, EMConditionTrue
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
    "AffineSensorModel",
    "ChippedImageSensorModel",
    "CompositeSensorModel",
    "ConditionalElevationModel",
    "ConstantElevationModel",
    "DigitalElevationModel",
    "DigitalElevationModelTileFactory",
    "DigitalElevationModelTileSet",
    "EMConditionFalse",
    "EMConditionTrue",
    "ElevationModel",
    "ElevationModelCondition",
    "ElevationRegionSummary",
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
