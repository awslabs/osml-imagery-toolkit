#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from abc import ABC
from enum import Enum
from math import floor, pi, radians, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .coordinates import (
    GeodeticWorldCoordinate,
    ImageCoordinate,
    WorldCoordinate,
    geocentric_to_geodetic,
    geodetic_to_geocentric,
)
from .elevation_model import ConstantElevationModel, ElevationModel
from .math_utils import equilateral_triangle
from .sensor_model import SensorModel, SensorModelOptions

# TODO: Add Support for Grid Based RSM Sensor Models
# TODO: Add Support for Adjustable RSM Sensor Models
# TODO: Add Support for Error Assessments
# TODO: Add typing for ArrayLike inputs once Numpy upgraded to 1.20+
# from numpy.typing import ArrayLike


class RSMImageDomain:
    """
    This RSM image domain is a rectangle defined by the minimum and maximum discrete row coordinate values, and the
    minimum and maximum discrete column coordinate values. It is typically constructed from values in the NITF RSMID
    TRE. For more information see section 5.5 of STDI-0002 Volume 1 Appendix U.
    """

    def __init__(
        self,
        min_row: int,
        max_row: int,
        min_column: int,
        max_column: int,
    ) -> None:
        """
        Construct an RSM Image domain from parameters

        :param min_row: min row
        :param max_row: max row
        :param min_column: min column
        :param max_column: max column

        :return: None
        """
        self.min_row = min_row
        self.max_row = max_row
        self.min_column = min_column
        self.max_column = max_column


class RSMGroundDomainForm(Enum):
    """
    The RSMGroundDomainForm defines how world coordinates (x, y, z) should be interpreted in this sensor model.

    If geodetic, X, Y, and Z, correspond to longitude, latitude, and height above the ellipsoid, respectively.
    Longitude is specified east of the prime meridian, and latitude is specified north of the equator. Units for X, Y,
    and Z, are radians, radians, and meters, respectively. The range for Y is (-pi/2 to pi/2). The range for X is
    (-pi to pi) when GEODETIC, and (0 to 2pi) when GEODETIC_2PI. The latter is specified when the RSM ground domain
    contains a longitude value near pi radians.

    If RECTANGULAR, X, Y, and Z correspond to a coordinate system that is defined as an offset from and rotation
    about the WGS 84 Rectangular coordinate system.

    For more information see the GRNDD TRE field definition in section 5.9 of STDI-0002 Volume 1 Appendix U.
    """

    GEODETIC = "G"
    GEODETIC_2PI = "H"
    RECTANGULAR = "R"


class RSMGroundDomain:
    """
    The RSM ground domain is an approximation of the ground area where the RSM representation is valid. It is a solid
    in three-dimensional space bounded by a hexahedron with quadrilateral faces specified using eight
    three-dimensional vertices.

    It is typically constructed from values in the NITF RSMID TRE. For more information see section 5.6 of
    STDI-0002 Volume 1 Appendix U.
    """

    def __init__(
        self,
        ground_domain_form: RSMGroundDomainForm,
        ground_domain_vertices: List[WorldCoordinate],
        rectangular_coordinate_origin: Optional[WorldCoordinate] = None,
        rectangular_coordinate_unit_vectors: Optional[Any] = None,
        ground_reference_point: Optional[WorldCoordinate] = None,
    ) -> None:
        """
        Construct a RSM ground domain from parameters typically found in the RSMID TRE.

        TODO: Change the type hint for rectangular_coordinate_unit_vectors to Optional[ArrayLike] once Numpy > 1.20

        :param ground_domain_form: form of coordinates used by this sensor model
        :param ground_domain_vertices: collection of vertices defining the valid model area
        :param rectangular_coordinate_origin: optional origin of the rectangular system
        :param rectangular_coordinate_unit_vectors: optional unit vectors to define rectangular system
        :param ground_reference_point: optional reference point for this image

        :return: None
        """
        if len(ground_domain_vertices) != 8:
            raise ValueError(
                f"Incorrect number of vertices in RSMGroundDomain. " f"{len(ground_domain_vertices)} provided, 8 expected."
            )

        if ground_domain_form == RSMGroundDomainForm.RECTANGULAR and (
            rectangular_coordinate_origin is None or rectangular_coordinate_unit_vectors is None
        ):
            raise ValueError("Rectangular coordinate system for RSMGroundDomain has not been defined")

        self.ground_domain_form = ground_domain_form
        self.ground_domain_vertices = ground_domain_vertices
        self.rectangular_coordinate_origin = rectangular_coordinate_origin
        if rectangular_coordinate_unit_vectors is not None:
            self.rectangular_coordinate_unit_vectors = np.array(rectangular_coordinate_unit_vectors, dtype=np.float64)
            self.rectangular_coordinate_unit_vectors_inverse = np.linalg.inv(self.rectangular_coordinate_unit_vectors)
        self.ground_reference_point = ground_reference_point

        self.geodetic_ground_domain_vertices = [
            self.ground_domain_coordinate_to_geodetic(vertex) for vertex in self.ground_domain_vertices
        ]

        # Calculate a bounding box and average elevation for this ground domain in geodetic coordinates
        elevation_sum = 0.0
        self.geodetic_lonlat_bbox = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        for geodetic_vertex in self.geodetic_ground_domain_vertices:
            elevation_sum += geodetic_vertex.elevation
            self.geodetic_lonlat_bbox[0] = min(self.geodetic_lonlat_bbox[0], geodetic_vertex.longitude)
            self.geodetic_lonlat_bbox[1] = min(self.geodetic_lonlat_bbox[1], geodetic_vertex.latitude)
            self.geodetic_lonlat_bbox[2] = max(self.geodetic_lonlat_bbox[2], geodetic_vertex.longitude)
            self.geodetic_lonlat_bbox[3] = max(self.geodetic_lonlat_bbox[3], geodetic_vertex.latitude)

        # If a ground reference point is included we will set the reference geodetic height to match that point.
        # If that point is not included then we will compute it from the average height of the ground domain
        # hexahedron in reference to the WGS84 ellipsoid. See section 5.7 of STDI-0002 Volume 1 Appendix U.
        if self.ground_reference_point is not None:
            geodetic_ground_reference_point = self.ground_domain_coordinate_to_geodetic(self.ground_reference_point)
            self.default_elevation_model = ConstantElevationModel(geodetic_ground_reference_point.elevation)
        else:
            average_elevation = elevation_sum / len(self.geodetic_ground_domain_vertices)
            self.default_elevation_model = ConstantElevationModel(average_elevation)

    def geodetic_to_ground_domain_coordinate(self, geodetic_coordinate: GeodeticWorldCoordinate) -> WorldCoordinate:
        """
        This function converts WGS-84 geodetic world coordinate into a world coordinate that uses the domain
        coordinate system for this sensor model.

        :param geodetic_coordinate: the WGS-84 longitude, latitude, elevation

        :return: the x, y, z domain coordinate
        """
        if self.ground_domain_form == RSMGroundDomainForm.RECTANGULAR:
            # This shouldn't happen due to the check in the constructor but the type checker is flagging this, so
            # we check again to ensure we have all the values necessary
            if self.rectangular_coordinate_origin is None or self.rectangular_coordinate_unit_vectors is None:
                raise TypeError("Rectangular ground domain missing origin or unit vectors")

            # The ground domain uses a rectangular coordinate system, so we need to convert the input geodetic
            # coordinate (longitude, latitude, elevation) to the rectangular coordinate system using the provided
            # parameters. See STDI-002 Volume 1 Appendix U Section 5.3 for more details.

            # The longitude, latitude, elevation in (radians, radians, meters) needs to be converted to a
            # WGS84 earth centric coordinate system before subtracting off the coordinate system origin. Then
            # the matrix of unit vectors is applied as shown in the specification.
            ecef_world_coordinate = geodetic_to_geocentric(geodetic_coordinate)
            coordinate_relative_to_origin = ecef_world_coordinate.coordinate - self.rectangular_coordinate_origin.coordinate
            ground_domain_coordinate = WorldCoordinate(
                np.dot(
                    self.rectangular_coordinate_unit_vectors,
                    coordinate_relative_to_origin,
                )
            )
        elif self.ground_domain_form == RSMGroundDomainForm.GEODETIC_2PI:
            # The ground domain uses geodetic coordinates but the range of longitude values is [0,2*PI). Adjust the
            # input value to ensure that range assumption is met.
            ground_domain_coordinate = GeodeticWorldCoordinate(
                [
                    (geodetic_coordinate.x + 2 * pi) % (2 * pi),
                    geodetic_coordinate.y,
                    geodetic_coordinate.z,
                ]
            )
        else:
            # The ground domain is using normal geodetic coordinates so there is nothing to do.
            ground_domain_coordinate = geodetic_coordinate

        return ground_domain_coordinate

    def ground_domain_coordinate_to_geodetic(self, ground_domain_coordinate: WorldCoordinate) -> GeodeticWorldCoordinate:
        """
        This function converts an x, y, z coordinate defined in the ground domain of this sensor model into a WGS-84
        longitude, latitude, elevation coordinate.

        :param ground_domain_coordinate: the x, y, z domain coordinate

        :return: the WGS-84 longitude, latitude, elevation coordinate
        """
        if self.ground_domain_form == RSMGroundDomainForm.RECTANGULAR:
            # This shouldn't happen due to the check in the constructor but the type checker is flagging this, so
            # we check again to ensure we have all the values necessary
            if self.rectangular_coordinate_origin is None or self.rectangular_coordinate_unit_vectors is None:
                raise TypeError("Rectangular ground domain missing origin or unit vectors")

            # The ground domain uses a rectangular coordinate system, so we need to convert the input rectangular
            # coordinate (x,y,z) to the geodetic coordinate system (longitude, latitude, elevation) using the inverse
            # of the provided parameters. See STDI-002 Volume 1 Appendix U Section 5.3 for more details.
            ecef_world_coordinate = GeodeticWorldCoordinate(
                np.dot(
                    self.rectangular_coordinate_unit_vectors_inverse,
                    ground_domain_coordinate.coordinate,
                )
                + self.rectangular_coordinate_origin.coordinate
            )
            world_coordinate = geocentric_to_geodetic(ecef_world_coordinate)
        else:
            # The ground domain is using normal geodetic coordinates so there is nothing to do.
            # TODO: Consider if we should adjust longitude to be in the [-PI,PI) range
            world_coordinate = GeodeticWorldCoordinate(ground_domain_coordinate.coordinate)

        return world_coordinate


class RSMContext:
    """
    The RSM context contains information necessary to apply and interpret results from the sensor models on an image.

    This current implementation only covers the ground domain and image domains necessary for georeferencing but it
    can be expanded as needed to support other RSM functions.

    TODO: Implement the TimeContext which can be used to identify the collection time of any x, y image coordinate
    TODO: Implement the IlluminationContext which can be used to predict shadow information on an image
    TODO: Implement the TrajectoryModel which captures the sensors 3D position in relation to the image
    """

    def __init__(self, ground_domain: RSMGroundDomain, image_domain: RSMImageDomain) -> None:
        """
        Construct the overall RSM context from parameters typically found in the RSMID TRE.

        :param ground_domain: the RSM ground domain
        :param image_domain: the RSM image domain

        :return: None
        """
        self.ground_domain = ground_domain
        self.image_domain = image_domain


class RSMPolynomial:
    """
    This is an implementation of a general polynomial that can be applied to a world coordinate (i.e. an x,y,z vector).
    For additional information see Section 7.2 of STDI-0002 Volume 1 Appendix U or Section 10.3.3.1.1 of the Manual of
    Photogrammetry Sixth Edition.
    """

    def __init__(
        self,
        max_power_x: int,
        max_power_y: int,
        max_power_z: int,
        coefficients: List[float],
    ) -> None:
        """
        Construct the polynomial from coefficients and information about the maximum power.
        Note that len(coefficients) must equal (max_power_x +1) * (max_power_y +1) * (max_power_z +1)

        :param max_power_x: maximum power of x
        :param max_power_y: maximum power of y
        :param max_power_z: maximum power of z
        :param coefficients: coefficients orders in sequence defined in the NITF RSMPCA TRE

        :return: None
        """
        # There must be one coefficient for every combination of x, y, z powers. For example a polynomial with
        # a maximum power of 0 for x, y, and z with only have a single coefficient (the constant C). A polynomial with
        # a maximum power of 1 for x, y, and z will have 8 coefficients for all the combinations of x, y, and z up to
        # a power of 1: [C, X, Y, XY, Z, XZ, YZ, XYZ].
        if len(coefficients) != (max_power_x + 1) * (max_power_y + 1) * (max_power_z + 1):
            raise ValueError("Incorrect number of coefficients in RSMPolynomial")

        self.max_power_x = max_power_x
        self.max_power_y = max_power_y
        self.max_power_z = max_power_z
        self.coefficients = coefficients

    def evaluate(self, normalized_world_coordinate: WorldCoordinate) -> float:
        """
        This function evaluates the polynomial for the given world coordinate by summing up the result of applying
        each coefficient to the world coordinate components. Note that these polynomials are usually defined with
        the assumption that the world coordinate has been normalized.

        :param normalized_world_coordinate: the world coordinate

        :return: the resulting value
        """
        result = 0.0
        a_index = 0
        # Black formatter doesn't play well with the **'s wrapped in brackets
        # fmt: off
        for k in range(self.max_power_z + 1):
            for j in range(self.max_power_y + 1):
                for i in range(self.max_power_x + 1):
                    result += (
                        self.coefficients[a_index]
                        * (normalized_world_coordinate.x ** i)
                        * (normalized_world_coordinate.y ** j)
                        * (normalized_world_coordinate.z ** k)
                    )
                    a_index += 1
        # fmt: on
        return result

    def __call__(self, *args, **kwargs):
        """
        This makes the polynomial object callable such that it can be applied to a world coordinate directly e.g.
        result = polynomial(coordinate).

        :param args: the world coordinate
        :param kwargs: no keyword arguments are accepted but this is required for the Callable interface
        :return: the result of the evaluation
        """
        return self.evaluate(args[0])


class RSMLowOrderPolynomial:
    """
    This is an implementation of a "low order" polynomial used when generating coarse image row and column coordinates
    from a world coordinate. For additional information see Section 6.2 of STDI-0002 Volume 1 Appendix U.
    """

    def __init__(self, coefficients: List[float]) -> None:
        """
        Construct a low order polynomial given a set of coefficients that are ordered to match the specific set of
        component powers [0, X, Y, Z, XX, XY, XZ, YY, YZ, ZZ]

        :param coefficients: the low order polynomial coefficients

        :return: None
        """
        if len(coefficients) != 10:
            raise ValueError("Incorrect number of coefficients when constructing RSMLowOderPolynomial")

        self.coefficients = coefficients

    def evaluate(self, world_coordinate: WorldCoordinate):
        """
        This function evaluates the polynomial for the given world coordinate by summing up the result of applying
        each coefficient to the world coordinate components.

        :param world_coordinate: the world coordinate

        :return: the resulting value
        """
        result = 0.0
        result += self.coefficients[0]  # constant
        result += self.coefficients[1] * world_coordinate.x  # X
        result += self.coefficients[2] * world_coordinate.y  # Y
        result += self.coefficients[3] * world_coordinate.z  # Z
        result += self.coefficients[4] * world_coordinate.x * world_coordinate.x  # XX
        result += self.coefficients[5] * world_coordinate.x * world_coordinate.y  # XY
        result += self.coefficients[6] * world_coordinate.x * world_coordinate.z  # XZ
        result += self.coefficients[7] * world_coordinate.y * world_coordinate.y  # YY
        result += self.coefficients[8] * world_coordinate.y * world_coordinate.z  # YZ
        result += self.coefficients[9] * world_coordinate.z * world_coordinate.z  # ZZ
        return result

    def __call__(self, *args, **kwargs):
        """
        This makes the polynomial object callable such that it can be applied to a world coordinate directly e.g.
        result = polynomial(coordinate).

        :param args: the world coordinate
        :param kwargs: no keyword arguments are accepted but this is required for the Callable interface
        :return: the result of the evaluation
        """
        return self.evaluate(args[0])


class RSMSensorModel(SensorModel, ABC):
    """
    This is an abstract base for all sensor models that use the RSM context information.
    """

    def __init__(self, context: RSMContext) -> None:
        """
        Constructor that accepts the RSM context as an input.

        :param context: contextual information describing the collection environment

        :return: None
        """
        super().__init__()
        self.context = context


class RSMPolynomialSensorModel(RSMSensorModel):
    """
    This is an implementation of a Rational Polynomial Camera as defined in section 10.3.3.1.1 of the Manual of
    Photogrammetry Sixth Edition.
    """

    def __init__(
        self,
        context: RSMContext,
        section_row: int,
        section_col: int,
        row_norm_offset: float,
        column_norm_offset: float,
        x_norm_offset: float,
        y_norm_offset: float,
        z_norm_offset: float,
        row_norm_scale: float,
        column_norm_scale: float,
        x_norm_scale: float,
        y_norm_scale: float,
        z_norm_scale: float,
        row_numerator_poly: RSMPolynomial,
        row_denominator_poly: RSMPolynomial,
        column_numerator_poly: RSMPolynomial,
        column_denominator_poly: RSMPolynomial,
    ) -> None:
        """
        This constructs the sensor model using parameters normally found in the NITF RSMPC TRE.

        :param context: contextual information describing the collection environment
        :param section_row: image row section number that the sensor model applies to
        :param section_col: image col section number that the sensor model applies to
        :param row_norm_offset: offset used to normalize/denormalize image row components
        :param column_norm_offset: offset used to normalize/denormalize image column components
        :param x_norm_offset: offset used to normalize/denormalize world x components
        :param y_norm_offset: offset used to normalize/denormalize world y components
        :param z_norm_offset: offset used to normalize/denormalize world z components
        :param row_norm_scale: scale used to normalize/denormalize image row components
        :param column_norm_scale: scale used to normalize/denormalize image column components
        :param x_norm_scale: scale used to normalize/denormalize world x components
        :param y_norm_scale: scale used to normalize/denormalize world y components
        :param z_norm_scale: scale used to normalize/denormalize world z components
        :param row_numerator_poly: polynomial used as the numerator in row calculations
        :param row_denominator_poly: polynomial used as the denominator in row calculations
        :param column_numerator_poly: polynomial used as the numerator in column calculations
        :param column_denominator_poly: polynomial used as the denominator in column calculations

        :return: None
        """
        super().__init__(context)
        self.section_row = section_row
        self.section_col = section_col
        self.row_norm_offset = row_norm_offset
        self.column_norm_offset = column_norm_offset
        self.x_norm_offset = x_norm_offset
        self.y_norm_offset = y_norm_offset
        self.z_norm_offset = z_norm_offset
        self.row_norm_scale = row_norm_scale
        self.column_norm_scale = column_norm_scale
        self.x_norm_scale = x_norm_scale
        self.y_norm_scale = y_norm_scale
        self.z_norm_scale = z_norm_scale
        self.row_numerator_poly = row_numerator_poly
        self.row_denominator_poly = row_denominator_poly
        self.column_numerator_poly = column_numerator_poly
        self.column_denominator_poly = column_denominator_poly

    def world_to_image(self, geodetic_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This function transforms a geodetic world coordinate (longitude, latitude, elevation) into an image coordinate
        (x, y).

        :param geodetic_coordinate: the world coordinate (longitude, latitude, elevation)

        :return: the resulting image coordinate (x,y)
        """

        # Convert the geodetic world coordinate into a coordinate in the ground domain then run the ground domain
        # to image function to get the final result.
        world_coordinate = self.context.ground_domain.geodetic_to_ground_domain_coordinate(geodetic_coordinate)
        return self.ground_domain_to_image(world_coordinate)

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function implements the image to world transform by iteratively invoking world to image within a
        minimization routine to find a matching image coordinate. The longitude and latitude parameters are searched
        independently while the elevation of the world coordinate comes from the surface provided with the ground
        domain.

        :param image_coordinate: the image coordinate (x, y)
        :param elevation_model: an optional elevation model used transform the coordinate
        :param options: optional hints, supports initial_guess and initial_search_distance

        :return: the corresponding world coordinate
        """

        options = options if options is not None else {}

        # This is the function we will be minimizing. Given a longitude, latitude coordinate we invoke the
        # world_to_image function to get a projection of that location in the image. Then we compute the
        # distance between that new image location and the input image location. When those locations match then
        # we know we have the world coordinate that corresponds to the input.
        def distance_to_target_coordinate(lonlat_coord: Tuple[float, float]) -> float:
            current_world_coordinate = GeodeticWorldCoordinate([lonlat_coord[0], lonlat_coord[1], 0.0])
            self.context.ground_domain.default_elevation_model.set_elevation(current_world_coordinate)
            if elevation_model:
                elevation_model.set_elevation(current_world_coordinate)
            new_image_coordinate = self.world_to_image(current_world_coordinate)
            return sqrt(
                (image_coordinate.x - new_image_coordinate.x) ** 2 + (image_coordinate.y - new_image_coordinate.y) ** 2
            )

        # Select an initial guess that is at the center of face 1 in the ground domain. Face 1 is defined as the
        # plane V1->V3->V4->V2 so taking a location at the center of the diagonal V1->V4 should start the search off
        # at the center of the ground domain.
        v1 = self.context.ground_domain.geodetic_ground_domain_vertices[0]
        v4 = self.context.ground_domain.geodetic_ground_domain_vertices[3]
        initial_guess = options.get(
            SensorModelOptions.INITIAL_GUESS,
            np.array([(v1.longitude + v4.longitude) / 2.0, (v1.latitude + v4.latitude) / 2.0]),
        )
        if isinstance(initial_guess, List):
            initial_guess = np.array(initial_guess)

        if options.get(SensorModelOptions.FORCE_INITIAL_GUESS, False) is True:
            world_coordinate = GeodeticWorldCoordinate(np.append(initial_guess, 0.0))
        else:
            initial_search_distance = options.get(
                SensorModelOptions.INITIAL_SEARCH_DISTANCE,
                sqrt(((v1.longitude - v4.longitude) ** 2) + ((v1.latitude - v4.latitude) ** 2)),
            )

            # Iteratively adjust the initial guess to minimize the distance to the target image coordinate. We are only
            # allowing the x,y components to vary here and the z is fixed to the elevation model used by the ground
            # domain. The starting simplex is estimated as a triangle centered in the ground domain.
            res = minimize(
                distance_to_target_coordinate,
                initial_guess,
                method="Nelder-Mead",
                bounds=[
                    (
                        self.context.ground_domain.geodetic_lonlat_bbox[0],
                        self.context.ground_domain.geodetic_lonlat_bbox[2],
                    ),
                    (
                        self.context.ground_domain.geodetic_lonlat_bbox[1],
                        self.context.ground_domain.geodetic_lonlat_bbox[3],
                    ),
                ],
                options={
                    "xatol": radians(0.000001),
                    "fatol": 0.5,
                    "initial_simplex": equilateral_triangle(initial_guess.tolist(), initial_search_distance),
                },
            )

            # The minimization result is a (longitude,latitude) tuple, so we need to expand it to
            # longitude,latitude,elevation and replace the z component with the height from the elevation model.
            world_coordinate = GeodeticWorldCoordinate(np.append(res.x, 0.0))
            success = res.success
            min_dist = options.get(SensorModelOptions.MIN_SUCCESS_DISTANCE_PIXELS, 1.0)
            if not success or distance_to_target_coordinate(world_coordinate.coordinate[:2]) > min_dist:
                if options.get(SensorModelOptions.FALLBACK_INITIAL_GUESS, False) is True:
                    world_coordinate = GeodeticWorldCoordinate(np.append(initial_guess, 0.0))
                elif options.get(SensorModelOptions.EXCEPTION_ON_FAILURE, False) is True:
                    raise RuntimeError(f"{self.__class__.__name__} failed to converge.")

        self.context.ground_domain.default_elevation_model.set_elevation(world_coordinate)
        if elevation_model:
            elevation_model.set_elevation(world_coordinate)

        return world_coordinate

    def ground_domain_to_image(self, domain_coordinate: WorldCoordinate) -> ImageCoordinate:
        """
        This function implements the polynomial ground-to-image transform as defined by section 10.3.3.1.1 of the
        Manual of Photogrammetry sixth edition. The world coordinate is first normalized using the offsets and scale
        factors provided. Then the rational polynomial equations are run to produce an x,y image coordinate. Those
        components are then denormalized to find the final image coordinate.

        :param domain_coordinate: the ground domain coordinate (x, y, z)

        :return: the image coordinate (x, y)
        """
        norm_domain_coordinate = self.normalize_world_coordinate(domain_coordinate)
        norm_row = self.row_numerator_poly(norm_domain_coordinate) / self.row_denominator_poly(norm_domain_coordinate)
        norm_column = self.column_numerator_poly(norm_domain_coordinate) / self.column_denominator_poly(
            norm_domain_coordinate
        )
        return self.denormalize_image_coordinate(ImageCoordinate([norm_column, norm_row]))

    def normalize_world_coordinate(self, world_coordinate: WorldCoordinate) -> WorldCoordinate:
        """
        This is a helper function used to normalize a world coordinate for use with the polynomials in this sensor
        model.

        :param world_coordinate: the world coordinate (longitude, latitude, elevation)

        :return: a world coordinate where each component has been normalized
        """
        norm_x = RSMPolynomialSensorModel.normalize(world_coordinate.x, self.x_norm_offset, self.x_norm_scale)
        norm_y = RSMPolynomialSensorModel.normalize(world_coordinate.y, self.y_norm_offset, self.y_norm_scale)
        norm_z = RSMPolynomialSensorModel.normalize(world_coordinate.z, self.z_norm_offset, self.z_norm_scale)
        return WorldCoordinate([norm_x, norm_y, norm_z])

    def denormalize_world_coordinate(self, world_coordinate: WorldCoordinate) -> WorldCoordinate:
        """
        This is a helper function used to denormalize a world coordinate for use with the polynomials in this sensor
        model.

        :param world_coordinate: the normalized world coordinate (longitude, latitude, elevation)

        :return: a world coordinate
        """
        denorm_x = RSMPolynomialSensorModel.denormalize(world_coordinate.x, self.x_norm_offset, self.x_norm_scale)
        denorm_y = RSMPolynomialSensorModel.denormalize(world_coordinate.y, self.y_norm_offset, self.y_norm_scale)
        denorm_z = RSMPolynomialSensorModel.denormalize(world_coordinate.z, self.z_norm_offset, self.z_norm_scale)
        return WorldCoordinate([denorm_x, denorm_y, denorm_z])

    def normalize_image_coordinate(self, image_coordinate: ImageCoordinate) -> ImageCoordinate:
        """
        This is a helper function used to normalize an image coordinate for use in these polynomials.

        :param image_coordinate: the image coordinate (x, y)

        :return: the normalized image coordinate (x, y)
        """
        norm_line = RSMPolynomialSensorModel.normalize(image_coordinate.r, self.row_norm_offset, self.row_norm_scale)
        norm_samp = RSMPolynomialSensorModel.normalize(image_coordinate.c, self.column_norm_offset, self.column_norm_scale)
        return ImageCoordinate([norm_samp, norm_line])

    def denormalize_image_coordinate(self, image_coordinate: ImageCoordinate) -> ImageCoordinate:
        """
        This is a helper function to denormalize an image coordinate after it has been processed by the polynomials.

        :param image_coordinate: the normalized image coordinate (x, y)

        :return: the image coordinate (x, y)
        """
        denorm_line = RSMPolynomialSensorModel.denormalize(image_coordinate.r, self.row_norm_offset, self.row_norm_scale)
        denorm_samp = RSMPolynomialSensorModel.denormalize(
            image_coordinate.c, self.column_norm_offset, self.column_norm_scale
        )
        return ImageCoordinate([denorm_samp, denorm_line])

    @staticmethod
    def normalize(value: float, offset: float, scale: float) -> float:
        """
        This function normalizes a value using an offset and scale using the equations defined in Section 7.2 of
        STDI-0002 Volume 1 Appendix U.

        :param value: the value to be normalized
        :param offset: the normalization offset
        :param scale: the normalization scale

        :return: the normalized value
        """
        return (value - offset) / scale

    @staticmethod
    def denormalize(value: float, offset: float, scale: float) -> float:
        """
        This function denormalizes a value using an offset and scale using the equations defined in Section 7.2 of
        STDI-0002 Volume 1 Appendix U.

        :param value: the normalized value
        :param offset: the normalization offset
        :param scale: the normalization scale

        :return: the denormalized value
        """
        return value * scale + offset


class RSMSectionedPolynomialSensorModel(RSMSensorModel):
    """
    This is an implementation of a sectioned sensor model that splits overall RSM domain into multiple regions each
    serviced by a dedicated sensor model. A low complexity sensor model covering the entire domain is used to first
    identify the general region of the image associated with a world coordinate then the final coordinate transform
    is delegated to a sensor model associated with that section.
    """

    def __init__(
        self,
        context: RSMContext,
        row_num_image_sections: int,
        column_num_image_sections: int,
        row_section_size: float,
        column_section_size: float,
        row_polynomial: RSMLowOrderPolynomial,
        column_polynomial: RSMLowOrderPolynomial,
        section_sensor_models: List[List[SensorModel]],
    ) -> None:
        """
        This constructs the sensor model using parameters normally found in the NITF RSMPI TRE. Note that the
        per-section sensor models should be constructed from their individual RSMPC TREs.

        :param context: contextual information describing the collection environment
        :param row_num_image_sections: the number of row image sections
        :param column_num_image_sections: the number of column image sections
        :param row_section_size: the size of row section
        :param column_section_size: the size of column section
        :param row_polynomial: polynomial used in row calculations
        :param column_polynomial: polynomial used in column calculations
        :param section_sensor_models: the list of sensor models

        :return: None
        """
        super().__init__(context)
        self.row_num_image_sections = row_num_image_sections
        self.column_num_image_sections = column_num_image_sections
        self.row_section_size = row_section_size
        self.column_section_size = column_section_size
        self.row_polynomial = row_polynomial
        self.column_polynomial = column_polynomial
        self.section_sensor_models = section_sensor_models

    def world_to_image(self, geodetic_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This function transforms a geodetic world coordinate (longitude, latitude, elevation) into an image coordinate
        (x, y).

        :param geodetic_coordinate: the world coordinate (longitude, latitude, elevation)

        :return: the resulting image coordinate (x,y)
        """

        # Use the low order polynomials to approximate an x,y location of this world coordinate in the image
        domain_coordinate = self.context.ground_domain.geodetic_to_ground_domain_coordinate(geodetic_coordinate)
        approximate_row = self.row_polynomial(domain_coordinate)
        approximate_column = self.column_polynomial(domain_coordinate)

        # Find the section of the image containing this approximate coordinate and select the corresponding sensor
        # model
        approximate_image_coordinate = ImageCoordinate([approximate_column, approximate_row])
        column_section_index, row_section_index = self.get_section_index(approximate_image_coordinate)
        section_sensor_model = self.section_sensor_models[row_section_index][column_section_index]

        # Use the selected sensor model to complete the full precision world to image transformation
        return section_sensor_model.world_to_image(geodetic_coordinate)

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function implements the image to world transform by selecting the sensor model responsible for coordinates
        in the image section and then delegating the image to world calculations to that sensor model.

        :param image_coordinate: the image coordinate (x, y)
        :param elevation_model: optional elevation model used to transform the coordinate
        :param options: optional dictionary of hints passed on to the section sensor models

        :return: the corresponding world coordinate (longitude, latitude, elevation)
        """

        # Find the section of the image containing this coordinate and select the corresponding sensor model
        column_section_index, row_section_index = self.get_section_index(image_coordinate)
        section_camera = self.section_sensor_models[row_section_index][column_section_index]

        # Use the selected sensor model to complete the full precision image to world transformation
        return section_camera.image_to_world(image_coordinate, elevation_model=elevation_model, options=options)

    def get_section_index(self, image_coordinate: ImageCoordinate) -> Tuple[int, int]:
        """
        Use the equations from STDO-0002 Volume 1 Appendix U Section 6.3 to calculate the section of this image
        containing the rough x, y. Note that these equations are slightly different from the documentation since
        those equations produce section numbers that start with 1, and we're starting with 0 to more naturally
        index into the array of sensor models. Note that if the value is outside the normal sections it is clamped
        to use the sensor model from the closest section available.

        :param image_coordinate: the image coordinate (x, y)

        :return: section index (x, y)
        """
        row_section_index = floor((image_coordinate.y - self.context.image_domain.min_row) / self.row_section_size)
        if row_section_index < 0:
            row_section_index = 0
        elif row_section_index >= self.row_num_image_sections:
            row_section_index = self.row_num_image_sections - 1

        column_section_index = floor((image_coordinate.x - self.context.image_domain.min_column) / self.column_section_size)
        if column_section_index < 0:
            column_section_index = 0
        elif column_section_index >= self.column_num_image_sections:
            column_section_index = self.column_num_image_sections - 1

        return column_section_index, row_section_index
