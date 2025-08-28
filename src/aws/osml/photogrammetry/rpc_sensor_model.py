#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from math import degrees, radians, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from . import ConstantElevationModel, ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, WorldCoordinate
from .math_utils import equilateral_triangle
from .sensor_model import SensorModel, SensorModelOptions


class RPCPolynomial:
    def __init__(self, coefficients: List[float]) -> None:
        """
        The order of terms in the coefficients is defined to match the RPC00B specification as defined in STDI-0002
        Volume 1 Appendix E Section E.2.4. Note that the order of terms for coefficients taken from other TREs (e.g.
        RPC00A is different).

        :param coefficients: the polynomial coefficients

        :return: None
        """
        self.coefficients = coefficients

    def evaluate(self, normalized_world_coordinate: WorldCoordinate) -> float:
        """
        This function evaluates the polynomial for the given world coordinate by summing up the result of applying
        each coefficient to the world coordinate components. Note that these polynomials are usually defined with
        the assumption that the world coordinate has been normalized.

        :param normalized_world_coordinate: the world coordinate

        :return: the resulting value
        """
        # The equations in the RPC00B definition use L, P, H to denote the normalized longitude, latitude, and
        # elevation components respectively.
        l_val = normalized_world_coordinate.x
        p_val = normalized_world_coordinate.y
        h_val = normalized_world_coordinate.z

        result = 0.0
        result += self.coefficients[0]  # constant
        result += self.coefficients[1] * l_val  # L
        result += self.coefficients[2] * p_val  # P
        result += self.coefficients[3] * h_val  # H
        result += self.coefficients[4] * l_val * p_val  # LP
        result += self.coefficients[5] * l_val * h_val  # LH
        result += self.coefficients[6] * p_val * h_val  # PH
        result += self.coefficients[7] * l_val * l_val  # LL
        result += self.coefficients[8] * p_val * p_val  # PP
        result += self.coefficients[9] * h_val * h_val  # HH
        result += self.coefficients[10] * l_val * p_val * h_val  # LPH
        result += self.coefficients[11] * l_val * l_val * l_val  # LLL
        result += self.coefficients[12] * l_val * p_val * p_val  # LPP
        result += self.coefficients[13] * l_val * h_val * h_val  # LHH
        result += self.coefficients[14] * l_val * l_val * p_val  # LLP
        result += self.coefficients[15] * p_val * p_val * p_val  # PPP
        result += self.coefficients[16] * p_val * h_val * h_val  # PHH
        result += self.coefficients[17] * l_val * l_val * h_val  # LLH
        result += self.coefficients[18] * p_val * p_val * h_val  # PPH
        result += self.coefficients[19] * h_val * h_val * h_val  # HHH
        return result

    def __call__(self, *args, **kwargs) -> float:
        """
        This makes the polynomial object callable such that it can be applied to a world coordinate directly e.g.
        result = polynomial(coordinate).

        :param args: the world coordinate
        :param kwargs: no keyword arguments are accepted but this is required for the Callable interface
        :return: the result of the evaluation
        """
        return self.evaluate(args[0])


class RPCSensorModel(SensorModel):
    """
    A Rational Polynomial Camera (RPC) sensor model is one where the world to image transform is approximated using
    a ratio of polynomials. The polynomials capture specific relationships between the latitude, longitude, and
    elevation and the image pixels.

    These cameras were common approximations for many years but started to be phased out in 2014 in favor of the
    more general Replacement Sensor Model (RSM). It is not uncommon to find historical imagery or imagery from
    older sensors still operating today that contain the metadata for these sensor models.
    """

    def __init__(
        self,
        err_bias: float,
        err_rand: float,
        line_off: float,
        samp_off: float,
        lat_off: float,
        long_off: float,
        height_off: float,
        line_scale: float,
        samp_scale: float,
        lat_scale: float,
        long_scale: float,
        height_scale: float,
        line_num_poly: RPCPolynomial,
        line_den_poly: RPCPolynomial,
        samp_num_poly: RPCPolynomial,
        samp_den_poly: RPCPolynomial,
    ) -> None:
        """
        This constructs the sensor model using parameters normally found in the NITF RPC00B TRE. Note that there also
        exists RPC00A, and RPC00C tags that contain the same information. Be careful and check the specifications. The
        order of the polynomial coefficients is different even though the structure of the TRE is the same.

        :param err_bias: non time-varying error estimate assumes correlated images
        :param err_rand: time-varying error estimate assumes uncorrelated images
        :param line_off: offset used to normalize/denormalize image y components
        :param samp_off: offset used to normalize/denormalize image x components
        :param lat_off: offset used to normalize/denormalize world y components
        :param long_off: offset used to normalize/denormalize world x components
        :param height_off: offset used to normalize/denormalize world z components
        :param line_scale: scale used to normalize/denormalize image y components
        :param samp_scale: scale used to normalize/denormalize image x components
        :param lat_scale: scale used to normalize/denormalize world y components
        :param long_scale: scale used to normalize/denormalize world x components
        :param height_scale: scale used to normalize/denormalize world z components
        :param line_num_poly: polynomial used as the numerator in line (row) calculations
        :param line_den_poly: polynomial used as the denominator in line (row) calculations
        :param samp_num_poly: polynomial used as the numerator in sample (column) calculations
        :param samp_den_poly: polynomial used as the denominator in sample (column) calculations

        :return: None
        """
        super().__init__()
        self.err_bias = err_bias
        self.err_rand = err_rand
        self.line_off = line_off
        self.samp_off = samp_off
        self.lat_off = lat_off
        self.long_off = long_off
        self.height_off = height_off
        self.line_scale = line_scale
        self.samp_scale = samp_scale
        self.lat_scale = lat_scale
        self.long_scale = long_scale
        self.height_scale = height_scale
        self.line_numerator_poly = line_num_poly
        self.line_denominator_poly = line_den_poly
        self.samp_numerator_poly = samp_num_poly
        self.samp_denominator_poly = samp_den_poly
        self.default_elevation_model = ConstantElevationModel(self.height_off)

    def world_to_image(self, geodetic_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This function transforms a geodetic world coordinate (longitude, latitude, elevation) into an image coordinate
        (x, y).

        :param geodetic_coordinate: the world coordinate (longitude, latitude, elevation)

        :return: the resulting image coordinate (x,y)
        """
        # The RPC spec assumes the ground coordinates are latitude and longitude in units of decimal degrees and
        # the geodetic elevation in units of meters. The ground coordinates are referenced to WGS-84. The
        # GeodeticWorldCoordinate is assumed to be in radians with meters elevation. Convert the coordinate to
        # degrees before normalizing the values.
        # TODO: Check that GeodeticWorldCoordinate in RSM is meters elevation
        l_val = (degrees(geodetic_coordinate.x) - self.long_off) / self.long_scale
        p_val = (degrees(geodetic_coordinate.y) - self.lat_off) / self.lat_scale
        h_val = (geodetic_coordinate.z - self.height_off) / self.height_scale
        norm_domain_coordinate = WorldCoordinate([l_val, p_val, h_val])

        # Evaluate the rational polynomials
        cn = self.samp_numerator_poly(norm_domain_coordinate) / self.samp_denominator_poly(norm_domain_coordinate)
        rn = self.line_numerator_poly(norm_domain_coordinate) / self.line_denominator_poly(norm_domain_coordinate)

        # Denormalize the rational polynomial results to get the actual x,y of the image coordinate
        col = cn * self.samp_scale + self.samp_off
        row = rn * self.line_scale + self.line_off
        return ImageCoordinate([col, row])

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function implements the image to world transform by iteratively invoking world to image within a
        minimization routine to find a matching image coordinate. The longitude and latitude parameters are searched
        independently while the elevation of the world coordinate comes from the elevation model.

        :param image_coordinate: the image coordinate (x, y)
        :param elevation_model: an optional elevation model used to transform the coordinate
        :param options: optional hints, supports initial_guess and initial_search_distance

        :return: the corresponding world coordinate
        """

        options = options if options is not None else {}

        # This is the function we will be minimizing. Given an x,y coordinate in the ground domain we use invoke the
        # ground_domain_to_image function to get a projection of that location in the image. Then we compute the
        # distance between that new image location and the input image location. When those locations match then
        # we know we have the ground domain coordinate that corresponds to the input.
        def distance_to_target_coordinate(lonlat_coord: Tuple[float, float]) -> float:
            ground_domain_coordinate = GeodeticWorldCoordinate([lonlat_coord[0], lonlat_coord[1], 0.0])
            self.default_elevation_model.set_elevation(ground_domain_coordinate)
            if elevation_model:
                elevation_model.set_elevation(ground_domain_coordinate)
            new_image_coordinate = self.world_to_image(ground_domain_coordinate)
            return sqrt(
                (image_coordinate.x - new_image_coordinate.x) ** 2 + (image_coordinate.y - new_image_coordinate.y) ** 2
            )

        # Select an initial guess using the normalization offsets for this camera model. Normally these are values
        # near an image corner or the center
        initial_guess = options.get(
            SensorModelOptions.INITIAL_GUESS,
            np.array([radians(self.long_off), radians(self.lat_off)]),
        )
        if isinstance(initial_guess, List):
            initial_guess = np.array(initial_guess)

        if options.get(SensorModelOptions.FORCE_INITIAL_GUESS, False) is True:
            world_coordinate = GeodeticWorldCoordinate(np.append(initial_guess, 0.0))
        else:
            initial_search_distance = options.get(SensorModelOptions.INITIAL_SEARCH_DISTANCE, radians(0.5))

            # Iteratively adjust the initial guess to minimize the distance to the target image coordinate. We are only
            # allowing the x,y components to vary here and the z is fixed to the elevation model. The starting simplex
            # is estimated as a triangle centered on the normalization offsets for this RPC.
            res = minimize(
                distance_to_target_coordinate,
                initial_guess,
                method="Nelder-Mead",
                options={
                    "xatol": radians(0.000001),
                    "fatol": 0.5,
                    "initial_simplex": equilateral_triangle(initial_guess.tolist(), initial_search_distance),
                },
            )

            # The minimization result is an (x,y) tuple, so we need to expand it to x,y,z and replace the z component with
            # the height from the elevation model. Note that the units of this are radians, radians, meters
            world_coordinate = GeodeticWorldCoordinate(np.append(res.x, 0.0))
            success = res.success
            min_dist = options.get(SensorModelOptions.MIN_SUCCESS_DISTANCE_PIXELS, 1.0)
            if not success or distance_to_target_coordinate(world_coordinate.coordinate[:2]) > min_dist:
                if options.get(SensorModelOptions.FALLBACK_INITIAL_GUESS, False) is True:
                    world_coordinate = GeodeticWorldCoordinate(np.append(initial_guess, 0.0))
                elif options.get(SensorModelOptions.EXCEPTION_ON_FAILURE, False) is True:
                    raise RuntimeError(f"{self.__class__.__name__} failed to converge.")

        self.default_elevation_model.set_elevation(world_coordinate)
        if elevation_model:
            elevation_model.set_elevation(world_coordinate)

        return world_coordinate
