#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from math import radians
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from . import eim_registry
from .coordinates import GeodeticWorldCoordinate
from .earth_intersection_minimizer import EarthIntersectionMinimizer
from .elevation_model import ElevationModel
from .math_utils import equilateral_triangle


class EIMNelderMead(EarthIntersectionMinimizer):
    """
    An optimization routine using Nelder-Mead.
    """

    def solve(
        self,
        minimization_function: Callable[
            [Tuple[float, float], ElevationModel],
            float,
        ],
        elevation_model: ElevationModel,
        initial_guess: npt.ArrayLike,
        search_distance: float,
        lon_bounds: Optional[Tuple[float, float]] = None,
        lat_bounds: Optional[Tuple[float, float]] = None,
        height_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[GeodeticWorldCoordinate, bool]:
        """
        Solve for the world coordinate.

        :param minimization_function: the function to minimize, taking a lon/lat tuple in radians and elevation model
        :param elevation_model: the elevation model to use during minimization
        :param initial_guess: initial lon/lat radians array guess
        :param search_distance: search distance from the initial guess, in radians
        :param lon_bounds: absolute longitude bounds in radians
        :param lat_bounds: absolute latitude bounds in radians
        :param height_bounds: absolute height bounds in meters

        :return: the world coordinate and a boolean that is True on success, False on failure
        """
        bounds = None
        if lon_bounds is not None and lat_bounds is not None:
            bounds = [lon_bounds, lat_bounds]
        res = minimize(
            lambda x: minimization_function(x, elevation_model),
            initial_guess,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "xatol": radians(0.000001),
                "fatol": 0.5,
                "initial_simplex": equilateral_triangle(initial_guess.tolist(), search_distance),
            },
        )
        world_coordinate = GeodeticWorldCoordinate(np.append(res.x, 0.0))
        elevation_model.set_elevation(world_coordinate)
        return world_coordinate, res.success


eim_registry.register("neldermead", EIMNelderMead())
