#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from math import radians
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from .bounded_solver import GenericBoundedSolver
from .coordinates import GeodeticWorldCoordinate
from .math_utils import equilateral_triangle
from .sensor_model import SensorModelOptions


class BoundedNelderMeadSolver(GenericBoundedSolver):
    """
    An optimization-based Solver using Nelder-Mead.
    """

    def solve(self) -> Tuple[GeodeticWorldCoordinate, bool]:
        """
        Solve for the world coordinate.

        :return: the world coordinate and a boolean that is True on success, False on failure
        """
        bounds = None
        if self.lon_bounds is not None and self.lat_bounds is not None:
            bounds = [self.lon_bounds, self.lat_bounds]
        res = minimize(
            lambda x: self.minimization_function(x, self.elevation_model),
            self.initial_guess,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "xatol": radians(0.000001),
                "fatol": 0.5,
                "initial_simplex": equilateral_triangle(self.initial_guess.tolist(), self.search_distance),
            },
        )
        world_coordinate = GeodeticWorldCoordinate(np.append(res.x, 0.0))
        self.elevation_model.set_elevation(world_coordinate)
        return world_coordinate, res.success


SensorModelOptions.register_bounded_solver("neldermead", BoundedNelderMeadSolver)
