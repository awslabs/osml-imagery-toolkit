#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy.typing as npt

from .coordinates import GeodeticWorldCoordinate
from .elevation_model import ElevationModel


class EarthIntersectionMinimizer(ABC):
    """
    A routine that minimizes a cost function with ElevationModel for Earth intersection.
    """

    @abstractmethod
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
