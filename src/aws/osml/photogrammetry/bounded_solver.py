#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from abc import abstractmethod
from typing import Callable, Optional, Tuple

import numpy.typing as npt

from .elevation_model import ElevationModel
from .solver import Solver


class BoundedSolver(Solver):
    """
    A Solver based on a minimization function and bounds.
    """

    @classmethod
    @abstractmethod
    def from_bounds(
        cls,
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
    ):
        """
        Create solver from common parameters.

        :param minimization_function: the function, taking a lon/lat tuple and elevation model, to minimize
        :param elevation_model: the elevation model to use during minimization
        :param initial_guess: initial lon/lat radians array guess
        :param search_distance: search distance from the initial guess, in radians
        :param lon_bounds: absolute longitude bounds in radians
        :param lat_bounds: absolute latitude bounds in radians
        :param height_bounds: absolute height bounds in meters
        """


class GenericBoundedSolver(BoundedSolver):
    """
    BoundedSolver that initializes from common parameters.
    """

    def __init__(
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
    ) -> None:
        """
        Store metadata to use during solve.

        :param minimization_function: the function, taking a lon/lat tuple and elevation model, to minimize
        :param elevation_model: the elevation model to use during minimization
        :param initial_guess: initial lon/lat radians array guess
        :param search_distance: search distance from the initial guess, in radians
        :param lon_bounds: absolute longitude bounds in radians
        :param lat_bounds: absolute latitude bounds in radians
        :param height_bounds: absolute height bounds in meters
        """
        self.minimization_function = minimization_function
        self.elevation_model = elevation_model
        self.initial_guess = initial_guess
        self.search_distance = search_distance
        self.lon_bounds = lon_bounds
        self.lat_bounds = lat_bounds
        self.height_bounds = height_bounds

    @classmethod
    def from_bounds(
        cls,
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
    ):
        """
        Create solver from common parameters.

        :param minimization_function: the function, taking a lon/lat tuple and elevation model, to minimize
        :param elevation_model: the elevation model to use during minimization
        :param initial_guess: initial lon/lat radians array guess
        :param search_distance: search distance from the initial guess, in radians
        :param lon_bounds: absolute longitude bounds in radians
        :param lat_bounds: absolute latitude bounds in radians
        :param height_bounds: absolute height bounds in meters
        """
        return cls(
            minimization_function,
            elevation_model,
            initial_guess,
            search_distance,
            lon_bounds,
            lat_bounds,
            height_bounds,
        )
