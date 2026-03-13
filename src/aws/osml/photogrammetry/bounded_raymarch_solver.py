#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from typing import List, Optional, Tuple

import numpy as np

from .bounded_neldermead_solver import BoundedNelderMeadSolver
from .bounded_solver import GenericBoundedSolver
from .coordinates import GeodeticWorldCoordinate
from .elevation_model import ConstantElevationModel, ElevationModel
from .sensor_model import SensorModelOptions


class BoundedRayMarchSolver(GenericBoundedSolver):
    """
    A Solver which first steps along a ray to get an approximate solution, then calls the BoundedNelderMeadSolver.
    """

    @staticmethod
    def _find_intersection_step(
        elevation_model: ElevationModel,
        start: np.ndarray,
        step: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Find the earth intersection of a ray by stepping along it from a starting point.

        :param elevation_model: the ElevationModel used to determine height
        :param start: the start position of the ray, in [lon, lat, height]
        :param step: the step of the ray, in [lon, lat, height], with positive height

        :return: the position below the intersection, and the position above it (or None if it failed)
        """
        above_position = start.copy()
        below_position = start.copy()
        world_coordinate = GeodeticWorldCoordinate(start.copy())
        if not elevation_model.set_elevation(world_coordinate):
            return None
        if start[2] > world_coordinate.z:
            while below_position[2] - world_coordinate.z > 0:
                above_position[:] = below_position[:]
                below_position -= step
                world_coordinate.x = below_position[0]
                world_coordinate.y = below_position[1]
                if not elevation_model.set_elevation(world_coordinate):
                    return None
        else:
            while above_position[2] - world_coordinate.z < 0:
                below_position[:] = above_position[:]
                above_position += step
                world_coordinate.x = above_position[0]
                world_coordinate.y = above_position[1]
                if not elevation_model.set_elevation(world_coordinate):
                    return None
        return (below_position, above_position)

    @staticmethod
    def _find_intersection_multistep(
        elevation_model: ElevationModel,
        start: np.ndarray,
        base_step: np.ndarray,
        step_multiples: List[float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Find the earth intersection of a ray by stepping along it from a starting point.

        :param elevation_model: the ElevationModel used to determine height
        :param start: the start position of the ray, in [lon, lat, height]
        :param base_step: the base step size of the ray, in [lon, lat, height], with positive height
        :param step_multiples: incremental, positive step multiples used to narrow intersection range

        :return: the positions below the intersection, and the position above it (or None if it failed)
        """
        if len(step_multiples) == 0:
            return None
        position = start
        for step_multiple in step_multiples:
            position_bounds = BoundedRayMarchSolver._find_intersection_step(
                elevation_model,
                position,
                step_multiple * base_step,
            )
            if position_bounds is None:
                return None
            position = position_bounds[1]
        return position_bounds

    def solve(self) -> Tuple[GeodeticWorldCoordinate, bool]:
        """
        Solve for the world coordinate.

        :return: the world coordinate and a boolean that is True on success, False on failure
        """
        # Get elevation at the initial guess
        initial_guess_world_coord = GeodeticWorldCoordinate(np.append(self.initial_guess, 0.0))
        self.elevation_model.set_elevation(initial_guess_world_coord)
        # Solve for longitude/latitude at constant height for two different heights.
        initial_guess_below_world_coord, below_success = BoundedNelderMeadSolver(
            self.minimization_function,
            ConstantElevationModel(initial_guess_world_coord.z - 1.0),
            self.initial_guess,
            self.search_distance,
        ).solve()
        if not below_success:
            return initial_guess_world_coord, False
        initial_guess_above_world_coord, above_success = BoundedNelderMeadSolver(
            self.minimization_function,
            ConstantElevationModel(initial_guess_world_coord.z + 1.0),
            self.initial_guess,
            self.search_distance,
        ).solve()
        if not above_success:
            return initial_guess_world_coord, False
        # Use those solutions to construct a vector, normalize to a 1m step in z.
        step = initial_guess_above_world_coord.coordinate - initial_guess_below_world_coord.coordinate
        step /= step[2]
        start = (initial_guess_above_world_coord.coordinate + initial_guess_below_world_coord.coordinate) / 2
        # Shoot up to the maximum height.
        new_z = min(max(self.height_bounds[1], start[2]) if self.height_bounds is not None else start[2], 10000.0)
        new_start = start + (new_z - start[2]) * step
        # Use step function to get to approx solution with altitude shifts of 50m, 10m, then 1m.
        position_bounds = BoundedRayMarchSolver._find_intersection_multistep(
            self.elevation_model,
            new_start,
            step,
            [50.0, 10.0, 1.0],
        )
        if position_bounds is None:
            world_coordinate = GeodeticWorldCoordinate(start)
            self.elevation_model.set_elevation(world_coordinate)
            return world_coordinate, False
        position = (position_bounds[0] + position_bounds[1]) / 2
        world_coordinate = GeodeticWorldCoordinate(position)
        self.elevation_model.set_elevation(world_coordinate)
        # Use the BoundedNelderMeadSolver to get the full solution.
        return BoundedNelderMeadSolver(
            self.minimization_function,
            self.elevation_model,
            position[:2],
            min(self.search_distance, 1e-5),
            lon_bounds=self.lon_bounds,
            lat_bounds=self.lat_bounds,
            height_bounds=self.height_bounds,
        ).solve()


SensorModelOptions.register_bounded_solver("raymarch", BoundedRayMarchSolver)
