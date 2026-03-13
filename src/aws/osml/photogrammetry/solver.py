#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from abc import ABC, abstractmethod
from typing import Tuple

from .coordinates import GeodeticWorldCoordinate


class Solver(ABC):
    """
    A sensor model solver abstracts the actual routine (such as an optimization routine) used to convert image to world
    coordinates.
    """

    @abstractmethod
    def solve(self) -> Tuple[GeodeticWorldCoordinate, bool]:
        """
        Solve for the world coordinate.

        :return: the world coordinate and a boolean that is True on success, False on failure
        """
