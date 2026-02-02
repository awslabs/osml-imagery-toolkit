#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from abc import ABC, abstractmethod
from typing import Optional

import shapely

from .coordinates import GeodeticWorldCoordinate


class GeometryQuery(ABC):
    """
    Define an abstraction for a query returning a geometry.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_geometry(
        self,
        world_coordinate: GeodeticWorldCoordinate,
    ) -> Optional[shapely.Geometry]:
        """
        Get a geometry (first, if many) containing a supplied point.

        :param world_coordinate: the point of interest

        :return: the geometry
        """
