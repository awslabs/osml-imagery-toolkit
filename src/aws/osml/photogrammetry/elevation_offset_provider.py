#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from abc import ABC, abstractmethod

from .coordinates import GeodeticWorldCoordinate


class ElevationOffsetProvider(ABC):
    """
    A base class functor for calculating a WGS84 height offset, in meters, given a coordinate. It is common for
    elevation data sources to reference a 0-level that does not match WGS84.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_offset(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> float:
        """
        Provide a WGS84 height offset, in meters, given a coordinate.

        :param geodetic_world_coordinate: the world coordinate of interest

        :return: meters above WGS84 ellipsoid
        """


class ConstantOffsetProvider(ElevationOffsetProvider):
    """
    Provide a constant WGS84 height offset.
    """

    def __init__(self, constant_offset: float) -> None:
        """
        Create the provider using a supplied offset.

        :param constant_offset: meters above WGS84 ellipsoid

        :return: None
        """
        super().__init__()
        self.constant_offset = constant_offset

    def get_offset(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> float:
        """
        Provide a WGS84 height offset, in meters, given a coordinate.

        :param geodetic_world_coordinate: the world coordinate of interest

        :return: meters above WGS84 ellipsoid
        """
        return self.constant_offset
