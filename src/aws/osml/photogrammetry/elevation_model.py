#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .coordinates import GeodeticWorldCoordinate


@dataclass
class ElevationRegionSummary:
    """
    This class contains a general summary of an elevation tile.
    """

    min_elevation: float
    max_elevation: float
    no_data_value: int
    post_spacing: float


class ElevationModel(ABC):
    """
    An elevation model associates a height z for a given x, y of a world coordinate. It typically provides information
    about the terrain associated with longitude, latitude locations of an ellipsoid, but it can also be used to model
    surfaces for other ground domains.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        This method updates the elevation component of a world coordinate to match the surface elevation at
        longitude, latitude.

        :param world_coordinate: the coordinate to update

        :return: True if the elevation was updated, else False
        """

    @abstractmethod
    def describe_region(self, world_coordinate: GeodeticWorldCoordinate) -> Optional[ElevationRegionSummary]:
        """
        Get a summary of the region near the provided world coordinate

        :param world_coordinate: the coordinate at the center of the region of interest
        :return: the summary information
        """


class ConstantElevationModel(ElevationModel):
    """
    A constant elevation model with a single value for all longitude, latitude.
    """

    def __init__(self, constant_elevation: float) -> None:
        """
        Constructs the constant elevation model.

        :param constant_elevation: the elevation value for all longitude, latitude

        :return: None
        """
        super().__init__()
        self.constant_elevation = constant_elevation

    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Updates world coordinate's elevation to match the constant elevation.

        :param world_coordinate: the coordinate to update

        :return: True if the elevation was updated, else False
        """
        world_coordinate.elevation = self.constant_elevation
        return True

    def describe_region(self, world_coordinate: GeodeticWorldCoordinate) -> Optional[ElevationRegionSummary]:
        """
        Get a summary of the region near the provided world coordinate

        :param world_coordinate: the coordinate at the center of the region of interest
        :return: [min elevation value, max elevation value, no elevation data value, post spacing]
        """
        return ElevationRegionSummary(
            min_elevation=self.constant_elevation,
            max_elevation=self.constant_elevation,
            no_data_value=-32767,
            post_spacing=30.0,
        )
