#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from typing import List, Optional

from .coordinates import GeodeticWorldCoordinate
from .elevation_model import ElevationModel, ElevationRegionSummary


class MultiElevationModel(ElevationModel):
    """
    Call multiple elevation models in order, using the result of the first to successfully update the elevation value.
    """

    def __init__(self, elevation_models: List[ElevationModel]) -> None:
        """
        Initialize using a list of elevation models.

        :param elevation_models: the ordered models

        :return: None
        """
        super().__init__()
        self.elevation_models = elevation_models

    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Set elevation by using multiple inner models.

        :param geodetic_world_coordinate: the coordinate to update

        :return: True if the elevation was updated, else False
        """
        for elevation_model in self.elevation_models:
            if elevation_model.set_elevation(world_coordinate):
                return True
        return False

    def describe_region(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[ElevationRegionSummary]:
        """
        Unimplemented summary of region near the provided world coordinate

        :param geodetic_world_coordinate: the coordinate at the center of the region of interest

        :return: None
        """
        return None
