#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from typing import Optional

from .coordinates import GeodeticWorldCoordinate
from .elevation_model import ElevationModel, ElevationRegionSummary
from .elevation_offset_provider import ElevationOffsetProvider


class OffsetElevationModel(ElevationModel):
    """
    An elevation model that adds an offset to the result of an inner elevation model when the elevation is set.
    """

    def __init__(
        self,
        inner_elevation_model: ElevationModel,
        offset_provider: ElevationOffsetProvider,
    ) -> None:
        """
        Create the model using a nested ElevationModel and an offset provider.

        :param inner_elevation_model: the inner model that sets elevation first
        :param offset_provider: provided offsets added second

        :return: None
        """
        self.inner_elevation_model = inner_elevation_model
        self.offset_provider = offset_provider

    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Set elevation using the inner model + offset.

        :param world_coordinate: the coordinate to update

        :return: True if the elevation was updated, else False
        """
        if self.inner_elevation_model.set_elevation(world_coordinate):
            world_coordinate.elevation += self.offset_provider.get_offset(world_coordinate)
            return True
        return False

    def describe_region(
        self,
        world_coordinate: GeodeticWorldCoordinate,
    ) -> Optional[ElevationRegionSummary]:
        """
        Unimplemented summary of region near the provided world coordinate

        :param world_coordinate: the coordinate at the center of the region of interest

        :return: None
        """
        return None
