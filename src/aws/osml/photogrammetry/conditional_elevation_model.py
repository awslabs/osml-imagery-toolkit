#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from typing import Optional

from .coordinates import GeodeticWorldCoordinate
from .elevation_model import ElevationModel, ElevationRegionSummary
from .em_condition import ElevationModelCondition


class ConditionalElevationModel(ElevationModel):
    """
    An elevation model that sets elevation using an inner elevation model only if it first passes a condition on the
    given coordinate.
    """

    def __init__(
        self,
        inner_elevation_model: ElevationModel,
        em_condition: ElevationModelCondition,
    ) -> None:
        """
        Create the model using a nested ElevationModel and a condition for its usage.

        :param inner_elevation_model: the inner model that actually sets elevation
        :param em_condition: the condition that must pass to set elevation

        :return: None
        """
        super().__init__()
        self.inner_elevation_model = inner_elevation_model
        self.em_condition = em_condition

    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Set elevation conditionally using the inner model.

        :param world_coordinate: the coordinate to update

        :return: True if the elevation was updated, else False
        """
        if self.em_condition.is_true(world_coordinate):
            return self.inner_elevation_model.set_elevation(world_coordinate)
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
