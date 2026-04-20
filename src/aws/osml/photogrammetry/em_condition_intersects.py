#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

from .coordinates import GeodeticWorldCoordinate
from .em_condition import ElevationModelCondition
from .geometry_query import GeometryQuery


class EMConditionIntersects(ElevationModelCondition):
    """
    An ElevationModel condition based on a geometry intersection query.
    """

    def __init__(
        self,
        geom_query: GeometryQuery,
        invert: bool = False,
    ) -> None:
        """
        :param geom_query: returns surrounding geometry for a point
        :param invert: if True, invert the query result

        :return: None
        """
        super().__init__()
        self.geom_query = geom_query
        self.invert = invert

    def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        Return True if the supplied coordinate intersects a geometry (or False if inverted).

        :param world_coordinate: the coordinate to evaluate

        :return: True if condition passes, else False
        """
        intersects = self.geom_query.get_geometry(world_coordinate) is not None
        return intersects if not self.invert else not intersects
