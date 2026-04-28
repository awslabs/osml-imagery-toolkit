#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from aws.osml.photogrammetry import ElevationModelCondition, GeodeticWorldCoordinate, GeometryQuery


class GeometryCondition(ElevationModelCondition):
    """
    Elevation model condition that evaluates True when a point is inside
    a geometry from the query (or False when invert=True).
    """

    def __init__(self, geometry_query: GeometryQuery, invert: bool = False) -> None:
        super().__init__()
        self.geometry_query = geometry_query
        self.invert = invert

    def is_true(self, world_coordinate: GeodeticWorldCoordinate) -> bool:
        result = self.geometry_query.get_geometry(world_coordinate) is not None
        return not result if self.invert else result
