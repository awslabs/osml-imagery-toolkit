#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import unittest
from typing import Optional

import shapely


class TestEMConditionIntersects(unittest.TestCase):
    def setUp(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.geometry_query import GeometryQuery

        class TestGeometryQuery(GeometryQuery):
            def get_geometry(
                self,
                world_coordinate: GeodeticWorldCoordinate,
            ) -> Optional[shapely.Geometry]:
                if abs(world_coordinate.x) <= 5.0 and abs(world_coordinate.y) <= 5.0:
                    return shapely.Polygon(
                        [
                            (-5, -5),
                            (5, -5),
                            (5, 5),
                            (-5, 5),
                            (-5, -5),
                        ],
                    )
                return None

        self.test_geometry_query = TestGeometryQuery()

    def test_intersection(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.em_condition_intersects import EMConditionIntersects

        em_condition_intersects = EMConditionIntersects(self.test_geometry_query)
        in_world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert em_condition_intersects.is_true(in_world_coordinate)
        out_world_coordinate = GeodeticWorldCoordinate([5.0, -6.0, 0.0])
        assert not em_condition_intersects.is_true(out_world_coordinate)

    def test_inverted_intersection(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.em_condition_intersects import EMConditionIntersects

        em_condition_intersects = EMConditionIntersects(self.test_geometry_query, invert=True)
        in_world_coordinate = GeodeticWorldCoordinate([5.0, -6.0, 0.0])
        assert em_condition_intersects.is_true(in_world_coordinate)
        out_world_coordinate = GeodeticWorldCoordinate([1.0, 2.0, 0.0])
        assert not em_condition_intersects.is_true(out_world_coordinate)


if __name__ == "__main__":
    unittest.main()
