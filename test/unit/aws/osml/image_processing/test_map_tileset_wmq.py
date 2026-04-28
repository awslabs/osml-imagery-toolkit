#  Copyright 2024 Amazon.com, Inc. or its affiliates.
from unittest import TestCase

import numpy as np

from aws.osml.image_processing import MapTileId, MapTileSetFactory, WellKnownMapTileSet
from aws.osml.photogrammetry import GeodeticWorldCoordinate


class TestWebMercatorQuadTileSet(TestCase):
    def setUp(self) -> None:
        self.wmq_tile_set = MapTileSetFactory.get_for_id(WellKnownMapTileSet.WEB_MERCATOR_QUAD)
        self.wmqx2_tile_set = MapTileSetFactory.get_for_id(WellKnownMapTileSet.WEB_MERCATOR_QUAD_X2)

    def test_well_known_ids(self):
        self.assertEqual(self.wmq_tile_set.tile_matrix_set_id, WellKnownMapTileSet.WEB_MERCATOR_QUAD.value)
        self.assertEqual(self.wmqx2_tile_set.tile_matrix_set_id, WellKnownMapTileSet.WEB_MERCATOR_QUAD_X2.value)

    def test_crs_id(self):
        self.assertEqual(self.wmq_tile_set.crs_id, "EPSG:3857")
        self.assertEqual(self.wmqx2_tile_set.crs_id, "EPSG:3857")

    def test_get_tile(self):
        top_tile_id = MapTileId(tile_matrix=0, tile_row=0, tile_col=0)
        top_tile_256 = self.wmq_tile_set.get_tile(top_tile_id)
        top_tile_512 = self.wmqx2_tile_set.get_tile(top_tile_id)

        self.assertEqual(top_tile_256.id, top_tile_id)
        np.testing.assert_almost_equal(
            top_tile_256.bounds,
            (np.radians(-180.0), np.radians(-85.051128), np.radians(180.0), np.radians(85.051128)),
            decimal=7,
        )
        self.assertEqual(top_tile_256.size, (256, 256))

        self.assertEqual(top_tile_512.id, top_tile_id)
        self.assertEqual(top_tile_512.bounds, top_tile_256.bounds)
        self.assertEqual(top_tile_512.size, (512, 512))

        test_tile_id = MapTileId(tile_matrix=10, tile_row=578, tile_col=856)
        test_tile = self.wmq_tile_set.get_tile(test_tile_id)
        self.assertEqual(test_tile.id, test_tile_id)
        np.testing.assert_almost_equal(test_tile.bounds, (2.1107576, 0.3943349, 2.1168935, 0.3999932), decimal=7)
        self.assertEqual(test_tile.size, (256, 256))

        test_tile = self.wmqx2_tile_set.get_tile(test_tile_id)
        self.assertEqual(test_tile.id, test_tile_id)
        np.testing.assert_almost_equal(test_tile.bounds, (2.1107576, 0.3943349, 2.1168935, 0.3999932), decimal=7)
        self.assertEqual(test_tile.size, (512, 512))

    def test_native_bounds_populated(self):
        tile_id = MapTileId(tile_matrix=0, tile_row=0, tile_col=0)
        tile = self.wmq_tile_set.get_tile(tile_id)
        xmin, ymin, xmax, ymax = tile.native_bounds
        self.assertAlmostEqual(xmin, -20037508.342789244, places=0)
        self.assertAlmostEqual(ymin, -20037508.342789244, places=0)
        self.assertAlmostEqual(xmax, 20037508.342789244, places=0)
        self.assertAlmostEqual(ymax, 20037508.342789244, places=0)

    def test_get_tile_for_location(self):
        expected_tile_id = MapTileId(tile_matrix=10, tile_row=578, tile_col=856)
        test_location = GeodeticWorldCoordinate([2.113, 0.395, 0.0])

        test_tile = self.wmq_tile_set.get_tile_for_location(test_location, tile_matrix=expected_tile_id.tile_matrix)
        self.assertEqual(test_tile.id, expected_tile_id)

        test_tile = self.wmqx2_tile_set.get_tile_for_location(test_location, tile_matrix=expected_tile_id.tile_matrix)
        self.assertEqual(test_tile.id, expected_tile_id)

    def test_get_tile_matrix_limits_for_area_row_first(self):
        """Verify get_tile_matrix_limits_for_area returns (min_row, min_col, max_row, max_col)."""
        loc1 = GeodeticWorldCoordinate([2.110, 0.394, 0.0])
        loc2 = GeodeticWorldCoordinate([2.117, 0.400, 0.0])
        min_row, min_col, max_row, max_col = self.wmq_tile_set.get_tile_matrix_limits_for_area([loc1, loc2], tile_matrix=10)
        self.assertLessEqual(min_row, max_row)
        self.assertLessEqual(min_col, max_col)
        tile_at_loc1 = self.wmq_tile_set.get_tile_for_location(loc1, tile_matrix=10)
        tile_at_loc2 = self.wmq_tile_set.get_tile_for_location(loc2, tile_matrix=10)
        self.assertEqual(min_row, min(tile_at_loc1.id.tile_row, tile_at_loc2.id.tile_row))
        self.assertEqual(min_col, min(tile_at_loc1.id.tile_col, tile_at_loc2.id.tile_col))
