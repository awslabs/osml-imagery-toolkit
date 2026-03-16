#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import unittest
from math import radians
from typing import Callable, List

import mock
import shapely
import shapely.wkt
from osgeo import gdal, ogr, osr

from configuration import TEST_ENV_CONFIG


@mock.patch.dict("os.environ", TEST_ENV_CONFIG, clear=True)
class TestGDALShapefileQuery(unittest.TestCase):
    def test_ogr_geom_to_shapely(self):
        from aws.osml.gdal.gdal_shapefile_query import GDALShapefileQuery

        wkt_1 = "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))"
        mask_1 = None
        res_1 = wkt_1
        assert shapely.equals_exact(
            shapely.normalize(
                GDALShapefileQuery._ogr_geom_to_shapely(
                    ogr.CreateGeometryFromWkt(wkt_1),
                    flatten=False,
                    spatial_mask=mask_1,
                ),
            ),
            shapely.normalize(shapely.wkt.loads(res_1)),
            tolerance=1e-3,
        )
        wkt_2 = "GEOMETRYCOLLECTION(MULTIPOLYGON(((0 0, 0 1, 1 1, 1 0, 0 0))))"
        mask_2 = None
        res_2 = wkt_1
        assert shapely.equals_exact(
            shapely.normalize(
                GDALShapefileQuery._ogr_geom_to_shapely(
                    ogr.CreateGeometryFromWkt(wkt_2),
                    flatten=True,
                    spatial_mask=mask_2,
                ),
            ),
            shapely.normalize(shapely.wkt.loads(res_2)),
            tolerance=1e-3,
        )
        wkt_3 = (
            "GEOMETRYCOLLECTION("
            "MULTIPOLYGON("
            "((0 0, 0 1, 1 1, 1 0, 0 0)),"
            "((1 1, 1 2, 2 2, 2 1, 1 1))"
            ")"
            ")"
        )  # fmt: skip
        mask_3 = "POLYGON((0.5 0.5, 0.5 1.5, 1.5 1.5, 1.5 0.5, 0.5 0.5))"
        res_3 = (
            "GEOMETRYCOLLECTION("
            "POLYGON((0.5 0.5, 0.5 1.0, 1.0 1.0, 1.0 0.5, 0.5 0.5)),"
            "POLYGON((1.0 1.0, 1.0 1.5, 1.5 1.5, 1.5 1.0, 1.0 1.0))"
            ")"
        )
        assert shapely.equals_exact(
            shapely.normalize(
                shapely.GeometryCollection(
                    GDALShapefileQuery._ogr_geom_to_shapely(
                        ogr.CreateGeometryFromWkt(wkt_3),
                        flatten=True,
                        spatial_mask=ogr.CreateGeometryFromWkt(mask_3),
                    ),
                ),
            ),
            shapely.normalize(shapely.wkt.loads(res_3)),
            tolerance=1e-3,
        )

    def test_get_geometry(self):
        from aws.osml.gdal.gdal_shapefile_query import GDALShapefileQuery
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate

        wkts = [
            "POLYGON((0.1 0.2, 0.2 0.2, 0.2 0.8, 0.1 0.8, 0.1 0.2))",
            "POLYGON((0.6 0.6, 0.7 0.6, 0.7 0.7, 0.6 0.7, 0.6 0.6))",
            "POLYGON((2.1 2.1, 3.1 2.1, 3.1 3.1, 2.1 3.1, 2.1 2.1))",
        ]

        geoms = [shapely.normalize(shapely.wkt.loads(wkt)) for wkt in wkts]

        with mock.patch(
            "osgeo.gdal.OpenEx",
            TestGDALShapefileQuery.create_gdal_fake_openex("layer", wkts),
        ):
            query = GDALShapefileQuery(
                "fakepath",
                "layer",
                tol=1e-6,
            )

            assert len(query._get_geometry(0, 0).geometries) == 2
            assert len(query._get_geometry(1, 1).geometries) == 0
            assert query.get_geometry(GeodeticWorldCoordinate((0, 0, 0))) is None
            assert (
                query.get_geometry(
                    GeodeticWorldCoordinate(
                        (
                            radians(0.5),
                            radians(0.5),
                            0,
                        )
                    )
                )
                is None
            )
            assert shapely.equals_exact(
                shapely.normalize(
                    query.get_geometry(
                        GeodeticWorldCoordinate(
                            (
                                radians(2.5),
                                radians(2.5),
                                0,
                            ),
                        ),
                    ),
                ),
                shapely.normalize(
                    shapely.Polygon(((2.1, 2.1), (3, 2.1), (3, 3), (2.1, 3), (2.1, 2.1))),
                ),
                tolerance=1e-3,
            )
            assert shapely.equals_exact(
                shapely.normalize(
                    query.get_geometry(
                        GeodeticWorldCoordinate(
                            (
                                radians(0.15),
                                radians(0.5),
                                0,
                            ),
                        ),
                    ),
                ),
                geoms[0],
                tolerance=1e-3,
            )
            assert shapely.equals_exact(
                shapely.normalize(
                    query.get_geometry(
                        GeodeticWorldCoordinate(
                            (
                                radians(0.6),
                                radians(0.6),
                                0,
                            ),
                        ),
                    ),
                ),
                geoms[1],
                tolerance=1e-3,
            )

    @staticmethod
    def create_gdal_fake_openex(
        layer_name: str,
        wkts: List[str],
    ) -> Callable[[str], gdal.Dataset]:
        """Create in-memory datasource from layer and geometries."""

        # Create datasource in memory.
        driver = ogr.GetDriverByName("Memory")
        f = driver.CreateDataSource("FAKEFILE")

        # Create layer, set WGS84.
        t = osr.SpatialReference()
        t.SetWellKnownGeogCS("WGS84")
        layer = f.CreateLayer(layer_name, t)
        feat_dfn = layer.GetLayerDefn()
        for wkt in wkts:
            feat = ogr.Feature(feat_dfn)
            feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt, t))
            layer.CreateFeature(feat)

        def gdal_fake_openex(filepath):
            return f

        return gdal_fake_openex


if __name__ == "__main__":
    unittest.main()
