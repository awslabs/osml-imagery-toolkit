#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import operator
from math import degrees, floor
from typing import Optional

import shapely
import shapely.wkb
import shapely.wkt
from cachetools import LRUCache, cachedmethod
from osgeo import gdal, ogr

from aws.osml.photogrammetry import GeodeticWorldCoordinate, GeometryQuery


class GDALShapefileQuery(GeometryQuery):
    """
    A geometry query that returns the first intersection geometry from a shapefile source.
    """

    def __init__(
        self,
        vector_filepath: str,
        table_name: str,
        geom_cache_size: int = 10,
        tol: float = 1e-6,
    ) -> None:
        """
        Store shapefile information, store the table name, and initialize a cache of geometries.

        :param vector_filepath: the path to a vector file
        :param table_name: the table name for the queries
        :param geom_cache_size: the size of the geometry cache
        :param tol: tolerance on geometry bounds in degrees

        :return: None
        """
        super().__init__()
        self.vector_filepath = vector_filepath
        self.ds = None
        self.table_name = table_name
        self.geom_cache: LRUCache = LRUCache(maxsize=geom_cache_size)
        self.tol = tol

    def __getstate__(self):
        """Clear the dataset for serialization."""
        data = self.__dict__.copy()
        data["ds"] = None
        return data

    @cachedmethod(operator.attrgetter("geom_cache"))
    def _get_geometry(
        self,
        lon_deg: int,
        lat_deg: int,
    ) -> shapely.STRtree:
        """
        Store a 1 degree by 1 degree 'tile' of geometries from the shapefile, split into polygons and indexed using a
        tree.

        :param lon_deg: floored longitude as an integer, in degrees
        :param lat_deg: floored latitude as an integer, in degrees

        :return: an R-tree wrapping geometries in the region
        """

        if self.ds is None:
            self.ds = gdal.OpenEx(self.vector_filepath)

        # Should be ints already, want to enforce.
        lon_deg = int(lon_deg)
        lat_deg = int(lat_deg)

        poly_str = (
            "POLYGON(("
            f"{lon_deg - self.tol} {lat_deg - self.tol},"
            f"{lon_deg + 1 + self.tol} {lat_deg - self.tol},"
            f"{lon_deg + 1 + self.tol} {lat_deg + 1 + self.tol},"
            f"{lon_deg - self.tol} {lat_deg + 1 + self.tol},"
            f"{lon_deg - self.tol} {lat_deg - self.tol}"
            "))"
        )
        spatial_filter = shapely.wkt.loads(poly_str)
        ogr_spatial_filter = ogr.CreateGeometryFromWkt(poly_str)

        # @todo Sanitize table name...
        result = self.ds.ExecuteSQL(
            f"SELECT * FROM {self.table_name}",
            spatialFilter=ogr_spatial_filter,
        )
        # The OGR dialect seems faster than SQLITE but it means that
        # intersections must be done afterwards.
        geoms = []
        while True:
            feat = result.GetNextFeature()
            if feat is None:
                break
            ogr_geom = feat.geometry()
            if ogr_geom.GetGeometryType() == 6:
                ogr_subgeoms = list(ogr_geom)
            else:
                ogr_subgeoms = [ogr_geom]
            for ogr_subgeom in ogr_subgeoms:
                geom = shapely.wkb.loads(
                    bytes(
                        ogr_subgeom.ExportToIsoWkb(),
                    ),
                )
                if spatial_filter.intersects(geom):
                    geoms.append(spatial_filter.intersection(geom))
        return shapely.STRtree(geoms)

    def get_geometry(
        self,
        world_coordinate: GeodeticWorldCoordinate,
    ) -> Optional[shapely.Geometry]:
        """
        Get a geometry (first, if many) containing a supplied point.

        :param world_coordinate: the point of interest

        :return: the geometry
        """
        search_tree = self._get_geometry(
            floor(degrees(world_coordinate.longitude)),
            floor(degrees(world_coordinate.latitude)),
        )
        nearest = search_tree.query_nearest(
            shapely.Point(
                degrees(world_coordinate.longitude),
                degrees(world_coordinate.latitude),
            ),
            max_distance=1e-12,
            all_matches=False,
        )
        return None if len(nearest) == 0 else search_tree.geometries[nearest[0]]
