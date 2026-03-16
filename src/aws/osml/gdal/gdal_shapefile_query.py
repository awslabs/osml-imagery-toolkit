# Copyright 2025-2026 General Atomics Integrated Intelligence, Inc.

import operator
from math import degrees, floor
from typing import List, Optional

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
        layer_name: str,
        geom_cache_size: int = 10,
        tol: float = 1e-6,
    ) -> None:
        """
        Store shapefile information, store the layer name, and initialize a cache of geometries.

        :param vector_filepath: the path to a vector file
        :param layer_name: the layer name for the queries
        :param geom_cache_size: the size of the geometry cache
        :param tol: tolerance on geometry bounds in degrees

        :return: None
        """
        super().__init__()
        self.vector_filepath = vector_filepath
        self.ds = None
        self.layer = None
        self.layer_name = layer_name
        self.geom_cache: LRUCache = LRUCache(maxsize=geom_cache_size)
        self.tol = tol

    def __getstate__(self):
        """Clear the dataset for serialization."""
        data = self.__dict__.copy()
        data["layer"] = None
        data["ds"] = None
        return data

    @staticmethod
    def _ogr_geom_to_shapely(
        ogr_geom: ogr.Geometry,
        flatten: bool = False,
        spatial_mask: Optional[ogr.Geometry] = None,
        shapely_geoms_init: Optional[List[shapely.Geometry]] = None,
    ) -> List[shapely.Geometry]:
        """
        Helper function to split up OGR geometry collecions while converting to
        shapely geometries.
        """
        if shapely_geoms_init is None:
            shapely_geoms_init = []
        ogr_geom_name = ogr_geom.GetGeometryName()
        if flatten and (ogr_geom_name[:5] == "MULTI" or ogr_geom_name == "GEOMETRYCOLLECTION"):
            for ogr_isubgeom in range(ogr_geom.GetGeometryCount()):
                GDALShapefileQuery._ogr_geom_to_shapely(
                    ogr_geom.GetGeometryRef(ogr_isubgeom),
                    flatten,
                    spatial_mask,
                    shapely_geoms_init,
                )
        else:
            if spatial_mask is not None:
                ogr_geom = spatial_mask.Intersection(ogr_geom)
                if ogr_geom is not None:
                    return GDALShapefileQuery._ogr_geom_to_shapely(
                        ogr_geom,
                        flatten,
                        None,
                        shapely_geoms_init,
                    )
                else:
                    return shapely_geoms_init
            shapely_geoms_init.append(
                shapely.wkb.loads(
                    bytes(
                        ogr_geom.ExportToIsoWkb(),
                    ),
                ),
            )
        return shapely_geoms_init

    @cachedmethod(operator.attrgetter("geom_cache"))
    def _get_geometry(
        self,
        lon_deg: int,
        lat_deg: int,
    ) -> shapely.STRtree:
        """
        Store a 1 degree by 1 degree 'tile' of geometries from the shapefile, with collections split and indexed using
        a tree.

        :param lon_deg: floored longitude as an integer, in degrees
        :param lat_deg: floored latitude as an integer, in degrees

        :return: an R-tree wrapping geometries in the region
        """

        if self.ds is None:
            self.ds = gdal.OpenEx(self.vector_filepath)
            if self.ds is None:
                raise RuntimeError(f"{self.vector_filepath} could not be opened")
            self.layer = self.ds.GetLayerByName(self.layer_name)
            if self.layer is None:
                self.ds = None
                raise RuntimeError(f"{self.layer_name} not in {self.vector_filepath}")

        # Should be ints already, want to enforce.
        lon_deg = int(lon_deg)
        lat_deg = int(lat_deg)

        self.layer.SetSpatialFilterRect(
            lon_deg - self.tol,
            lat_deg - self.tol,
            lon_deg + 1 + self.tol,
            lat_deg + 1 + self.tol,
        )
        spatial_mask = self.layer.GetSpatialFilter()

        geoms = []
        while True:
            feat = self.layer.GetNextFeature()
            if feat is None:
                break
            ogr_geom = feat.geometry()
            GDALShapefileQuery._ogr_geom_to_shapely(
                ogr_geom,
                True,
                spatial_mask,
                geoms,
            )

        return shapely.STRtree(geoms)

    def get_geometry(
        self,
        world_coordinate: GeodeticWorldCoordinate,
    ) -> Optional[shapely.Geometry]:
        """
        Get a geometry (first, if many) containing a supplied point.

        :param world_coordinate: the point of interest

        :return: the geometry

        :note: Shapefile geometries use clockwise for positive orientation.
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
