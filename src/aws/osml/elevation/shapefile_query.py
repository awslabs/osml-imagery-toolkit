#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import operator
from math import degrees, floor
from typing import List, Optional

import shapefile
import shapely
import shapely.geometry
from cachetools import LRUCache, cachedmethod

from aws.osml.photogrammetry import GeodeticWorldCoordinate, GeometryQuery


class ShapefileQuery(GeometryQuery):
    """
    Geometry query backed by pyshp (pure Python) and Shapely STRtree.
    Returns the first intersection geometry from a shapefile source.
    Uses 1-degree tile caching with LRU eviction.
    """

    def __init__(
        self,
        vector_filepath: str,
        geom_cache_size: int = 10,
        tol: float = 1e-6,
    ) -> None:
        """
        :param vector_filepath: path to a .shp file (or basename without extension)
        :param geom_cache_size: max number of 1-degree tiles cached
        :param tol: tolerance in degrees added to tile bounding boxes
        """
        super().__init__()
        self.vector_filepath = vector_filepath
        self._reader: Optional[shapefile.Reader] = None
        self.geom_cache: LRUCache = LRUCache(maxsize=geom_cache_size)
        self.tol = tol

    def _open_reader(self) -> shapefile.Reader:
        if self._reader is None:
            self._reader = shapefile.Reader(shp=self.vector_filepath)
        return self._reader

    def __getstate__(self):
        """Exclude file handle and unpicklable cached-method state."""
        return {
            "vector_filepath": self.vector_filepath,
            "geom_cache_maxsize": self.geom_cache.maxsize,
            "tol": self.tol,
        }

    def __setstate__(self, state):
        self.vector_filepath = state["vector_filepath"]
        self._reader = None
        self.geom_cache = LRUCache(maxsize=state["geom_cache_maxsize"])
        self.tol = state["tol"]

    @cachedmethod(operator.attrgetter("geom_cache"))
    def _get_geometry(self, lon_deg: int, lat_deg: int) -> shapely.STRtree:
        """
        Load and cache geometries for a 1-degree tile, indexed via STRtree.

        :param lon_deg: floored longitude as an integer, in degrees
        :param lat_deg: floored latitude as an integer, in degrees
        :return: an STRtree wrapping geometries in the region
        """
        bbox = (
            lon_deg - self.tol,
            lat_deg - self.tol,
            lon_deg + 1 + self.tol,
            lat_deg + 1 + self.tol,
        )
        spatial_mask = shapely.box(*bbox)
        reader = self._open_reader()

        geoms: List[shapely.Geometry] = []
        if len(reader) == 0:
            return shapely.STRtree(geoms)
        for shp in reader.iterShapes(bbox=bbox):
            if shp is None:
                continue
            geom = shapely.geometry.shape(shp.__geo_interface__)
            clipped = geom.intersection(spatial_mask)
            if clipped.is_empty:
                continue
            if hasattr(clipped, "geoms"):
                geoms.extend(clipped.geoms)
            else:
                geoms.append(clipped)

        return shapely.STRtree(geoms)

    def get_geometry(
        self,
        world_coordinate: GeodeticWorldCoordinate,
    ) -> Optional[shapely.Geometry]:
        """
        Get a geometry (first, if many) containing a supplied point.

        :param world_coordinate: the point of interest (radians)
        :return: the geometry, or None if the point is not within any geometry
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
