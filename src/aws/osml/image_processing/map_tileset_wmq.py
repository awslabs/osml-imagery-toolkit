#  Copyright 2024 Amazon.com, Inc. or its affiliates.

import math

import pyproj
from pyproj.enums import TransformDirection

from ..photogrammetry import GeodeticWorldCoordinate
from .map_tileset import MapTile, MapTileId, MapTileSet

# Axis Info: X[east], Y[north]
WEB_MERCATOR_PROJ = pyproj.Proj("EPSG:3857")

# Axis Info: Lat[north], Lon[east]
LLE_PROJ = pyproj.Proj("EPSG:4326")
LLE_TO_WEB_MERCATOR = pyproj.Transformer.from_proj(LLE_PROJ, WEB_MERCATOR_PROJ)


class WebMercatorQuadMapTileSet(MapTileSet):
    """
    Tile matrix set definition for WebMercatorQuad, expressed in http://www.opengis.net/def/crs/EPSG/0/3857 This tile
    matrix set is the most used tile matrix set in the mass market.(Google Maps, Microsoft Bing Maps, and Open Street
    Map tiles all use this projection). Note that this class should not be imported / used directly; it is an
    implementation detail of the abstract MapTileSet class. The correct way to utilize this class is to get an
    implementation from the MapTileSetFactory using a value from the WellKnownTileSet enumeration.

    It has been long criticized because it is a based on a spherical Mercator instead of an ellipsoid. The use of
    WebMercatorQuad should be limited to visualization. Any additional use (including distance measurements, routing,
    etc.) needs to use the Mercator spherical expressions to transform the coordinate to an appropriate CRS first. The
    risks caused by imprecision in the use of Web Mercator have been noted by the US National Geospatial-Intelligence
    Agency (NGA). NGA has issued an Advisory Notice on web Mercator noting its risks and limiting approved usage to
    visualizations: https://nsgreg.nga.mil/doc/view?i=4478
    """

    def __init__(self, tile_size: int = 256, tile_matrix_set_id="WebMercatorQuad"):
        """
        Construct a spherical web mercator map tile set.

        :param tile_size: the tile size, defaults to 256 to align with the WebMercatorQuad well known tile set
        :param tile_matrix_set_id: name of the tile set, defaults to "WebMercatorQuad"
        """
        self.tile_size = tile_size
        self._tile_matrix_set_id = tile_matrix_set_id

        spherical_circumference = 2 * math.pi * WEB_MERCATOR_PROJ.crs.ellipsoid.semi_major_metre
        self.initial_resolution = spherical_circumference / self.tile_size
        self.origin_shift = spherical_circumference / 2.0

    @property
    def tile_matrix_set_id(self) -> str:
        """
        Get the identifier for this map tile set. This is the tile matrix set ID in the OGC definitions.

        :return: the tile matrix set ID
        """
        return self._tile_matrix_set_id

    @property
    def crs_id(self) -> str:
        return "EPSG:3857"

    def get_tile(self, tile_id: MapTileId) -> MapTile:
        """
        Get a description of the tile identified by a specific map tile ID.

        :param tile_id: the tile ID
        :return: the tile description
        """
        native_bounds = self._calculate_tile_bounds_meters(tile_id)
        return MapTile(
            id=tile_id,
            size=(self.tile_size, self.tile_size),
            bounds=self._calculate_tile_bounds_lle(tile_id),
            native_bounds=native_bounds,
        )

    def get_tile_for_location(self, world_coordinate: GeodeticWorldCoordinate, tile_matrix: int) -> MapTile:
        """
        Get a description of the tile containing a specific world location.

        :param world_coordinate: the location in the world
        :param tile_matrix: the tile_matrix or zoom level of interest
        :return: the tile description
        """
        x_meters, y_meters = LLE_TO_WEB_MERCATOR.transform(
            world_coordinate.latitude,
            world_coordinate.longitude,
            radians=True,
            direction=TransformDirection.FORWARD,
        )
        x_pixels, y_pixels = self._meters_to_pixels(x_meters, y_meters, tile_matrix)
        tile_col, tile_row = self._pixels_to_tile(x_pixels, y_pixels)
        return self.get_tile(MapTileId(tile_matrix=tile_matrix, tile_row=tile_row, tile_col=tile_col))

    def _resolution(self, tile_matrix: int) -> float:
        """
        Compute the resolution of pixels in meters for a given zoom level in this tile set.

        :param tile_matrix: the zoom level
        :return: the resolution in meters
        """
        return self.initial_resolution / (2**tile_matrix)

    def _pixels_to_meters(self, x_pixels: float, y_pixels: float, tile_matrix: int) -> tuple[float, float]:
        """
        Convert an x, y location in pixels to x (east), y (north) location in meters. This includes shifting the
        origin for the image pixels (0, 0) in upper left corner to the origin in meters (0, 0) in the center of the
        image.

        :param x_pixels: the x location in pixels
        :param y_pixels: the y location in pixels
        :param tile_matrix: the zoom level
        :return: the (x, y) tuple in meters
        """
        tile_matrix_resolution = self._resolution(tile_matrix)
        x_meters = x_pixels * tile_matrix_resolution - self.origin_shift
        y_meters = y_pixels * tile_matrix_resolution - self.origin_shift
        return x_meters, y_meters

    def _meters_to_pixels(self, x_meters: float, y_meters: float, tile_matrix: int) -> tuple[float, float]:
        """
        Convert an x (east), y (north) location in meters to x, y location in pixels. This includes shifting the (0, 0)
        origin in meters from the center of the image to the upper left corner (0, 0) in image pixels.

        :param x_meters: the x location in meters
        :param y_meters: the y location in meters
        :param tile_matrix: the zoom level
        :return: the (x, y) tuple in pixels
        """
        tile_matrix_resolution = self._resolution(tile_matrix)
        x_pixels = (x_meters + self.origin_shift) / tile_matrix_resolution
        y_pixels = (y_meters + self.origin_shift) / tile_matrix_resolution
        return x_pixels, y_pixels

    def _pixels_to_tile(self, x_pixels: float, y_pixels: float) -> tuple[int, int]:
        """
        This function converts x, y pixel coordinates to col, row tile indexes.

        :param x_pixels: the x location in pixels
        :param y_pixels: the y location in pixels
        :return: the col, row tuple for the corresponding tile index
        """
        tile_col = int(math.ceil(x_pixels / float(self.tile_size)) - 1)
        tile_row = int(math.ceil(y_pixels / float(self.tile_size)) - 1)
        return tile_col, tile_row

    def _calculate_tile_bounds_meters(self, tile_id: MapTileId) -> tuple[float, float, float, float]:
        """
        This function calculates the boundary of a tile in meters.

        :param tile_id: the requested tile id
        :return: the [min_x, min_y, max_x, max_y] bounding box in EPSG:3587
        """
        min_x, min_y = self._pixels_to_meters(
            tile_id.tile_col * self.tile_size, tile_id.tile_row * self.tile_size, tile_id.tile_matrix
        )
        max_x, max_y = self._pixels_to_meters(
            (tile_id.tile_col + 1) * self.tile_size, (tile_id.tile_row + 1) * self.tile_size, tile_id.tile_matrix
        )
        return min_x, min_y, max_x, max_y

    def _calculate_tile_bounds_lle(self, tile_id: MapTileId):
        """
        This function calculates the boundary of a tile in latitude, longitude

        :param tile_id: the requested tile id
        :return: the [min_lon, min_lat, max_lon, max_lat] bounding box in EPSG:4326
        """
        min_x, min_y, max_x, max_y = self._calculate_tile_bounds_meters(tile_id)
        min_lat, min_lon = LLE_TO_WEB_MERCATOR.transform(min_x, min_y, radians=True, direction=TransformDirection.INVERSE)
        max_lat, max_lon = LLE_TO_WEB_MERCATOR.transform(max_x, max_y, radians=True, direction=TransformDirection.INVERSE)
        return min_lon, min_lat, max_lon, max_lat
