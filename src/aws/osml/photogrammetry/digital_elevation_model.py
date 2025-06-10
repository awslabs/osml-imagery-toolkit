#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

# TODO: Add typing for ArrayLike once Numpy upgraded to 1.20+
# from numpy.typing import ArrayLike

import operator
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from cachetools import LRUCache, cachedmethod
from scipy.interpolate import RectBivariateSpline

from .coordinates import GeodeticWorldCoordinate
from .elevation_model import ElevationModel, ElevationRegionSummary
from .sensor_model import SensorModel


class DigitalElevationModelTileSet(ABC):
    """
    This class defines an abstraction that is capable of identifying which elevation tile in a
    DEM contains the elevations for a given world coordinate. It is common to split a DEM with
    global coverage up into a collection of files with well understood coverage areas. Those files
    may follow a simple naming convention but could be cataloged in an external spatial index. This
    class abstracts those details away from the DigitalElevationModel allowing us to easily extend
    this design to various tile sets.

    :return: None
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def find_tile_id(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[str]:
        """
        Converts the latitude, longitude of the world coordinate into a tile path.

        :param geodetic_world_coordinate: GeodeticWorldCoordinate = the world coordinate of interest

        :return: the tile path or None if the DEM does not have coverage for this location
        """


class DigitalElevationModelTileFactory(ABC):
    """
    This class defines an abstraction that is able to load a tile and convert it to a Numpy array
    of elevation data along with a SensorModel that can be used to identify the grid locations
    associated with a latitude, longitude.

    :return: None
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_tile(self, tile_path: str) -> Tuple[Optional[Any], Optional[SensorModel], Optional[ElevationRegionSummary]]:
        """
        Retrieve a numpy array of elevation values and a sensor model.

        TODO: Replace Any with numpy.typing.ArrayLike once we move to numpy >1.20

        :param tile_path: the location of the tile to load

        :return: an array of elevation values, a sensor model, and a summary
        """


class DigitalElevationModel(ElevationModel):
    """
    A Digital Elevation Model (DEM) is a representation of the topographic surface of the Earth. Theoretically
    these representations exclude trees, buildings, and any other surface objects but in practice elevations
    from those objects are likely captured by the sensors use to capture the elevation data.

    These datasets are normally stored as a pre-tiled collection of images with a well established resolution and
    coverage.
    """

    def __init__(
        self,
        tile_set: DigitalElevationModelTileSet,
        tile_factory: DigitalElevationModelTileFactory,
        raster_cache_size: int = 10,
    ) -> None:
        """
        This constructor accepts a tile set and a tile factory which specify how to index into and
        load elevations from a TBD collection of elevation tiles that make up a DEM. This inversion of
        control pattern isolates the generic processing of interpolating elevation values from a regular
        grid away from the details of how the elevation tiles are stored and accessed.

        :param tile_set: DigitalElevationModelTileSet = a class used to identify the DEM tile associated with a location
        :param tile_factory: DigitalElevationModelTileFactory = a class used to load DEM tile and convert to numpy array
        :param raster_cache_size: int = the number of DEM arrays to store in memory preventing frequent loading

        :return: None
        """
        super().__init__()
        self.tile_set = tile_set
        self.tile_factory = tile_factory
        self.raster_cache: LRUCache = LRUCache(maxsize=raster_cache_size)
        # TODO: Think about raster_cache_size parameter. This is the number of rasters we will keep open at any
        #       one time. Look at the size of those tiles and add a comment about how much memory will be used by this
        #       setting. Pick a default that is reasonable and also likely to cover most images

    def set_elevation(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> bool:
        """
        This method updates the elevation component of a geodetic world coordinate to match the surface
        elevation at the provided latitude and longitude. Note that if the DEM does not have elevation
        values for this region or if there is an error loading the associated image the elevation will
        be unchanged.

        :param geodetic_world_coordinate: the coordinate to update

        :return: True if the elevation was updated, else False
        """
        tile_id = self.tile_set.find_tile_id(geodetic_world_coordinate)
        if not tile_id:
            return False

        interpolation_grid, sensor_model, summary = self.get_interpolation_grid(tile_id)

        if interpolation_grid is not None and sensor_model is not None:
            image_coordinate = sensor_model.world_to_image(geodetic_world_coordinate)
            geodetic_world_coordinate.elevation = interpolation_grid(image_coordinate.x, image_coordinate.y)[0][0]
            return True

        # else can't set elevation without grid / model
        return False

    def describe_region(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[ElevationRegionSummary]:
        """
        Get a summary of the region near the provided world coordinate

        :param geodetic_world_coordinate: the coordinate at the center of the region of interest
        :return: a summary of the elevation data in this tile
        """

        tile_id = self.tile_set.find_tile_id(geodetic_world_coordinate)
        if not tile_id:
            return

        interpolation_grid, sensor_model, summary = self.get_interpolation_grid(tile_id)
        return summary

    @cachedmethod(operator.attrgetter("raster_cache"))
    def get_interpolation_grid(
        self, tile_path: str
    ) -> Tuple[Optional[RectBivariateSpline], Optional[SensorModel], Optional[ElevationRegionSummary]]:
        """
        This method loads and converts an array of elevation values into a class that can
        interpolate values that lie between measured elevations. The sensor model is also
        returned to allow us to precisely identify the location in the grid of a
        world coordinate.

        Note that the results of this method are cached by tile_id. It is very common for
        the set_elevation() method to be called multiple times for locations that are in a
        narrow region of interest. This will prevent unnecessary repeated loading of tiles.

        :param tile_path: the location of the tile to load

        :return: the cached interpolation object, sensor model, and summary
        """
        elevations_array, sensor_model, summary = self.tile_factory.get_tile(tile_path)
        if elevations_array is not None and sensor_model is not None:
            height, width = elevations_array.shape
            x = range(0, width)
            y = range(0, height)
            return RectBivariateSpline(x, y, elevations_array.T, kx=1, ky=1), sensor_model, summary
        else:
            return None, None, None
