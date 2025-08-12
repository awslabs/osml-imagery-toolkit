#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from math import degrees, floor
from typing import Optional

from .coordinates import GeodeticWorldCoordinate
from .digital_elevation_model import DigitalElevationModelTileSet


class GenericDEMTileSet(DigitalElevationModelTileSet):
    """
    A generalizable tile set with a naming convention that can be described as a format string.
    """

    def __init__(
        self,
        format_spec: str = "%od%oh/%ld%lh.dt2",
        min_latitude_degrees: float = -90.0,
        max_latitude_degrees: float = 90.0,
        min_longitude_degrees: float = -180.0,
        max_longitude_degrees: float = 180.0,
    ) -> None:
        """
        Construct a tile set from a limited collection of configurable parameters. This implementation uses the
        custom formatting directives %od (longitude degrees), %oh (longitude hemisphere), %ld (latitude degrees),
        and %lh (latitude hemisphere) to allow users to create tile IDs that match a variety of situations.
        For example the default format_spec of '%od%oh/%ld%lh.dt2' will generate tile ids like: '115e/45s.dt2'
        which would match some common 1-degree cell based DEM file hierarchies.

        :param format_spec: the format specification for the GeodeteticWorldCoordinate


        :return: None
        """
        super().__init__()
        self.format_string = format_spec
        self.min_latitude_degrees = min_latitude_degrees
        self.max_latitude_degrees = max_latitude_degrees
        self.min_longitude_degrees = min_longitude_degrees
        self.max_longitude_degrees = max_longitude_degrees

    def find_tile_id(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[str]:
        """
        This method creates tile IDs that based on the format string provided.

        :param geodetic_world_coordinate: the world coordinate of interest

        :return: the tile path or None if the DEM does not have coverage for this location
        """
        longitude_degrees = floor(degrees(geodetic_world_coordinate.longitude))
        latitude_degrees = floor(degrees(geodetic_world_coordinate.latitude))

        # The SRTM mission only covers latitudes N59 through S56 so if the requested location is outside those
        # ranges we know there is no file available for it.
        if (
            latitude_degrees > self.max_latitude_degrees
            or latitude_degrees < self.min_latitude_degrees
            or longitude_degrees > self.max_longitude_degrees
            or longitude_degrees < self.min_longitude_degrees
        ):
            return None

        return f"{geodetic_world_coordinate:{self.format_string}}"
