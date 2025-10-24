#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import math
from typing import Any, Dict, Optional

from .coordinates import GeodeticWorldCoordinate, ImageCoordinate
from .elevation_model import ElevationModel
from .sensor_model import SensorModel, SensorModelOptions


class DefaultedSensorModel(SensorModel):
    """
    Sensor model wrapper that adds default options for image_to_world.
    """

    def __init__(
        self,
        inner_sensor_model: SensorModel,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create the defaulted model using an inner sensor model and defaults for the elevation model and options
        dictionary.

        :param inner_sensor_model: inner model that actually does calculations
        :param elevation_model: default elevation model for the sensor model
        :param options: default options dictionary for the sensor model

        :return: None
        """
        self.inner_sensor_model = inner_sensor_model
        self.elevation_model = elevation_model
        self.options = options

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function returns the longitude, latitude, elevation world coordinate associated with the x, y coordinate
        of any pixel in the full image.

        :param image_coordinate: the x, y full image coordinate
        :param elevation_model: optional elevation model used to transform the coordinate
        :param options: optional dictionary of hints, passed to nested sensor models

        :return: the longitude, latitude, elevation world coordinate
        """
        if options is None:
            new_options = self.options
        elif self.options is None:
            new_options = options
        else:
            new_options = {**self.options, **options}
        ignore_default_elevation_model = False
        if new_options is not None:
            ignore_default_elevation_model = new_options.get(
                SensorModelOptions.IGNORE_DEFAULT_ELEVATION_MODEL,
                False,
            )
        return self.inner_sensor_model.image_to_world(
            image_coordinate,
            self.elevation_model if elevation_model is None and not ignore_default_elevation_model else elevation_model,
            new_options,
        )

    def world_to_image(
        self,
        world_coordinate: GeodeticWorldCoordinate,
    ) -> ImageCoordinate:
        """
        This function returns the x, y full image coordinate associated with a given longitude, latitude, elevation
        world coordinate. If elevation is NaN, it will be set by the default elevation model if it exists.

        :param world_coordinate: the longitude, latitude, elevation world coordinate

        :return: the x, y full image coordinate
        """
        if math.isnan(world_coordinate.elevation) and self.elevation_model is not None:
            new_world_coordinate = GeodeticWorldCoordinate(world_coordinate.coordinate)
            self.elevation_model.set_elevation(new_world_coordinate)
            return self.inner_sensor_model.world_to_image(new_world_coordinate)
        return self.inner_sensor_model.world_to_image(world_coordinate)
