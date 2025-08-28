#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

from typing import Any, Dict, Optional

from .coordinates import GeodeticWorldCoordinate, ImageCoordinate
from .elevation_model import ElevationModel
from .sensor_model import SensorModel, SensorModelOptions


class CompositeSensorModel(SensorModel):
    """
    A CompositeSensorModel is a SensorModel that combines an approximate but fast model with an accurate but slower
    model.
    """

    def __init__(self, approximate_sensor_model: SensorModel, precision_sensor_model: SensorModel) -> None:
        """
        Constructs the model given the two models to aggregate.

        :param approximate_sensor_model: a faster but less accurate model
        :param precision_sensor_model: a slower but more accurate model

        :return: None
        """
        super().__init__()
        self.approximate_sensor_model = approximate_sensor_model
        self.precision_sensor_model = precision_sensor_model

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function first calls the approximate model's image_to_world function to get an initial guess and then
        passes that information to the more accurate model through the options parameter's 'initial_guess' and
        'initial_search_distance' options.

        :param image_coordinate: the x, y image coordinate
        :param elevation_model: an optional elevation model used to transform the coordinate
        :param options: the options that will be augmented and then passed along

        :return: the longitude, latitude, elevation world coordinate
        """
        approximate_coord = self.approximate_sensor_model.image_to_world(
            image_coordinate, elevation_model=elevation_model, options=options
        )
        updated_options = options.copy() if options is not None else {}
        updated_options[SensorModelOptions.INITIAL_GUESS] = [
            approximate_coord.longitude,
            approximate_coord.latitude,
        ]
        return self.precision_sensor_model.image_to_world(
            image_coordinate, elevation_model=elevation_model, options=updated_options
        )

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This is just a pass through to the more accurate sensor model's world_to_image. These calculations tend to
        be quicker so there is no need to involve the approximate model.

        :param world_coordinate: the longitude, latitude, elevation world coordinate

        :return: the x, y image coordinate
        """
        return self.precision_sensor_model.world_to_image(world_coordinate)
