#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from .coordinates import GeodeticWorldCoordinate, ImageCoordinate
from .elevation_model import ElevationModel

logger = logging.getLogger(__name__)


class SensorModel(ABC):
    """
    A sensor model is an abstraction that maps the information in a georeferenced image to the real world. The
    concrete implementations of this abstraction will either capture the physical service model characteristics or
    more frequently an approximation of that physical model that allow users to transform world coordinates to
    image coordinates.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function returns the longitude, latitude, elevation world coordinate associated with the x, y coordinate
        of any pixel in the image.

        :param image_coordinate: the x, y image coordinate
        :param elevation_model: optional elevation model used to transform the coordinate
        :param options: optional dictionary of hints that will be passed on to sensor models

        :return: GeodeticWorldCoordinate = the longitude, latitude, elevation world coordinate
        """

    @abstractmethod
    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This function returns the x, y image coordinate associated with a given longitude, latitude, elevation world
        coordinate.

        :param world_coordinate: the longitude, latitude, elevation world coordinate

        :return: the x, y image coordinate
        """


class SensorModelOptions(str, Enum):
    """
    These are common options for SensorModel operations. Not all implementations will support these, but they are
    included here to encourage convention.
    """

    EXCEPTION_ON_FAILURE = "exception_on_failure"
    FALLBACK_INITIAL_GUESS = "fallback_initial_guess"
    FORCE_INITIAL_GUESS = "force_initial_guess"
    INITIAL_GUESS = "initial_guess"
    INITIAL_SEARCH_DISTANCE = "initial_search_distance"
    MIN_SUCCESS_DISTANCE_PIXELS = "min_success_distance_pixels"
