#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from abc import ABC, abstractmethod
from typing import Optional

from aws.osml.photogrammetry import SensorModel


class SensorModelBuilder(ABC):
    """
    Abstract base for all classes that construct SensorModels from various types of metadata.
    """

    @abstractmethod
    def build(self) -> Optional[SensorModel]:
        """
        Construct a sensor model from the available information.

        In cases where not enough information is available to provide any solution,
        this method will return None.

        :return: the sensor model if available in the metadata provided
        """
