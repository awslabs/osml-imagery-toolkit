#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from typing import List, Optional

from aws.osml.photogrammetry import AffineSensorModel

from .sensor_model_builder import SensorModelBuilder

logger = logging.getLogger(__name__)


class AffineSensorModelBuilder(SensorModelBuilder):
    """
    This builder constructs sensor models for images that have a 6-coefficient
    affine geo transform (e.g. from GeoTIFF). It produces an AffineSensorModel
    from the transform coefficients and an optional CRS projection string.
    """

    def __init__(self, geo_transform: List[float], proj_wkt: Optional[str] = None) -> None:
        """
        Construct the builder with the required geo transform.

        :param geo_transform: the 6-coefficient affine transform
        :param proj_wkt: optional CRS well-known text string

        :return: None
        """
        super().__init__()
        self.geo_transform = geo_transform
        self.proj_wkt = proj_wkt

    def build(self) -> Optional[AffineSensorModel]:
        """
        Use the geo transform to construct an affine sensor model.

        :return: an AffineSensorModel, or None if geo_transform is None
        """
        if self.geo_transform is None:
            return None
        return AffineSensorModel(self.geo_transform, self.proj_wkt)
