#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from math import radians
from typing import Dict, Optional

from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, ProjectiveSensorModel

from .dicttre_utils import get_tre_field_value
from .sensor_model_builder import SensorModelBuilder

logger = logging.getLogger(__name__)


class ProjectiveSensorModelBuilder(SensorModelBuilder):
    """
    This builder constructs sensor models for images that have a CSCRNA TRE. The inputs are TRE
    metadata provided as Python dicts (from osml-imagery-io's MetadataProvider) rather than GDAL
    XML elements.

    The CSCRNA TRE contains corner coordinates (upper-left, upper-right, lower-right, lower-left)
    that define the geographic extent of the image. These are used to construct a projective sensor
    model that maps between image pixel coordinates and world coordinates.

    See STDI-0002 Volume 1 Appendix AW for more detailed information.
    """

    def __init__(self, tre_dicts: Dict[str, dict], full_image_width: float, full_image_height: float) -> None:
        """
        Constructor for the builder accepting the required TRE dicts and image dimensions.

        :param tre_dicts: dict mapping TRE names to their field dicts (expects "CSCRNA")
        :param full_image_width: the width of the full image in pixels
        :param full_image_height: the height of the full image in pixels
        """
        super().__init__()
        self.tre_dicts = tre_dicts
        self.full_image_width = full_image_width
        self.full_image_height = full_image_height

    def build(self) -> Optional[ProjectiveSensorModel]:
        """
        Examine the TRE metadata for CSCRNA corner coordinate information, parse the necessary
        values, and construct a projective sensor model.

        :return: a ProjectiveSensorModel if one can be constructed, None otherwise
        """
        cscrna_dict = self.tre_dicts.get("CSCRNA")
        if cscrna_dict is None:
            logging.debug("No CSCRNA TRE found. Skipping projective sensor model build.")
            return None

        try:
            return ProjectiveSensorModelBuilder.build_projective_sensor_model(
                cscrna_dict, self.full_image_width, self.full_image_height
            )
        except ValueError as ve:
            logging.warning("Unable to parse CSCRNA TRE found in metadata. No SensorModel created.")
            logging.warning(str(ve))
            return None

    @staticmethod
    def build_projective_sensor_model(
        cscrna_dict: dict, full_image_width: float, full_image_height: float
    ) -> ProjectiveSensorModel:
        """
        Construct a projective sensor model from a CSCRNA TRE dict and image dimensions.

        :param cscrna_dict: the CSCRNA TRE field dict
        :param full_image_width: the width of the image in pixels
        :param full_image_height: the height of the image in pixels
        :return: the projective sensor model
        """
        world_coordinates = [
            GeodeticWorldCoordinate(
                [
                    radians(get_tre_field_value(cscrna_dict, f"{corner}CNR_LONG", float)),
                    radians(get_tre_field_value(cscrna_dict, f"{corner}CNR_LAT", float)),
                    radians(get_tre_field_value(cscrna_dict, f"{corner}CNR_HT", float)),
                ]
            )
            for corner in ["UL", "UR", "LR", "LL"]
        ]
        image_coordinates = [
            ImageCoordinate([0, 0]),
            ImageCoordinate([full_image_width, 0]),
            ImageCoordinate([full_image_width, full_image_height]),
            ImageCoordinate([0, full_image_height]),
        ]
        return ProjectiveSensorModel(world_coordinates, image_coordinates)
