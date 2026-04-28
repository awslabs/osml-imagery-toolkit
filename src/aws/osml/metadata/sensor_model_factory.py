#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from enum import Enum
from typing import Dict, List, Optional, Union

from aws.osml.photogrammetry import ChippedImageSensorModel, CompositeSensorModel, ImageCoordinate, SensorModel

from .affine_sensor_model_builder import AffineSensorModelBuilder
from .dicttre_utils import get_tre_field_value
from .gcp_sensor_model_builder import GCPSensorModelBuilder, GroundControlPoint
from .projective_sensor_model_builder import ProjectiveSensorModelBuilder
from .rpc_sensor_model_builder import RPCSensorModelBuilder
from .rsm_sensor_model_builder import RSMSensorModelBuilder
from .sicd_sensor_model_builder import SICDSensorModelBuilder
from .sidd_sensor_model_builder import SIDDSensorModelBuilder

logger = logging.getLogger(__name__)


class ChippedImageInfoFacade:
    """
    This is a facade class that can be initialized with an ICHIPB TRE dict. It provides accessors for the values
    so that they can easily be used to create a ChippedImageSensorModel.
    """

    def __init__(self, ichipb_dict: dict) -> None:
        """
        Constructor initializes the properties from values in the TRE dict.

        :param ichipb_dict: the ICHIPB TRE fields as a Python dict

        :return: None
        """
        try:
            # Loop through the Output Product (OP) and Full Image (FI) fields in the ICHIPB TRE and construct
            # the corresponding image coordinates needed to create a chipped sensor model.
            self.full_image_coordinates: List[ImageCoordinate] = []
            self.chipped_image_coordinates: List[ImageCoordinate] = []
            for grid_point in ["11", "12", "21", "22"]:
                op_col = get_tre_field_value(ichipb_dict, f"OP_COL_{grid_point}", float)
                op_row = get_tre_field_value(ichipb_dict, f"OP_ROW_{grid_point}", float)
                fi_col = get_tre_field_value(ichipb_dict, f"FI_COL_{grid_point}", float)
                fi_row = get_tre_field_value(ichipb_dict, f"FI_ROW_{grid_point}", float)
                self.full_image_coordinates.append(ImageCoordinate([fi_col, fi_row]))
                self.chipped_image_coordinates.append(ImageCoordinate([op_col, op_row]))

            self.full_image_width: int = get_tre_field_value(ichipb_dict, "FI_COL", int)
            self.full_image_height: int = get_tre_field_value(ichipb_dict, "FI_ROW", int)
        except ValueError as ve:
            logging.warning("Unable to parse ICHIPB TRE found in metadata. SensorModel is unchanged.")
            logging.warning(str(ve))


class SensorModelTypes(Enum):
    """
    This enumeration defines the various sensor model types this factory can build.
    """

    AFFINE = "AFFINE"
    PROJECTIVE = "PROJECTIVE"
    RPC = "RPC"
    RSM = "RSM"
    SICD = "SICD"


ALL_SENSOR_MODEL_TYPES = [item for item in SensorModelTypes]


class SensorModelFactory:
    """
    This class encapsulates the logic necessary to construct SensorModels from imagery metadata provided by
    osml-imagery-io. Users initialize the factory by providing whatever metadata is available and this class
    will decide how to create the most accurate SensorModel from the available information.
    """

    def __init__(
        self,
        actual_image_width: int,
        actual_image_height: int,
        tre_dicts: Optional[Dict[str, Union[dict, List[dict]]]] = None,
        des_xml_strings: Optional[List[str]] = None,
        geo_transform: Optional[List[float]] = None,
        proj_wkt: Optional[str] = None,
        ground_control_points: Optional[List[GroundControlPoint]] = None,
        selected_sensor_model_types: Optional[List[SensorModelTypes]] = None,
    ) -> None:
        """
        Construct the factory providing whatever metadata is available from the image. All of the parameters
        are named and optional allowing users to provide whatever they can and trusting that this factory will
        make use of as much of the information as possible.

        :param actual_image_width: width of the current image in pixels
        :param actual_image_height: height of the current image in pixels
        :param tre_dicts: mapping of TRE name to field dict or list of field dicts
        :param des_xml_strings: list of XML strings from DES segments
        :param geo_transform: a 6-coefficient affine transform
        :param proj_wkt: the well known text string of the CRS used by the image
        :param ground_control_points: a list of ground control point correspondences
        :param selected_sensor_model_types: a list of sensor models that should be attempted by this factory

        :return: None
        """
        if selected_sensor_model_types is None:
            selected_sensor_model_types = ALL_SENSOR_MODEL_TYPES
        self.actual_image_width = actual_image_width
        self.actual_image_height = actual_image_height
        self.tre_dicts = tre_dicts
        self.des_xml_strings = des_xml_strings
        self.geo_transform = geo_transform
        self.proj_wkt = proj_wkt
        self.ground_control_points = ground_control_points
        self.selected_sensor_model_types = selected_sensor_model_types

    def build(self) -> Optional[SensorModel]:
        """
        Constructs the sensor model from the available information. Note that in cases where not enough
        information is available to provide any solution this method will return None.

        :return: the highest quality sensor model available given the information provided
        """

        approximate_sensor_model = None
        precision_sensor_model = None

        if SensorModelTypes.AFFINE in self.selected_sensor_model_types:
            if self.geo_transform is not None:
                approximate_sensor_model = AffineSensorModelBuilder(self.geo_transform, self.proj_wkt).build()

        if SensorModelTypes.PROJECTIVE in self.selected_sensor_model_types:
            if self.ground_control_points is not None and len(self.ground_control_points) > 3:
                approximate_sensor_model = GCPSensorModelBuilder(self.ground_control_points).build()

        if self.tre_dicts is not None:
            # Start with the assumption that the raster we have is the full image. We will update this later if
            # it turns out we're working with an image chip.
            full_image_width = self.actual_image_width
            full_image_height = self.actual_image_height

            # Check to see if this image is a chip from a larger image and if so extract the chip corner
            # information from the ICHIPB TRE.
            chipped_image_info = None
            ichipb_dict = self.tre_dicts.get("ICHIPB")
            if ichipb_dict is not None:
                chipped_image_info = ChippedImageInfoFacade(ichipb_dict)
                full_image_width = chipped_image_info.full_image_width
                full_image_height = chipped_image_info.full_image_height

            # Attempt to build a robust sensor model from either RSM or RPC metadata in the TREs. These
            # sensor models always reference the full image so if this is a chip we wrap the resulting sensor
            # model using information taken from ICHIPB. Note that in the unlikely event that an image has both
            # RSM and RPC metadata the RSM will be used because it has been developed as a replacement for RPC.
            precision_sensor_model = None
            if SensorModelTypes.RSM in self.selected_sensor_model_types:
                precision_sensor_model = RSMSensorModelBuilder(self.tre_dicts).build()
            if precision_sensor_model is None and SensorModelTypes.RPC in self.selected_sensor_model_types:
                precision_sensor_model = RPCSensorModelBuilder(self.tre_dicts).build()
            if precision_sensor_model is not None and chipped_image_info is not None:
                precision_sensor_model = ChippedImageSensorModel(
                    chipped_image_info.full_image_coordinates,
                    chipped_image_info.chipped_image_coordinates,
                    precision_sensor_model,
                )

            # Attempt to build an approximate sensor model from information in a corner coordinate TRE. The
            # CSCRNA TRE is considered more precise than IGEOLO so we will use it whenever possible.
            if SensorModelTypes.PROJECTIVE in self.selected_sensor_model_types:
                cscrna_dict = self.tre_dicts.get("CSCRNA")
                if cscrna_dict is not None:
                    approximate_sensor_model = ProjectiveSensorModelBuilder(
                        self.tre_dicts, full_image_width, full_image_height
                    ).build()
                    if approximate_sensor_model is not None and chipped_image_info is not None:
                        approximate_sensor_model = ChippedImageSensorModel(
                            chipped_image_info.full_image_coordinates,
                            chipped_image_info.chipped_image_coordinates,
                            approximate_sensor_model,
                        )

        if self.des_xml_strings is not None and len(self.des_xml_strings) > 0:
            for xml_str in self.des_xml_strings:
                if not xml_str:
                    continue

                if "SIDD" in xml_str:
                    # SIDD images will often contain SICD XML metadata as well but the SIDD should come first
                    # so we can stop processing other XML data segments
                    precision_sensor_model = SIDDSensorModelBuilder(sidd_xml=xml_str).build()
                    break
                elif "SICD" in xml_str and SensorModelTypes.SICD in self.selected_sensor_model_types:
                    precision_sensor_model = SICDSensorModelBuilder(sicd_xml=xml_str).build()
                    break

        # If we have both an approximate and a precision sensor model return them as a composite so applications
        # can choose which model best meets their needs. If we were only able to construct one or the other then
        # return what we were able to build.
        if approximate_sensor_model is not None and precision_sensor_model is not None:
            return CompositeSensorModel(
                approximate_sensor_model=approximate_sensor_model,
                precision_sensor_model=precision_sensor_model,
            )
        elif precision_sensor_model is not None:
            return precision_sensor_model
        else:
            return approximate_sensor_model
