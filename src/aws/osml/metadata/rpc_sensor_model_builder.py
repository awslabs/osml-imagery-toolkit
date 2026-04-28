#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from typing import Dict, Optional

from aws.osml.photogrammetry import RPCPolynomial, RPCSensorModel

from .dicttre_utils import get_tre_field_value, parse_tre_coefficient_list
from .sensor_model_builder import SensorModelBuilder

logger = logging.getLogger(__name__)


class RPCSensorModelBuilder(SensorModelBuilder):
    """
    This builder constructs sensor models for images that have RPC TREs. The inputs are TRE metadata
    provided as Python dicts (from osml-imagery-io's MetadataProvider) rather than GDAL XML elements.

    This builder only supports the RPC00B format. Support for other TREs can be added in the future
    if we find ourselves working with imagery containing that metadata.

    See STDI-0002 Volume 1 Appendix E for more detailed information.
    """

    def __init__(self, tre_dicts: Dict[str, dict]) -> None:
        """
        Constructor for the builder accepting the required TRE dicts.

        :param tre_dicts: dict mapping TRE names to their field dicts
        """
        super().__init__()
        self.tre_dicts = tre_dicts

    def build(self) -> Optional[RPCSensorModel]:
        """
        Examine the TRE metadata for RPC information, parse the necessary values out of those TREs,
        and construct an RPC sensor model.

        :return: an RPC SensorModel if one can be constructed, None otherwise
        """
        # Check to see if an RPC00B TRE is included with the metadata.
        rpc_dict = self.tre_dicts.get("RPC00B")
        if rpc_dict is None:
            logging.debug("No RPC00B TRE found. Skipping RPC sensor model build.")
            return None

        # Attempt to construct the RPC camera model from the metadata provided
        try:
            success = get_tre_field_value(rpc_dict, "SUCCESS", int)
            if success != 1:
                logging.info("RPC00B TRE SUCCESS field was not '1'. Skipping RPC sensor model build.")
                return None

            return RPCSensorModelBuilder.build_rpc_sensor_model(rpc_dict)

        except ValueError as ve:
            logging.warning("Unable to parse RPC00B TRE found in metadata. No SensorModel created.")
            logging.warning(str(ve))
            return None

    @staticmethod
    def build_rpc_sensor_model(rpc_dict: dict) -> RPCSensorModel:
        """
        Construct an RPC sensor model from an RPC00B TRE dict.

        :param rpc_dict: the RPC00B TRE field dict
        :return: the RPC sensor model
        """
        return RPCSensorModel(
            get_tre_field_value(rpc_dict, "ERR_BIAS", float),
            get_tre_field_value(rpc_dict, "ERR_RAND", float),
            get_tre_field_value(rpc_dict, "LINE_OFF", float),
            get_tre_field_value(rpc_dict, "SAMP_OFF", float),
            get_tre_field_value(rpc_dict, "LAT_OFF", float),
            get_tre_field_value(rpc_dict, "LONG_OFF", float),
            get_tre_field_value(rpc_dict, "HEIGHT_OFF", float),
            get_tre_field_value(rpc_dict, "LINE_SCALE", float),
            get_tre_field_value(rpc_dict, "SAMP_SCALE", float),
            get_tre_field_value(rpc_dict, "LAT_SCALE", float),
            get_tre_field_value(rpc_dict, "LONG_SCALE", float),
            get_tre_field_value(rpc_dict, "HEIGHT_SCALE", float),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_dict, "LINE_NUM_COEFF"),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_dict, "LINE_DEN_COEFF"),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_dict, "SAMP_NUM_COEFF"),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_dict, "SAMP_DEN_COEFF"),
        )

    @staticmethod
    def build_rpc_polynomial(rpc_dict: dict, polynomial_name: str) -> RPCPolynomial:
        """
        Construct an RPC polynomial from coefficients found in the RPC00B TRE dict. There are 4
        repeating groups of these coefficients for the polynomials associated with line or sample
        numerators and denominators.

        :param rpc_dict: the RPC00B TRE field dict
        :param polynomial_name: the prefix of the polynomial coefficients to extract
        :return: the RPC polynomial
        """
        return RPCPolynomial(parse_tre_coefficient_list(rpc_dict, polynomial_name, 20))
