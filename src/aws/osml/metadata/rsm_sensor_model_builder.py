#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from typing import Dict, List, Optional, Union

from aws.osml.photogrammetry import (
    RSMContext,
    RSMGroundDomain,
    RSMGroundDomainForm,
    RSMImageDomain,
    RSMLowOrderPolynomial,
    RSMPolynomial,
    RSMPolynomialSensorModel,
    RSMSectionedPolynomialSensorModel,
    SensorModel,
    WorldCoordinate,
)

from .dicttre_utils import get_tre_field_value, parse_tre_coefficient_list
from .sensor_model_builder import SensorModelBuilder

logger = logging.getLogger(__name__)


class RSMSensorModelBuilder(SensorModelBuilder):
    """
    This builder constructs sensor models for images that have RSM TREs. The inputs are TRE metadata
    provided as Python dicts (from osml-imagery-io's MetadataProvider) rather than GDAL XML elements.

    The actual type and number of RSM TREs included with an image will vary depending on the type of
    RSM sensor model defined. In general all images with these sensor models must have an RSMIDA TRE
    that defines the overall context of the sensor model.

    The polynomial based sensor models will then have at least one RSMPCA TRE and may have multiple.
    If there are multiple then a RSMPIA TRE will also be present to describe how the various polynomial
    models cover the overall image domain.

    See STDI-0002 Volume 1 Appendix U for more detailed information.
    """

    def __init__(self, tre_dicts: Dict[str, Union[dict, List[dict]]]) -> None:
        """
        Constructor for the builder accepting the required TRE dicts.

        :param tre_dicts: dict mapping TRE names to their field dicts or list of field dicts
        """
        super().__init__()
        self.tre_dicts = tre_dicts

    def build(self) -> Optional[SensorModel]:
        """
        Examine the TRE metadata for RSM information, parse the necessary values out of those TREs,
        and construct an RSM sensor model.

        :return: an RSM SensorModel if one can be constructed, None otherwise
        """
        # Check to see if an RSMIDA TRE is included with the metadata. This is a mandatory TRE that
        # will be available for all RSM based sensor models.
        rsmid_dict = self.tre_dicts.get("RSMIDA")
        if rsmid_dict is None:
            logging.debug("No RSMIDA TRE found. Skipping RSM sensor model build.")
            return None

        try:
            # Use the information in the RSMIDA TRE to build the context containing the ground and
            # image domains that bound the valid regions for the RSM model.
            rsm_context = RSMSensorModelBuilder._build_rsm_context(rsmid_dict)

            # If an RSM model is using polynomial coefficients to define the sensor model then those
            # coefficients will be stored in RSMPCA TREs. The value can be a single dict or a list
            # of dicts.
            rsmpca_value = self.tre_dicts.get("RSMPCA")
            rsmpc_dicts: List[dict] = []
            if rsmpca_value is not None:
                if isinstance(rsmpca_value, list):
                    rsmpc_dicts = rsmpca_value
                else:
                    rsmpc_dicts = [rsmpca_value]

            rsm_polynomial_sensor_models = [
                RSMSensorModelBuilder._build_rsm_polynomial_sensor_model(rsmpc_dict, rsm_context)
                for rsmpc_dict in rsmpc_dicts
            ]

            # If we only have one RSM polynomial sensor model then it applies to the entire RSM
            # domain. If we have multiple then we are dealing with a sectioned sensor model which
            # will require additional TREs to be parsed.
            if len(rsm_polynomial_sensor_models) > 0:
                if len(rsm_polynomial_sensor_models) == 1:
                    return rsm_polynomial_sensor_models[0]
                else:
                    # Parse RSMPIA and construct a sectioned polynomial sensor model
                    rsmpi_dict = self.tre_dicts.get("RSMPIA")
                    if rsmpi_dict is None:
                        logging.warning(
                            "Image has multiple RSMPCA TREs but is missing a RSMPIA that assigns "
                            "them to sections. No sensor model can be built!"
                        )
                        return None
                    return RSMSensorModelBuilder._build_rsm_sectioned_polynomial_sensor_model(
                        rsmpi_dict, rsm_context, rsm_polynomial_sensor_models
                    )

            # TODO: Check for RSMGGA and RSMGIA TREs and construct a RSM grid interpolation sensor model
            logging.warning(
                "Image has RSMIDA TRE but no polynomials. Grid based RSM not implemented so no sensor model returned."
            )
            return None

        except ValueError as ve:
            logging.warning("Unable to parse RSM TREs found in metadata. No SensorModel created.")
            logging.warning(str(ve))
            return None

    @staticmethod
    def _build_rsm_ground_domain(rsmid_dict: dict) -> RSMGroundDomain:
        """
        Construct the ground domain from information in the RSMIDA TRE dict.

        :param rsmid_dict: the RSMIDA TRE field dict
        :return: the ground domain
        """
        # This is the type of ground domain for this RSM sensor model
        ground_domain_form = RSMGroundDomainForm(get_tre_field_value(rsmid_dict, "GRNDD", str))

        # The valid region of the ground domain is defined by 8 sets of world coordinates (V1 - V8)
        ground_domain_vertices = [
            WorldCoordinate(
                [
                    get_tre_field_value(rsmid_dict, f"V{vertex_number}X", float),
                    get_tre_field_value(rsmid_dict, f"V{vertex_number}Y", float),
                    get_tre_field_value(rsmid_dict, f"V{vertex_number}Z", float),
                ]
            )
            for vertex_number in range(1, 9)
        ]

        # Ground domains are either rectangular or geodetic. Rectangular ground domains have
        # additional values that define a cartesian coordinate system anchored at a point on earth.
        rectangular_coordinate_origin = None
        rectangular_coordinate_unit_vectors = None
        if ground_domain_form == RSMGroundDomainForm.RECTANGULAR:
            # The world location for the origin (0, 0, 0) of the rectangular coordinate system
            rectangular_coordinate_origin = WorldCoordinate(
                [
                    get_tre_field_value(rsmid_dict, "XUOR", float),
                    get_tre_field_value(rsmid_dict, "YUOR", float),
                    get_tre_field_value(rsmid_dict, "ZUOR", float),
                ]
            )

            # Unit vectors defining the cartesian coordinate system
            rectangular_coordinate_unit_vectors = []
            for row_coefficient in ["XR", "YR", "ZR"]:
                row = [
                    get_tre_field_value(rsmid_dict, f"{col_coefficient}U{row_coefficient}", float)
                    for col_coefficient in ["X", "Y", "Z"]
                ]
                rectangular_coordinate_unit_vectors.append(row)

        try:
            ground_reference_point = WorldCoordinate(
                [
                    get_tre_field_value(rsmid_dict, "GRPX", float),
                    get_tre_field_value(rsmid_dict, "GRPY", float),
                    get_tre_field_value(rsmid_dict, "GRPZ", float),
                ]
            )
        except ValueError:
            # The ground reference point is optional and these fields may be filled with spaces.
            # If we can't parse floating point numbers from these fields we should assume this
            # information has not been provided.
            ground_reference_point = None

        return RSMGroundDomain(
            ground_domain_form,
            ground_domain_vertices,
            rectangular_coordinate_origin=rectangular_coordinate_origin,
            rectangular_coordinate_unit_vectors=rectangular_coordinate_unit_vectors,
            ground_reference_point=ground_reference_point,
        )

    @staticmethod
    def _build_rsm_image_domain(rsmid_dict: dict) -> RSMImageDomain:
        """
        Construct the image domain from information in the RSMIDA TRE dict.

        :param rsmid_dict: the RSMIDA TRE field dict
        :return: the image domain
        """
        return RSMImageDomain(
            get_tre_field_value(rsmid_dict, "MINR", int),
            get_tre_field_value(rsmid_dict, "MAXR", int),
            get_tre_field_value(rsmid_dict, "MINC", int),
            get_tre_field_value(rsmid_dict, "MAXC", int),
        )

    @staticmethod
    def _build_rsm_context(rsmid_dict: dict) -> RSMContext:
        """
        Construct an RSM context from information in the RSMIDA TRE dict.

        :param rsmid_dict: the RSMIDA TRE field dict
        :return: the RSM context
        """
        return RSMContext(
            RSMSensorModelBuilder._build_rsm_ground_domain(rsmid_dict),
            RSMSensorModelBuilder._build_rsm_image_domain(rsmid_dict),
        )

    @staticmethod
    def _build_rsm_polynomial(rsmpc_dict: dict, polynomial_prefix: str) -> RSMPolynomial:
        """
        Construct an RSM polynomial from a group of related fields in the RSMPCA TRE dict. These
        TREs have similar fields grouped by the RN, RD, CN, and CD prefixes which correspond to the
        row or column (R or C) numerator or denominator (N or D) identifiers for the polynomial.

        :param rsmpc_dict: the RSMPCA TRE field dict
        :param polynomial_prefix: the prefix identifying the polynomial
        :return: the RSM polynomial
        """
        if polynomial_prefix not in ["RN", "RD", "CN", "CD"]:
            raise ValueError(f"Unexpected prefix {polynomial_prefix}. Expecting RN, RD, CN, or CD")

        max_power_x = get_tre_field_value(rsmpc_dict, f"{polynomial_prefix}PWRX", int)
        max_power_y = get_tre_field_value(rsmpc_dict, f"{polynomial_prefix}PWRY", int)
        max_power_z = get_tre_field_value(rsmpc_dict, f"{polynomial_prefix}PWRZ", int)

        # The number of coefficients is determined by the polynomial powers:
        # (max_power_x + 1) * (max_power_y + 1) * (max_power_z + 1)
        num_coefficients = (max_power_x + 1) * (max_power_y + 1) * (max_power_z + 1)

        coefficients = parse_tre_coefficient_list(rsmpc_dict, f"{polynomial_prefix}PCF", num_coefficients)

        return RSMPolynomial(max_power_x, max_power_y, max_power_z, coefficients)

    @staticmethod
    def _build_rsm_polynomial_sensor_model(rsmpc_dict: dict, rsm_context: RSMContext) -> RSMPolynomialSensorModel:
        """
        Construct an RSM polynomial sensor model from an RSMPCA TRE dict and the context object.

        :param rsmpc_dict: the RSMPCA TRE field dict
        :param rsm_context: the corresponding RSM context
        :return: the RSM polynomial sensor model
        """
        return RSMPolynomialSensorModel(
            rsm_context,
            get_tre_field_value(rsmpc_dict, "RSN", int),
            get_tre_field_value(rsmpc_dict, "CSN", int),
            get_tre_field_value(rsmpc_dict, "RNRMO", float),
            get_tre_field_value(rsmpc_dict, "CNRMO", float),
            get_tre_field_value(rsmpc_dict, "XNRMO", float),
            get_tre_field_value(rsmpc_dict, "YNRMO", float),
            get_tre_field_value(rsmpc_dict, "ZNRMO", float),
            get_tre_field_value(rsmpc_dict, "RNRMSF", float),
            get_tre_field_value(rsmpc_dict, "CNRMSF", float),
            get_tre_field_value(rsmpc_dict, "XNRMSF", float),
            get_tre_field_value(rsmpc_dict, "YNRMSF", float),
            get_tre_field_value(rsmpc_dict, "ZNRMSF", float),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_dict, "RN"),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_dict, "RD"),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_dict, "CN"),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_dict, "CD"),
        )

    @staticmethod
    def _build_loworder_rsm_polynomial(rsmpi_dict: dict, polynomial_prefix: str) -> RSMLowOrderPolynomial:
        """
        Construct a low order RSM polynomial from a group of related fields in the RSMPIA TRE dict.
        These TREs have similar fields grouped by the R and C prefixes which correspond to the row
        or column identifiers for the polynomial.

        :param rsmpi_dict: the RSMPIA TRE field dict
        :param polynomial_prefix: the prefix identifying the polynomial
        :return: the low order RSM polynomial
        """
        if polynomial_prefix not in ["R", "C"]:
            raise ValueError(f"Unexpected prefix {polynomial_prefix}. Expecting R or C")

        coefficients = []
        for coeff_suffix in ["0", "X", "Y", "Z", "XX", "XY", "XZ", "YY", "YZ", "ZZ"]:
            coefficients.append(get_tre_field_value(rsmpi_dict, f"{polynomial_prefix}{coeff_suffix}", float))
        return RSMLowOrderPolynomial(coefficients)

    @staticmethod
    def _build_rsm_sectioned_polynomial_sensor_model(
        rsmpi_dict: dict,
        rsm_context: RSMContext,
        rsm_polynomial_sensor_models: List[RSMPolynomialSensorModel],
    ) -> RSMSectionedPolynomialSensorModel:
        """
        Construct an RSM sectioned polynomial sensor model from an RSMPIA TRE dict, the context
        object, and a collection of RSMPolynomialSensorModels.

        :param rsmpi_dict: the RSMPIA TRE field dict
        :param rsm_context: the corresponding RSM context
        :param rsm_polynomial_sensor_models: the per-section polynomial sensor models
        :return: the RSM sectioned polynomial sensor model
        """
        num_section_rows = get_tre_field_value(rsmpi_dict, "RNIS", int)
        num_section_cols = get_tre_field_value(rsmpi_dict, "CNIS", int)
        sensor_model_grid_map = {}
        for sensor_model in rsm_polynomial_sensor_models:
            sensor_model_grid_map[(sensor_model.section_row, sensor_model.section_col)] = sensor_model

        section_sensor_model_grid: List[List[RSMPolynomialSensorModel]] = []
        for row in range(1, num_section_rows + 1):
            row_of_sensor_models: List[RSMPolynomialSensorModel] = []
            for col in range(1, num_section_cols + 1):
                row_of_sensor_models.append(sensor_model_grid_map[(row, col)])
            section_sensor_model_grid.append(row_of_sensor_models)

        return RSMSectionedPolynomialSensorModel(
            rsm_context,
            num_section_rows,
            num_section_cols,
            get_tre_field_value(rsmpi_dict, "RSSIZ", float),
            get_tre_field_value(rsmpi_dict, "CSSIZ", float),
            RSMSensorModelBuilder._build_loworder_rsm_polynomial(rsmpi_dict, "R"),
            RSMSensorModelBuilder._build_loworder_rsm_polynomial(rsmpi_dict, "C"),
            section_sensor_model_grid,
        )
