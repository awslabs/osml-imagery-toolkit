#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from typing import Optional, Union

import numpy as np

import aws.osml.formats.sicd.models.sicd_v1_2_1 as sicd121
import aws.osml.formats.sicd.models.sicd_v1_3_0 as sicd130
from aws.osml.formats.model_utils import sicd_parser

from ..photogrammetry import (
    ImageCoordinate,
    INCAProjectionSet,
    PFAProjectionSet,
    PlaneProjectionSet,
    Polynomial2D,
    PolynomialXYZ,
    RGAZCOMPProjectionSet,
    SARImageCoordConverter,
    SensorModel,
    SICDSensorModel,
    WorldCoordinate,
)
from .sensor_model_builder import SensorModelBuilder

logger = logging.getLogger(__name__)


def xyztype_to_ndarray(xyztype: Union[sicd121.XYZType, sicd130.XYZType]) -> np.ndarray:
    """
    Convert the XYZType to a 1-d NumPy array.

    :param xyztype: the XYZType dataclass
    :return: the NumPy array
    """
    return np.array([xyztype.x, xyztype.y, xyztype.z])


def poly1d_to_native(poly1d: Union[sicd121.Poly1DType, sicd130.XYZType]) -> np.polynomial.Polynomial:
    """
    Convert the Poly1DType to a NumPy Polynomial.

    :param poly1d: the Poly1D dataclass
    :return: the NumPy polynomial with matching coefficients
    """
    coefficients = [0] * (poly1d.order1 + 1)
    for coef in poly1d.coef:
        coefficients[coef.exponent1] = coef.value
    return np.polynomial.Polynomial(coefficients)


def poly2d_to_native(poly2d: Union[sicd121.Poly2DType, sicd130.Poly2DType]) -> Polynomial2D:
    """
    Convert the Poly2D dataclass to a Polynomial2D.

    :param poly2d: the Poly2D dataclass
    :return: the Polynomial2D with matching coefficients
    """
    coefficients = [0] * (poly2d.order1 + 1)
    for row in range(0, len(coefficients)):
        coefficients[row] = [0] * (poly2d.order2 + 1)
    for coef in poly2d.coef:
        coefficients[coef.exponent1][coef.exponent2] = coef.value
    return Polynomial2D(np.array(coefficients))


def xyzpoly_to_native(xyzpoly: Union[sicd121.XYZPolyType, sicd130.XYZPolyType]) -> PolynomialXYZ:
    """
    Convert the XYZPoly dataclass into a PolynomialXYZ.

    :param xyzpoly: the XYZPoly dataclass
    :return: the PolynomialXYZ with matching coefficients
    """
    return PolynomialXYZ(
        x_polynomial=poly1d_to_native(xyzpoly.x),
        y_polynomial=poly1d_to_native(xyzpoly.y),
        z_polynomial=poly1d_to_native(xyzpoly.z),
    )


class SICDSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have SICD metadata. The metadata is provided
    as XML that conforms to the SICD specifications. We intend to support multiple SICD versions but the current
    software was implemented using the v1.2.1 and v1.3.0 specifications.
    """

    def __init__(self, sicd_xml: str):
        """
        Construct the builder given the SICD XML.

        :param sicd_xml: the XML string
        """
        super().__init__()
        self.sicd_xml = sicd_xml

    def build(self) -> Optional[SensorModel]:
        """
        Attempt to build a precise SAR sensor model. This sensor model handles chipped images natively.

        :return: the sensor model; if available
        """
        try:
            if self.sicd_xml is None or len(self.sicd_xml) == 0:
                return None

            sicd = sicd_parser.from_string(self.sicd_xml)
            return SICDSensorModelBuilder.from_dataclass(sicd)
        except Exception as e:
            logging.error("Exception caught attempting to build SICD sensor model. %s", e)
        return None

    @staticmethod
    def from_dataclass(sicd: Union[sicd121.SICD, sicd130.SICD]) -> Optional[SensorModel]:
        """
        This method constructs a SICD sensor model from the python dataclasses generated when parsing the XML.

        :param sicd: the dataclass object constructed from the XML
        :return: the sensor model; if available
        """

        scp_ecf = WorldCoordinate(xyztype_to_ndarray(sicd.geo_data.scp.ecf))
        scp_pixel = ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])
        time_coa_poly = poly2d_to_native(sicd.grid.time_coapoly)
        arp_poly = xyzpoly_to_native(sicd.position.arppoly)

        coord_converter = SARImageCoordConverter(
            scp_pixel=scp_pixel,
            scp_ecf=scp_ecf,
            u_row=xyztype_to_ndarray(sicd.grid.row.uvect_ecf),
            u_col=xyztype_to_ndarray(sicd.grid.col.uvect_ecf),
            row_ss=sicd.grid.row.ss,
            col_ss=sicd.grid.col.ss,
            first_pixel=ImageCoordinate([sicd.image_data.first_col, sicd.image_data.first_row]),
        )

        projection_set = None
        ugpn = None
        if sicd.grid.type_value.value == sicd121.ImageGridType.RGAZIM.value:
            if sicd.image_formation.image_form_algo.value == sicd121.ImageFormAlgo.PFA.value:
                ugpn = xyztype_to_ndarray(sicd.pfa.fpn)
                projection_set = PFAProjectionSet(
                    scp_ecf=scp_ecf,
                    polar_ang_poly=poly1d_to_native(sicd.pfa.polar_ang_poly),
                    spatial_freq_sf_poly=poly1d_to_native(sicd.pfa.spatial_freq_sfpoly),
                    coa_time_poly=time_coa_poly,
                    arp_poly=arp_poly,
                )
            elif sicd.image_formation.image_form_algo.value == sicd121.ImageFormAlgo.RGAZCOMP.value:
                projection_set = RGAZCOMPProjectionSet(
                    scp_ecf=scp_ecf, az_scale_factor=sicd.rg_az_comp.az_sf, coa_time_poly=time_coa_poly, arp_poly=arp_poly
                )
        elif sicd.grid.type_value.value == sicd121.ImageGridType.RGZERO.value:
            projection_set = INCAProjectionSet(
                r_ca_scp=sicd.rma.inca.r_ca_scp,
                inca_time_coa_poly=poly1d_to_native(sicd.rma.inca.time_capoly),
                drate_sf_poly=poly2d_to_native(sicd.rma.inca.drate_sfpoly),
                coa_time_poly=time_coa_poly,
                arp_poly=arp_poly,
            )
        elif sicd.grid.type_value.value in [
            sicd121.ImageGridType.PLANE.value,
            sicd121.ImageGridType.XCTYAT.value,
            sicd121.ImageGridType.XRGYCR.value,
        ]:
            projection_set = PlaneProjectionSet(
                scp_ecf=scp_ecf,
                image_plane_urow=xyztype_to_ndarray(sicd.grid.row.uvect_ecf),
                image_plane_ucol=xyztype_to_ndarray(sicd.grid.col.uvect_ecf),
                coa_time_poly=time_coa_poly,
                arp_poly=arp_poly,
            )
        else:
            logger.warning(f"SICD image with unknown grid type {sicd.grid.type_value}. No sensor model created.")
            return None

        sicd_sensor_model = SICDSensorModel(
            coord_converter=coord_converter,
            coa_projection_set=projection_set,
            u_spn=SICDSensorModel.compute_u_spn(
                scp_ecf=scp_ecf,
                scp_arp=xyztype_to_ndarray(sicd.scpcoa.arppos),
                scp_varp=xyztype_to_ndarray(sicd.scpcoa.arpvel),
                side_of_track=str(sicd.scpcoa.side_of_track.value),
            ),
            side_of_track=str(sicd.scpcoa.side_of_track.value),
            u_gpn=ugpn,
        )

        return sicd_sensor_model
