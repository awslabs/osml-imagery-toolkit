#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import logging
from typing import Optional, Union

from xsdata.formats.dataclass.parsers import XmlParser

import aws.osml.formats.sidd.models.sidd_v1_0_0 as sidd100
import aws.osml.formats.sidd.models.sidd_v2_0_0 as sidd200
import aws.osml.formats.sidd.models.sidd_v3_0_0 as sidd300

from ..photogrammetry import (
    ChippedImageSensorModel,
    ImageCoordinate,
    PlaneProjectionSet,
    SARImageCoordConverter,
    SensorModel,
    SICDSensorModel,
    WorldCoordinate,
)
from .sensor_model_builder import SensorModelBuilder
from .sicd_sensor_model_builder import poly2d_to_native, xyzpoly_to_native, xyztype_to_ndarray

logger = logging.getLogger(__name__)


class SIDDSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have SIDD metadata. The metadata is provided
    as XML that conforms to the SIDD specifications. We intend to support multiple SIDD versions but the current
    software was implemented using the v2.0.0 and v3.0.0 specifications.

    Note that the SIDD sensor models rely heavily on the SICD projections so the class of the returned model
    will be a SICDSensorModel. Future versions may rename this to SISensorModel or SARSensorModel.
    """

    def __init__(self, sidd_xml: str):
        """
        Construct the builder given the SIDD XML.

        :param sidd_xml: the XML string
        """
        super().__init__()
        self.sidd_xml = sidd_xml

    def build(self) -> Optional[SensorModel]:
        """
        Attempt to build a precise SAR sensor model. This sensor model handles chipped images natively.

        :return: the sensor model; if available
        """
        try:
            if self.sidd_xml is None or len(self.sidd_xml) == 0:
                return None

            parser = XmlParser()
            sicd = parser.from_string(self.sidd_xml)
            return SIDDSensorModelBuilder.from_dataclass(sicd)
        except Exception as e:
            logging.error("Exception caught attempting to build SIDD sensor model.", e)
        return None

    @staticmethod
    def from_dataclass(sidd: Union[sidd100.SIDD, sidd200.SIDD, sidd300.SIDD]) -> Optional[SensorModel]:
        """
        This method constructs a SIDD sensor model from the python dataclasses generated when parsing the XML. If
        the metadata shows that this is a chip then a ChippedImageSensorModel will be constructed to wrap the
        SICDSensorModel used for the full image.

        :param sidd: the dataclass object constructed from the XML
        :return: the sensor model; if available
        """

        plane_projection = sidd.measurement.plane_projection
        scp_ecf = WorldCoordinate(xyztype_to_ndarray(plane_projection.reference_point.ecef))
        scp_pixel = ImageCoordinate([plane_projection.reference_point.point.col, plane_projection.reference_point.point.row])
        time_coa_poly = poly2d_to_native(plane_projection.time_coapoly)
        arp_poly = xyzpoly_to_native(sidd.measurement.arppoly)

        u_row = xyztype_to_ndarray(plane_projection.product_plane.row_unit_vector)
        u_col = xyztype_to_ndarray(plane_projection.product_plane.col_unit_vector)
        coord_converter = SARImageCoordConverter(
            scp_pixel=scp_pixel,
            scp_ecf=scp_ecf,
            u_row=u_row,
            u_col=u_col,
            row_ss=plane_projection.sample_spacing.row,
            col_ss=plane_projection.sample_spacing.col,
            first_pixel=ImageCoordinate([0, 0]),
        )

        projection_set = PlaneProjectionSet(
            scp_ecf=scp_ecf,
            image_plane_urow=u_row,
            image_plane_ucol=u_col,
            coa_time_poly=time_coa_poly,
            arp_poly=arp_poly,
        )

        center_time = projection_set.coa_time_poly(0, 0)
        side_of_track = SICDSensorModel.compute_side_of_track(
            scp_ecf=scp_ecf,
            scp_arp=projection_set.arp_poly(center_time),
            scp_varp=projection_set.varp_poly(center_time),
        )
        u_spn = SICDSensorModel.compute_u_spn(
            scp_ecf=scp_ecf,
            scp_arp=projection_set.arp_poly(center_time),
            scp_varp=projection_set.varp_poly(center_time),
            side_of_track=side_of_track,
        )

        u_gpn = SICDSensorModel.compute_u_gpn(scp_ecf=scp_ecf, u_row=u_row, u_col=u_col)

        sidd_sensor_model = SICDSensorModel(
            coord_converter=coord_converter,
            coa_projection_set=projection_set,
            u_spn=u_spn,
            side_of_track=side_of_track,
            u_gpn=u_gpn,
        )

        if sidd.downstream_reprocessing is None or sidd.downstream_reprocessing.geometric_chip is None:
            return sidd_sensor_model
        else:
            # Since this SIDD image is a chip of a full image wrap the regular sensor model in a sensor model that
            # will handle the conversions between the chipped image coordinates and the full image coordinates.
            # This 4 corner transformation handles images that are cropped, rotated, and scaled, from the full
            # SIDD image grid.
            geo_chip = sidd.downstream_reprocessing.geometric_chip
            chip_num_rows = geo_chip.chip_size.row
            chip_num_cols = geo_chip.chip_size.col

            chipped_image_coords = [
                ImageCoordinate(coord)
                for coord in [[0, 0], [chip_num_cols, 0], [chip_num_cols, chip_num_rows], [0, chip_num_rows]]
            ]
            full_image_coords = [
                ImageCoordinate([x.col, x.row])
                for x in [
                    geo_chip.original_upper_left_coordinate,
                    geo_chip.original_upper_right_coordinate,
                    geo_chip.original_lower_right_coordinate,
                    geo_chip.original_lower_left_coordinate,
                ]
            ]

            return ChippedImageSensorModel(
                full_image_coords,
                chipped_image_coords,
                sidd_sensor_model,
            )
