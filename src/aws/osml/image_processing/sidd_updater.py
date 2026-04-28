#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import logging
from typing import List, Optional, Tuple

import aws.osml.formats.sidd.models.sidd_v1_0_0 as sidd100
import aws.osml.formats.sidd.models.sidd_v2_0_0 as sidd200
import aws.osml.formats.sidd.models.sidd_v3_0_0 as sidd300
from aws.osml.formats.model_utils import sidd_parser, sidd_serializer

logger = logging.getLogger(__name__)


def chipped_coordinate_to_full(
    chip_coordinate: Tuple[float, float],
    chip_size: Tuple[int, int],
    original_corner_coordinates: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Convert pixel locations in a chip to pixel locations in the full image.

    Uses a bi-linear interpolation method described in section 5.1.1 of the
    Sensor Independent Derived Data (SIDD) specification v3.0 Volume 1.

    :param chip_coordinate: the (col, row) coordinate of the pixel in the chip
    :param chip_size: the size of the chip (width, height)
    :param original_corner_coordinates: the (col, row) location of the UL, UR, LR, LL corners in the original image
    :return: the (col, row) coordinate of the pixel in the original image
    """
    u = chip_coordinate[1] / (chip_size[1] - 1)
    v = chip_coordinate[0] / (chip_size[0] - 1)

    a_r = original_corner_coordinates[0][1]
    b_r = original_corner_coordinates[3][1] - original_corner_coordinates[0][1]
    d_r = original_corner_coordinates[1][1] - original_corner_coordinates[0][1]
    f_r = (
        original_corner_coordinates[0][1]
        + original_corner_coordinates[2][1]
        - original_corner_coordinates[1][1]
        - original_corner_coordinates[3][1]
    )

    a_c = original_corner_coordinates[0][0]
    b_c = original_corner_coordinates[3][0] - original_corner_coordinates[0][0]
    d_c = original_corner_coordinates[1][0] - original_corner_coordinates[0][0]
    f_c = (
        original_corner_coordinates[0][0]
        + original_corner_coordinates[2][0]
        - original_corner_coordinates[1][0]
        - original_corner_coordinates[3][0]
    )

    r = a_r + u * b_r + v * d_r + u * v * f_r
    c = a_c + u * b_c + v * d_c + u * v * f_c

    return c, r


def update_sidd_for_chip(xml_str: str, chip_bounds: List[int], output_size: Optional[Tuple[int, int]] = None) -> str:
    """Update SIDD XML metadata for a chipped region (stateless).

    Parses the provided SIDD XML, updates the GeometricChip structure
    for the chip bounds, and returns the serialized updated XML. Does
    not mutate any shared state — safe for concurrent calls.

    :param xml_str: the SIDD XML metadata to update
    :param chip_bounds: the [col, row, width, height] of the chip boundary
    :param output_size: the [width, height] of the output chip if different from the chip boundary
    :return: updated SIDD XML string
    """
    sidd = sidd_parser.from_string(xml_str)

    if not output_size:
        output_size = chip_bounds[2], chip_bounds[3]

    if isinstance(sidd, sidd100.SIDD):
        sidd_namespace = sidd100
    elif isinstance(sidd, sidd200.SIDD):
        sidd_namespace = sidd200
    elif isinstance(sidd, sidd300.SIDD):
        sidd_namespace = sidd300
    else:
        logger.warning("sidd_updater.py has not been updated to support a new SIDD version. Defaulting to 3.0")
        sidd_namespace = sidd300

    if not sidd.downstream_reprocessing:
        sidd.downstream_reprocessing = sidd_namespace.DownstreamReprocessingType()

    full_image_chip_corners = [
        (chip_bounds[0], chip_bounds[1]),
        (chip_bounds[0] + chip_bounds[2] - 1, chip_bounds[1]),
        (chip_bounds[0] + chip_bounds[2] - 1, chip_bounds[1] + chip_bounds[3] - 1),
        (chip_bounds[0], chip_bounds[1] + chip_bounds[3] - 1),
    ]
    if sidd.downstream_reprocessing.geometric_chip:
        original_chip_size = (
            sidd.downstream_reprocessing.geometric_chip.chip_size.col,
            sidd.downstream_reprocessing.geometric_chip.chip_size.row,
        )
        original_corners = [
            (
                sidd.downstream_reprocessing.geometric_chip.original_upper_left_coordinate.col,
                sidd.downstream_reprocessing.geometric_chip.original_upper_left_coordinate.row,
            ),
            (
                sidd.downstream_reprocessing.geometric_chip.original_upper_right_coordinate.col,
                sidd.downstream_reprocessing.geometric_chip.original_upper_right_coordinate.row,
            ),
            (
                sidd.downstream_reprocessing.geometric_chip.original_lower_right_coordinate.col,
                sidd.downstream_reprocessing.geometric_chip.original_lower_right_coordinate.row,
            ),
            (
                sidd.downstream_reprocessing.geometric_chip.original_lower_left_coordinate.col,
                sidd.downstream_reprocessing.geometric_chip.original_lower_left_coordinate.row,
            ),
        ]

        full_image_chip_corners = [
            chipped_coordinate_to_full(corner, original_chip_size, original_corners) for corner in full_image_chip_corners
        ]

    sidd.downstream_reprocessing.geometric_chip = sidd_namespace.GeometricChipType(
        chip_size=sidd_namespace.RowColIntType(row=output_size[1], col=output_size[0]),
        original_upper_left_coordinate=sidd_namespace.RowColDoubleType(
            row=full_image_chip_corners[0][1], col=full_image_chip_corners[0][0]
        ),
        original_upper_right_coordinate=sidd_namespace.RowColDoubleType(
            row=full_image_chip_corners[1][1], col=full_image_chip_corners[1][0]
        ),
        original_lower_left_coordinate=sidd_namespace.RowColDoubleType(
            row=full_image_chip_corners[3][1], col=full_image_chip_corners[3][0]
        ),
        original_lower_right_coordinate=sidd_namespace.RowColDoubleType(
            row=full_image_chip_corners[2][1], col=full_image_chip_corners[2][0]
        ),
    )

    return sidd_serializer.render(sidd)
