#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

import logging
from math import floor
from typing import List, Optional, Tuple

import aws.osml.formats.sicd.models.sicd_v1_2_1 as sicd121  # noqa: F401 — registers model with xsdata context
import aws.osml.formats.sicd.models.sicd_v1_3_0 as sicd130  # noqa: F401
from aws.osml.formats.model_utils import sicd_parser, sicd_serializer

logger = logging.getLogger(__name__)


def update_sicd_for_chip(xml_str: str, chip_bounds: List[int], output_size: Optional[Tuple[int, int]] = None) -> str:
    """Update SICD XML metadata for a chipped region (stateless).

    Parses the provided SICD XML, updates ImageData fields for the chip
    bounds, and returns the serialized updated XML. Does not mutate any
    shared state — safe for concurrent calls.

    :param xml_str: the SICD XML metadata to update
    :param chip_bounds: the [col, row, width, height] of the chip boundary
    :param output_size: the [width, height] of the output chip
    :return: updated SICD XML string
    :raises ValueError: if output_size differs from chip dimensions (SICD does not support decimation)
    """
    if output_size is not None and (output_size[0] != chip_bounds[2] or output_size[1] != chip_bounds[3]):
        raise ValueError("SICD chipping does not support scaling operations.")

    sicd = sicd_parser.from_string(xml_str)

    original_first_row = sicd.image_data.first_row
    original_first_col = sicd.image_data.first_col

    sicd.image_data.first_row = floor(float(original_first_row)) + int(chip_bounds[1])
    sicd.image_data.first_col = floor(float(original_first_col)) + int(chip_bounds[0])
    sicd.image_data.num_rows = int(chip_bounds[3])
    sicd.image_data.num_cols = int(chip_bounds[2])

    return sicd_serializer.render(sicd)
