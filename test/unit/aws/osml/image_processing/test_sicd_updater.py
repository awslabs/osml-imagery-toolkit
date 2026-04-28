#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.sicd_updater`.

Tests cover the stateless ``update_sicd_for_chip()`` function, verifying
correct ImageData updates for various chipping scenarios including offset
images and the scaling-rejection guard.
"""

from pathlib import Path

import pytest

from aws.osml.formats.model_utils import sicd_parser, sicd_serializer
from aws.osml.image_processing.sicd_updater import update_sicd_for_chip

TEST_DATA = Path(__file__).resolve().parents[4] / "data" / "sicd"
SICD_XML_PATH = TEST_DATA / "example.sicd121.capella.xml"


@pytest.fixture
def sicd_xml():
    return SICD_XML_PATH.read_text()


class TestUpdateSicdForChip:
    """Tests for the stateless update_sicd_for_chip function."""

    def test_basic_chip_updates_image_data(self, sicd_xml):
        """Chipping a region updates FirstRow, FirstCol, NumRows, NumCols."""
        chip_bounds = [100, 200, 512, 256]  # col, row, width, height
        result_xml = update_sicd_for_chip(sicd_xml, chip_bounds)

        sicd = sicd_parser.from_string(result_xml)
        assert sicd.image_data.first_row == 200
        assert sicd.image_data.first_col == 100
        assert sicd.image_data.num_rows == 256
        assert sicd.image_data.num_cols == 512

    def test_chip_with_matching_output_size(self, sicd_xml):
        """output_size matching chip dimensions is accepted."""
        chip_bounds = [50, 75, 1024, 768]
        result_xml = update_sicd_for_chip(sicd_xml, chip_bounds, output_size=(1024, 768))

        sicd = sicd_parser.from_string(result_xml)
        assert sicd.image_data.first_row == 75
        assert sicd.image_data.first_col == 50
        assert sicd.image_data.num_rows == 768
        assert sicd.image_data.num_cols == 1024

    def test_scaling_raises_value_error(self, sicd_xml):
        """output_size different from chip dimensions raises ValueError."""
        chip_bounds = [0, 0, 512, 512]
        with pytest.raises(ValueError, match="does not support scaling"):
            update_sicd_for_chip(sicd_xml, chip_bounds, output_size=(256, 256))

    def test_chip_at_origin(self, sicd_xml):
        """Chip at (0, 0) preserves the original first_row/first_col offsets."""
        chip_bounds = [0, 0, 100, 100]
        result_xml = update_sicd_for_chip(sicd_xml, chip_bounds)

        sicd = sicd_parser.from_string(result_xml)
        assert sicd.image_data.first_row == 0
        assert sicd.image_data.first_col == 0
        assert sicd.image_data.num_rows == 100
        assert sicd.image_data.num_cols == 100

    def test_result_is_valid_xml_roundtrip(self, sicd_xml):
        """Output can be re-parsed by the SICD parser (valid XML)."""
        chip_bounds = [10, 20, 300, 400]
        result_xml = update_sicd_for_chip(sicd_xml, chip_bounds)

        sicd = sicd_parser.from_string(result_xml)
        assert sicd is not None
        assert sicd.image_data is not None

    def test_does_not_mutate_original(self, sicd_xml):
        """Calling update_sicd_for_chip does not alter the original XML string."""
        original_copy = sicd_xml
        update_sicd_for_chip(sicd_xml, [10, 20, 100, 100])
        assert sicd_xml == original_copy

    def test_chip_from_already_offset_image(self):
        """Chipping an image where FirstRow/FirstCol are already non-zero
        accumulates the offsets correctly."""
        sicd = sicd_parser.from_string(SICD_XML_PATH.read_text())
        sicd.image_data.first_row = 1000
        sicd.image_data.first_col = 2000
        offset_xml = sicd_serializer.render(sicd)

        chip_bounds = [50, 100, 256, 128]
        result_xml = update_sicd_for_chip(offset_xml, chip_bounds)

        result = sicd_parser.from_string(result_xml)
        assert result.image_data.first_row == 1100  # 1000 + 100
        assert result.image_data.first_col == 2050  # 2000 + 50
        assert result.image_data.num_rows == 128
        assert result.image_data.num_cols == 256
