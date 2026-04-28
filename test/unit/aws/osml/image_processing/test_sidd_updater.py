#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for :mod:`aws.osml.image_processing.sidd_updater`.

Tests cover the stateless ``update_sidd_for_chip()`` function and the
``chipped_coordinate_to_full()`` bi-linear mapping, verifying correct
GeometricChip creation and updates for fresh images, already-chipped
images, and scaled chips.
"""

from pathlib import Path

import pytest

from aws.osml.image_processing.sidd_updater import chipped_coordinate_to_full, update_sidd_for_chip

TEST_DATA = Path(__file__).resolve().parents[4] / "data" / "sidd"
SIDD_XML_PATH = TEST_DATA / "example.sidd.xml"
SIDD_CHIP_XML_PATH = TEST_DATA / "example.sidd-chip.xml"


@pytest.fixture
def sidd_xml():
    """Full SIDD image — no existing GeometricChip."""
    return SIDD_XML_PATH.read_text()


@pytest.fixture
def sidd_chip_xml():
    """Already-chipped SIDD image — has GeometricChip in DownstreamReprocessing."""
    return SIDD_CHIP_XML_PATH.read_text()


class TestUpdateSiddForChip:
    """Tests for the stateless update_sidd_for_chip function."""

    def test_creates_geometric_chip_on_fresh_image(self, sidd_xml):
        """First chip of a full image creates DownstreamReprocessing.GeometricChip."""
        chip_bounds = [100, 200, 512, 256]  # col, row, width, height
        result_xml = update_sidd_for_chip(sidd_xml, chip_bounds)

        from aws.osml.formats.model_utils import sidd_parser

        sidd = sidd_parser.from_string(result_xml)
        gc = sidd.downstream_reprocessing.geometric_chip
        assert gc is not None
        assert gc.chip_size.row == 256
        assert gc.chip_size.col == 512

    def test_corner_coordinates_match_chip_bounds(self, sidd_xml):
        """Corner coordinates reflect the chip position in the full image."""
        chip_bounds = [100, 200, 512, 256]
        result_xml = update_sidd_for_chip(sidd_xml, chip_bounds)

        from aws.osml.formats.model_utils import sidd_parser

        sidd = sidd_parser.from_string(result_xml)
        gc = sidd.downstream_reprocessing.geometric_chip

        # UL = (col, row)
        assert gc.original_upper_left_coordinate.col == 100
        assert gc.original_upper_left_coordinate.row == 200
        # UR = (col + width - 1, row)
        assert gc.original_upper_right_coordinate.col == 611
        assert gc.original_upper_right_coordinate.row == 200
        # LR = (col + width - 1, row + height - 1)
        assert gc.original_lower_right_coordinate.col == 611
        assert gc.original_lower_right_coordinate.row == 455
        # LL = (col, row + height - 1)
        assert gc.original_lower_left_coordinate.col == 100
        assert gc.original_lower_left_coordinate.row == 455

    def test_output_size_defaults_to_chip_dimensions(self, sidd_xml):
        """When output_size is None, ChipSize equals chip_bounds width/height."""
        chip_bounds = [0, 0, 300, 400]
        result_xml = update_sidd_for_chip(sidd_xml, chip_bounds)

        from aws.osml.formats.model_utils import sidd_parser

        sidd = sidd_parser.from_string(result_xml)
        gc = sidd.downstream_reprocessing.geometric_chip
        assert gc.chip_size.col == 300
        assert gc.chip_size.row == 400

    def test_explicit_output_size_for_scaled_chip(self, sidd_xml):
        """output_size different from chip dimensions records the scaled size."""
        chip_bounds = [0, 0, 1024, 1024]
        result_xml = update_sidd_for_chip(sidd_xml, chip_bounds, output_size=(512, 512))

        from aws.osml.formats.model_utils import sidd_parser

        sidd = sidd_parser.from_string(result_xml)
        gc = sidd.downstream_reprocessing.geometric_chip
        assert gc.chip_size.col == 512
        assert gc.chip_size.row == 512
        # Corner coordinates still reference the full chip region
        assert gc.original_upper_left_coordinate.col == 0
        assert gc.original_upper_left_coordinate.row == 0
        assert gc.original_lower_right_coordinate.col == 1023
        assert gc.original_lower_right_coordinate.row == 1023

    def test_rechip_of_already_chipped_image(self, sidd_chip_xml):
        """Chipping an already-chipped image maps coordinates back to full image."""
        # The fixture has a 512x512 chip at UL=(512, 512) in the full image
        chip_bounds = [0, 0, 256, 256]
        result_xml = update_sidd_for_chip(sidd_chip_xml, chip_bounds)

        from aws.osml.formats.model_utils import sidd_parser

        sidd = sidd_parser.from_string(result_xml)
        gc = sidd.downstream_reprocessing.geometric_chip
        assert gc.chip_size.row == 256
        assert gc.chip_size.col == 256
        assert gc.original_upper_left_coordinate.col == pytest.approx(512.0, abs=1.0)
        assert gc.original_upper_left_coordinate.row == pytest.approx(512.0, abs=1.0)

    def test_rechip_center_region(self, sidd_chip_xml):
        """Sub-chip from center of existing chip maps correctly to full image."""
        # Parent chip: 512x512, UL at (512, 512) in full image, axis-aligned
        chip_bounds = [128, 128, 256, 256]
        result_xml = update_sidd_for_chip(sidd_chip_xml, chip_bounds)

        from aws.osml.formats.model_utils import sidd_parser

        sidd = sidd_parser.from_string(result_xml)
        gc = sidd.downstream_reprocessing.geometric_chip
        assert gc.original_upper_left_coordinate.col == pytest.approx(640.0, abs=1.0)
        assert gc.original_upper_left_coordinate.row == pytest.approx(640.0, abs=1.0)

    def test_result_is_valid_xml_roundtrip(self, sidd_xml):
        """Output can be re-parsed (valid XML structure)."""
        result_xml = update_sidd_for_chip(sidd_xml, [50, 50, 200, 200])

        from aws.osml.formats.model_utils import sidd_parser

        parsed = sidd_parser.from_string(result_xml)
        assert parsed is not None
        assert parsed.downstream_reprocessing is not None


class TestChippedCoordinateToFull:
    """Tests for chipped_coordinate_to_full() bi-linear mapping."""

    def test_identity_mapping_corners(self):
        """When chip occupies the same region as original corners, mapping is identity."""
        chip_size = (100, 100)
        original_corners = [
            (0, 0),  # UL (col, row)
            (99, 0),  # UR
            (99, 99),  # LR
            (0, 99),  # LL
        ]

        result = chipped_coordinate_to_full((0, 0), chip_size, original_corners)
        assert result == pytest.approx((0, 0))

        result = chipped_coordinate_to_full((99, 0), chip_size, original_corners)
        assert result == pytest.approx((99, 0))

        result = chipped_coordinate_to_full((99, 99), chip_size, original_corners)
        assert result == pytest.approx((99, 99))

        result = chipped_coordinate_to_full((0, 99), chip_size, original_corners)
        assert result == pytest.approx((0, 99))

    def test_offset_chip(self):
        """Chip offset into the full image maps corners correctly."""
        chip_size = (512, 512)
        original_corners = [
            (100, 200),  # UL
            (611, 200),  # UR
            (611, 711),  # LR
            (100, 711),  # LL
        ]

        result = chipped_coordinate_to_full((0, 0), chip_size, original_corners)
        assert result == pytest.approx((100, 200))

        result = chipped_coordinate_to_full((511, 511), chip_size, original_corners)
        assert result == pytest.approx((611, 711))

    def test_center_point(self):
        """Center of an axis-aligned chip maps to center of the original region."""
        chip_size = (100, 100)
        original_corners = [
            (0, 0),
            (99, 0),
            (99, 99),
            (0, 99),
        ]
        result = chipped_coordinate_to_full((49.5, 49.5), chip_size, original_corners)
        assert result[0] == pytest.approx(49.5, abs=0.1)
        assert result[1] == pytest.approx(49.5, abs=0.1)

    def test_rotated_chip(self):
        """Non-axis-aligned (rotated) chip uses bi-linear interpolation correctly."""
        chip_size = (100, 100)
        # A chip that's rotated 90 degrees
        original_corners = [
            (0, 99),  # UL
            (0, 0),  # UR
            (99, 0),  # LR
            (99, 99),  # LL
        ]

        result = chipped_coordinate_to_full((0, 0), chip_size, original_corners)
        assert result == pytest.approx((0, 99))

        result = chipped_coordinate_to_full((99, 0), chip_size, original_corners)
        assert result == pytest.approx((0, 0))

    def test_scaled_chip_bilinear_interpolation(self):
        """Quarter-point of a scaled chip interpolates correctly."""
        chip_size = (200, 200)
        original_corners = [
            (1000, 2000),  # UL
            (1199, 2000),  # UR
            (1199, 2199),  # LR
            (1000, 2199),  # LL
        ]
        result = chipped_coordinate_to_full((50, 50), chip_size, original_corners)
        assert result[0] == pytest.approx(1000 + (50.0 / 199.0) * 199, abs=0.5)
        assert result[1] == pytest.approx(2000 + (50.0 / 199.0) * 199, abs=0.5)
