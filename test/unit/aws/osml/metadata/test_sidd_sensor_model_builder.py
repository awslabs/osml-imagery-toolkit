#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from aws.osml.metadata.sidd_sensor_model_builder import SIDDSensorModelBuilder
from aws.osml.photogrammetry import ChippedImageSensorModel, SICDSensorModel

# Real SIDD XML loaded from test data files and passed as inline strings to the builder.
# We read the XML files once at module level so the tests exercise the full build() path
# (XML parsing → dataclass construction → sensor model construction) without GDAL.
SIDD_XML_PATH = Path("./test/data/sidd/example.sidd.xml")
SIDD_CHIP_XML_PATH = Path("./test/data/sidd/example.sidd-chip.xml")


def _read_xml(path: Path) -> str:
    """Read XML file content as a string."""
    return path.read_text(encoding="utf-8")


class TestSIDDSensorModelBuilder:
    """Unit tests for SIDDSensorModelBuilder. Requirements: 6.1, 6.2, 6.3."""

    def test_build_from_valid_sidd_xml_produces_sicd_sensor_model(self):
        """Build from valid SIDD planar projection XML produces a SICDSensorModel."""
        xml_str = _read_xml(SIDD_XML_PATH)
        builder = SIDDSensorModelBuilder(xml_str)
        model = builder.build()

        assert model is not None
        assert isinstance(model, SICDSensorModel)

    def test_build_from_chipped_sidd_xml_produces_chipped_image_sensor_model(self):
        """Build from chipped SIDD XML produces a ChippedImageSensorModel."""
        xml_str = _read_xml(SIDD_CHIP_XML_PATH)
        builder = SIDDSensorModelBuilder(xml_str)
        model = builder.build()

        assert model is not None
        assert isinstance(model, ChippedImageSensorModel)

    def test_returns_none_for_empty_string(self):
        """Empty XML string should return None."""
        builder = SIDDSensorModelBuilder("")
        assert builder.build() is None

    def test_returns_none_for_none_xml(self):
        """None XML should return None."""
        builder = SIDDSensorModelBuilder(None)
        assert builder.build() is None

    def test_returns_none_for_malformed_xml(self, caplog):
        """Malformed XML should return None and log an error."""
        builder = SIDDSensorModelBuilder("<SIDD><broken>")
        with caplog.at_level(logging.ERROR):
            result = builder.build()

        assert result is None

    def test_returns_none_for_invalid_sidd_content(self, caplog):
        """Valid XML but not valid SIDD content should return None and log an error."""
        builder = SIDDSensorModelBuilder("<root><child>value</child></root>")
        with caplog.at_level(logging.ERROR):
            result = builder.build()

        assert result is None

    def test_build_delegates_to_from_dataclass(self):
        """Verify build() calls sidd_parser.from_string and from_dataclass."""
        mock_sidd = MagicMock()
        mock_sensor_model = MagicMock(spec=SICDSensorModel)

        with (
            patch(
                "aws.osml.metadata.sidd_sensor_model_builder.sidd_parser.from_string",
                return_value=mock_sidd,
            ) as mock_from_string,
            patch(
                "aws.osml.metadata.sidd_sensor_model_builder.SIDDSensorModelBuilder.from_dataclass",
                return_value=mock_sensor_model,
            ) as mock_from_dataclass,
        ):
            builder = SIDDSensorModelBuilder("<SIDD>valid</SIDD>")
            result = builder.build()

            mock_from_string.assert_called_once_with("<SIDD>valid</SIDD>")
            mock_from_dataclass.assert_called_once_with(mock_sidd)
            assert result is mock_sensor_model
