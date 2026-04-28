#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from aws.osml.metadata.sicd_sensor_model_builder import SICDSensorModelBuilder
from aws.osml.photogrammetry import SICDSensorModel

# Real SICD XML loaded from test data files and passed as inline strings to the builder.
# We read the XML files once at module level so the tests exercise the full build() path
# (XML parsing → dataclass construction → sensor model construction) without GDAL.
SICD_PFA_XML_PATH = Path("./test/data/sicd/example.sicd121.pfa.xml")
SICD_RMA_XML_PATH = Path("./test/data/sicd/example.sicd121.rma.xml")
SICD_CAPELLA_XML_PATH = Path("./test/data/sicd/example.sicd121.capella.xml")


def _read_xml(path: Path) -> str:
    """Read XML file content as a string."""
    return path.read_text(encoding="utf-8")


class TestSICDSensorModelBuilder:
    """Unit tests for SICDSensorModelBuilder. Requirements: 5.1, 5.2, 5.3."""

    def test_build_from_valid_pfa_xml_produces_sicd_sensor_model(self):
        """Build from valid SICD PFA XML produces a SICDSensorModel instance."""
        xml_str = _read_xml(SICD_PFA_XML_PATH)
        builder = SICDSensorModelBuilder(xml_str)
        model = builder.build()

        assert model is not None
        assert isinstance(model, SICDSensorModel)

    def test_build_from_valid_rma_xml_produces_sicd_sensor_model(self):
        """Build from valid SICD RMA XML produces a SICDSensorModel instance."""
        xml_str = _read_xml(SICD_RMA_XML_PATH)
        builder = SICDSensorModelBuilder(xml_str)
        model = builder.build()

        assert model is not None
        assert isinstance(model, SICDSensorModel)

    def test_build_from_valid_capella_xml_produces_sicd_sensor_model(self):
        """Build from valid SICD Capella (INCA) XML produces a SICDSensorModel instance."""
        xml_str = _read_xml(SICD_CAPELLA_XML_PATH)
        builder = SICDSensorModelBuilder(xml_str)
        model = builder.build()

        assert model is not None
        assert isinstance(model, SICDSensorModel)

    def test_returns_none_for_empty_string(self):
        """Empty XML string should return None."""
        builder = SICDSensorModelBuilder("")
        assert builder.build() is None

    def test_returns_none_for_none_xml(self):
        """None XML should return None."""
        builder = SICDSensorModelBuilder(None)
        assert builder.build() is None

    def test_returns_none_for_malformed_xml(self, caplog):
        """Malformed XML should return None and log an error."""
        builder = SICDSensorModelBuilder("<SICD><broken>")
        with caplog.at_level(logging.ERROR):
            result = builder.build()

        assert result is None

    def test_returns_none_for_invalid_sicd_content(self, caplog):
        """Valid XML but not valid SICD content should return None and log an error."""
        builder = SICDSensorModelBuilder("<root><child>value</child></root>")
        with caplog.at_level(logging.ERROR):
            result = builder.build()

        assert result is None

    def test_build_delegates_to_from_dataclass(self):
        """Verify build() calls sicd_parser.from_string and from_dataclass."""
        mock_sicd = MagicMock()
        mock_sensor_model = MagicMock(spec=SICDSensorModel)

        with (
            patch(
                "aws.osml.metadata.sicd_sensor_model_builder.sicd_parser.from_string",
                return_value=mock_sicd,
            ) as mock_from_string,
            patch(
                "aws.osml.metadata.sicd_sensor_model_builder.SICDSensorModelBuilder.from_dataclass",
                return_value=mock_sensor_model,
            ) as mock_from_dataclass,
        ):
            builder = SICDSensorModelBuilder("<SICD>valid</SICD>")
            result = builder.build()

            mock_from_string.assert_called_once_with("<SICD>valid</SICD>")
            mock_from_dataclass.assert_called_once_with(mock_sicd)
            assert result is mock_sensor_model
