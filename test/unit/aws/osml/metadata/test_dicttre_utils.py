#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import pytest

from aws.osml.metadata.dicttre_utils import get_tre_field_value, parse_tre_coefficient_list


class TestGetTreFieldValue:
    """Tests for get_tre_field_value with various type conversions and error cases."""

    def test_int_conversion(self):
        tre_dict = {"LINE_OFF": "2048", "SAMP_OFF": "1024"}
        assert get_tre_field_value(tre_dict, "LINE_OFF", int) == 2048

    def test_float_conversion(self):
        tre_dict = {"ERR_BIAS": "3.09", "ERR_RAND": "1.44"}
        assert get_tre_field_value(tre_dict, "ERR_BIAS", float) == pytest.approx(3.09)

    def test_str_conversion(self):
        tre_dict = {"GRNDD": "G", "SUCCESS": "1"}
        assert get_tre_field_value(tre_dict, "GRNDD", str) == "G"

    def test_missing_field_raises_value_error(self):
        tre_dict = {"LINE_OFF": "2048"}
        with pytest.raises(ValueError, match="Unable to find TRE field named 'MISSING'"):
            get_tre_field_value(tre_dict, "MISSING", int)

    def test_unconvertible_value_raises_value_error(self):
        tre_dict = {"LINE_OFF": "not_a_number"}
        with pytest.raises(ValueError, match="Failed to convert field 'LINE_OFF'"):
            get_tre_field_value(tre_dict, "LINE_OFF", int)

    def test_empty_dict_raises_value_error(self):
        with pytest.raises(ValueError, match="Unable to find TRE field named"):
            get_tre_field_value({}, "ANY_FIELD", str)

    def test_whitespace_value_float_conversion(self):
        tre_dict = {"VALUE": "  3.14  "}
        assert get_tre_field_value(tre_dict, "VALUE", float) == pytest.approx(3.14)

    def test_negative_float_conversion(self):
        tre_dict = {"COEFF": "-0.00345"}
        assert get_tre_field_value(tre_dict, "COEFF", float) == pytest.approx(-0.00345)


class TestParseTreCoefficientList:
    """Tests for parse_tre_coefficient_list with list-based coefficient dicts."""

    def test_valid_coefficients(self):
        tre_dict = {"LINE_NUM_COEFF": ["0.006", "-0.003", "1.0"]}
        result = parse_tre_coefficient_list(tre_dict, "LINE_NUM_COEFF", 3)
        assert result == pytest.approx([0.006, -0.003, 1.0])

    def test_single_coefficient(self):
        tre_dict = {"COEFF": ["42.5"]}
        result = parse_tre_coefficient_list(tre_dict, "COEFF", 1)
        assert result == pytest.approx([42.5])

    def test_missing_field_raises_value_error(self):
        tre_dict = {"OTHER": ["0.006"]}
        with pytest.raises(ValueError, match="Unable to find TRE field named 'LINE_NUM_COEFF'"):
            parse_tre_coefficient_list(tre_dict, "LINE_NUM_COEFF", 3)

    def test_not_a_list_raises_value_error(self):
        tre_dict = {"COEFF": "0.5"}
        with pytest.raises(ValueError, match="Expected list for 'COEFF'"):
            parse_tre_coefficient_list(tre_dict, "COEFF", 1)

    def test_too_few_coefficients_raises_value_error(self):
        tre_dict = {"COEFF": ["0.5", "0.6"]}
        with pytest.raises(ValueError, match="Expected 3 coefficients for 'COEFF', got 2"):
            parse_tre_coefficient_list(tre_dict, "COEFF", 3)

    def test_non_numeric_coefficient_raises_value_error(self):
        tre_dict = {"COEFF": ["0.5", "abc"]}
        with pytest.raises(ValueError, match="Failed to parse coefficients for 'COEFF'"):
            parse_tre_coefficient_list(tre_dict, "COEFF", 2)

    def test_twenty_coefficients(self):
        """Verify extraction of 20 coefficients, matching RPC polynomial size."""
        tre_dict = {"POLY_COEFF": [str(float(i) * 0.001) for i in range(1, 21)]}
        result = parse_tre_coefficient_list(tre_dict, "POLY_COEFF", 20)
        expected = [float(i) * 0.001 for i in range(1, 21)]
        assert result == pytest.approx(expected)

    def test_extra_coefficients_ignored(self):
        tre_dict = {"COEFF": ["1.0", "2.0", "3.0", "4.0", "5.0"]}
        result = parse_tre_coefficient_list(tre_dict, "COEFF", 3)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_zero_count_returns_empty_list(self):
        tre_dict = {"COEFF": ["1.0"]}
        result = parse_tre_coefficient_list(tre_dict, "COEFF", 0)
        assert result == []

    def test_none_field_raises_value_error(self):
        tre_dict = {"COEFF": None}
        with pytest.raises(ValueError, match="Unable to find TRE field named 'COEFF'"):
            parse_tre_coefficient_list(tre_dict, "COEFF", 1)
