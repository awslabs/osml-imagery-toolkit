#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from typing import Callable, List, TypeVar

logger = logging.getLogger(__name__)

# This is a type placeholder needed by the get_tre_field_value() type hints
T = TypeVar("T")


def get_tre_field_value(tre_dict: dict, field_name: str, type_conversion: Callable[[str], T]) -> T:
    """
    Extract a named field from a TRE dict and convert it to the requested type.

    This is the dict-based equivalent of the XML-based get_tre_field_value in
    xmltre_utils.py. Instead of searching for a named "field" element in XML,
    it looks up field_name as a key in the provided dict and converts the
    string value using the provided callable.

    :param tre_dict: dict mapping field names to string values
    :param field_name: the field to extract
    :param type_conversion: callable to convert the string value (e.g., int, float, str)
    :return: the converted value
    :raises ValueError: if the field is missing or conversion fails
    """
    if field_name not in tre_dict:
        raise ValueError(f"Unable to find TRE field named '{field_name}' in dict")

    str_value = tre_dict[field_name]
    try:
        return type_conversion(str_value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert field '{field_name}' value '{str_value}': {e}") from e


def parse_tre_coefficient_list(tre_dict: dict, field_name: str, count: int) -> List[float]:
    """
    Extract a list of coefficients from a TRE dict.

    osml-imagery-io represents repeated coefficient groups as a list of
    strings under a single key (e.g., ``"LINE_NUM_COEFF": ["+1.0E+0", ...]``).

    :param tre_dict: dict mapping field names to values
    :param field_name: the key for the coefficient list
    :param count: the expected number of coefficients
    :return: list of float coefficients in order
    :raises ValueError: if the field is missing, not a list, or cannot be parsed
    """
    value = tre_dict.get(field_name)
    if value is None:
        raise ValueError(f"Unable to find TRE field named '{field_name}' in dict")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list for '{field_name}', got {type(value).__name__}")
    if len(value) < count:
        raise ValueError(f"Expected {count} coefficients for '{field_name}', got {len(value)}")
    try:
        return [float(v) for v in value[:count]]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse coefficients for '{field_name}': {e}") from e
