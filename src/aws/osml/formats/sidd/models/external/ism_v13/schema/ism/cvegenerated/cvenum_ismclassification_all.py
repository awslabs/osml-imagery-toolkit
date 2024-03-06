#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""This file was generated by xsdata, v23.8, on 2023-10-05 09:59:45

Generator: DataclassGenerator
See: https://xsdata.readthedocs.io/
"""
from enum import Enum

__NAMESPACE__ = "urn:us:gov:ic:cvenum:ism:classification:all"


class CVEnumISMClassificationAll(Enum):
    """(U) All currently valid classification marks
    PERMISSIBLE VALUES
    The permissible values for this simple type are defined in the Controlled Value Enumeration:
    CVEnumISMClassificationAll.xml

    :cvar R: RESTRICTED
    :cvar C: CONFIDENTIAL
    :cvar S: SECRET
    :cvar TS: TOP SECRET
    :cvar U: UNCLASSIFIED
    """

    R = "R"
    C = "C"
    S = "S"
    TS = "TS"
    U = "U"
