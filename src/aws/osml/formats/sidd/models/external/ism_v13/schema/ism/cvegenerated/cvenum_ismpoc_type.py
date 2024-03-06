#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""This file was generated by xsdata, v23.8, on 2023-10-05 09:59:45

Generator: DataclassGenerator
See: https://xsdata.readthedocs.io/
"""
from enum import Enum

__NAMESPACE__ = "urn:us:gov:ic:cvenum:ism:pocType"


class CVEnumISMPocTypeValues(Enum):
    """(U) All currently authorized types for ISM-related points-of-contact.

    PERMISSIBLE VALUES
    The permissible values for this simple type are defined in the Controlled Value Enumeration:
    CVEnumISMPocType.xml

    :cvar ICD_710: Point-of-contact for an ICD-710 notice.
    :cvar DO_D_DIST_B: DoD Distribution statement B from DoD Directive 5230.24
    :cvar DO_D_DIST_C: DoD Distribution statement C from DoD Directive 5230.24
    :cvar DO_D_DIST_D: DoD Distribution statement D from DoD Directive 5230.24
    :cvar DO_D_DIST_E: DoD Distribution statement E from DoD Directive 5230.24
    :cvar DO_D_DIST_F: DoD Distribution statement F from DoD Directive 5230.24
    :cvar DO_D_DIST_X: DoD Distribution statement X from DoD Directive 5230.24
    """

    ICD_710 = "ICD-710"
    DO_D_DIST_B = "DoD-Dist-B"
    DO_D_DIST_C = "DoD-Dist-C"
    DO_D_DIST_D = "DoD-Dist-D"
    DO_D_DIST_E = "DoD-Dist-E"
    DO_D_DIST_F = "DoD-Dist-F"
    DO_D_DIST_X = "DoD-Dist-X"
