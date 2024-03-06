#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""This file was generated by xsdata, v23.8, on 2023-10-05 09:59:45

Generator: DataclassGenerator
See: https://xsdata.readthedocs.io/
"""
from enum import Enum

__NAMESPACE__ = "urn:us:gov:ic:ism-cvenum"


class CVEnumISMNonUSControlsValues(Enum):
    """(U) NonUS Control markings supported by ISM
    PERMISSIBLE VALUES
    The permissible values for this simple type are defined in the Controlled Value Enumeration:
    CVEnumISMNonUSControls.xml

    :cvar ATOMAL: NATO Atomal mark
    :cvar BOHEMIA: NATO Bohemia mark
    :cvar BALK: NATO Balk mark
    """

    ATOMAL = "ATOMAL"
    BOHEMIA = "BOHEMIA"
    BALK = "BALK"
