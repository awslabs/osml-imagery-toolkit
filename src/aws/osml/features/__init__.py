#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

# Telling flake8 to not flag errors in this file. It is normal that these classes are imported but not used in an
# __init__.py file.
# flake8: noqa
"""
The features package contains classes that assist with working with geospatial features derived from imagery.

-------------------------

APIs
****
"""

from .feature_index import Feature2DSpatialIndex, STRFeature2DSpatialIndex
from .geolocation import Geolocator
from .imaged_feature_property_accessor import ImagedFeaturePropertyAccessor
from .projection import Projector

__all__ = [
    "Geolocator",
    "ImagedFeaturePropertyAccessor",
    "Feature2DSpatialIndex",
    "Projector",
    "STRFeature2DSpatialIndex",
]
