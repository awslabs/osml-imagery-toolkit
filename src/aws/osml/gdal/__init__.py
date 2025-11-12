#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

# Telling flake8 to not flag errors in this file. It is normal that these classes are imported but not used in an
# __init__.py file.
# flake8: noqa
"""
The gdal package contains utilities that assist with loading imagery and metadata using the OSGeo GDAL library.

Loading Imagery and Sensor Models with OSML
*******************************************

OSML provides utilities to load a dataset and automatically construct an appropriate sensor model from metadata
available in the image. Metadata handled by GDAL (e.g. GeoTIFF tags or NITF segment metadata and TREs) is available
through the dataset accessors.

.. code-block:: python
    :caption: Example of loading a dataset and sensor model using OSML

    from aws.osml.gdal import load_gdal_dataset

    # Load the image and create a sensor model
    dataset, sensor_model = load_gdal_dataset("./imagery/sample.nitf")
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    print(f"Loaded image with dimensions: ({height}, {width}) (rows, cols)")
    print(f"Using Sensor Model Implementation: {type(sensor_model).__name__}")
    print(dataset.GetMetadata())


Access to NITF Data Extension Segments
**************************************

SICD and SIDD imagery contains additional metadata in a XML Data Extension Segment that is not currently processed
by GDAL. This information can be accessed with the help of the NITFDESAccessor.

.. code-block:: python
    :caption: Example of loading a dataset and sensor model using OSML

    import base64
    import xml.dom.minidom
    from aws.osml.gdal import load_gdal_dataset, NITFDESAccessor

    dataset, sensor_model = load_gdal_dataset("./sample-sicd.nitf")

    des_accessor = NITFDESAccessor(dataset.GetMetadata("xml:DES"))
    xml_data_content_segments = des_accessor.get_segments_by_name("XML_DATA_CONTENT")
    if xml_data_content_segments is not None:
        for xml_data_segment in xml_data_content_segments:
            xml_str = des_accessor.extract_desdata_xml(xml_data_segment)
            if "SICD" in xml_str:
                temp = xml.dom.minidom.parseString(xml_str)
                new_xml = temp.toprettyxml()
                print(new_xml)
                break

-------------------------

APIs
****
"""

from .gdal_config import GDALConfigEnv, set_gdal_default_configuration
from .gdal_dem_tile_factory import GDALDigitalElevationModelTileFactory
from .gdal_shapefile_query import GDALShapefileQuery
from .gdal_utils import get_image_extension, get_type_and_scales, load_gdal_dataset
from .nitf_des_accessor import NITFDESAccessor
from .sensor_model_factory import ChippedImageInfoFacade, SensorModelFactory, SensorModelTypes
from .typing import GDALCompressionOptions, GDALImageFormats, RangeAdjustmentType

__all__ = [
    "set_gdal_default_configuration",
    "load_gdal_dataset",
    "get_image_extension",
    "get_type_and_scales",
    "GDALCompressionOptions",
    "GDALConfigEnv",
    "GDALDigitalElevationModelTileFactory",
    "GDALImageFormats",
    "GDALShapefileQuery",
    "RangeAdjustmentType",
    "NITFDESAccessor",
    "ChippedImageInfoFacade",
    "SensorModelFactory",
    "SensorModelTypes",
]
