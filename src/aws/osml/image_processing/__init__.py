#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

# Telling flake8 to not flag errors in this file. It is normal that these classes are imported but not used in an
# __init__.py file.
# flake8: noqa
"""
The image_processing package contains various utilities for manipulating overhead imagery.

Image Tiling: Tiling With Updated Image Metadata
************************************************

Many applications break large remote sensing images into smaller chips or tiles for distributed processing or
dissemination. GDAL's Translate function provides basic capabilities, but it does not correctly update geospatial
metadata to reflect the new image extent. These utilities provide those functions so tile consumers can correctly
interpret the pixel information they have been provided.

.. code-block:: python
    :caption: Example showing creation of a NITF tile from the upper left corner of an image

    # Load the image and create a sensor model
    ds, sensor_model = load_gdal_dataset("./imagery/sample.nitf")
    tile_factory = GDALTileFactory(ds,
                                   sensor_model,
                                   GDALImageFormats.NITF,
                                   GDALCompressionOptions.NONE
                                   )

    # Bounds are [left_x, top_y, width, height]
    nitf_encoded_tile_bytes = tile_factory.create_encoded_tile([0, 0, 1024, 1024])


Image Tiling: Tiles for Display
*******************************

Some images, for example 11-bit panchromatic images or SAR imagery with floating point complex data, can not be
displayed directly without remapping the pixels into an 8-bit per pixel grayscale or RGB color model. The TileFactory
supports creation of tiles suitable for human review by setting both the output_type and range_adjustment options.
Note that the output_size parameter can be used to generate lower resolution tiles. This operation will make use of
GDAL generated overviews if they are available to the dataset.

.. code-block:: python
    :caption: Example showing creation of a PNG tile scaled down from the full resolution image

    viz_tile_factory = GDALTileFactory(ds,
                                       sensor_model,
                                       GDALImageFormats.PNG,
                                       GDALCompressionOptions.NONE,
                                       output_type=gdalconst.GDT_Byte,
                                       range_adjustment=RangeAdjustmentType.DRA)

    viz_tile = viz_tile_factory.create_encoded_tile([0, 0, 1024, 1024], output_size=(512, 512))

Image Tiling: Map Tiles / Orthophotos
*************************************

The TileFactory supports creation of tiles suitable for use by geographic information systems (GIS) or map-based
visualization tools. Given a north-east aligned bounding box in geographic coordinates the tile factory can use
the sensor models to orthorectify imagery to remove the perspective and terrain effects.

.. code-block:: python
    :caption: Example showing creation of a map tile from the WebMercatorQuad tile set

    # Look up the tile boundary for a tile in a well known tile set
    tile_set = MapTileSetFactory.get_for_id("WebMercatorQuad")
    tile_id = MapTileId(tile_matrix=16, tile_row=37025, tile_col=54816)
    tile = tile_set.get_tile(tile_id)

    # Create an orthophoto for this tile
    image_bytes = viz_tile_factory.create_orthophoto_tile(geo_bbox=tile.bounds, tile_size=tile.size)

.. figure:: ../images/MapTileExample-BeforeAfter.png
    :width: 600
    :alt: Original image with perspective effects and same area after orthorectification

    Example showing original image with perspective effects and same area after orthorectification.

.. figure:: ../images/MapTileExample-MapOverlay.png
    :width: 400
    :alt: Orthophoto tile overlaid on Google Maps

    Example showing map tile overlaid on Google Maps

Complex SAR Data Display
************************

There are a variety of different techniques to convert complex SAR data to a simple image suitable for human display.
The toolkit contains two helper functions that can convert complex image data into an 8-bit grayscle representation
The equations implemented are described in Sections 3.1 and 3.2 of SAR Image Scaling, Dynamic Range, Radiometric
Calibration, and Display (SAND2019-2371).

.. code-block:: python
    :caption: Example converting complex SAR data into a 8-bit per pixel image for display

    import numpy as np
    from aws.osml.image_processing import histogram_stretch, quarter_power_image

    sicd_dataset, sensor_model = load_gdal_dataset("./sample-sicd.nitf")
    complex_pixels = sicd_dataset.ReadAsArray()

    histo_stretch_pixels = histogram_stretch(complex_pixels)
    quarter_power_pixels = quarter_power_image(complex_pixels)


.. figure:: ../images/SAR-HistogramStretchImage.png
   :width: 400
   :alt: Histogram Stretch Applied to Sample SICD Image

   Example of applying histogram_stretch to a sample SICD image.


.. figure:: ../images/SAR-QuarterPowerImage.png
   :width: 400
   :alt: Quarter Power Image Applied to Sample SICD Image

   Example of applying quarter_power_image to a sample SICD image.


-------------------------

APIs
****
"""

from .gdal_tile_factory import GDALTileFactory
from .map_tileset import MapTile, MapTileId, MapTileSet
from .map_tileset_factory import MapTileSetFactory, WellKnownMapTileSet
from .sar_complex_imageop import histogram_stretch, linear_mapping_complex, quarter_power_image

__all__ = [
    "GDALTileFactory",
    "MapTile",
    "MapTileId",
    "MapTileSet",
    "MapTileSetFactory",
    "WellKnownMapTileSet",
    "histogram_stretch",
    "quarter_power_image",
    "linear_mapping_complex",
]
