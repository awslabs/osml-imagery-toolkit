#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

# Telling flake8 to not flag errors in this file. It is normal that these classes are imported but not used in an
# __init__.py file.
# flake8: noqa
"""
The image_processing package contains various utilities for manipulating overhead imagery.

Image Chipping: Chips With Updated Image Metadata
**************************************************

Many applications break large remote sensing images into smaller chips for distributed processing or
dissemination. These utilities extract arbitrary sub-regions and derive updated geospatial metadata
(ICHIPB, IGEOLO, GeoTransform, SICD/SIDD XML) so chip consumers can correctly interpret the pixel
information they have been provided.

.. code-block:: python
    :caption: Example showing creation of a NITF chip from the upper left corner of an image

    from aws.osml.io import IO
    from aws.osml.image_processing import ChipFactory, TiledImagePyramid, PixelWindow

    with IO.open("./imagery/sample.nitf", "r") as reader:
        pyramid = TiledImagePyramid.from_dataset(reader)
        chip_factory = ChipFactory(source=pyramid, output_format="nitf")
        nitf_encoded_chip_bytes = chip_factory.create_chip(PixelWindow(0, 0, 1024, 1024))


Image Chipping: Chips for Display
**********************************

Some images, for example 11-bit panchromatic images or SAR imagery with floating point complex data, can not be
displayed directly without remapping the pixels into an 8-bit per pixel grayscale or RGB color model. The ChipFactory
supports creation of chips suitable for human review by attaching a processing chain.
Note that the output_size parameter can be used to generate lower resolution chips. This operation will make use of
pyramid overviews if they are available.

.. code-block:: python
    :caption: Example showing creation of a PNG chip scaled down from the full resolution image

    from aws.osml.image_processing import ChipFactory, DisplayChainFactory, ImageSize

    chain = DisplayChainFactory.build(source)
    chip_factory = ChipFactory(
        source=pyramid,
        output_format="png",
        processing_chain=chain,
    )
    viz_chip = chip_factory.create_chip(PixelWindow(0, 0, 1024, 1024), output_size=ImageSize(512, 512))

Image Chipping: Map Tiles / Orthophotos
****************************************

The ChipFactory supports creation of chips suitable for use by geographic information systems (GIS) or map-based
visualization tools. Given a north-east aligned bounding box in geographic coordinates the chip factory can use
the sensor models to orthorectify imagery to remove the perspective and terrain effects.

.. code-block:: python
    :caption: Example showing creation of a map tile from the WebMercatorQuad tile set

    # Look up the tile boundary for a tile in a well known tile set
    tile_set = MapTileSetFactory.get_for_id("WebMercatorQuad")
    tile_id = MapTileId(tile_matrix=16, tile_row=37025, tile_col=54816)
    tile = tile_set.get_tile(tile_id)

    # Orthorectify and warp pixels into the tile's geographic extent
    grid_builder = OrthoGridBuilder(sensor_model=sensor_model, ...)
    warped = WarpedImageProvider(source, grid_builder)

Complex SAR Data Display
************************

Complex SAR imagery (I/Q data) must be remapped to scalar magnitude before display. The
:func:`is_complex` utility detects complex assets, and :class:`ComplexRemapFactory` wraps them in a
:class:`MappedImageProvider` that performs a domain transform (complex I/Q → scalar magnitude).
The resulting scalar data can then be processed through the standard DRA-based display pipeline.

.. code-block:: python
    :caption: Example converting complex SAR data for display via ComplexRemapFactory

    from aws.osml.io import IO
    from aws.osml.image_processing import (
        is_complex,
        ComplexRemapFactory,
        DisplayChainFactory,
        TiledImagePyramid,
        compute_image_statistics,
    )

    with IO.open("./sample-sicd.nitf", "r") as reader:
        asset = reader.get_asset("image:0")
        if is_complex(asset):
            remapped = ComplexRemapFactory.build(asset, band_interpretation=["real", "imaginary"])
            pyramid = TiledImagePyramid.from_asset(remapped)
            stats = compute_image_statistics(remapped)
            chain = DisplayChainFactory.build(remapped, stats=stats)



-------------------------

APIs
****
"""

from .block_utils import read_block_or_pad, read_window, stitch_source_blocks
from .cached_provider import CachedImageProvider
from .color_space import color_space_transform
from .convolution import sips_convolve, sips_correlate
from .display_chain_factory import DisplayChainFactory
from .downsampled_provider import DownsampledImageProvider
from .dynamic_range_adjustment import DRAParameters, dynamic_range_adjust
from .image_to_image_grid_builder import ImageToImageGridBuilder
from .lut import apply_lut
from .map_tileset import MapTile, MapTileId, MapTileSet
from .map_tileset_factory import MapTileSetFactory, WellKnownMapTileSet
from .mapped_provider import MappedImageProvider
from .ortho_grid_builder import OrthoGridBuilder
from .processing_chain import ProcessingChain, band_select, compose
from .projected_image_tileset import ProjectedImageTileSet
from .pyramid import TiledImagePyramid, build_pyramid_levels, iter_blocks
from .pyramid_builder import ProgressCallback, PyramidBuilder
from .resample import ResampleFunc, area_resample, bilinear_resample, lanczos_resample, nearest_neighbor_resample
from .retiled_provider import RetiledImageProvider
from .complex_remap import (
    ComplexRemapFactory,
    ROLE_AMPLITUDE_INDEX,
    ROLE_IMAGINARY,
    ROLE_MAGNITUDE,
    ROLE_PHASE,
    ROLE_REAL,
    complex_to_power,
    decode_to_iq,
    is_complex,
    load_complex_remap,
    magnitude_remap,
    power_to_decibels,
    quarter_power_remap,
)
from .sips_resample import (
    SIPS_ANTIALIAS_KERNEL_7x7,
    build_lagrange_kernel_2d,
    compute_compromise_coefficients,
    compute_lagrange_coefficients,
    sips_rrds_resample,
)
from .tile_cache import TileCache
from .statistics import (
    BandStatistics,
    ImageStatistics,
    SamplingStrategy,
    compute_image_statistics,
    compute_statistics,
    merge_statistics,
    statistics_from_gdal_metadata,
    statistics_to_gdal_metadata,
)
from .chip_factory import ChipFactory, ImageSize, PixelWindow
from .chip_metadata_builder import ChipMetadataBuilder, GeoTiffChipMetadataBuilder, NitfChipMetadataBuilder
from .warp_grid import GridBuilder, OcclusionMode, WarpGrid, WarpGridOptions
from .warped_provider import WarpedImageProvider

__all__ = [
    "BandStatistics",
    "CachedImageProvider",
    "ChipFactory",
    "ChipMetadataBuilder",
    "ComplexRemapFactory",
    "DRAParameters",
    "DisplayChainFactory",
    "DownsampledImageProvider",
    "GeoTiffChipMetadataBuilder",
    "GridBuilder",
    "ImageSize",
    "ImageStatistics",
    "ImageToImageGridBuilder",
    "MapTile",
    "MapTileId",
    "MapTileSet",
    "MapTileSetFactory",
    "MappedImageProvider",
    "NitfChipMetadataBuilder",
    "OcclusionMode",
    "OrthoGridBuilder",
    "PixelWindow",
    "ProcessingChain",
    "ProgressCallback",
    "ProjectedImageTileSet",
    "PyramidBuilder",
    "ROLE_AMPLITUDE_INDEX",
    "ROLE_IMAGINARY",
    "ROLE_MAGNITUDE",
    "ROLE_PHASE",
    "ROLE_REAL",
    "ResampleFunc",
    "RetiledImageProvider",
    "SIPS_ANTIALIAS_KERNEL_7x7",
    "SamplingStrategy",
    "TileCache",
    "TiledImagePyramid",
    "WarpGrid",
    "WarpGridOptions",
    "WarpedImageProvider",
    "WellKnownMapTileSet",
    "apply_lut",
    "area_resample",
    "band_select",
    "bilinear_resample",
    "build_lagrange_kernel_2d",
    "build_pyramid_levels",
    "color_space_transform",
    "complex_to_power",
    "compose",
    "compute_compromise_coefficients",
    "compute_image_statistics",
    "compute_lagrange_coefficients",
    "compute_statistics",
    "decode_to_iq",
    "dynamic_range_adjust",
    "is_complex",
    "iter_blocks",
    "lanczos_resample",
    "load_complex_remap",
    "magnitude_remap",
    "merge_statistics",
    "nearest_neighbor_resample",
    "power_to_decibels",
    "quarter_power_remap",
    "read_block_or_pad",
    "read_window",
    "sips_convolve",
    "sips_correlate",
    "sips_rrds_resample",
    "statistics_from_gdal_metadata",
    "statistics_to_gdal_metadata",
    "stitch_source_blocks",
]
