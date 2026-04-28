#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from typing import Any, Optional, Tuple

import numpy as np

from aws.osml.io import IO, AssetType
from aws.osml.photogrammetry import (
    AffineSensorModel,
    DigitalElevationModelTileFactory,
    ElevationRegionSummary,
    ImageCoordinate,
    SensorModel,
    geodetic_to_geocentric,
)

from ._geo_transform import derive_geo_transform
from ._raster_utils import read_single_band

logger = logging.getLogger(__name__)


class StoredDEMTileFactory(DigitalElevationModelTileFactory):
    """
    Tile factory that loads DEM rasters (DTED, GeoTIFF) using osml-imagery-io.
    """

    def __init__(self, tile_directory: str) -> None:
        """
        :param tile_directory: root directory containing DEM tile files
        """
        super().__init__()
        self.tile_directory = tile_directory

    def get_tile(self, tile_path: str) -> Tuple[Optional[Any], Optional[SensorModel], Optional[ElevationRegionSummary]]:
        """
        Load a DEM tile and return its elevation data, sensor model, and region summary.

        :param tile_path: relative path to the tile within the tile directory
        :return: (elevation_array, sensor_model, summary) or (None, None, None) if unavailable
        """
        tile_location = f"{self.tile_directory}/{tile_path}"
        try:
            with IO.open(tile_location, "r") as reader:
                image = _get_first_image_asset(reader)
                metadata = dict(image.metadata) if image.metadata else {}

                geo_transform = derive_geo_transform(metadata)
                if geo_transform is None:
                    logger.warning("No geo transform for DEM tile: %s", tile_location)
                    return None, None, None

                sensor_model = AffineSensorModel(geo_transform)
                band_array = read_single_band(image)
                height, width = band_array.shape

                ul_ecf = geodetic_to_geocentric(sensor_model.image_to_world(ImageCoordinate([0, 0]))).coordinate
                lr_ecf = geodetic_to_geocentric(sensor_model.image_to_world(ImageCoordinate([width, height]))).coordinate
                post_spacing = np.linalg.norm(ul_ecf - lr_ecf) / np.sqrt(width * width + height * height)

                no_data = _get_no_data_value(image, metadata)
                valid = band_array if no_data is None else band_array[band_array != no_data]

                summary = ElevationRegionSummary(
                    min_elevation=float(np.min(valid)) if valid.size > 0 else 0.0,
                    max_elevation=float(np.max(valid)) if valid.size > 0 else 0.0,
                    no_data_value=int(no_data) if no_data is not None else 0,
                    post_spacing=post_spacing,
                )
                return band_array, sensor_model, summary

        except OSError as e:
            if "No such file or directory" in str(e):
                logger.debug("No DEM tile available for %s", tile_path)
            else:
                logger.warning("Failed to load DEM tile: %s", tile_path, exc_info=True)
            return None, None, None
        except Exception:
            logger.warning("Failed to load DEM tile: %s", tile_path, exc_info=True)
            return None, None, None


def _get_first_image_asset(reader):
    """Return the first image asset from the reader."""
    keys = reader.get_asset_keys(asset_type=AssetType.Image)
    if not keys:
        raise ValueError(f"No image asset found. Available: {reader.get_asset_keys()}")
    return reader.get_asset(keys[0])


def _get_no_data_value(image_asset, metadata: dict) -> Optional[float]:
    """Determine the no-data sentinel value for a DEM image asset.

    Checks two sources in priority order:
    1. GeoTIFF tag 42113 (GDAL_NODATA) — a private tag registered by GDAL that
       stores the no-data value as ASCII. Present in most DEM GeoTIFFs produced
       by GDAL-based tools (gdalwarp, USGS downloads, etc.).
    2. The image asset's pad_pixel_value — a format-native fill value provided
       by osml-imagery-io. For DTED this is -32767 (the standard void sentinel).
       For GeoTIFF this defaults to 0 which is a valid elevation, so it is only
       used as a fallback when tag 42113 is absent and the value is non-zero.
    """
    tag_value = metadata.get("42113")
    if tag_value is not None:
        try:
            return float(tag_value)
        except (ValueError, TypeError):
            pass

    pad = image_asset.pad_pixel_value
    if pad is not None and pad != 0.0:
        return float(pad)

    return None
