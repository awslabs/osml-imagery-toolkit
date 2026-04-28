#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from typing import Dict, List, Optional

from aws.osml.metadata.dataset_utils import derive_geotiff_georeference


def derive_geo_transform(metadata: Dict) -> Optional[List[float]]:
    """
    Derive a 6-coefficient affine geo transform from image metadata.
    Tries GeoTIFF tags first, then DTED header fields.

    :param metadata: image metadata dictionary from osml-imagery-io
    :return: 6-coefficient geo transform [x_origin, x_res, x_rot, y_origin, y_rot, y_res], or None
    """
    geo_transform, _ = derive_geotiff_georeference(metadata)
    if geo_transform is not None:
        return geo_transform
    return _derive_dted_geo_transform(metadata)


def _derive_dted_geo_transform(metadata: Dict) -> Optional[List[float]]:
    """
    Derive geo transform from DTED header fields as reported by osml-imagery-io.

    DTED metadata uses the following keys:
    - dted:origin_longitude: float, SW corner longitude in degrees
    - dted:origin_latitude: float, SW corner latitude in degrees
    - dted:longitude_interval: int, post spacing in tenths of arcseconds
    - dted:latitude_interval: int, post spacing in tenths of arcseconds
    - dted:num_latitude_points: int, number of latitude posts per profile

    The geo transform maps pixel (col, row) to geographic (lon, lat) with
    origin at the NW corner, matching GeoTIFF pixel-is-point semantics.

    :param metadata: image metadata dictionary from osml-imagery-io
    :return: 6-coefficient geo transform, or None if DTED keys are missing
    """
    origin_lon = metadata.get("dted:origin_longitude")
    origin_lat = metadata.get("dted:origin_latitude")
    lon_interval = metadata.get("dted:longitude_interval")
    lat_interval = metadata.get("dted:latitude_interval")
    num_lat_points = metadata.get("dted:num_latitude_points")

    if None in (origin_lon, origin_lat, lon_interval, lat_interval, num_lat_points):
        return None

    # Intervals are in tenths of arcseconds; convert to degrees
    x_res = lon_interval / 10.0 / 3600.0
    y_res = lat_interval / 10.0 / 3600.0

    # Geo transform origin is NW corner; DTED origin is SW corner.
    # NW lat = SW lat + (num_lat_points - 1) * y_res
    y_origin = origin_lat + y_res * (num_lat_points - 1)

    return [origin_lon, x_res, 0.0, y_origin, 0.0, -y_res]
