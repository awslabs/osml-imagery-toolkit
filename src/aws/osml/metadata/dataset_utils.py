#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from aws.osml.photogrammetry import SensorModel

from .gcp_sensor_model_builder import GroundControlPoint
from .sensor_model_factory import SensorModelFactory

logger = logging.getLogger(__name__)


def load_sensor_model(reader: Any, asset_key: Optional[str] = None) -> Optional[SensorModel]:
    """
    Convenience function that extracts metadata from an osml-imagery-io DatasetReader
    and constructs the best available SensorModel.

    This function handles all format-specific metadata extraction (NITF TREs, GeoTIFF
    tags, DES XML segments, IGEOLO corner coordinates) and passes normalized inputs
    to SensorModelFactory for model construction.

    :param reader: an osml-imagery-io DatasetReader (from IO.open)
    :param asset_key: specific image asset key to use (default: first "image:" asset)
    :return: the best available SensorModel, or None if no model can be built
    """
    try:
        # Find the image asset
        if asset_key is None:
            asset_keys = reader.get_asset_keys()
            image_keys = [k for k in asset_keys if k.startswith("image:")]
            if not image_keys:
                logger.warning("No image assets found in dataset")
                return None
            asset_key = image_keys[0]

        image_asset = reader.get_asset(asset_key)
        actual_image_width = image_asset.num_columns
        actual_image_height = image_asset.num_rows

        # The metadata dict contains TRE names (NITF) or numeric tag IDs (TIFF) as keys.
        # Pass it directly as tre_dicts to the factory.
        metadata_dict = dict(image_asset.metadata) if image_asset.metadata else {}
        tre_dicts: Optional[Dict[str, Union[dict, List[dict]]]] = metadata_dict or None

        # Derive geo_transform from GeoTIFF tags.
        # GDAL's GetGeoTransform() computes this from:
        #   - ModelPixelScale (tag 33550) + ModelTiepoint (tag 33922) for axis-aligned images
        #   - ModelTransformation (tag 34264) for rotated/skewed images
        geo_transform: Optional[List[float]] = None
        ground_control_points: Optional[List[GroundControlPoint]] = None

        if metadata_dict:
            geo_transform, ground_control_points = derive_geotiff_georeference(metadata_dict)

        # Derive GCPs from IGEOLO corner coordinates when no GeoTIFF GCPs exist
        if ground_control_points is None and metadata_dict:
            igeolo_gcps = _extract_igeolo_gcps(metadata_dict, actual_image_width, actual_image_height)
            if igeolo_gcps:
                ground_control_points = igeolo_gcps

        # Derive CRS WKT from GeoKey directory (tag 34735).
        # GDAL's GetProjection() parses the GeoKeys to produce an OGC WKT string that
        # the AffineSensorModel needs to transform between the image CRS and WGS84.
        proj_wkt: Optional[str] = None
        if metadata_dict:
            proj_wkt = _derive_proj_wkt(metadata_dict)

        # Extract DES XML strings (SICD/SIDD) from data extension segments.
        # Only segments with DESID "XML_DATA_CONTENT" contain SICD/SIDD XML;
        # other DES types (CSATTA, CSSHPA, etc.) are binary and should not be decoded.
        xml_des_ids = {"XML_DATA_CONTENT"}
        des_xml_strings: Optional[List[str]] = None
        all_keys = reader.get_asset_keys()
        des_keys = [k for k in all_keys if k.startswith("des:")]
        if des_keys:
            des_xml_strings = []
            for key in des_keys:
                try:
                    des_asset = reader.get_asset(key)
                    des_meta = dict(des_asset.metadata) if des_asset.metadata else {}
                    desid = des_meta.get("DESID", "").strip()
                    if desid not in xml_des_ids:
                        continue
                    xml_str = des_asset.raw_asset.read().decode("utf-8")
                    if xml_str:
                        des_xml_strings.append(xml_str)
                except Exception as e:
                    logger.warning("Failed to read DES asset %s: %s", key, e)

            if not des_xml_strings:
                des_xml_strings = None

        return SensorModelFactory(
            actual_image_width=actual_image_width,
            actual_image_height=actual_image_height,
            tre_dicts=tre_dicts,
            des_xml_strings=des_xml_strings,
            geo_transform=geo_transform,
            proj_wkt=proj_wkt,
            ground_control_points=ground_control_points,
        ).build()

    except Exception as e:
        logger.error("Failed to load sensor model from dataset reader: %s", e)
        return None


def _extract_igeolo_gcps(metadata_dict: Dict[str, Any], width: int, height: int) -> Optional[List[GroundControlPoint]]:
    """
    Parse IGEOLO corner coordinates into GroundControlPoints.

    IGEOLO provides 4 geographic corners (UL, UR, LR, LL) that map to the
    image corners. Supports ICORDS values:
    - "G": geographic (DMS)
    - "D": decimal degrees
    - "N": UTM northern hemisphere
    - "S": UTM southern hemisphere

    :param metadata_dict: image metadata as a flat dict
    :param width: image width in pixels
    :param height: image height in pixels
    :return: list of 4 GroundControlPoints, or None if IGEOLO is unavailable
    """
    igeolo = metadata_dict.get("IGEOLO")
    icords = metadata_dict.get("ICORDS")
    if not igeolo or not icords:
        return None

    # TODO: Support ICORDS="U" (MGRS). Requires an MGRS-to-lat/lon decoder —
    # either the `mgrs` package or a hand-rolled grid-square lookup.
    if icords not in ("G", "D", "N", "S"):
        return None

    try:
        from aws.osml.io.jbp.utils import IGEOLOAdapter

        parsed = IGEOLOAdapter.parse(igeolo, icords)
    except Exception as e:
        logger.debug("Failed to parse IGEOLO: %s", e)
        return None

    # Convert to (lat, lon) tuples in decimal degrees
    if icords in ("G", "D"):
        corners = parsed
    elif icords in ("N", "S"):
        corners = _utm_to_latlon(parsed, icords)
        if corners is None:
            return None

    if not _validate_igeolo_corners(corners):
        return None

    # IGEOLO corners are ordered: UL, UR, LR, LL → (lat, lon) tuples
    image_corners = [
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ]

    gcps = []
    for (lat, lon), (ix, iy) in zip(corners, image_corners):
        gcps.append(
            GroundControlPoint(
                image_x=ix,
                image_y=iy,
                world_longitude=lon,
                world_latitude=lat,
                world_elevation=0.0,
            )
        )
    return gcps


def _utm_to_latlon(utm_coords: list, icords: str) -> Optional[List[tuple]]:
    """
    Convert UTM coordinates to (lat, lon) decimal degree tuples.

    :param utm_coords: list of UTMCoordinate objects from IGEOLOAdapter.parse()
    :param icords: "N" for northern hemisphere, "S" for southern hemisphere
    :return: list of (lat, lon) tuples, or None on failure
    """
    try:
        import pyproj
    except ImportError:
        logger.debug("pyproj not available for UTM conversion")
        return None

    corners = []
    for utm in utm_coords:
        zone = utm.zone
        if icords == "N":
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        try:
            crs_code = f"EPSG:{epsg}"  # noqa: E231
            transformer = pyproj.Transformer.from_crs(crs_code, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(utm.easting, utm.northing)
            corners.append((lat, lon))
        except Exception as e:
            logger.debug("Failed to convert UTM zone %d: %s", zone, e)
            return None

    return corners


def _validate_igeolo_corners(corners: list) -> bool:
    """
    Validate that IGEOLO corners have coordinates within valid WGS84 range.

    :param corners: list of 4 (lat, lon) tuples from IGEOLOAdapter.parse()
    :return: True if all coordinates are in range
    """
    if len(corners) != 4:
        return False

    for lat, lon in corners:
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            logger.debug("IGEOLO corner out of range: lat=%f, lon=%f", lat, lon)
            return False

    return True


def derive_geotiff_georeference(
    metadata_dict: Dict[str, Any],
) -> tuple:
    """
    Derive geo_transform and/or GCPs from GeoTIFF tags.

    Replicates GDAL's GetGeoTransform() and GetGCPs() logic:

    1. If ModelPixelScale (33550) is present with a single ModelTiepoint (33922),
       compute the 6-coefficient affine geo_transform.
    2. If ModelTransformation (34264) is present (rotated/skewed image), extract
       the 6 affine coefficients from the 4x4 matrix.
    3. If multiple ModelTiepoints exist without ModelPixelScale, treat them as
       ground control points (GDAL's behavior for multi-tiepoint GeoTIFFs).

    :param metadata_dict: image metadata as a flat dict
    :return: tuple of (geo_transform, ground_control_points) — either or both may be None
    """
    geo_transform: Optional[List[float]] = None
    ground_control_points: Optional[List[GroundControlPoint]] = None

    scale = metadata_dict.get("33550")
    tiepoints = metadata_dict.get("33922")
    model_transform = metadata_dict.get("34264")

    if scale and tiepoints and len(scale) >= 2 and len(tiepoints) >= 6:
        if len(tiepoints) == 6:
            # Single tiepoint + scale → affine geo_transform
            geo_transform = [
                tiepoints[3] - tiepoints[0] * scale[0],
                scale[0],
                0.0,
                tiepoints[4] + tiepoints[1] * scale[1],
                0.0,
                -scale[1],
            ]
        else:
            # Multiple tiepoints with scale is unusual but we still derive the
            # affine from the first tiepoint (matching GDAL behavior)
            geo_transform = [
                tiepoints[3] - tiepoints[0] * scale[0],
                scale[0],
                0.0,
                tiepoints[4] + tiepoints[1] * scale[1],
                0.0,
                -scale[1],
            ]
    elif model_transform and len(model_transform) >= 16:
        # 4x4 ModelTransformation matrix → extract affine coefficients
        # Row-major: [a, b, 0, tx, d, e, 0, ty, ...]
        # Maps to GDAL geo_transform: [tx, a, b, ty, d, e]
        geo_transform = [
            model_transform[3],
            model_transform[0],
            model_transform[1],
            model_transform[7],
            model_transform[4],
            model_transform[5],
        ]
    elif tiepoints and len(tiepoints) > 6 and not scale:
        # Multiple tiepoints without scale → treat as GCPs (GDAL behavior).
        # Each tiepoint is 6 values: [pixel_x, pixel_y, pixel_z, geo_x, geo_y, geo_z]
        num_points = len(tiepoints) // 6
        if num_points >= 4:
            ground_control_points = []
            for i in range(num_points):
                offset = i * 6
                ground_control_points.append(
                    GroundControlPoint(
                        image_x=float(tiepoints[offset]),
                        image_y=float(tiepoints[offset + 1]),
                        world_longitude=float(tiepoints[offset + 3]),
                        world_latitude=float(tiepoints[offset + 4]),
                        world_elevation=float(tiepoints[offset + 5]),
                    )
                )

    return geo_transform, ground_control_points


def _derive_proj_wkt(metadata_dict: Dict[str, Any]) -> Optional[str]:
    """
    Derive CRS WKT from the GeoKey directory (tag 34735).

    Replicates GDAL's GetProjection() logic: parses the GeoKeyDirectory to find
    the EPSG code (GeographicTypeGeoKey 2048 or ProjectedCSTypeGeoKey 3072),
    then uses pyproj to produce WKT.

    :param metadata_dict: image metadata as a flat dict
    :return: CRS WKT string, or None if no GeoKeys are present
    """
    geokey_dir = metadata_dict.get("34735")
    if not geokey_dir or len(geokey_dir) < 4:
        return None

    # GeoKey directory structure:
    # Header: [version, revision, minor_revision, num_keys]
    # Each key entry: [key_id, tiff_tag_location, count, value_offset]
    # When tiff_tag_location=0, value_offset contains the value directly.
    num_keys = int(geokey_dir[3])
    epsg_code = None

    for i in range(num_keys):
        offset = 4 + i * 4
        if offset + 3 >= len(geokey_dir):
            break
        key_id = int(geokey_dir[offset])
        tiff_tag_location = int(geokey_dir[offset + 1])
        value_offset = int(geokey_dir[offset + 3])

        # ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
        if key_id in (2048, 3072) and tiff_tag_location == 0:
            epsg_code = value_offset
            break

    if epsg_code is None or epsg_code == 0 or epsg_code == 32767:
        return None

    try:
        import pyproj

        crs = pyproj.CRS.from_epsg(epsg_code)
        return crs.to_wkt()
    except Exception as e:
        logger.debug("Failed to derive CRS WKT from EPSG %d: %s", epsg_code, e)
        return None
