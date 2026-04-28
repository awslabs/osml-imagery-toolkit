#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Chip metadata builder protocol and implementations.

This module defines the :class:`ChipMetadataBuilder` protocol and
provides :class:`GeoTiffChipMetadataBuilder` and
:class:`NitfChipMetadataBuilder` for deriving format-appropriate
metadata for chipped output images.

Builders are constructed once per image (caching source metadata and
sensor model reference) and called per chip with only the varying
parameters. All ``build()`` calls are stateless — no mutable instance
state is modified — enabling thread-safe concurrent use from
:class:`~aws.osml.image_processing.chip_factory.ChipFactory`.
"""

from math import degrees
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .chip_factory import ImageSize, PixelWindow


class ChipMetadataBuilder(Protocol):
    """Protocol for chip metadata derivation.

    Implementations derive format-appropriate metadata for a chipped
    image given source metadata, a sensor model, and the chip bounds.
    """

    def __init__(
        self,
        reader: Any,
        sensor_model: Any = None,
    ) -> None: ...

    def build(
        self,
        src_window: PixelWindow,
        output_size: ImageSize,
    ) -> Any:
        """Derive chip metadata for the given chip bounds.

        Returns a ``BufferedMetadataProvider`` compatible with
        ``DatasetWriter``'s asset metadata API.

        Must be stateless per call — no mutable instance state modified.
        """
        ...


def _parse_geotransform(metadata: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Extract a 6-element GeoTransform from TIFF metadata dict.

    Looks for ModelTiepoint (33922) + ModelPixelScale (33550) and
    constructs the equivalent GeoTransform array. Returns None if
    the required tags are not present.

    GeoTransform semantics:
        [0] = x-origin (upper-left corner of upper-left pixel)
        [1] = pixel width (west-east resolution)
        [2] = row rotation (typically 0)
        [3] = y-origin (upper-left corner of upper-left pixel)
        [4] = column rotation (typically 0)
        [5] = pixel height (negative for north-up)
    """
    scale_raw = metadata.get("33550")
    tiepoint_raw = metadata.get("33922")

    if scale_raw is None or tiepoint_raw is None:
        return None

    if isinstance(scale_raw, str):
        scale = [float(v) for v in scale_raw.split(",")]
    elif isinstance(scale_raw, (list, tuple)):
        scale = [float(v) for v in scale_raw]
    else:
        return None

    if isinstance(tiepoint_raw, str):
        tiepoint = [float(v) for v in tiepoint_raw.split(",")]
    elif isinstance(tiepoint_raw, (list, tuple)):
        tiepoint = [float(v) for v in tiepoint_raw]
    else:
        return None

    if len(scale) < 2 or len(tiepoint) < 6:
        return None

    # Tiepoint: [i, j, k, x, y, z] — pixel (i,j) maps to geo (x,y)
    i, j = tiepoint[0], tiepoint[1]
    x, y = tiepoint[3], tiepoint[4]

    pixel_width = scale[0]
    pixel_height = -scale[1]

    x_origin = x - i * pixel_width
    y_origin = y - j * pixel_height

    return (x_origin, pixel_width, 0.0, y_origin, 0.0, pixel_height)


def _geotransform_for_chip(
    geo_transform: Tuple[float, float, float, float, float, float],
    src_window: PixelWindow,
    output_size: ImageSize,
) -> Tuple[float, float, float, float, float, float]:
    """Compute a new GeoTransform for a chip extracted from the source.

    Accounts for pixel offset (chip origin) and resolution change
    (when output_size differs from src_window dimensions).
    """
    x_origin, pixel_width, x_rot, y_origin, y_rot, pixel_height = geo_transform

    new_x_origin = x_origin + src_window.x * pixel_width + src_window.y * x_rot
    new_y_origin = y_origin + src_window.x * y_rot + src_window.y * pixel_height

    scale_x = src_window.width / output_size.width
    scale_y = src_window.height / output_size.height

    new_pixel_width = pixel_width * scale_x
    new_x_rot = x_rot * scale_x
    new_y_rot = y_rot * scale_y
    new_pixel_height = pixel_height * scale_y

    return (new_x_origin, new_pixel_width, new_x_rot, new_y_origin, new_y_rot, new_pixel_height)


def _geotransform_from_sensor_model(
    sensor_model: Any,
    src_window: PixelWindow,
    output_size: ImageSize,
) -> Tuple[float, float, float, float, float, float]:
    """Derive a linearized GeoTransform from a sensor model.

    Computes corner coordinates of the chip and fits a north-up
    affine approximation. This is accurate for small tiles but
    introduces linearization error for large extents with non-affine
    sensor models (RPC, RSM).
    """
    from aws.osml.photogrammetry import ImageCoordinate

    ul = sensor_model.image_to_world(ImageCoordinate([src_window.x, src_window.y]))
    ur = sensor_model.image_to_world(ImageCoordinate([src_window.x + src_window.width, src_window.y]))
    ll = sensor_model.image_to_world(ImageCoordinate([src_window.x, src_window.y + src_window.height]))

    ul_lon, ul_lat = degrees(ul.longitude), degrees(ul.latitude)
    ur_lon, ur_lat = degrees(ur.longitude), degrees(ur.latitude)
    ll_lon, ll_lat = degrees(ll.longitude), degrees(ll.latitude)

    pixel_width = (ur_lon - ul_lon) / output_size.width
    x_rot = (ll_lon - ul_lon) / output_size.height
    y_rot = (ur_lat - ul_lat) / output_size.width
    pixel_height = (ll_lat - ul_lat) / output_size.height

    return (ul_lon, pixel_width, x_rot, ul_lat, y_rot, pixel_height)


def _compute_corner_coords(
    sensor_model: Any,
    src_window: PixelWindow,
) -> List[Tuple[float, float]]:
    """Compute (lat, lon) in decimal degrees for the four tile corners.

    Corner order follows NITF IGEOLO convention:
    UL (0,0), UR (maxX, 0), LR (maxX, maxY), LL (0, maxY).
    """
    from aws.osml.photogrammetry import ImageCoordinate

    corners_px = [
        (src_window.x, src_window.y),
        (src_window.x + src_window.width, src_window.y),
        (src_window.x + src_window.width, src_window.y + src_window.height),
        (src_window.x, src_window.y + src_window.height),
    ]
    result: List[Tuple[float, float]] = []
    for px, py in corners_px:
        world = sensor_model.image_to_world(ImageCoordinate([px, py]))
        lat_deg = degrees(world.latitude)
        lon_deg = degrees(world.longitude)
        result.append((lat_deg, lon_deg))
    return result


def _build_ichipb_fields(
    src_window: PixelWindow,
    output_size: ImageSize,
    source_ichipb: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build ICHIPB TRE field dict for the chip.

    The ICHIPB TRE records the geometric relationship between the
    output product (OP — the chip) and the full image (FI — the
    original). When the source is itself a chip (has its own ICHIPB),
    coordinates are chained back to the original full image.

    Grid points map corners of the output product to the full image:
        11 = UL, 12 = UR, 21 = LL, 22 = LR
    """
    # Output Product corner coordinates (0-based pixel centers)
    op_corners = {
        "11": (0.0, 0.0),
        "12": (float(output_size.width - 1), 0.0),
        "21": (0.0, float(output_size.height - 1)),
        "22": (float(output_size.width - 1), float(output_size.height - 1)),
    }

    # Full Image coordinates — if source has ICHIPB, chain through it
    fi_col_offset = float(src_window.x)
    fi_row_offset = float(src_window.y)
    scale_x = src_window.width / output_size.width
    scale_y = src_window.height / output_size.height

    # Source full image dimensions (default to src_window extent)
    fi_col = 0
    fi_row = 0

    if source_ichipb is not None:
        # Chain: the source's FI coordinates become our reference frame.
        # The source ICHIPB maps source OP coords → original FI coords.
        # Our src_window is in source pixel (OP) coordinates; we need to
        # map those through the source's ICHIPB to get original FI coords.
        src_fi_col_11 = _get_ichipb_float(source_ichipb, "FI_COL_11")
        src_fi_row_11 = _get_ichipb_float(source_ichipb, "FI_ROW_11")
        src_fi_col_12 = _get_ichipb_float(source_ichipb, "FI_COL_12")
        src_fi_row_21 = _get_ichipb_float(source_ichipb, "FI_ROW_21")

        src_op_col_11 = _get_ichipb_float(source_ichipb, "OP_COL_11")
        src_op_row_11 = _get_ichipb_float(source_ichipb, "OP_ROW_11")
        src_op_col_12 = _get_ichipb_float(source_ichipb, "OP_COL_12")
        src_op_row_21 = _get_ichipb_float(source_ichipb, "OP_ROW_21")

        # Derive the affine mapping: OP → FI from the source ICHIPB
        src_op_w = src_op_col_12 - src_op_col_11
        src_op_h = src_op_row_21 - src_op_row_11
        if src_op_w != 0 and src_op_h != 0:
            fi_scale_x = (src_fi_col_12 - src_fi_col_11) / src_op_w
            fi_scale_y = (src_fi_row_21 - src_fi_row_11) / src_op_h
        else:
            fi_scale_x = 1.0
            fi_scale_y = 1.0

        fi_col_offset = src_fi_col_11 + (src_window.x - src_op_col_11) * fi_scale_x
        fi_row_offset = src_fi_row_11 + (src_window.y - src_op_row_11) * fi_scale_y
        scale_x = scale_x * fi_scale_x
        scale_y = scale_y * fi_scale_y

        fi_col = _get_ichipb_int(source_ichipb, "FI_COL")
        fi_row = _get_ichipb_int(source_ichipb, "FI_ROW")

    # ICHIPB header fields
    fields: Dict[str, Any] = {
        "XFRM_FLAG": "00",
        "SCALE_FACTOR": f"{scale_x:010.5f}",  # noqa: E231
        "ANAMRPH_CORR": "00",
        "SCANBLK_NUM": "00",
    }

    # OP/FI coordinate fields are 12 chars each per STDI-0002
    for point, (op_col, op_row) in op_corners.items():
        fields[f"OP_COL_{point}"] = f"{op_col:012.2f}"  # noqa: E231
        fields[f"OP_ROW_{point}"] = f"{op_row:012.2f}"  # noqa: E231
        fi_col_val = fi_col_offset + op_col * scale_x
        fi_row_val = fi_row_offset + op_row * scale_y
        fields[f"FI_COL_{point}"] = f"{fi_col_val:012.2f}"  # noqa: E231
        fields[f"FI_ROW_{point}"] = f"{fi_row_val:012.2f}"  # noqa: E231

    # FI_ROW/FI_COL are 8 chars each
    if fi_col > 0 and fi_row > 0:
        fields["FI_COL"] = f"{fi_col:08d}"  # noqa: E231
        fields["FI_ROW"] = f"{fi_row:08d}"  # noqa: E231
    else:
        fields["FI_COL"] = f"{src_window.x + src_window.width:08d}"  # noqa: E231
        fields["FI_ROW"] = f"{src_window.y + src_window.height:08d}"  # noqa: E231

    return fields


def _get_ichipb_float(ichipb: Dict[str, Any], field: str) -> float:
    """Extract a float field from an ICHIPB dict."""
    val = ichipb.get(field, "0.0")
    if isinstance(val, str):
        return float(val)
    return float(val)


def _get_ichipb_int(ichipb: Dict[str, Any], field: str) -> int:
    """Extract an int field from an ICHIPB dict."""
    val = ichipb.get(field, "0")
    if isinstance(val, str):
        return int(val)
    return int(val)


# NITF file header security fields to propagate to chip file headers
_NITF_FILE_SECURITY_FIELDS = (
    "FSCLAS",
    "FSCLSY",
    "FSCODE",
    "FSCTLH",
    "FSREL",
    "FSDCTP",
    "FSDCDT",
    "FSDCXM",
    "FSDG",
    "FSDGDT",
    "FSCLTX",
    "FSCATP",
    "FSCAUT",
    "FSCRSN",
    "FSSRDT",
    "FSCTLN",
)

# NITF subheader fields that are safe to propagate from source to chip
_NITF_PROPAGATED_FIELDS = (
    "IREP",
    "ICAT",
    "IMODE",
    "PVTYPE",
    "IC",
    "COMRAT",
    "ISCLSY",
    "ISCODE",
    "ISCTLH",
    "ISREL",
    "ISDCTP",
    "ISDCDT",
    "ISDCXM",
    "ISDG",
    "ISDGDT",
    "ISCLTX",
    "ISCATP",
    "ISCAUT",
    "ISCRSN",
    "ISSRDT",
    "ISCTLN",
)

# TREs that reference the full image coordinate system and should be
# propagated unchanged to chips (ICHIPB maps chip coords back to full).
_NITF_PROPAGATED_TRES = (
    "RPC00B",
    "RPC00A",
    "CSCRNA",
)


class GeoTiffChipMetadataBuilder:
    """Derives GeoTIFF chip metadata (GeoTransform + CRS) for chips.

    For same-format (GeoTIFF source → GeoTIFF output), the source
    GeoTransform is adjusted for the chip offset and any resolution
    change. For cross-format (non-GeoTIFF source → GeoTIFF output),
    the GeoTransform is derived from the sensor model by linearizing
    corner coordinates — accurate for small chips but an approximation
    for large extents with non-affine models.
    """

    def __init__(
        self,
        reader: Any = None,
        sensor_model: Any = None,
    ) -> None:
        self._sensor_model = sensor_model
        self._source_geotransform: Optional[Tuple[float, float, float, float, float, float]] = None
        self._crs_tags: Dict[str, Any] = {}

        if reader is not None:
            metadata = getattr(reader, "metadata", None)
            if metadata is not None:
                meta_dict = dict(metadata) if metadata is not None else metadata
                if isinstance(meta_dict, dict):
                    self._source_geotransform = _parse_geotransform(meta_dict)
                    for tag_id in ("34735", "34736", "34737"):
                        if tag_id in meta_dict:
                            self._crs_tags[tag_id] = meta_dict[tag_id]

    def build(
        self,
        src_window: PixelWindow,
        output_size: ImageSize,
    ) -> Any:
        """Derive GeoTIFF metadata for the given chip bounds.

        Returns a ``BufferedMetadataProvider`` with ModelTiepoint,
        ModelPixelScale, and CRS tags set for the chip extent.
        """
        from aws.osml.io import BufferedMetadataProvider

        metadata = BufferedMetadataProvider()

        geo_transform = None
        if self._source_geotransform is not None:
            geo_transform = _geotransform_for_chip(self._source_geotransform, src_window, output_size)
        elif self._sensor_model is not None:
            geo_transform = _geotransform_from_sensor_model(self._sensor_model, src_window, output_size)

        if geo_transform is not None:
            x_origin, pixel_width, x_rot, y_origin, y_rot, pixel_height = geo_transform

            scale_x = abs(pixel_width)
            scale_y = abs(pixel_height)
            metadata["33550"] = [scale_x, scale_y, 0.0]
            metadata["33922"] = [0.0, 0.0, 0.0, x_origin, y_origin, 0.0]

        for tag_id, value in self._crs_tags.items():
            if isinstance(value, (list, tuple)):
                metadata[tag_id] = list(value)
            else:
                metadata[tag_id] = value

        return metadata


class NitfChipMetadataBuilder:
    """Derives NITF chip metadata for chipped EO and SAR imagery.

    Computes IGEOLO (60-character geographic coordinate string),
    creates ICHIPB TRE (chip-to-parent geometric relationship),
    propagates source NITF subheader fields, and updates SICD/SIDD
    DES XML metadata for the chip bounds.

    For NITF-to-NITF, source fields are copied with targeted updates.
    For non-NITF-to-NITF, metadata is derived from the sensor model.

    SICD/SIDD DES handling:
    - When the source contains SICD or SIDD XML DES segments, the
      builder updates the XML for the chip bounds using stateless
      updater functions.
    - DES XML is omitted from output when ``skip_des=True`` (set by
      ChipFactory when a processing chain is applied).
    """

    def __init__(
        self,
        reader: Any = None,
        sensor_model: Any = None,
    ) -> None:
        self._sensor_model = sensor_model
        self._source_fields: Dict[str, str] = {}
        self._source_tres: Dict[str, Any] = {}
        self._file_security: Dict[str, str] = {}
        self._source_ichipb: Optional[Dict[str, Any]] = None
        self._sicd_xml: Optional[str] = None
        self._sidd_xml: Optional[str] = None
        self._des_metadata: Optional[Dict[str, str]] = None

        if reader is not None:
            self._extract_file_security(reader)

            meta_dict = self._get_image_metadata(reader)
            if isinstance(meta_dict, dict):
                for field in _NITF_PROPAGATED_FIELDS:
                    if field in meta_dict:
                        val = meta_dict[field]
                        if isinstance(val, str):
                            self._source_fields[field] = val

                for tre_name in _NITF_PROPAGATED_TRES:
                    if tre_name in meta_dict:
                        self._source_tres[tre_name] = meta_dict[tre_name]

                ichipb = meta_dict.get("ICHIPB")
                if isinstance(ichipb, dict):
                    self._source_ichipb = ichipb

            self._extract_des_xml(reader)

    def _extract_file_security(self, reader: Any) -> None:
        """Extract file-level security fields from the reader."""
        metadata_attr = getattr(reader, "metadata", None)
        if metadata_attr is None:
            return
        try:
            file_dict = dict(metadata_attr) if metadata_attr is not None else {}
        except Exception:
            return
        if not isinstance(file_dict, dict):
            return
        for field in _NITF_FILE_SECURITY_FIELDS:
            if field in file_dict:
                val = file_dict[field]
                if isinstance(val, str) and val.strip():
                    self._file_security[field] = val

    @staticmethod
    def _get_image_metadata(reader: Any) -> Optional[Dict[str, Any]]:
        """Extract image metadata from a reader, trying asset then reader level."""
        # Try the base image asset first (where TREs live in NITF)
        get_asset = getattr(reader, "get_asset", None)
        if callable(get_asset):
            try:
                asset = get_asset("image:0")
                metadata = getattr(asset, "metadata", None)
                if metadata is not None:
                    return dict(metadata)
            except (KeyError, AttributeError):
                pass

        # Fallback to reader-level metadata
        metadata = getattr(reader, "metadata", None)
        if metadata is not None:
            if isinstance(metadata, dict):
                return metadata
            return dict(metadata)

        return None

    def _extract_des_xml(self, reader: Any) -> None:
        """Extract SICD or SIDD XML from DES assets on the reader."""
        get_asset_keys = getattr(reader, "get_asset_keys", None)
        get_asset = getattr(reader, "get_asset", None)
        if not callable(get_asset_keys) or not callable(get_asset):
            return

        try:
            keys = get_asset_keys()
        except Exception:
            return

        for key in keys:
            if not (isinstance(key, str) and key.startswith("des:")):
                continue
            try:
                asset = get_asset(key)
                raw_io = asset.raw_asset
                xml_str = raw_io.read().decode("utf-8", errors="replace")
            except Exception:
                continue

            if "SIDD" in xml_str:
                self._sidd_xml = xml_str
                self._des_metadata = self._extract_des_metadata(asset)
                break
            elif "SICD" in xml_str:
                self._sicd_xml = xml_str
                self._des_metadata = self._extract_des_metadata(asset)
                break

    @staticmethod
    def _extract_des_metadata(asset: Any) -> Optional[Dict[str, str]]:
        """Extract DES subheader metadata fields from an asset."""
        metadata = getattr(asset, "metadata", None)
        if metadata is None:
            return None
        try:
            meta_dict = dict(metadata) if metadata is not None else {}
            if isinstance(meta_dict, dict):
                return {k: str(v) for k, v in meta_dict.items()}
        except Exception:
            pass
        return None

    @property
    def has_sicd(self) -> bool:
        """True if source contains SICD DES metadata."""
        return self._sicd_xml is not None

    @property
    def has_sidd(self) -> bool:
        """True if source contains SIDD DES metadata."""
        return self._sidd_xml is not None

    @property
    def file_security(self) -> Dict[str, str]:
        """File-level security fields from the source NITF header."""
        return self._file_security

    def build(
        self,
        src_window: PixelWindow,
        output_size: ImageSize,
        skip_des: bool = False,
    ) -> Any:
        """Derive NITF metadata for the given chip bounds.

        Returns a ``BufferedMetadataProvider`` with IGEOLO, ICORDS,
        ICHIPB TRE, propagated subheader fields, and optionally
        updated SICD/SIDD DES XML.

        :param src_window: The source pixel window (R0 coordinates).
        :param output_size: The output tile dimensions.
        :param skip_des: When True, SICD/SIDD DES XML is omitted from
            output (used when a processing chain transforms the pixels).
        """
        from aws.osml.io import BufferedMetadataProvider

        metadata = BufferedMetadataProvider()

        for field, value in self._source_fields.items():
            metadata[field] = value

        for tre_name, tre_value in self._source_tres.items():
            metadata[tre_name] = tre_value

        if self._sensor_model is not None:
            corners = _compute_corner_coords(self._sensor_model, src_window)
            metadata["ICORDS"] = "G"
            from aws.osml.io.jbp.utils import IGEOLOAdapter

            igeolo = IGEOLOAdapter.format(corners, "G")
            metadata["IGEOLO"] = igeolo

        ichipb_fields = _build_ichipb_fields(src_window, output_size, self._source_ichipb)
        metadata["ICHIPB"] = ichipb_fields

        if not skip_des:
            des_xml = self._build_des_xml(src_window, output_size)
            if des_xml is not None:
                metadata["_DES_XML"] = des_xml
                if self._des_metadata:
                    metadata["_DES_METADATA"] = self._des_metadata

        return metadata

    def _build_des_xml(self, src_window: PixelWindow, output_size: ImageSize) -> Optional[str]:
        """Build updated SICD or SIDD XML for the chip bounds."""
        chip_bounds = [src_window.x, src_window.y, src_window.width, src_window.height]
        out_size = (output_size.width, output_size.height)

        if self._sicd_xml is not None:
            from .sicd_updater import update_sicd_for_chip

            return update_sicd_for_chip(self._sicd_xml, chip_bounds, out_size)

        if self._sidd_xml is not None:
            from .sidd_updater import update_sidd_for_chip

            return update_sidd_for_chip(self._sidd_xml, chip_bounds, out_size)

        return None
