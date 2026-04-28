#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""ChipFactory — GDAL-free chip extraction with encoding.

This module provides :class:`ChipFactory`, which reads arbitrary pixel
windows from a :class:`~aws.osml.image_processing.pyramid.TiledImagePyramid`,
optionally applies a processing chain, and encodes the result via
osml-imagery-io's ``DatasetWriter``.

Named tuples :class:`ImageSize` and :class:`PixelWindow` eliminate
width/height ordering ambiguity in the public API.
"""

from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from .block_utils import read_window
from .resample import area_resample, bilinear_resample


class ImageSize(NamedTuple):
    """Output image dimensions (width, height)."""

    width: int
    height: int


class PixelWindow(NamedTuple):
    """Pixel region in image coordinates."""

    x: int
    y: int
    width: int
    height: int


def _resample_to_size(pixels: NDArray, output_size: ImageSize) -> NDArray:
    """Resize a CHW array to the given output dimensions.

    Uses bilinear for upsampling, area-based for downsampling.
    """
    current_h, current_w = pixels.shape[1], pixels.shape[2]
    target_h, target_w = output_size.height, output_size.width

    if current_h == target_h and current_w == target_w:
        return pixels

    if target_h > current_h or target_w > current_w:
        return bilinear_resample(pixels, target_h, target_w)
    return area_resample(pixels, target_h, target_w)


class ChipFactory:
    """Orchestrator for extracting and encoding image chips.

    Selects the appropriate pyramid level, reads pixels via
    :func:`~aws.osml.image_processing.block_utils.read_window`,
    optionally applies a processing chain, resizes to the requested
    output dimensions, and encodes the result.

    Thread-safe for concurrent :meth:`create_chip` calls.
    """

    def __init__(
        self,
        source: Any,
        sensor_model: Any = None,
        output_format: str = "nitf",
        processing_chain: Any = None,
        metadata_builder: Any = None,
        metadata_overrides: Any = None,
    ):
        """
        Args:
            source: Pre-built :class:`TiledImagePyramid`. Single-level is
                valid for images without overviews.
            sensor_model: For geospatial metadata derivation. Optional.
            output_format: Format string matching osml-imagery-io's
                ``IO.open`` conventions (e.g., "nitf", "geotiff", "png",
                "jpeg").
            processing_chain: Optional
                :class:`~aws.osml.image_processing.processing_chain.ProcessingChain`
                applied before encoding. When ``None``, chips contain raw
                pixels.
            metadata_builder: Derives chip metadata from source metadata
                + chip bounds. When ``None``, auto-selected based on
                output_format (GeoTiffChipMetadataBuilder for geotiff).
            metadata_overrides: User-supplied ``BufferedMetadataProvider``
                whose fields are applied on top of builder output.
        """
        self._source = source
        self._sensor_model = sensor_model
        self._output_format = output_format
        self._processing_chain = processing_chain
        self._metadata_builder = metadata_builder
        self._metadata_overrides = metadata_overrides

        if self._metadata_builder is None:
            self._metadata_builder = self._auto_select_metadata_builder()

    def create_chip(
        self,
        src_window: PixelWindow,
        output_size: Optional[ImageSize] = None,
    ) -> Optional[bytearray]:
        """Extract and encode a chip.

        Args:
            src_window: Pixel window in R0 coordinates.
            output_size: Optional output dimensions. When different from
                src_window dimensions, the best pyramid level is selected
                and a final resize is applied after the processing chain.

        Returns:
            Encoded image bytes, or ``None`` when the window region
            contains no pixel data (sparse/empty blocks).

        Raises:
            ValueError: Invalid window dimensions (negative or zero), or
                SICD source with scaled output requested in NITF format.
        """
        if src_window.width <= 0 or src_window.height <= 0:
            raise ValueError(f"src_window dimensions must be > 0, got width={src_window.width}, height={src_window.height}")

        if output_size is None:
            output_size = ImageSize(src_window.width, src_window.height)

        if output_size.width <= 0 or output_size.height <= 0:
            raise ValueError(
                f"output_size dimensions must be > 0, got width={output_size.width}, height={output_size.height}"
            )

        self._validate_sicd_scaling(src_window, output_size)

        # 1. Select best pyramid level
        src_size = (src_window.width, src_window.height)
        level = self._source.best_level_for(src_size, (output_size.width, output_size.height))
        provider = self._source.get_level(level)

        # 2. Check for fully out-of-bounds window (short-circuit)
        provider_rows = getattr(provider, "num_rows", None)
        provider_cols = getattr(provider, "num_columns", None)
        if provider_rows is not None and provider_cols is not None:
            divisor_check = self._source.scale_factor**level
            win_x_end = src_window.x + src_window.width
            win_y_end = src_window.y + src_window.height
            scaled_provider_cols = provider_cols * divisor_check
            scaled_provider_rows = provider_rows * divisor_check
            fully_oob = (
                win_x_end <= 0
                or win_y_end <= 0
                or src_window.x >= scaled_provider_cols
                or src_window.y >= scaled_provider_rows
            )
            if fully_oob:
                return None

        # 3. Scale src_window to that level's coordinate space
        divisor = self._source.scale_factor**level
        scaled_x = src_window.x // divisor
        scaled_y = src_window.y // divisor
        x_end = (src_window.x + src_window.width + divisor - 1) // divisor
        y_end = (src_window.y + src_window.height + divisor - 1) // divisor
        scaled_window = (scaled_x, scaled_y, x_end - scaled_x, y_end - scaled_y)

        # 4. Read pixels from that level (band-selective if chain specifies)
        bands = None
        if self._processing_chain is not None:
            bands = self._processing_chain.input_bands
        pixels = read_window(provider, scaled_window, bands=bands)

        # 5. Check for empty/sparse data
        if pixels is None:
            return None

        # 6. Apply processing chain (if provided)
        if self._processing_chain is not None:
            pixels = self._processing_chain(pixels)

        # 7. Final resize to output_size
        if (pixels.shape[2], pixels.shape[1]) != (output_size.width, output_size.height):
            pixels = _resample_to_size(pixels, output_size)

        # 8. Derive tile metadata
        tile_metadata = None
        if self._metadata_builder is not None:
            build_kwargs: dict = {
                "src_window": src_window,
                "output_size": output_size,
            }
            if self._processing_chain is not None and hasattr(self._metadata_builder, "build"):
                import inspect

                sig = inspect.signature(self._metadata_builder.build)
                if "skip_des" in sig.parameters:
                    build_kwargs["skip_des"] = True
            tile_metadata = self._metadata_builder.build(**build_kwargs)

        # 9. Encode via DatasetWriter (in-memory)
        file_metadata = self._build_file_metadata()
        return self._encode(pixels, output_size, tile_metadata, file_metadata)

    def _validate_sicd_scaling(self, src_window: PixelWindow, output_size: ImageSize) -> None:
        """Raise ValueError if SICD source is scaled to NITF output."""
        if self._output_format != "nitf":
            return
        if self._metadata_builder is None:
            return
        from .chip_metadata_builder import NitfChipMetadataBuilder

        if not isinstance(self._metadata_builder, NitfChipMetadataBuilder):
            return
        if not self._metadata_builder.has_sicd:
            return
        if output_size.width != src_window.width or output_size.height != src_window.height:
            raise ValueError(
                "SICD metadata does not support decimation. Cannot produce scaled NITF output "
                f"(src_window={src_window.width}x{src_window.height}, "
                f"output_size={output_size.width}x{output_size.height}). "
                "Use a non-NITF output format or request 1:1 output size."
            )

    def _build_file_metadata(self) -> Optional[Any]:
        """Build file-level metadata from the metadata builder (if NITF)."""
        if self._output_format != "nitf":
            return None
        if self._metadata_builder is None:
            return None
        file_security = getattr(self._metadata_builder, "file_security", None)
        if not file_security:
            return None
        from aws.osml.io import BufferedMetadataProvider

        file_meta = BufferedMetadataProvider()
        for field, value in file_security.items():
            file_meta[field] = value
        return file_meta

    def _auto_select_metadata_builder(self) -> Optional[Any]:
        """Auto-select a metadata builder based on output_format."""
        if self._output_format in ("geotiff", "tiff"):
            from .chip_metadata_builder import GeoTiffChipMetadataBuilder

            reader = getattr(self._source, "reader", None)
            return GeoTiffChipMetadataBuilder(reader=reader, sensor_model=self._sensor_model)
        if self._output_format == "nitf":
            from .chip_metadata_builder import NitfChipMetadataBuilder

            reader = getattr(self._source, "reader", None)
            if reader is not None or self._sensor_model is not None:
                return NitfChipMetadataBuilder(reader=reader, sensor_model=self._sensor_model)
        return None

    def _encode(
        self, pixels: NDArray, output_size: ImageSize, tile_metadata: Any = None, file_metadata: Any = None
    ) -> bytearray:
        """Encode pixels to the configured output format."""
        import io as stdlib_io

        from aws.osml.io import IO, BufferedDataAssetProvider, BufferedImageAssetProvider, BufferedMetadataProvider

        pixel_type = _numpy_dtype_to_pixel_type(pixels.dtype)
        num_bands, height, width = pixels.shape

        metadata = BufferedMetadataProvider()
        if tile_metadata is not None:
            metadata = tile_metadata
        if self._metadata_overrides is not None:
            metadata = _merge_metadata(metadata, self._metadata_overrides)

        # Extract DES XML before writing (stored as internal metadata keys)
        des_xml = None
        des_meta_dict = None
        meta_dict = dict(metadata) if metadata is not None else {}
        if isinstance(meta_dict, dict) and "_DES_XML" in meta_dict:
            des_xml = meta_dict["_DES_XML"]
            des_meta_dict = meta_dict.get("_DES_METADATA")

        pixels = np.ascontiguousarray(pixels)

        provider = BufferedImageAssetProvider.create(
            key="image:0",
            num_columns=width,
            num_rows=height,
            num_bands=num_bands,
            block_width=width,
            block_height=height,
            pixel_type=pixel_type,
            metadata=metadata,
        )
        provider.set_full_image(pixels)

        io_format = "tiff" if self._output_format == "geotiff" else self._output_format
        buf = stdlib_io.BytesIO()
        with IO.open(buf, "w", io_format) as writer:
            if file_metadata is not None:
                writer.metadata = file_metadata
            writer.add_asset(
                key="image:0",
                provider=provider,
                title="Image",
                description=f"{width}x{height} {num_bands}-band",
                roles=["data"],
            )
            if des_xml is not None and io_format == "nitf":
                des_metadata = BufferedMetadataProvider()
                if isinstance(des_meta_dict, dict):
                    for k, v in des_meta_dict.items():
                        des_metadata[k] = str(v)
                else:
                    des_metadata["DESID"] = "XML_DATA_CONTENT"
                    des_metadata["DESVER"] = "01"
                    des_metadata["DESCLAS"] = "U"
                des_provider = BufferedDataAssetProvider.create(
                    key="des:0",
                    data=des_xml.encode("utf-8"),
                    mime_type="text/xml",
                    metadata=des_metadata,
                )
                writer.add_asset(
                    key="des:0",
                    provider=des_provider,
                    title="XML_DATA_CONTENT",
                    description="SICD/SIDD XML DES",
                    roles=["data"],
                )
        return bytearray(buf.getvalue())


def _merge_metadata(base: Any, overrides: Any) -> Any:
    """Merge override metadata on top of base, returning overrides.

    Copies all fields from base into overrides (which takes precedence
    for any key it already defines).
    """
    if base is overrides:
        return base
    base_dict = dict(base) if base is not None else {}
    overrides_dict = dict(overrides) if overrides is not None else {}
    if isinstance(base_dict, dict):
        for key, value in base_dict.items():
            if key not in overrides_dict:
                if isinstance(value, (list, tuple)):
                    overrides[key] = list(value)
                else:
                    overrides[key] = value
    return overrides


def _numpy_dtype_to_pixel_type(dtype: np.dtype) -> Any:
    """Map a numpy dtype to an osml-imagery-io PixelType."""
    from aws.osml.io import PixelType

    mapping = {
        np.dtype(np.uint8): PixelType.UInt8,
        np.dtype(np.uint16): PixelType.UInt16,
        np.dtype(np.int16): PixelType.Int16,
        np.dtype(np.int32): PixelType.Int32,
        np.dtype(np.uint32): PixelType.UInt32,
        np.dtype(np.float32): PixelType.Float32,
        np.dtype(np.float64): PixelType.Float64,
    }
    result = mapping.get(dtype)
    if result is None:
        raise ValueError(f"Unsupported pixel dtype: {dtype}")
    return result
