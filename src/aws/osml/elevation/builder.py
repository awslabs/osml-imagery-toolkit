#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from typing import List, Optional, Tuple, Union

from aws.osml.photogrammetry import (
    ConditionalElevationModel,
    ConstantElevationModel,
    DigitalElevationModel,
    DigitalElevationModelTileFactory,
    DigitalElevationModelTileSet,
    ElevationModel,
    ElevationModelCondition,
    GeometryQuery,
    MultiElevationModel,
    NormalizedElevationModel,
    OffsetElevationModel,
)

from .geometry_condition import GeometryCondition
from .raster_offset_provider import RasterOffsetProvider


class ElevationModelBuilder:
    """
    Fluent builder for composing multi-source elevation models.

    Sources are tried in the order they are added (first = highest priority).
    Each source can optionally be guarded by a spatial condition. A geoid
    offset is applied globally to the final result. Coordinate normalization
    is applied by default.

    Example::

        model = (
            ElevationModelBuilder()
            .add_source(StoredDEMTileFactory("/srtm"), SRTMTileSet())
            .with_geoid("/path/to/egm96.tif")
            .build()
        )
    """

    def __init__(self) -> None:
        self._sources: List[Tuple[ElevationModel, Optional[ElevationModelCondition]]] = []
        self._geoid_path: Optional[str] = None
        self._geoid_scale: float = 1.0

    def add_source(
        self,
        tile_factory: DigitalElevationModelTileFactory,
        tile_set: DigitalElevationModelTileSet,
        condition: Optional[Union[GeometryQuery, ElevationModelCondition]] = None,
        invert_condition: bool = False,
        raster_cache_size: int = 10,
    ) -> "ElevationModelBuilder":
        """Add a DEM source. Sources are tried in order; first success wins."""
        dem = DigitalElevationModel(tile_set, tile_factory, raster_cache_size=raster_cache_size)
        em_condition = self._resolve_condition(condition, invert_condition)
        self._sources.append((dem, em_condition))
        return self

    def add_elevation_model(
        self,
        elevation_model: ElevationModel,
        condition: Optional[Union[GeometryQuery, ElevationModelCondition]] = None,
        invert_condition: bool = False,
    ) -> "ElevationModelBuilder":
        """Add an arbitrary ElevationModel as a source."""
        em_condition = self._resolve_condition(condition, invert_condition)
        self._sources.append((elevation_model, em_condition))
        return self

    def add_fallback(self, elevation: float = 0.0) -> "ElevationModelBuilder":
        """Add a constant elevation fallback as the lowest-priority source."""
        self._sources.append((ConstantElevationModel(elevation), None))
        return self

    def with_geoid(self, offset_path: str, scale_factor: float = 1.0) -> "ElevationModelBuilder":
        """
        Apply geoid correction to the final elevation result.

        The offset is applied globally to all sources. This assumes all sources
        use the same vertical datum (e.g., all orthometric heights). If mixing
        sources with different vertical datums, compose the OffsetElevationModel
        manually around only the orthometric source and use add_elevation_model().
        """
        self._geoid_path = offset_path
        self._geoid_scale = scale_factor
        return self

    def build(self, normalize: bool = True) -> ElevationModel:
        """
        Assemble the elevation model composition.

        :param normalize: wrap in NormalizedElevationModel (default True, opt-out)
        """
        if not self._sources:
            raise ValueError("No elevation sources added to builder.")

        models: List[ElevationModel] = []
        for model, condition in self._sources:
            if condition is not None:
                models.append(ConditionalElevationModel(model, condition))
            else:
                models.append(model)

        if len(models) == 1:
            result = models[0]
        else:
            result = MultiElevationModel(models)

        if self._geoid_path is not None:
            offset_provider = RasterOffsetProvider(self._geoid_path, self._geoid_scale)
            result = OffsetElevationModel(result, offset_provider)

        if normalize:
            result = NormalizedElevationModel(result)

        return result

    @staticmethod
    def _resolve_condition(
        condition: Optional[Union[GeometryQuery, ElevationModelCondition]],
        invert: bool,
    ) -> Optional[ElevationModelCondition]:
        if condition is None:
            return None
        if isinstance(condition, GeometryQuery):
            return GeometryCondition(condition, invert=invert)
        return condition
