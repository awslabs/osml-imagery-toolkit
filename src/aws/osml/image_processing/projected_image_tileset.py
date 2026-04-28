#  Copyright 2024 Amazon.com, Inc. or its affiliates.

import math
from typing import Optional, Tuple

import pyproj

from ..photogrammetry import ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, SensorModel
from .map_tileset import MapTile, MapTileBounds, MapTileId, MapTileSet, MapTileSize


class ProjectedImageTileSet(MapTileSet):
    """A MapTileSet covering a single image's projected footprint with multi-level tile matrices.

    Level 0 tiles are block_size pixels at native GSD. Level N tiles each cover
    2^N x 2^N level-0 tiles (same pixel size, coarser GSD). Grid dimensions at
    level N are ceil(level_0_cols / 2^N) x ceil(level_0_rows / 2^N).
    """

    def __init__(
        self,
        target_crs: pyproj.CRS,
        origin: Tuple[float, float],
        gsd: float,
        block_size: Tuple[int, int],
        grid_cols: int,
        grid_rows: int,
        num_levels: int,
    ) -> None:
        self._target_crs = target_crs
        self._origin = origin
        self._gsd = gsd
        self._block_width, self._block_height = block_size
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows
        self._num_levels = num_levels

        authority = target_crs.to_authority()
        self._crs_id = ":".join(authority) if authority else target_crs.to_wkt()

        is_geographic = target_crs.is_geographic
        crs_4326 = pyproj.CRS.from_epsg(4326)
        if is_geographic and target_crs == crs_4326:
            self._to_4326 = None
        else:
            self._to_4326 = pyproj.Transformer.from_crs(target_crs, crs_4326, always_xy=True)

    @staticmethod
    def from_sensor_model(
        sensor_model: SensorModel,
        source_width: int,
        source_height: int,
        target_crs: pyproj.CRS = pyproj.CRS.from_epsg(4326),
        gsd: Optional[float] = None,
        block_size: Tuple[int, int] = (1024, 1024),
        elevation_model: Optional[ElevationModel] = None,
        num_levels: Optional[int] = None,
    ) -> "ProjectedImageTileSet":
        """Construct a ProjectedImageTileSet from a sensor model and image dimensions.

        :param sensor_model: Sensor model for the source image.
        :param source_width: Source image width in pixels.
        :param source_height: Source image height in pixels.
        :param target_crs: Target CRS for the tile set. Defaults to EPSG:4326.
        :param gsd: Pixel spacing in target CRS units. None to estimate from source.
        :param block_size: (width, height) of each tile in pixels.
        :param elevation_model: Optional elevation model for corner projection.
        :param num_levels: Number of tile matrix levels. None to auto-calculate.
        :return: A new ProjectedImageTileSet instance.
        """
        from_4326 = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), target_crs, always_xy=True)

        corners = [
            ImageCoordinate([0.0, 0.0]),
            ImageCoordinate([source_width - 1.0, 0.0]),
            ImageCoordinate([source_width - 1.0, source_height - 1.0]),
            ImageCoordinate([0.0, source_height - 1.0]),
        ]

        lons_deg = []
        lats_deg = []
        for corner in corners:
            world = sensor_model.image_to_world(corner, elevation_model)
            lons_deg.append(math.degrees(world.longitude))
            lats_deg.append(math.degrees(world.latitude))

        target_x, target_y = from_4326.transform(lons_deg, lats_deg)
        xmin, ymin = min(target_x), min(target_y)
        xmax, ymax = max(target_x), max(target_y)

        if gsd is None:
            center = ImageCoordinate([source_width / 2.0, source_height / 2.0])
            right = ImageCoordinate([source_width / 2.0 + 1.0, source_height / 2.0])
            world_center = sensor_model.image_to_world(center, elevation_model)
            world_right = sensor_model.image_to_world(right, elevation_model)
            cx_deg = math.degrees(world_center.longitude)
            cy_deg = math.degrees(world_center.latitude)
            rx_deg = math.degrees(world_right.longitude)
            ry_deg = math.degrees(world_right.latitude)
            tx_c, ty_c = from_4326.transform(cx_deg, cy_deg)
            tx_r, ty_r = from_4326.transform(rx_deg, ry_deg)
            dx = tx_r - tx_c
            dy = ty_r - ty_c
            gsd = math.sqrt(dx * dx + dy * dy)

        block_width, block_height = block_size
        x_span = xmax - xmin
        y_span = ymax - ymin
        grid_cols = max(1, int(math.ceil(x_span / (gsd * block_width))))
        grid_rows = max(1, int(math.ceil(y_span / (gsd * block_height))))

        if num_levels is None:
            max_dim = max(grid_cols, grid_rows)
            num_levels = max(1, math.ceil(math.log2(max_dim)) + 1) if max_dim > 1 else 1

        origin = (xmin, ymax)

        return ProjectedImageTileSet(
            target_crs=target_crs,
            origin=origin,
            gsd=gsd,
            block_size=block_size,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            num_levels=num_levels,
        )

    @property
    def tile_matrix_set_id(self) -> str:
        return f"ProjectedImage_{self._crs_id}"

    @property
    def crs_id(self) -> str:
        return self._crs_id

    @property
    def num_tile_matrices(self) -> int:
        return self._num_levels

    def tile_matrix_dimensions(self, tile_matrix: int) -> Tuple[int, int]:
        """(num_cols, num_rows) at the given tile matrix level."""
        scale = 2**tile_matrix
        cols = max(1, math.ceil(self._grid_cols / scale))
        rows = max(1, math.ceil(self._grid_rows / scale))
        return (cols, rows)

    def get_tile(self, tile_id: MapTileId) -> MapTile:
        tile_matrix = tile_id.tile_matrix
        tile_row = tile_id.tile_row
        tile_col = tile_id.tile_col

        scale = 2**tile_matrix
        tile_gsd = self._gsd * scale

        x0 = self._origin[0] + tile_col * self._block_width * tile_gsd
        y1 = self._origin[1] - tile_row * self._block_height * tile_gsd
        x1 = x0 + self._block_width * tile_gsd
        y0 = y1 - self._block_height * tile_gsd

        native_bounds = (x0, y0, x1, y1)

        if self._to_4326 is not None:
            corner_xs = [x0, x1, x1, x0]
            corner_ys = [y0, y0, y1, y1]
            lon_deg, lat_deg = self._to_4326.transform(corner_xs, corner_ys)
            min_lon_rad = math.radians(min(lon_deg))
            max_lon_rad = math.radians(max(lon_deg))
            min_lat_rad = math.radians(min(lat_deg))
            max_lat_rad = math.radians(max(lat_deg))
        else:
            min_lon_rad = math.radians(x0)
            max_lon_rad = math.radians(x1)
            min_lat_rad = math.radians(y0)
            max_lat_rad = math.radians(y1)

        bounds = MapTileBounds(min_lon_rad, min_lat_rad, max_lon_rad, max_lat_rad)
        size = MapTileSize(self._block_width, self._block_height)

        return MapTile(id=tile_id, size=size, bounds=bounds, native_bounds=native_bounds)

    def get_tile_for_location(self, world_coordinate: GeodeticWorldCoordinate, tile_matrix: int) -> MapTile:
        lon_deg = math.degrees(world_coordinate.longitude)
        lat_deg = math.degrees(world_coordinate.latitude)

        if self._to_4326 is not None:
            from_4326 = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), self._target_crs, always_xy=True)
            native_x, native_y = from_4326.transform(lon_deg, lat_deg)
        else:
            native_x, native_y = lon_deg, lat_deg

        scale = 2**tile_matrix
        tile_gsd = self._gsd * scale

        tile_col = int((native_x - self._origin[0]) / (self._block_width * tile_gsd))
        tile_row = int((self._origin[1] - native_y) / (self._block_height * tile_gsd))

        cols, rows = self.tile_matrix_dimensions(tile_matrix)
        tile_col = max(0, min(tile_col, cols - 1))
        tile_row = max(0, min(tile_row, rows - 1))

        return self.get_tile(MapTileId(tile_matrix=tile_matrix, tile_row=tile_row, tile_col=tile_col))
