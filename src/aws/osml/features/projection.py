#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import math
from typing import List, Optional, Tuple

import geojson
import shapely
import shapely.geometry

from aws.osml.photogrammetry import ElevationModel, GeodeticWorldCoordinate, SensorModel

from .imaged_feature_property_accessor import ImagedFeaturePropertyAccessor

logger = logging.getLogger(__name__)


class Projector:
    """Projects GeoJSON features from geographic to image pixel coordinates.

    Symmetric inverse of Geolocator. Reads feature.geometry (lon/lat) and
    populates feature.properties["imageGeometry"] and ["imageBBox"] with
    pixel coordinates. Returns the subset of features that intersect the
    provided image bounds.
    """

    def __init__(
        self,
        property_accessor: ImagedFeaturePropertyAccessor,
        sensor_model: SensorModel,
        image_bounds: Tuple[float, float, float, float],
        elevation_model: Optional[ElevationModel] = None,
        force: bool = False,
    ) -> None:
        """
        Construct a Projector given the context objects necessary for performing the calculations.

        :param property_accessor: facade used to access standard properties of an imaged feature
        :param sensor_model: sensor model for the image
        :param image_bounds: pixel-space bounding box (min_x, min_y, max_x, max_y) defining the valid projection region
        :param elevation_model: optional external elevation model
        :param force: if True, re-project all features even if imageGeometry already exists
        """
        self.property_accessor = property_accessor
        self.sensor_model = sensor_model
        self.image_bounds = image_bounds
        self.elevation_model = elevation_model
        self.force = force
        self._bounds_box = shapely.geometry.box(image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3])

    def project_features(self, features: List[geojson.Feature]) -> List[geojson.Feature]:
        """Project features from geographic to image coordinates.

        For each feature:
        - If force=False and imageGeometry already exists (preferred property only), skip projection.
        - Otherwise, project geometry vertices through world_to_image().
        - Only features whose projected geometry intersects image_bounds are
          mutated (imageGeometry/imageBBox written) and included in the result.
        - Features that fail projection or fall outside bounds are left unmodified.

        :param features: list of GeoJSON features to project
        :return: the subset of features that intersect image_bounds
        """
        if not features:
            return []

        results = []
        for feature in features:
            projected = self._project_single_feature(feature)
            if projected:
                results.append(feature)

        return results

    def _project_single_feature(self, feature: geojson.Feature) -> bool:
        """Attempt to project a single feature. Returns True if it passes bounds filtering."""
        if not self.force:
            existing = ImagedFeaturePropertyAccessor.get_image_geometry(feature)
            if existing is not None:
                return self._check_bounds_with_existing(feature, existing)

        geojson_geometry = feature.get("geometry")
        if geojson_geometry is None:
            return False

        projected_geometry = self._project_geometry(geojson_geometry)
        if projected_geometry is None:
            return False

        image_bbox = self._compute_image_bbox(feature, projected_geometry)
        if image_bbox is None:
            return False

        bbox_box = shapely.geometry.box(image_bbox[0], image_bbox[1], image_bbox[2], image_bbox[3])
        if not bbox_box.intersects(self._bounds_box):
            return False

        if not projected_geometry.intersects(self._bounds_box):
            return False

        ImagedFeaturePropertyAccessor.set_image_geometry(feature, projected_geometry)
        ImagedFeaturePropertyAccessor.set_image_bbox(feature, bbox_box)
        return True

    def _check_bounds_with_existing(self, feature: geojson.Feature, existing_geometry: shapely.Geometry) -> bool:
        """Check if an existing imageGeometry intersects image_bounds."""
        image_bbox_geom = ImagedFeaturePropertyAccessor.get_image_bbox(feature)
        if image_bbox_geom is not None:
            if not image_bbox_geom.intersects(self._bounds_box):
                return False

        return existing_geometry.intersects(self._bounds_box)

    def _compute_image_bbox(self, feature: geojson.Feature, projected_geometry: shapely.Geometry) -> Optional[List[float]]:
        """Compute imageBBox from feature bbox (4-corner method) or geometry bounds."""
        feature_bbox = feature.get("bbox")
        if feature_bbox is not None and len(feature_bbox) >= 4:
            return self._project_bbox_corners(feature_bbox)
        return list(projected_geometry.bounds)

    def _project_bbox_corners(self, bbox: List[float]) -> Optional[List[float]]:
        """Project the 4 corners of a geographic bbox and compute the axis-aligned envelope."""
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
        corners = [
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, max_lat),
            (max_lon, min_lat),
        ]

        projected_xs = []
        projected_ys = []
        for lon_deg, lat_deg in corners:
            pixel = self._project_coordinate([lon_deg, lat_deg])
            if pixel is None:
                return None
            projected_xs.append(pixel[0])
            projected_ys.append(pixel[1])

        return [min(projected_xs), min(projected_ys), max(projected_xs), max(projected_ys)]

    def _project_geometry(self, geojson_geometry: dict) -> Optional[shapely.Geometry]:
        """Convert GeoJSON geometry dict to shapely geometry in pixel space.

        Handles: Point, LineString, Polygon (with interior rings), MultiPoint,
        MultiLineString, MultiPolygon, GeometryCollection (recursive).

        Returns None if any coordinate fails to project.
        """
        geom_type = geojson_geometry.get("type")
        coordinates = geojson_geometry.get("coordinates")

        if geom_type == "Point":
            pixel = self._project_coordinate(coordinates)
            if pixel is None:
                return None
            return shapely.geometry.Point(pixel)

        elif geom_type == "LineString":
            projected = self._project_coordinate_sequence(coordinates)
            if projected is None:
                return None
            return shapely.geometry.LineString(projected)

        elif geom_type == "Polygon":
            rings = []
            for ring in coordinates:
                projected_ring = self._project_coordinate_sequence(ring)
                if projected_ring is None:
                    return None
                rings.append(projected_ring)
            if not rings:
                return None
            return shapely.geometry.Polygon(shell=rings[0], holes=rings[1:])

        elif geom_type == "MultiPoint":
            points = []
            for coord in coordinates:
                pixel = self._project_coordinate(coord)
                if pixel is None:
                    return None
                points.append(pixel)
            return shapely.geometry.MultiPoint(points)

        elif geom_type == "MultiLineString":
            lines = []
            for line_coords in coordinates:
                projected = self._project_coordinate_sequence(line_coords)
                if projected is None:
                    return None
                lines.append(projected)
            return shapely.geometry.MultiLineString(lines)

        elif geom_type == "MultiPolygon":
            polygons = []
            for polygon_coords in coordinates:
                rings = []
                for ring in polygon_coords:
                    projected_ring = self._project_coordinate_sequence(ring)
                    if projected_ring is None:
                        return None
                    rings.append(projected_ring)
                if not rings:
                    return None
                polygons.append(shapely.geometry.Polygon(shell=rings[0], holes=rings[1:]))
            return shapely.geometry.MultiPolygon(polygons)

        elif geom_type == "GeometryCollection":
            geometries = geojson_geometry.get("geometries", [])
            projected_geoms = []
            for sub_geom in geometries:
                projected = self._project_geometry(sub_geom)
                if projected is None:
                    return None
                projected_geoms.append(projected)
            return shapely.geometry.GeometryCollection(projected_geoms)

        return None

    def _project_coordinate_sequence(self, coordinates: List) -> Optional[List[Tuple[float, float]]]:
        """Project a sequence of coordinates. Returns None if any fails."""
        result = []
        for coord in coordinates:
            pixel = self._project_coordinate(coord)
            if pixel is None:
                return None
            result.append(pixel)
        return result

    def _project_coordinate(self, coord: List) -> Optional[Tuple[float, float]]:
        """Project a single geographic coordinate to pixel space.

        Elevation precedence: explicit Z > elevation model > 0.0
        """
        lon_rad = math.radians(coord[0])
        lat_rad = math.radians(coord[1])

        world_coord = GeodeticWorldCoordinate([lon_rad, lat_rad, 0.0])
        if len(coord) > 2:
            world_coord.elevation = coord[2]
        elif self.elevation_model is not None:
            if not self.elevation_model.set_elevation(world_coord):
                logger.debug("Elevation model failed for (%f, %f), defaulting to 0.0", coord[0], coord[1])

        try:
            pixel = self.sensor_model.world_to_image(world_coord)
            return (pixel.x, pixel.y)
        except Exception:
            return None
