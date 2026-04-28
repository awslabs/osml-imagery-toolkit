#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import math
from typing import List, Optional, Tuple, Union

import geojson
import numpy as np
import shapely
from scipy.interpolate import RectBivariateSpline

from aws.osml.photogrammetry import ElevationModel, GeodeticWorldCoordinate, ImageCoordinate, SensorModel

from .imaged_feature_property_accessor import ImagedFeaturePropertyAccessor


class LocationGridInterpolator:
    """
    This class can be used to approximate geodetic world coordinates from a grid of correspondences that is
    computed over a given area.
    """

    def __init__(
        self,
        sensor_model: SensorModel,
        elevation_model: Optional[ElevationModel],
        grid_area_ulx: float,
        grid_area_uly: float,
        grid_area_width: float,
        grid_area_height: float,
        grid_resolution: int,
    ) -> None:
        """
        Construct the grid of correspondences of the requested size/resolution from using the sensor model provided.

        :param sensor_model: SensorModel = the sensor model for the image
        :param elevation_model: Optional[Elevationmodel] = an optional external elevation model
        :param grid_area_ulx: float = the x component of the upper left corner of the grid in pixel space
        :param grid_area_uly: float = the y component of the upper left corner of the grid in pixel space
        :param grid_area_width: float = the width of the grid in pixels
        :param grid_area_height: float = the height of the grid in pixels
        :param grid_resolution: int = the number of points to calculate across the grid. Total points will be resolution^^2

        :return: None
        """
        xs = np.linspace(grid_area_ulx, grid_area_ulx + grid_area_width, grid_resolution)
        ys = np.linspace(grid_area_uly, grid_area_uly + grid_area_height, grid_resolution)
        longitude_values = np.empty(len(xs) * len(ys))
        latitude_values = np.empty(len(longitude_values))
        elevation_values = np.empty(len(longitude_values))
        i = 0
        for x in xs:
            for y in ys:
                world_coordinate = sensor_model.image_to_world(ImageCoordinate([x, y]), elevation_model=elevation_model)
                longitude_values[i] = world_coordinate.longitude
                latitude_values[i] = world_coordinate.latitude
                elevation_values[i] = world_coordinate.elevation
                i += 1
        longitude_values.shape = len(xs), len(ys)
        latitude_values.shape = longitude_values.shape
        elevation_values.shape = longitude_values.shape

        self.longitude_interpolator = RectBivariateSpline(xs, ys, longitude_values, kx=1, ky=1)
        self.latitude_interpolator = RectBivariateSpline(xs, ys, latitude_values, kx=1, ky=1)
        self.elevation_interpolator = RectBivariateSpline(xs, ys, elevation_values, kx=1, ky=1)
        self.elevation_model = elevation_model

    def __call__(self, *args, **kwargs):
        """
        Call this interpolation function given an image coordinate array.

        :param args: a single argument for the coordinate array
        :param kwargs: not used
        :return: a GeodeticWorldCoordinate for that image location
        """
        image_coord = args[0]
        world_coord = [
            self.longitude_interpolator(image_coord[0], image_coord[1])[0][0],
            self.latitude_interpolator(image_coord[0], image_coord[1])[0][0],
            self.elevation_interpolator(image_coord[0], image_coord[1])[0][0],
        ]
        world_coordinate = GeodeticWorldCoordinate(world_coord)
        if self.elevation_model is not None:
            self.elevation_model.set_elevation(world_coordinate)
        return world_coordinate


class Geolocator:
    """
    A Geolocator is a class that assign geographic coordinates for the features that are currently defined in image
    coordinates.
    """

    def __init__(
        self,
        property_accessor: ImagedFeaturePropertyAccessor,
        sensor_model: SensorModel,
        elevation_model: Optional[ElevationModel] = None,
        approximation_grid_size: int = 11,
        force: bool = False,
    ) -> None:
        """
        Construct a geolocator given the context objects necessary for performing the calculations.

        :param property_accessor: facade used to access standard properties of an imaged feature
        :param sensor_model: sensor model for the image
        :param elevation_model: external elevation model
        :param approximation_grid_size: resolution of the approximation grid to use
        :param force: if True, re-geolocate all features even if they already have geometry set

        :return: None
        """
        self.sensor_model = sensor_model
        self.property_accessor = property_accessor
        self.elevation_model = elevation_model
        self.approximation_grid_size = approximation_grid_size
        self.force = force

    def geolocate_features(self, features: List[geojson.Feature]) -> None:
        """
        Update the features to contain additional information from the context provided.

        When force=False (default), features that already have a non-None geometry property are skipped.
        When force=True, all features are re-geolocated regardless of existing geometry.

        :param features: List[geojson.Feature] = the input features to refine
        :return: None, the features are updated in place
        """

        if not features:
            return

        if self.force:
            features_to_process = features
        else:
            features_to_process = [f for f in features if f.get("geometry") is None]

        if not features_to_process:
            return

        self._geolocate_features_using_approximation_grid(features_to_process)

    def _geolocate_features_using_approximation_grid(self, features: List[geojson.Feature]) -> None:
        """
        This method computes geolocations for features using an approximation grid. It is useful for dense feature
        sets where the cost of computing precise locations for a set of close features is expensive and unnecessary.
        Here we first compute a grid of locations covering the features in question that is then used to efficiently
        assign geolocations for all the features.

        :param features: List[geojson.Feature] = the input features
        :return: None, but the individual features have their geometry property updated
        """

        # Compute the boundary of these features. Normally this will be similar to the tile boundary but the
        # interpolation needs to be setup to cover the entire extent, so we calculate it explicitly here. If the
        # features happen to be very tightly packed and only occupy a small portion of the tile we will gain some
        # benefit by creating the same resolution of approximation grid over the smaller area.
        feature_bounds = [math.inf, math.inf, -math.inf, -math.inf]
        for feature in features:
            image_geometry = self.property_accessor.find_image_geometry(feature)
            if image_geometry is None:
                continue
            image_geometry_bbox = image_geometry.bounds
            feature_bounds[0] = min(feature_bounds[0], image_geometry_bbox[0])
            feature_bounds[1] = min(feature_bounds[1], image_geometry_bbox[1])
            feature_bounds[2] = max(feature_bounds[2], image_geometry_bbox[2])
            feature_bounds[3] = max(feature_bounds[3], image_geometry_bbox[3])

        # Expand the bounds to handle edge case where the features are all located at a single point or line.
        feature_bounds[0] -= 10
        feature_bounds[1] -= 10
        feature_bounds[2] += 10
        feature_bounds[3] += 10

        # Use the feature boundary to set up an approximation grid for the region
        grid_area_ulx = feature_bounds[0]
        grid_area_uly = feature_bounds[1]
        grid_area_width = feature_bounds[2] - feature_bounds[0]
        grid_area_height = feature_bounds[3] - feature_bounds[1]
        tile_interpolation_grid = LocationGridInterpolator(
            self.sensor_model,
            self.elevation_model,
            grid_area_ulx,
            grid_area_uly,
            grid_area_width,
            grid_area_height,
            self.approximation_grid_size,
        )

        for feature in features:
            # If the feature has the "imageBBox" property set then we will convert it to the "bbox" property defined
            # in the GeoJSON spec. This is a [minx, miny, maxx, maxy] bounds for this object where x is degrees
            # longitude and y is degrees latitude.
            image_bbox = self.property_accessor.get_image_bbox(feature)
            if image_bbox is not None:
                bbox = image_bbox.bounds
                center_xy = [
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                ]
                bbox_corners_image_coords = [
                    [bbox[0], bbox[1]],
                    [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]],
                ]
                bbox_corners_world_coords = [
                    Geolocator.radians_coordinate_to_degrees(tile_interpolation_grid(corner))
                    for corner in bbox_corners_image_coords
                ]
                bbox_corners_world_geometry = shapely.MultiPoint(bbox_corners_world_coords)
                feature["bbox"] = bbox_corners_world_geometry.bounds

            # If the feature has the "imageGeometry" property set then we will convert it to the "geometry" property
            # defined in the GeoJSON spec. The "geometry" property will have the same type (e.g. Point, LineString,
            # Polygon) as the "imageGeometry" property. If the feature does not have the "imageGeometry" property
            # defined this
            image_geometry = self.property_accessor.get_image_geometry(feature)
            if image_bbox is None and image_geometry is None:
                logging.info(f"Feature may be using deprecated attributes: {feature}")
                image_geometry = self.property_accessor.find_image_geometry(feature)
                if image_geometry is None:
                    logging.warning(f"There isn't a valid detection shape for feature: {feature}")
                    continue

            if image_geometry is not None:
                center_xy = (image_geometry.centroid.x, image_geometry.centroid.y)
                feature["geometry"] = self._geolocate_image_geometry(image_geometry, tile_interpolation_grid)

            # Adding these because some visualization tools (e.g. kepler.gl) can perform more
            # advanced rendering (e.g. cluster layers) if the data points have single coordinates.
            center_location = tile_interpolation_grid(center_xy)
            feature["properties"]["center_longitude"] = math.degrees(center_location.longitude)
            feature["properties"]["center_latitude"] = math.degrees(center_location.latitude)

    @staticmethod
    def _geolocate_image_geometry(
        image_geometry: shapely.Geometry, interpolation_grid: LocationGridInterpolator
    ) -> Union[geojson.geometry.Geometry, geojson.GeometryCollection]:
        """
        This function converts a shape in image coordinates into a GeoJSON geometry using an interpolation grid.
        The image geometry, represented as a shapely Geometry object, is assumed to have coordinates in [x, y]
        pixel values. The resulting feature geometry, represented as a geojson Geometry object, will have coordinates
        in [longitude, latitude, elevation] where longitude and latitude are in degrees and elevation is in meters.

        :param image_geometry: the image shape
        :param interpolation_grid: the interpolation grid derived from the sensor model
        :return: a GeoJSON geometry object
        """
        if image_geometry is None:
            raise ValueError("Unable to geolocate features without a geometry property")

        if isinstance(image_geometry, shapely.Point):
            logging.debug("Geolocating point: %s", image_geometry.coords[0])
            return geojson.Point(Geolocator.radians_coordinate_to_degrees(interpolation_grid(image_geometry.coords[0])))
        elif isinstance(image_geometry, (shapely.LineString, shapely.LinearRing)):
            return geojson.LineString(
                [Geolocator.radians_coordinate_to_degrees(interpolation_grid(coord)) for coord in image_geometry.coords]
            )
        elif isinstance(image_geometry, shapely.Polygon):
            boundary_coords = [
                [
                    Geolocator.radians_coordinate_to_degrees(interpolation_grid(coord))
                    for coord in image_geometry.exterior.coords
                ]
            ]
            for i in image_geometry.interiors:
                boundary_coords.append(
                    [Geolocator.radians_coordinate_to_degrees(interpolation_grid(coord)) for coord in i.coords]
                )
            return geojson.Polygon(boundary_coords)
        elif (
            image_geometry.__class__.__name__.startswith("Multi")
            or image_geometry.__class__.__name__ == "GeometryCollection"
        ):
            geometry_list = [Geolocator._geolocate_image_geometry(part, interpolation_grid) for part in image_geometry.geoms]
            logging.debug("Geolocated %s with %d parts", image_geometry.__class__.__name__, len(geometry_list))
            return getattr(geojson, image_geometry.__class__.__name__)(geometry_list)
        else:
            raise ValueError(f"Unhandled geometry type: {image_geometry.__class__}")

    @staticmethod
    def radians_coordinate_to_degrees(
        coordinate: GeodeticWorldCoordinate,
    ) -> Tuple[float, float, float]:
        """
        GeoJSON coordinate order is a decimal longitude, latitude with an optional height as a 3rd value
        (i.e. [lon, lat, ht]). The WorldCoordinate uses the same ordering but the longitude and latitude are expressed
        in radians rather than degrees.

        :param coordinate: GeodeticWorldCoordinate = the geodetic world coordinate (longitude, latitude, elevation)

        :return: Tuple[float, float, float] = degrees(longitude), degrees(latitude), elevation
        """
        return (
            math.degrees(coordinate.longitude),
            math.degrees(coordinate.latitude),
            coordinate.elevation,
        )
