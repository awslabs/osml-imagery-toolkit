#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import geojson
import shapely


class TestImagedFeaturePropertiesAccessor(unittest.TestCase):
    def test_find_imagegeometry_point(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        point_feature = geojson.Feature(
            geometry=geojson.Point((-1.0, 2.0)),
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    ImagedFeaturePropertyAccessor.TYPE: "Point",
                    ImagedFeaturePropertyAccessor.COORDINATES: [5.1, 10.2],
                }
            },
        )

        image_geometry = accessor.find_image_geometry(point_feature)

        assert image_geometry == shapely.Point(5.1, 10.2)

    def test_find_imagegeometry_polygon(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        polygon_feature = geojson.Feature(
            geometry=geojson.Point((0.0, 0.0)),
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    ImagedFeaturePropertyAccessor.TYPE: "Polygon",
                    ImagedFeaturePropertyAccessor.COORDINATES: [
                        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
                    ],
                }
            },
        )

        image_geometry = accessor.find_image_geometry(polygon_feature)

        assert image_geometry == shapely.Polygon(shell=[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    def test_find_imagebbox(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        bbox_feature = geojson.Feature(
            geometry=geojson.Point((0.0, 0.0)), properties={ImagedFeaturePropertyAccessor.IMAGE_BBOX: [0.0, 0.0, 1.0, 1.0]}
        )

        image_geometry = accessor.find_image_geometry(bbox_feature)

        assert image_geometry == shapely.Polygon(shell=[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    def test_find_bounds_imcoords(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        bbox_feature = geojson.Feature(
            geometry=geojson.Point((0.0, 0.0)),
            properties={ImagedFeaturePropertyAccessor.BOUNDS_IMCORDS: [0.0, 0.0, 1.0, 1.0]},
        )

        image_geometry = accessor.find_image_geometry(bbox_feature)

        assert image_geometry == shapely.Polygon(shell=[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    def test_find_geom_imcoords(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        polygon_feature = geojson.Feature(
            geometry=geojson.Point((0.0, 0.0)),
            properties={
                ImagedFeaturePropertyAccessor.GEOM_IMCOORDS: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
            },
        )

        image_geometry = accessor.find_image_geometry(polygon_feature)

        assert image_geometry == shapely.Polygon(shell=[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    def test_find_detection(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        detection_feature = geojson.Feature(
            geometry=geojson.Point((0.0, 0.0)),
            properties={
                ImagedFeaturePropertyAccessor.DETECTION: {
                    ImagedFeaturePropertyAccessor.TYPE: "Polygon",
                    ImagedFeaturePropertyAccessor.PIXEL_COORDINATES: [
                        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
                    ],
                }
            },
        )

        image_geometry = accessor.find_image_geometry(detection_feature)

        assert image_geometry == shapely.Polygon(shell=[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    def test_update_all(self):
        from aws.osml.features import ImagedFeaturePropertyAccessor

        accessor = ImagedFeaturePropertyAccessor()

        image_feature = geojson.Feature(
            geometry=geojson.Point((0.0, 0.0)),
            properties={
                ImagedFeaturePropertyAccessor.DETECTION: {
                    ImagedFeaturePropertyAccessor.TYPE: "Polygon",
                    ImagedFeaturePropertyAccessor.PIXEL_COORDINATES: [
                        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
                    ],
                },
                ImagedFeaturePropertyAccessor.GEOM_IMCOORDS: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
                ImagedFeaturePropertyAccessor.BOUNDS_IMCORDS: [0.0, 0.0, 1.0, 1.0],
                ImagedFeaturePropertyAccessor.IMAGE_BBOX: [0.0, 0.0, 1.0, 1.0],
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    ImagedFeaturePropertyAccessor.TYPE: "Polygon",
                    ImagedFeaturePropertyAccessor.COORDINATES: [
                        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
                    ],
                },
            },
        )

        accessor.update_existing_image_geometries(image_feature, shapely.box(3.0, 4.0, 5.0, 6.0))

        assert image_feature.properties[ImagedFeaturePropertyAccessor.DETECTION][
            ImagedFeaturePropertyAccessor.PIXEL_COORDINATES
        ] != [[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]

        assert image_feature.properties[ImagedFeaturePropertyAccessor.IMAGE_BBOX] == [3.0, 4.0, 5.0, 6.0]
        assert image_feature.properties[ImagedFeaturePropertyAccessor.BOUNDS_IMCORDS] == [3.0, 4.0, 5.0, 6.0]
