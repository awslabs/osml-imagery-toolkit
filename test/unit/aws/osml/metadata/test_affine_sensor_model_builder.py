#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from math import radians

from aws.osml.metadata.affine_sensor_model_builder import AffineSensorModelBuilder
from aws.osml.photogrammetry import AffineSensorModel, ImageCoordinate


class TestAffineSensorModelBuilder:
    """Unit tests for AffineSensorModelBuilder. Requirements: 7.1, 7.2."""

    def test_successful_construction_produces_affine_sensor_model(self):
        """Build from a valid geo transform produces a AffineSensorModel."""
        geo_transform = [0.0, 0.002, 0.0, 0.0, 0.0, 0.003]
        builder = AffineSensorModelBuilder(geo_transform)
        model = builder.build()

        assert model is not None
        assert isinstance(model, AffineSensorModel)

    def test_known_coordinate_transform(self):
        """A simple geo transform maps pixel (200, 300) to the expected world coordinate."""
        geo_transform = [0.0, 0.002, 0.0, 0.0, 0.0, 0.003]
        builder = AffineSensorModelBuilder(geo_transform)
        model = builder.build()
        assert model is not None

        world = model.image_to_world(ImageCoordinate([200, 300]))
        # With this transform: lon = 0.002*200 = 0.4 degrees, lat = 0.003*300 = 0.9 degrees
        assert abs(world.coordinate[0] - radians(0.4)) < 1e-6
        assert abs(world.coordinate[1] - radians(0.9)) < 1e-6

    def test_with_proj_wkt(self):
        """Build with a CRS WKT string produces a valid model."""
        geo_transform = [0.0, 0.002, 0.0, 0.0, 0.0, 0.003]
        proj_wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
        builder = AffineSensorModelBuilder(geo_transform, proj_wkt=proj_wkt)
        model = builder.build()

        assert model is not None
        assert isinstance(model, AffineSensorModel)

    def test_returns_none_for_none_geo_transform(self):
        """None geo_transform should return None."""
        builder = AffineSensorModelBuilder(None)
        assert builder.build() is None

    def test_real_geo_transform(self):
        """A realistic geo transform produces correct image-to-world mapping."""
        # Real-world-like transform: origin at (9.0, 52.0), pixel size ~0.001 degrees
        geo_transform = [9.0, 0.001, 0.0, 52.0, 0.0, -0.001]
        builder = AffineSensorModelBuilder(geo_transform)
        model = builder.build()
        assert model is not None

        # Pixel (50, 50) should map to approximately (9.05, 51.95)
        world = model.image_to_world(ImageCoordinate([50, 50]))
        assert abs(world.coordinate[0] - radians(9.05)) < 1e-6
        assert abs(world.coordinate[1] - radians(51.95)) < 1e-6
