#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from math import degrees, radians

import numpy as np

from aws.osml.metadata.gcp_sensor_model_builder import GCPSensorModelBuilder, GroundControlPoint
from aws.osml.photogrammetry import ImageCoordinate, ProjectiveSensorModel

# Real GCPs extracted from the existing GDAL GCP sensor model builder test.
# These correspond to the four corners of a real image.
SAMPLE_GCPS = [
    GroundControlPoint(image_x=0.5, image_y=0.5, world_longitude=121.67722222222223, world_latitude=13.924722222222222),
    GroundControlPoint(image_x=10239.5, image_y=0.5, world_longitude=121.8261111111111, world_latitude=13.91861111111111),
    GroundControlPoint(
        image_x=10239.5, image_y=4095.5, world_longitude=121.8261111111111, world_latitude=13.858333333333333
    ),
    GroundControlPoint(image_x=0.5, image_y=4095.5, world_longitude=121.67722222222223, world_latitude=13.864166666666666),
]


class TestGCPSensorModelBuilder:
    """Unit tests for GCPSensorModelBuilder. Requirements: 8.1, 8.2, 8.3."""

    def test_successful_construction_produces_projective_sensor_model(self):
        """Build from 4 valid GCPs produces a ProjectiveSensorModel."""
        builder = GCPSensorModelBuilder(SAMPLE_GCPS)
        model = builder.build()

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_known_coordinate_transform_image_center(self):
        """Image center (5120, 2048) maps to the expected geodetic coordinate."""
        builder = GCPSensorModelBuilder(SAMPLE_GCPS)
        model = builder.build()
        assert model is not None

        image_center = ImageCoordinate([5120, 2048])
        geodetic_image_center = model.image_to_world(image_center)

        assert np.allclose(
            geodetic_image_center.coordinate,
            np.array([radians(121.7518378), radians(13.89145147), 0.0]),
        )

    def test_more_than_4_gcps(self):
        """Build from more than 4 GCPs also produces a valid model."""
        extra_gcp = GroundControlPoint(image_x=5120.0, image_y=2048.0, world_longitude=121.7517, world_latitude=13.8915)
        gcps = list(SAMPLE_GCPS) + [extra_gcp]
        builder = GCPSensorModelBuilder(gcps)
        model = builder.build()

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_returns_none_for_fewer_than_4_points(self):
        """Fewer than 4 GCPs should return None."""
        builder = GCPSensorModelBuilder(SAMPLE_GCPS[:3])
        assert builder.build() is None

    def test_returns_none_for_empty_list(self):
        """Empty GCP list should return None."""
        builder = GCPSensorModelBuilder([])
        assert builder.build() is None

    def test_returns_none_for_none_input(self):
        """None GCP list should return None."""
        builder = GCPSensorModelBuilder(None)
        assert builder.build() is None

    def test_gcps_with_elevation(self):
        """GCPs with non-zero elevation are accepted and produce a valid model."""
        gcps_with_elev = [
            GroundControlPoint(
                image_x=gcp.image_x,
                image_y=gcp.image_y,
                world_longitude=gcp.world_longitude,
                world_latitude=gcp.world_latitude,
                world_elevation=100.0,
            )
            for gcp in SAMPLE_GCPS
        ]
        builder = GCPSensorModelBuilder(gcps_with_elev)
        model = builder.build()

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_returns_none_for_all_same_world_coordinate(self):
        """GCPs that all share the same world coordinate return None."""
        gcps = [
            GroundControlPoint(image_x=0.0, image_y=0.0, world_longitude=85.0, world_latitude=33.0),
            GroundControlPoint(image_x=1024.0, image_y=0.0, world_longitude=85.0, world_latitude=33.0),
            GroundControlPoint(image_x=1024.0, image_y=1024.0, world_longitude=85.0, world_latitude=33.0),
            GroundControlPoint(image_x=0.0, image_y=1024.0, world_longitude=85.0, world_latitude=33.0),
        ]
        builder = GCPSensorModelBuilder(gcps)
        assert builder.build() is None

    def test_returns_none_for_bowtie_polygon(self):
        """GCPs forming a self-intersecting (bowtie) polygon return None."""
        gcps = [
            GroundControlPoint(image_x=0.0, image_y=0.0, world_longitude=-110.350, world_latitude=31.580),
            GroundControlPoint(image_x=1024.0, image_y=0.0, world_longitude=-110.340, world_latitude=31.570),
            GroundControlPoint(image_x=1024.0, image_y=1024.0, world_longitude=-110.350, world_latitude=31.570),
            GroundControlPoint(image_x=0.0, image_y=1024.0, world_longitude=-110.340, world_latitude=31.580),
        ]
        builder = GCPSensorModelBuilder(gcps)
        assert builder.build() is None

    def test_triangle_pattern_produces_valid_model(self):
        """GCPs with two adjacent corners at same world point (triangle) produce a model."""
        gcps = [
            GroundControlPoint(image_x=0.0, image_y=0.0, world_longitude=-110.345, world_latitude=31.580),
            GroundControlPoint(image_x=1024.0, image_y=0.0, world_longitude=-110.345, world_latitude=31.580),
            GroundControlPoint(image_x=1024.0, image_y=1024.0, world_longitude=-110.340, world_latitude=31.570),
            GroundControlPoint(image_x=0.0, image_y=1024.0, world_longitude=-110.350, world_latitude=31.570),
        ]
        builder = GCPSensorModelBuilder(gcps)
        model = builder.build()

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

        # Top corners should both map near the apex
        ul = model.image_to_world(ImageCoordinate([0.0, 0.0]))
        ur = model.image_to_world(ImageCoordinate([1024.0, 0.0]))
        assert abs(degrees(ul.longitude) - (-110.345)) < 0.001
        assert abs(degrees(ur.longitude) - (-110.345)) < 0.001
