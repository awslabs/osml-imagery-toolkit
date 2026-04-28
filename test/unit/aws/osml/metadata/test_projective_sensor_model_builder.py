#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

from math import radians

from aws.osml.metadata.projective_sensor_model_builder import ProjectiveSensorModelBuilder
from aws.osml.photogrammetry import ImageCoordinate, ProjectiveSensorModel

# Real CSCRNA TRE field values extracted from test/data/sample-metadata-ms-rpc00b.xml.
# Hardcoded as an inline dict so this test does not depend on GDAL XML parsing.
SAMPLE_CSCRNA_DICT = {
    "PREDICT_CORNERS": "Y",
    "ULCNR_LAT": "+25.02860",
    "ULCNR_LONG": "+121.48749",
    "ULCNR_HT": "+00027.1",
    "URCNR_LAT": "+25.01000",
    "URCNR_LONG": "+121.68566",
    "URCNR_HT": "+00234.7",
    "LRCNR_LAT": "+24.91148",
    "LRCNR_LONG": "+121.68595",
    "LRCNR_HT": "+00403.1",
    "LLCNR_LAT": "+24.92772",
    "LLCNR_LONG": "+121.48975",
    "LLCNR_HT": "+00431.4",
}

FULL_IMAGE_WIDTH = 8820.0
FULL_IMAGE_HEIGHT = 5212.0


class TestProjectiveSensorModelBuilder:
    """Unit tests for ProjectiveSensorModelBuilder. Requirements: 4.1, 4.2, 4.3."""

    def test_successful_construction_produces_projective_sensor_model(self):
        """Build from valid CSCRNA dict produces a ProjectiveSensorModel instance."""
        tre_dicts = {"CSCRNA": SAMPLE_CSCRNA_DICT}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        model = builder.build()

        assert model is not None
        assert isinstance(model, ProjectiveSensorModel)

    def test_corner_coordinate_mapping_ul(self):
        """UL image corner (0, 0) maps to UL world coordinate."""
        tre_dicts = {"CSCRNA": SAMPLE_CSCRNA_DICT}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        model = builder.build()
        assert model is not None

        # The projective model should map the UL image corner back to approximately the UL world coordinate
        world = model.image_to_world(ImageCoordinate([0.0, 0.0]))
        expected_lon = radians(121.48749)
        expected_lat = radians(25.02860)
        assert abs(world.coordinate[0] - expected_lon) < 1e-4
        assert abs(world.coordinate[1] - expected_lat) < 1e-4

    def test_corner_coordinate_mapping_ur(self):
        """UR image corner (width, 0) maps to UR world coordinate."""
        tre_dicts = {"CSCRNA": SAMPLE_CSCRNA_DICT}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        model = builder.build()
        assert model is not None

        world = model.image_to_world(ImageCoordinate([FULL_IMAGE_WIDTH, 0.0]))
        expected_lon = radians(121.68566)
        expected_lat = radians(25.01000)
        assert abs(world.coordinate[0] - expected_lon) < 1e-4
        assert abs(world.coordinate[1] - expected_lat) < 1e-4

    def test_corner_coordinate_mapping_lr(self):
        """LR image corner (width, height) maps to LR world coordinate."""
        tre_dicts = {"CSCRNA": SAMPLE_CSCRNA_DICT}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        model = builder.build()
        assert model is not None

        world = model.image_to_world(ImageCoordinate([FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT]))
        expected_lon = radians(121.68595)
        expected_lat = radians(24.91148)
        assert abs(world.coordinate[0] - expected_lon) < 1e-4
        assert abs(world.coordinate[1] - expected_lat) < 1e-4

    def test_corner_coordinate_mapping_ll(self):
        """LL image corner (0, height) maps to LL world coordinate."""
        tre_dicts = {"CSCRNA": SAMPLE_CSCRNA_DICT}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        model = builder.build()
        assert model is not None

        world = model.image_to_world(ImageCoordinate([0.0, FULL_IMAGE_HEIGHT]))
        expected_lon = radians(121.48975)
        expected_lat = radians(24.92772)
        assert abs(world.coordinate[0] - expected_lon) < 1e-4
        assert abs(world.coordinate[1] - expected_lat) < 1e-4

    def test_returns_none_when_cscrna_key_missing(self):
        """No CSCRNA key in tre_dicts should return None."""
        tre_dicts = {"RPC00B": {"SUCCESS": "1"}}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        assert builder.build() is None

    def test_returns_none_when_tre_dicts_empty(self):
        """Empty tre_dicts should return None."""
        builder = ProjectiveSensorModelBuilder({}, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        assert builder.build() is None

    def test_returns_none_when_required_field_missing(self):
        """CSCRNA dict missing a required field should return None."""
        cscrna_dict = dict(SAMPLE_CSCRNA_DICT)
        del cscrna_dict["ULCNR_LAT"]
        tre_dicts = {"CSCRNA": cscrna_dict}
        builder = ProjectiveSensorModelBuilder(tre_dicts, FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT)
        assert builder.build() is None
