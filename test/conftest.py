#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2026-2026 General Atomics Integrated Intelligence, Inc.

# Place test specific configurations and fixtures here

from typing import Tuple

import pytest

from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
from aws.osml.photogrammetry.elevation_model import ElevationModel


@pytest.fixture(scope="class")
def fake_minimization(request):
    """Create a fake Earth intersection error function to minimize."""

    def _fake_minimization(
        self,
        image_coordinate: ImageCoordinate,
    ):
        def _inner_minimization(
            lonlat_coord: Tuple[float, float],
            elevation_model: ElevationModel,
        ):
            current_world_coordinate = GeodeticWorldCoordinate([*lonlat_coord, 0.0])
            elevation_model.set_elevation(current_world_coordinate)
            lon, lat, z = current_world_coordinate.coordinate
            # Parallel LOS down from right, varying longitude aligned latitude.
            x, y = (100 * lat, 100 * lon - z)
            return ((x - image_coordinate.x) ** 2 + (y - image_coordinate.y) ** 2) ** 0.5

        return _inner_minimization

    request.cls.fake_minimization = _fake_minimization
