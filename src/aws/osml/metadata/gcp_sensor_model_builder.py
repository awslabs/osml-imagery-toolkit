#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import logging
from dataclasses import dataclass
from math import radians
from typing import List, Optional

from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, ProjectiveSensorModel

from .sensor_model_builder import SensorModelBuilder

logger = logging.getLogger(__name__)


@dataclass
class GroundControlPoint:
    """A ground control point mapping image coordinates to world coordinates."""

    image_x: float
    image_y: float
    world_longitude: float
    world_latitude: float
    world_elevation: float = 0.0


class GCPSensorModelBuilder(SensorModelBuilder):
    """
    This builder constructs a ProjectiveSensorModel from a list of ground control
    point correspondences. It replaces the GDAL-based GCPSensorModelBuilder by
    accepting a list of GroundControlPoint dataclass instances instead of gdal.GCP
    objects.

    Handles degenerate cases:
    - Fewer than 3 distinct world-coordinate points → returns None
    - Self-intersecting (bowtie) polygons for exactly 4 GCPs → returns None
    - Adjacent duplicate points (triangle patterns) → perturbs them apart before solving
    """

    def __init__(self, ground_control_points: List[GroundControlPoint]) -> None:
        """
        Construct the builder with the required ground control points.

        :param ground_control_points: list of GCP correspondences

        :return: None
        """
        super().__init__()
        self.ground_control_points = ground_control_points

    def build(self) -> Optional[ProjectiveSensorModel]:
        """
        Use the GCPs to construct a projective sensor model.

        Requires at least 4 ground control points to estimate the projective
        transform. Returns None if the list is None, empty, has fewer than
        4 points, or has geometrically degenerate world coordinates.

        :return: a ProjectiveSensorModel, or None if GCPs are insufficient or degenerate
        """
        if not self.ground_control_points or len(self.ground_control_points) < 4:
            return None

        gcps = self.ground_control_points

        if not _validate_gcp_geometry(gcps):
            return None

        gcps = _split_coincident_adjacent_gcps(gcps)

        world_coordinates = [
            GeodeticWorldCoordinate([radians(gcp.world_longitude), radians(gcp.world_latitude), gcp.world_elevation])
            for gcp in gcps
        ]
        image_coordinates = [ImageCoordinate([gcp.image_x, gcp.image_y]) for gcp in gcps]
        return ProjectiveSensorModel(world_coordinates, image_coordinates)


def _validate_gcp_geometry(gcps: List[GroundControlPoint]) -> bool:
    """
    Validate that GCPs have sufficient geometric diversity for a projective solve.

    Rejects if:
    - Fewer than 3 distinct world-coordinate positions
    - Exactly 4 GCPs forming a self-intersecting (bowtie) polygon (signed area ~ 0)

    :param gcps: list of GroundControlPoint instances
    :return: True if GCPs are geometrically adequate
    """
    lons = [gcp.world_longitude for gcp in gcps]
    lats = [gcp.world_latitude for gcp in gcps]

    distinct = []
    for lon, lat in zip(lons, lats):
        is_dup = False
        for dlon, dlat in distinct:
            if abs(lon - dlon) < 1e-10 and abs(lat - dlat) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            distinct.append((lon, lat))

    if len(distinct) < 3:
        logger.debug("GCPs have fewer than 3 distinct world-coordinate positions")
        return False

    # For exactly 4 GCPs, check for self-intersecting polygon via signed area
    if len(gcps) == 4 and len(distinct) == 4:
        signed_area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            signed_area += lons[i] * lats[j] - lons[j] * lats[i]
        if abs(signed_area) < 1e-10:
            logger.debug("4 GCPs form a degenerate or self-intersecting polygon")
            return False

    return True


def _split_coincident_adjacent_gcps(gcps: List[GroundControlPoint]) -> List[GroundControlPoint]:
    """
    Handle triangle patterns by splitting coincident adjacent GCPs.

    When two adjacent GCPs share the same world coordinate but map to different
    image locations, the projective transform is underdetermined. This function
    perturbs coincident adjacent points slightly apart along the direction of the
    opposite edge, preserving the triangular footprint while giving the solver
    4 distinct source points.

    Only operates on lists of exactly 4 GCPs (the projective quadrilateral case).

    :param gcps: list of GroundControlPoint instances
    :return: list with coincident adjacent pairs perturbed apart
    """
    if len(gcps) != 4:
        return gcps

    for i in range(4):
        j = (i + 1) % 4
        if (
            abs(gcps[i].world_latitude - gcps[j].world_latitude) < 1e-10
            and abs(gcps[i].world_longitude - gcps[j].world_longitude) < 1e-10
        ):
            opp_a = (i + 2) % 4
            opp_b = (i + 3) % 4
            dlat = gcps[opp_a].world_latitude - gcps[opp_b].world_latitude
            dlon = gcps[opp_a].world_longitude - gcps[opp_b].world_longitude

            eps_lat = dlat * 0.001
            eps_lon = dlon * 0.001

            gcps = list(gcps)
            gcps[i] = GroundControlPoint(
                image_x=gcps[i].image_x,
                image_y=gcps[i].image_y,
                world_longitude=gcps[i].world_longitude - eps_lon,
                world_latitude=gcps[i].world_latitude - eps_lat,
                world_elevation=gcps[i].world_elevation,
            )
            gcps[j] = GroundControlPoint(
                image_x=gcps[j].image_x,
                image_y=gcps[j].image_y,
                world_longitude=gcps[j].world_longitude + eps_lon,
                world_latitude=gcps[j].world_latitude + eps_lat,
                world_elevation=gcps[j].world_elevation,
            )
            break

    return gcps
