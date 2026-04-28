#  Copyright 2025-2026 Amazon.com, Inc. or its affiliates.

import numpy as np

from aws.osml.photogrammetry.transforms import ProjectiveTransform


class TestProjectiveTransformEstimate:
    """Unit tests for ProjectiveTransform.estimate and forward/inverse roundtrips."""

    def test_identity_transform(self):
        """When src == dst, the transform should be (near) identity."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        t = ProjectiveTransform.estimate(pts, pts)

        result = t.forward(pts)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_pure_translation(self):
        """A shift of all points by a constant offset."""
        src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        dst = src + np.array([10.0, 20.0])
        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-10)

    def test_pure_scaling(self):
        """Uniform scaling of coordinates."""
        src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        dst = src * 100.0
        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-8)

    def test_affine_rotation(self):
        """A 90-degree rotation about the origin."""
        src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # 90° CCW: (x,y) -> (-y, x)
        dst = np.array([[0.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]])
        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-10)

    def test_forward_inverse_roundtrip(self):
        """Forward then inverse should recover the original source points."""
        src = np.array([[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]])
        dst = np.array([[0.0, 0.0], [1024.0, 0.0], [1024.0, 768.0], [0.0, 768.0]])
        t = ProjectiveTransform.estimate(src, dst)

        fwd = t.forward(src)
        np.testing.assert_allclose(fwd, dst, atol=1e-8)

        recovered = t.inverse(fwd)
        np.testing.assert_allclose(recovered, src, atol=1e-8)

    def test_projective_trapezoid(self):
        """A true projective (non-affine) mapping — trapezoid to rectangle."""
        src = np.array([[1.0, 1.0], [3.0, 1.0], [4.0, 3.0], [0.0, 3.0]])
        dst = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-8)

        recovered = t.inverse(result)
        np.testing.assert_allclose(recovered, src, atol=1e-8)

    def test_more_than_4_points_least_squares(self):
        """With >4 points, estimate uses least squares and reproduces well."""
        src = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [1.0, 0.5],
                [0.5, 1.0],
                [0.0, 0.5],
            ]
        )
        # Affine: scale by 100, translate by (10, 20)
        dst = src * 100.0 + np.array([10.0, 20.0])
        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-6)


class TestProjectiveTransformNumericalStability:
    """Tests verifying Hartley normalization handles poorly-conditioned inputs."""

    def test_tiny_footprint_in_radians(self):
        """Coordinates with large absolute values but tiny range (the small.ntf case)."""
        # Simulates lon/lat in radians: absolute ~1.48, range ~4.85e-6
        center_x = 1.483532
        center_y = 0.575665
        half_w = 2.426e-6
        half_h = 2.417e-6

        src = np.array(
            [
                [center_x - half_w, center_y + half_h],
                [center_x + half_w, center_y + half_h],
                [center_x + half_w, center_y - half_h],
                [center_x - half_w, center_y - half_h],
            ]
        )
        dst = np.array([[0.0, 0.0], [1024.0, 0.0], [1024.0, 1024.0], [0.0, 1024.0]])

        t = ProjectiveTransform.estimate(src, dst)

        # Forward must reproduce the destination corners exactly
        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-4)

        # Inverse must recover the source corners
        recovered = t.inverse(dst)
        np.testing.assert_allclose(recovered, src, atol=1e-12)

    def test_large_pixel_coordinates(self):
        """Large image (43008 x 41984 pixels) with normal geographic extent."""
        src = np.array(
            [
                [2.1145, 0.3975],
                [2.1160, 0.3975],
                [2.1160, 0.3960],
                [2.1145, 0.3960],
            ]
        )
        dst = np.array(
            [
                [0.0, 0.0],
                [43008.0, 0.0],
                [43008.0, 41984.0],
                [0.0, 41984.0],
            ]
        )

        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-4)

        recovered = t.inverse(dst)
        np.testing.assert_allclose(recovered, src, atol=1e-10)

    def test_very_small_source_range(self):
        """Source range of ~1e-8 relative to absolute value of ~100."""
        base = 100.0
        eps = 1e-8

        src = np.array(
            [
                [base, base],
                [base + eps, base],
                [base + eps, base + eps],
                [base, base + eps],
            ]
        )
        dst = np.array([[0.0, 0.0], [512.0, 0.0], [512.0, 512.0], [0.0, 512.0]])

        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-2)

        recovered = t.inverse(dst)
        np.testing.assert_allclose(recovered, src, atol=1e-14)

    def test_asymmetric_scale(self):
        """Source with very different x and y scales."""
        src = np.array(
            [
                [0.001, 1000.0],
                [0.002, 1000.0],
                [0.002, 2000.0],
                [0.001, 2000.0],
            ]
        )
        dst = np.array([[0.0, 0.0], [256.0, 0.0], [256.0, 256.0], [0.0, 256.0]])

        t = ProjectiveTransform.estimate(src, dst)

        result = t.forward(src)
        np.testing.assert_allclose(result, dst, atol=1e-6)

        recovered = t.inverse(dst)
        np.testing.assert_allclose(recovered, src, atol=1e-10)
