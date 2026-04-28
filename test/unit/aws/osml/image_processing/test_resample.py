#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Unit tests for the OpenCV-based resamplers in ``resample.py``."""

import unittest

import cv2
import numpy as np

from aws.osml.image_processing.resample import (
    ResampleFunc,
    area_resample,
    bilinear_resample,
    lanczos_resample,
    nearest_neighbor_resample,
)

# Every resampler in the module paired with the OpenCV interpolation
# constant it wraps. Used to parametrize the reference-match, identity,
# layout, dtype, and mutation tests.
_RESAMPLERS = [
    ("nearest_neighbor", nearest_neighbor_resample, cv2.INTER_NEAREST),
    ("bilinear", bilinear_resample, cv2.INTER_LINEAR),
    ("area", area_resample, cv2.INTER_AREA),
    ("lanczos", lanczos_resample, cv2.INTER_LANCZOS4),
]


def _reference_chw(image: np.ndarray, target_rows: int, target_cols: int, interpolation: int) -> np.ndarray:
    """Call ``cv2.resize`` directly on a CHW array via an HWC transpose.

    This is what each resampler is expected to return, modulo the
    identity short-circuit — we use it as the oracle for the
    "matches cv2.resize" tests.
    """
    hwc = np.ascontiguousarray(image.transpose(1, 2, 0))
    resized = cv2.resize(hwc, (target_cols, target_rows), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    if resized.dtype != image.dtype:
        resized = resized.astype(image.dtype)
    return resized.transpose(2, 0, 1)


class TestResampleTypeAlias(unittest.TestCase):
    """The ``ResampleFunc`` type alias should cover every shipped resampler."""

    def test_resamplers_are_assignable_to_resample_func(self):
        # This is a structural (type-checker-style) check: assign each
        # callable to a name annotated as ResampleFunc. If the alias is
        # wrong, callers typed on it would break.
        fn: ResampleFunc
        for _, resampler, _ in _RESAMPLERS:
            fn = resampler
            self.assertTrue(callable(fn))


class TestResampleAgainstCv2Reference(unittest.TestCase):
    """Each resampler should match a direct ``cv2.resize`` reference."""

    def test_4x4_to_2x2_matches_cv2_resize(self):
        # Single-band 4x4 → 2x2. Using a float64 input so none of the
        # interpolation modes quantize away the reference value.
        src = np.arange(16, dtype=np.float64).reshape(1, 4, 4)
        for name, resampler, interpolation in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 2, 2)
                expected = _reference_chw(src, 2, 2, interpolation)
                self.assertEqual(result.shape, (1, 2, 2))
                np.testing.assert_array_equal(result, expected)

    def test_multiband_4x4_to_2x2_matches_cv2_resize(self):
        # Three-band CHW input with distinct values per band.
        rng = np.random.default_rng(0)
        src = rng.integers(0, 255, size=(3, 4, 4), dtype=np.uint8)
        for name, resampler, interpolation in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 2, 2)
                expected = _reference_chw(src, 2, 2, interpolation)
                self.assertEqual(result.shape, (3, 2, 2))
                np.testing.assert_array_equal(result, expected)


class TestResampleLayoutHandling(unittest.TestCase):
    """2-D inputs should round-trip as 2-D; 3-D inputs preserve band count."""

    def test_single_band_2d_input_returns_2d(self):
        src = np.arange(16, dtype=np.float32).reshape(4, 4)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 2, 2)
                self.assertEqual(result.ndim, 2)
                self.assertEqual(result.shape, (2, 2))

    def test_chw_multiband_preserves_band_count(self):
        # 5-band input — forces the HWC transpose path through the
        # multi-band branch.
        src = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 2, 2)
                self.assertEqual(result.shape, (5, 2, 2))

    def test_single_band_chw_input_returns_chw(self):
        # Input is (1, H, W) — cv2.resize would squeeze the channel
        # axis on HWC, so the resampler must re-add it.
        src = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 2, 2)
                self.assertEqual(result.shape, (1, 2, 2))


class TestResampleDtypePreservation(unittest.TestCase):
    """Output dtype must equal input dtype across common numeric types."""

    def test_dtype_preserved(self):
        # Pick dtypes that round-trip cleanly through cv2.resize. float16
        # is not supported by cv2.INTER_AREA for all cases, so we stick
        # to widely supported numeric types.
        dtypes = [np.uint8, np.uint16, np.int16, np.float32, np.float64]
        for dtype in dtypes:
            src = np.arange(16, dtype=dtype).reshape(1, 4, 4)
            for name, resampler, _ in _RESAMPLERS:
                with self.subTest(resampler=name, dtype=dtype):
                    result = resampler(src, 2, 2)
                    self.assertEqual(result.dtype, np.dtype(dtype))


class TestResampleIdentityCase(unittest.TestCase):
    """When target dims match source dims, output equals input element-wise."""

    def test_identity_chw(self):
        src = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 4, 5)
                np.testing.assert_array_equal(result, src)

    def test_identity_2d(self):
        src = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                result = resampler(src, 4, 5)
                np.testing.assert_array_equal(result, src)


class TestResampleDoesNotMutateInput(unittest.TestCase):
    """No resampler should write back into the caller's array."""

    def test_input_not_mutated(self):
        src = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                original = src.copy()
                _ = resampler(src, 2, 2)
                np.testing.assert_array_equal(src, original)

    def test_input_not_mutated_identity_case(self):
        # Identity short-circuits to return the input. The returned
        # array IS the input, which is fine — the caller just needs to
        # observe that no values changed.
        src = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                original = src.copy()
                _ = resampler(src, 4, 4)
                np.testing.assert_array_equal(src, original)


class TestResampleValidation(unittest.TestCase):
    """Error handling for invalid inputs."""

    def test_invalid_ndim_1d_raises(self):
        src = np.arange(8, dtype=np.float32)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                with self.assertRaises(ValueError):
                    resampler(src, 2, 2)

    def test_invalid_ndim_4d_raises(self):
        src = np.zeros((2, 3, 4, 4), dtype=np.float32)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                with self.assertRaises(ValueError):
                    resampler(src, 2, 2)

    def test_non_positive_target_rows_raises(self):
        src = np.zeros((1, 4, 4), dtype=np.float32)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                with self.assertRaises(ValueError):
                    resampler(src, 0, 2)
                with self.assertRaises(ValueError):
                    resampler(src, -1, 2)

    def test_non_positive_target_cols_raises(self):
        src = np.zeros((1, 4, 4), dtype=np.float32)
        for name, resampler, _ in _RESAMPLERS:
            with self.subTest(resampler=name):
                with self.assertRaises(ValueError):
                    resampler(src, 2, 0)
                with self.assertRaises(ValueError):
                    resampler(src, 2, -3)


if __name__ == "__main__":
    unittest.main()
