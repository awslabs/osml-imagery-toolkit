#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import numpy as np


class TestMathUtils(unittest.TestCase):
    def test_equilateral_triangle(self):
        from aws.osml.photogrammetry.math_utils import equilateral_triangle

        triangle = equilateral_triangle([10, 10], 5)
        assert len(triangle) == 3
        assert np.allclose(triangle[0], [10.0, 12.886751345948129])
        assert np.allclose(triangle[1], [7.5, 8.556624327025936])
        assert np.allclose(triangle[2], [12.5, 8.556624327025936])


if __name__ == "__main__":
    unittest.main()
