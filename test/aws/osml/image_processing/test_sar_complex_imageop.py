#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase

import numpy as np
from osgeo import gdal

from aws.osml.gdal import load_gdal_dataset
from aws.osml.image_processing import histogram_stretch, quarter_power_image
from aws.osml.image_processing.sar_complex_imageop import image_pixels_to_complex, linear_mapping, linear_mapping_complex

gdal.UseExceptions()


class TestSARImageOPs(TestCase):
    def test_histogram_stretch(self):
        dataset, sensor_model = load_gdal_dataset("./test/data/sicd/capella-sicd121-chip1.ntf")
        pixels = dataset.ReadAsArray()

        # Sanity check to ensure the image needs to be scaled
        self.assertLessEqual(np.min(pixels), 0.0)
        self.assertGreaterEqual(np.max(pixels), 255.0)

        grayscale = histogram_stretch(pixels)

        min_grayscale_value = np.min(grayscale)
        self.assertGreaterEqual(min_grayscale_value, 0)
        self.assertLessEqual(min_grayscale_value, 15)

        max_grayscale_value = np.max(grayscale)
        self.assertGreaterEqual(max_grayscale_value, 240)
        self.assertLessEqual(max_grayscale_value, 255)

    def test_quarter_power_image(self):
        dataset, sensor_model = load_gdal_dataset("./test/data/sicd/umbra-sicd121-chip1.ntf")
        pixels = dataset.ReadAsArray()

        # Sanity check to ensure the image needs to be scaled
        self.assertLessEqual(np.min(pixels), -1.0)
        self.assertGreaterEqual(np.max(pixels), 3.0)

        grayscale = quarter_power_image(pixels)

        min_grayscale_value = np.min(grayscale)
        self.assertGreaterEqual(min_grayscale_value, 0)
        self.assertLessEqual(min_grayscale_value, 15)

        max_grayscale_value = np.max(grayscale)
        self.assertGreaterEqual(max_grayscale_value, 240)
        self.assertLessEqual(max_grayscale_value, 255)

    def test_linear_mapping(self):
        dataset, sensor_model = load_gdal_dataset("./test/data/sicd/capella-sicd121-chip2.ntf")
        pixels = dataset.ReadAsArray()

        # Sanity check to ensure the image needs to be scaled
        self.assertLessEqual(np.min(pixels), 0.0)
        self.assertGreaterEqual(np.max(pixels), 255.0)

        linear = linear_mapping_complex(pixels)
        min_value = np.min(linear)
        self.assertGreaterEqual(min_value, 0.0)
        self.assertLessEqual(min_value, 0.02)

        max_value = np.max(linear)
        self.assertGreaterEqual(max_value, 0.98)
        self.assertLessEqual(max_value, 1.0)

    def test_linear_mapping_all_same(self):
        fake_pixels = np.full((3, 3), 42.0)
        linear = linear_mapping(fake_pixels)
        self.assertTrue(np.allclose(linear, np.full(fake_pixels.shape, 0.5)))

    def test_pixel_lut(self):
        amp = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        phase = np.full((3, 3), 256)
        fake_pixels = np.array([amp, phase])
        lut = [10, 11, 12, 13, 14, 15, 16, 17, 18]

        fake_complex = image_pixels_to_complex(fake_pixels, pixel_type="AMP8I_PHS8I", amplitude_table=lut)
        self.assertTrue(np.allclose(fake_complex[1], np.full((3, 3), 0.0)))
        self.assertTrue(np.allclose(fake_complex[0], np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])))
