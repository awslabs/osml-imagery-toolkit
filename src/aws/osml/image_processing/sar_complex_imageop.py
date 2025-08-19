#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
from typing import Optional, Tuple

import numpy as np

TWO_PI = np.pi * 2.0

logger = logging.getLogger(__name__)


def image_pixels_to_complex(
    image_pixels: np.ndarray, pixel_type: Optional[str] = None, amplitude_table: Optional[np.typing.ArrayLike] = None
) -> np.ndarray:
    """
    This function converts SAR pixels from SICD imagery into complex values using equations
    found in SICD Volume 1 Section 4.2.

    :param image_pixels: the SAR image pixels
    :param pixel_type: "AMP8I_PHS8I", "RE32F_IM32F", or "RE16I_IM16I"
    :param amplitude_table: optional lookup table of amplitude values for AMP8I_PHS8I image pixels
    :return:
    """

    if pixel_type is None or pixel_type in ["RE32F_IM32F", "RE16I_IM16I"]:
        # For these pixel types the complex value is already stored in the file
        return image_pixels
    elif pixel_type == "AMP8I_PHS8I":
        # If the data is 8-bit amplitude/phase with an optional amplitude lookup table need to
        # convert it to the complex image value
        amplitude = image_pixels[0]
        phase = image_pixels[1] / 256.0
        if amplitude_table is not None:
            amplitude_lut = np.array(amplitude_table)
            amplitude = amplitude_lut[amplitude]
        return np.array([amplitude * np.cos(TWO_PI * phase), amplitude * np.sin(TWO_PI * phase)])
    else:
        raise ValueError(f"Unknown SAR Pixel Type: {pixel_type}")


def complex_to_power_value(complex_data: np.ndarray) -> np.ndarray:
    """
    This function converts SAR complex data into the pixel power values (sometimes
    called pixel intensity) using the equation found in SICD Volume 1 Section 4.10.

    :param complex_data: the SAR complex image signal with real and imaginary components
    :return: the power values
    """
    return np.sum(np.square(complex_data), axis=0)


def power_value_in_decibels(power_values: np.ndarray) -> np.ndarray:
    """
    This function converts SAR power values to decibels using the equation found in SICD Volume 1 Section 4.10.

    :param power_values: the SAR power values
    :return: the power values in decibels
    """
    return 10.0 * np.log10(power_values)


def get_value_bounds(magnitude_values: np.ndarray) -> Tuple[float, float]:
    """
    This function calculates the minimum and maximum of a set of values.

    :param magnitude_values: SAR magnitude values
    :return: (min value, max value)
    """
    return np.min(magnitude_values), np.max(magnitude_values)


def linear_mapping(magnitude_values: np.ndarray) -> np.ndarray:
    """
    This function accepts an array of magnitude values and scales them to be in the range [0:1].

    :param magnitude_values: SAR magnitude values
    :return: the scaled values in range [0:1]
    """
    min_value, max_value = get_value_bounds(magnitude_values)
    if max_value == min_value:
        return np.full(magnitude_values.shape, 0.5)

    return np.clip((magnitude_values.astype(np.float32) - min_value) / (np.float32(max_value) - min_value), 0, 1.0)


def histogram_stretch_mag_values(magnitude_values: np.ndarray, scale_factor: float = 8.0):
    """
    This function converts image pixel magnitudes to an 8-bit image by scaling the pixels and
    cropping to the desired range. This is histogram stretching without any gamma correction.

    :param magnitude_values: SAR magnitude values
    :param scale_factor: a scale factor, default = 8.0
    :return: the quantized grayscale image clipped to the range of [0:255]
    """
    mean_value = np.mean(magnitude_values[np.isfinite(magnitude_values)])
    u = 1 / (scale_factor * mean_value)
    return np.clip(255.0 * u * magnitude_values, 0.0, 255.0)


def quarter_power_mag_values(
    magnitude_values: np.ndarray, scale_factor: float = 3.0, precomputed_mean: Optional[float] = None
):
    """
    This function converts image pixel magnitudes to a Quarter-Power Image using equations
    found in Section 3.2 of SAR Image Scaling, Dynamic Range, Radiometric Calibration, and Display
    (SAND2019-2371).

    :param magnitude_values: SAR magnitude values
    :param scale_factor: a brightness factor that is typically between 5 and 3
    :param precomputed_mean: a precomputed mean of magnitude usually taken from a larger set of pixels
    :return: the quantized grayscale image clipped to the range of [0:255]
    """
    sqrt_magnitude = np.sqrt(np.abs(magnitude_values))
    mean_value = precomputed_mean
    if mean_value is None:
        mean_value = np.mean(sqrt_magnitude[np.isfinite(sqrt_magnitude)])
    b = 1 / (scale_factor * mean_value)
    return np.clip(255.0 * b * sqrt_magnitude, 0.0, 255.0)


def linear_mapping_complex(
    image_pixels: np.ndarray,
    pixel_type: Optional[str] = None,
    amplitude_table: Optional[np.typing.ArrayLike] = None,
) -> np.ndarray:
    """
    This function converts SAR complex image pixels to linearly mapped values in the range [0:1].
    It first converts the complex data to power values, then applies linear mapping.

    :param image_pixels: the SAR image pixels (can be complex data)
    :param pixel_type: "AMP8I_PHS8I", "RE32F_IM32F", or "RE16I_IM32F"
    :param amplitude_table: optional lookup table of amplitude values for AMP8I_PHS8I image pixels
    :return: the linearly mapped values in range [0:1]
    """
    complex_data = image_pixels_to_complex(image_pixels, pixel_type=pixel_type, amplitude_table=amplitude_table)
    power_values = complex_to_power_value(complex_data)
    return linear_mapping(power_values)


def histogram_stretch(
    image_pixels: np.ndarray,
    pixel_type: Optional[str] = None,
    amplitude_table: Optional[np.typing.ArrayLike] = None,
    scale_factor: float = 8.0,
) -> np.ndarray:
    """
    This function converts SAR image pixels to an 8-bit grayscale image by scaling the pixels and
    cropping to the desired range [0:255]. This is histogram stretching without any gamma correction.
    The equations are described in Section 3.1 of SAR Image Scaling, Dynamic Range, Radiometric Calibration,
    and Display (SAND2019-2371).

    :param image_pixels: the SAR image pixels
    :param pixel_type: "AMP8I_PHS8I", "RE32F_IM32F", or "RE16I_IM16I"
    :param amplitude_table: optional lookup table of amplitude values for AMP8I_PHS8I image pixels
    :param scale_factor: a scale factor, default = 8.0
    :return: the quantized grayscale image clipped to the range of [0:255]
    """
    complex_data = image_pixels_to_complex(image_pixels, pixel_type=pixel_type, amplitude_table=amplitude_table)
    power_values = complex_to_power_value(complex_data)
    return histogram_stretch_mag_values(power_values, scale_factor=scale_factor)


def quarter_power_image(
    image_pixels: np.ndarray,
    pixel_type: Optional[str] = None,
    amplitude_table: Optional[np.typing.ArrayLike] = None,
    scale_factor: float = 3.0,
    precomputed_mean: Optional[float] = None,
) -> np.ndarray:
    """
    This function converts SAR image pixels to an 8-bit grayscale image pixel magnitudes to a Quarter-Power
    Image using equations found in Section 3.2 of SAR Image Scaling, Dynamic Range, Radiometric Calibration,
    and Display (SAND2019-2371).

    :param image_pixels: the SAR image pixels
    :param pixel_type: "AMP8I_PHS8I", "RE32F_IM32F", or "RE16I_IM16I"
    :param amplitude_table: optional lookup table of amplitude values for AMP8I_PHS8I image pixels
    :param scale_factor: a brightness factor that is typically between 5 and 3
    :param precomputed_mean: a precomputed mean of magnitude usually taken from a larger set of pixels
    :return: the quantized grayscale image clipped to the range of [0:255]
    """
    complex_data = image_pixels_to_complex(image_pixels, pixel_type=pixel_type, amplitude_table=amplitude_table)
    power_values = complex_to_power_value(complex_data)
    return quarter_power_mag_values(power_values, scale_factor=scale_factor, precomputed_mean=precomputed_mean)
