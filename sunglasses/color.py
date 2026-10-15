"""Normalized SDR sRGB / linear-light conversion. HDR is intentionally rejected."""

import numpy as np

from .i18n import ValidationError

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def normalized_rgb(values: np.ndarray) -> np.ndarray:
    rgb = np.asarray(values, dtype=np.float32)
    if rgb.ndim < 2 or rgb.shape[-1] != 3 or rgb.size == 0:
        raise ValidationError("rgb_shape")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValidationError("rgb_range")
    return rgb


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    rgb = normalized_rgb(values)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    rgb = normalized_rgb(values)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    return np.sum(linear_rgb * LUMA, axis=-1)
