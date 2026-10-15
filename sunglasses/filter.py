"""Stateless highlight shoulder. Input/output are linear-light sRGB, not nits."""

from dataclasses import dataclass
import math

import numpy as np

from .color import luminance, normalized_rgb
from .i18n import ValidationError


@dataclass(frozen=True)
class FilterConfig:
    knee: float = 0.20
    ceiling: float = 0.55
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.knee, self.ceiling, self.strength)):
            raise ValidationError("parameters_finite")
        if not 0 < self.knee < self.ceiling <= 1:
            raise ValidationError("parameters_range")
        if not 0 <= self.strength <= 1:
            raise ValidationError("strength_range")


def compress_highlights(linear_rgb: np.ndarray, config: FilterConfig = FilterConfig()) -> np.ndarray:
    rgb = normalized_rgb(linear_rgb)
    if config.strength == 0:
        return rgb.copy()
    y = luminance(rgb)
    distance = np.maximum(y - config.knee, 0)
    span = config.ceiling - config.knee
    shoulder = config.knee - span * np.expm1(-distance / span)
    mapped = np.where(y > config.knee, y + config.strength * (shoulder - y), y)
    gain = np.ones_like(y)
    np.divide(mapped, y, out=gain, where=y > config.knee)
    # Roundoff must never increase an RGB channel or produce out-of-gamut SDR.
    return rgb * np.clip(gain, 0, 1)[..., None]
