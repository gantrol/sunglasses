"""Offline port of the Apple VideoFlashingReduction MATLAB temporal core.

This module is an independently written interoperability baseline around the
official MIT-licensed UFM kernel data at a pinned upstream commit.  It is not an
Apple product, a medical risk score, or a standards-conformance test.  The
reference assumes constant frame rate and whole-frame average luminance.

Inputs to :meth:`AppleVFMReference.process_linear_rgb` are normalized,
linear-light RGB.  The upstream MATLAB implementation starts from encoded
8-bit video and applies a power-2.2 decode; callers comparing video files must
perform that decode before using the temporal core.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import math

import numpy as np

from .color import normalized_rgb
from .i18n import ValidationError


UPSTREAM_COMMIT = "6e9604d92446c8e6763bc53c5e12ccc4e0046796"
KERNEL_RESOURCE = "apple_ufm_kernels_6e9604d.txt"
KERNEL_SHA256 = "209bc2e47a8a8dcaab6172787dc2721efe38a6b5d9d49fdeecd5938100a8dc79"
STANDARD_FRAME_RATES = (24, 25, 30, 50, 60, 90, 120)
STANDARD_SIZES_DEGREES = (6.0, 20.0, 45.0)
STANDARD_LUMINANCE_NITS = (.2, 1.0, 10.0, 150.0, 500.0)
MATLAB_LUMA = np.array([.299, .587, .114], dtype=np.float64)


@dataclass(frozen=True)
class AppleVFMConfig:
    frame_rate: int
    peak_luminance_nits: float
    area_square_degrees: float
    adaptation_floor_nits: float = 1e-6

    def __post_init__(self) -> None:
        if type(self.frame_rate) is not int or not 1 <= self.frame_rate <= 240:
            raise ValidationError("apple_vfm_frame_rate")
        values = (self.peak_luminance_nits, self.area_square_degrees, self.adaptation_floor_nits)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValidationError("apple_vfm_finite")
        if not (0 < self.peak_luminance_nits <= 10_000 and
                0 < self.area_square_degrees <= 100_000 and
                0 < self.adaptation_floor_nits < self.peak_luminance_nits):
            raise ValidationError("apple_vfm_range")


@dataclass(frozen=True)
class AppleVFMFrame:
    frame_index: int
    apl_relative: float
    average_luminance_nits: float
    adaptation_luminance_nits: float
    contrast: float
    response_energy: float
    normalized_step_energy: float
    unmasked_risk: float
    risk: float
    applied_mitigation: float
    next_mitigation: float
    output_adaptation_luminance_nits: float
    adaptation_floor_applied: bool


@lru_cache(maxsize=1)
def load_ufm_kernels():
    """Return kernels indexed by display size, adaptation level and frame rate."""
    resource = files("sunglasses").joinpath("data", KERNEL_RESOURCE)
    payload = resource.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if KERNEL_SHA256 and digest != KERNEL_SHA256:
        raise RuntimeError("apple_vfm_kernel_hash")
    lines = payload.decode("ascii").splitlines()
    expected = len(STANDARD_SIZES_DEGREES) * len(STANDARD_LUMINANCE_NITS) * len(STANDARD_FRAME_RATES)
    if len(lines) != expected:
        raise RuntimeError("apple_vfm_kernel_count")
    parsed = []
    for line in lines:
        values = tuple(float(value.strip()) for value in line.split(","))
        if not values or len(values) > 256 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("apple_vfm_kernel_value")
        parsed.append(values)
    offset = 0
    sizes = []
    for _ in STANDARD_SIZES_DEGREES:
        luminances = []
        for _ in STANDARD_LUMINANCE_NITS:
            rates = tuple(parsed[offset:offset + len(STANDARD_FRAME_RATES)])
            offset += len(STANDARD_FRAME_RATES)
            luminances.append(rates)
        sizes.append(tuple(luminances))
    return tuple(sizes)


def matlab_apl_from_linear_rgb(frame) -> float:
    """Whole-frame APL after decode, using the MATLAB reference luma weights."""
    rgb = normalized_rgb(frame)
    if rgb.ndim != 3:
        raise ValidationError("apple_vfm_shape")
    return float(np.sum(rgb.astype(np.float64) * MATLAB_LUMA, axis=-1).mean(dtype=np.float64))


def apply_matlab_mitigation(frame, mitigation: float, adaptation_relative: float):
    """Apply RiskComputePass0's luminance+contrast mapping in linear light."""
    rgb = normalized_rgb(frame)
    if rgb.ndim != 3:
        raise ValidationError("apple_vfm_shape")
    if (not math.isfinite(mitigation) or not 0 <= mitigation <= 1 or
            not math.isfinite(adaptation_relative) or adaptation_relative < 0):
        raise ValidationError("apple_vfm_mitigation")
    if mitigation == 0:
        return rgb.copy()
    scale = 1 - mitigation
    output = scale * scale * (rgb.astype(np.float64) - adaptation_relative) + scale * adaptation_relative
    return np.clip(output, 0, 1).astype(np.float32)


def _gamma_kernel(frame_rate: int) -> np.ndarray:
    # MATLAB GammaKernelFunction: shape=2, scale=.15, quantile=.99.
    values = []
    total = 0.0
    time = 0.0
    while total / frame_rate <= .99:
        time += 1 / frame_rate
        value = math.exp(-(time / .15)) * (.15 ** -2) * time
        values.append(value)
        total += value
        if len(values) >= 256:
            raise RuntimeError("apple_vfm_gamma_length")
    return np.asarray(values, dtype=np.float64)


class AppleVFMReference:
    """Fixed-rate, causal MATLAB-reference detector and mitigation mapper."""

    def __init__(self, config: AppleVFMConfig):
        self.config = config
        self.kernel_rate_index = min(range(len(STANDARD_FRAME_RATES)),
                                     key=lambda index: abs(STANDARD_FRAME_RATES[index] - config.frame_rate))
        equivalent_size = math.sqrt(config.area_square_degrees * 1.6)
        self.kernel_size_index = min(range(len(STANDARD_SIZES_DEGREES)),
                                     key=lambda index: abs(STANDARD_SIZES_DEGREES[index] - equivalent_size))
        self.kernel_frame_rate = STANDARD_FRAME_RATES[self.kernel_rate_index]
        self.kernel_size_degrees = STANDARD_SIZES_DEGREES[self.kernel_size_index]
        all_kernels = load_ufm_kernels()
        self.kernels = tuple(np.asarray(all_kernels[self.kernel_size_index][level][self.kernel_rate_index],
                                        dtype=np.float64)
                             for level in range(len(STANDARD_LUMINANCE_NITS)))
        self.gamma_kernel = _gamma_kernel(config.frame_rate)
        self.response_adjust = ((equivalent_size / self.kernel_size_degrees) ** (2 * .263) /
                                math.sqrt(config.frame_rate))
        self.mu_adaptation = 1 - math.exp(-1 / config.frame_rate)
        self.mu_mitigation = 1 - math.exp(-1 / (2 * config.frame_rate))
        self.reset()

    def reset(self) -> None:
        self.frame_index = 0
        self.adaptation = None
        self.mitigation = 0.0
        self.contrast_buffers = [np.zeros(len(kernel), dtype=np.float64) for kernel in self.kernels]
        shape = (len(STANDARD_LUMINANCE_NITS), len(self.gamma_kernel))
        self.energy_buffer = np.zeros(shape, dtype=np.float64)
        self.normalized_energy_buffer = np.zeros(shape, dtype=np.float64)

    def process_apl(self, apl_relative: float) -> AppleVFMFrame:
        if not isinstance(apl_relative, (int, float)) or not math.isfinite(apl_relative) or not 0 <= apl_relative <= 1:
            raise ValidationError("apple_vfm_apl")
        average_nits = float(apl_relative) * self.config.peak_luminance_nits
        previous_adaptation = self.adaptation
        output_adaptation = average_nits if previous_adaptation is None else previous_adaptation
        applied_mitigation = self.mitigation

        if previous_adaptation is None:
            adaptation = average_nits
        else:
            adaptation = self.mu_adaptation * average_nits + (1 - self.mu_adaptation) * previous_adaptation
        floor_applied = adaptation < self.config.adaptation_floor_nits
        safe_adaptation = max(adaptation, self.config.adaptation_floor_nits)
        contrast = (average_nits - safe_adaptation) / safe_adaptation

        responses, normalized_responses = [], []
        for index, kernel in enumerate(self.kernels):
            buffer = self.contrast_buffers[index]
            buffer[:-1] = buffer[1:]
            buffer[-1] = contrast
            response = float(np.dot(buffer, kernel[::-1]))
            reverse_response = float(np.dot(buffer, kernel))
            denominator = math.sqrt(float(np.dot(buffer, buffer) * np.dot(kernel, kernel)) + 1e-7)
            responses.append(response)
            normalized_responses.append(reverse_response / denominator)

        self.energy_buffer[:, :-1] = self.energy_buffer[:, 1:]
        self.normalized_energy_buffer[:, :-1] = self.normalized_energy_buffer[:, 1:]
        self.energy_buffer[:, -1] = np.square(responses)
        self.normalized_energy_buffer[:, -1] = np.square(normalized_responses)
        pooled = np.sqrt(self.energy_buffer @ self.gamma_kernel[::-1])
        normalized_pooled = np.sqrt(self.normalized_energy_buffer @ self.gamma_kernel[::-1])

        log_adaptation = math.log10(safe_adaptation)
        log_standard = np.log10(np.asarray(STANDARD_LUMINANCE_NITS, dtype=np.float64))
        energy = float(np.interp(log_adaptation, log_standard, pooled) * self.response_adjust)
        # RiskComputePass2 explicitly uses the 500-nit normalized pool for step masking.
        normalized_step_energy = float(normalized_pooled[-1])
        unmasked_risk = 0.0
        if energy > 33:
            unmasked_risk = 100 * (1 - math.exp(-((energy - 33) / 200) ** 3))
        risk = unmasked_risk if normalized_step_energy > 1.8 else 0.0
        next_mitigation = min(1.0, max(0.0, math.log10(risk) * .5)) if risk > 0 else 0.0
        if next_mitigation < applied_mitigation:
            next_mitigation = (next_mitigation * self.mu_mitigation +
                               applied_mitigation * (1 - self.mu_mitigation))
            if next_mitigation < .01:
                next_mitigation = 0.0

        self.adaptation = adaptation
        self.mitigation = next_mitigation
        measurement = AppleVFMFrame(
            frame_index=self.frame_index, apl_relative=float(apl_relative),
            average_luminance_nits=average_nits, adaptation_luminance_nits=adaptation,
            contrast=contrast, response_energy=energy, normalized_step_energy=normalized_step_energy,
            unmasked_risk=unmasked_risk, risk=risk, applied_mitigation=applied_mitigation,
            next_mitigation=next_mitigation, output_adaptation_luminance_nits=output_adaptation,
            adaptation_floor_applied=floor_applied)
        self.frame_index += 1
        return measurement

    def process_linear_rgb(self, frame):
        rgb = normalized_rgb(frame)
        apl = matlab_apl_from_linear_rgb(rgb)
        measurement = self.process_apl(apl)
        output = apply_matlab_mitigation(
            rgb, measurement.applied_mitigation,
            measurement.output_adaptation_luminance_nits / self.config.peak_luminance_nits)
        return output, measurement
