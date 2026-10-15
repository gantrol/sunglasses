"""Causal protection candidate for pre-existing small bright screen details.

This is an offline experiment, not semantic text or HUD recognition.  A pixel
may be protected only after it has remained stable, is locally brighter than
an angular surround, and was armed while the base filter was not dimming it.
The last rule prevents a newly detected detail from releasing active
compression and creating an algorithm-added flash.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from . import demo_filter as screen_reference
from .automatic import FrameClock
from .i18n import ValidationError
from .perceptual import DisplayProfile


@dataclass(frozen=True)
class StableDetailConfig:
    stable_seconds: float = .20
    arm_seconds: float = .20
    release_seconds: float = .10
    stable_log_delta: float = math.log(1.01)
    local_log_contrast: float = math.log(1.25)
    surround_degrees: float = .50
    minimum_relative_luminance: float = .65
    maximum_area_fraction: float = .05
    area_overflow_policy: str = "reject_batch"
    motion_propagation: bool = False
    maximum_motion_degrees: float = .25
    motion_minimum_support: float = .85
    motion_minimum_support_gain: float = .10
    max_added_log_rise_per_second: float = .50

    def __post_init__(self) -> None:
        scalars = (self.stable_seconds, self.arm_seconds, self.release_seconds,
                   self.stable_log_delta, self.local_log_contrast, self.surround_degrees,
                   self.minimum_relative_luminance, self.maximum_area_fraction,
                   self.maximum_motion_degrees, self.motion_minimum_support,
                   self.motion_minimum_support_gain, self.max_added_log_rise_per_second)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scalars):
            raise ValidationError("detail_finite")
        if not (.01 <= self.stable_seconds <= 5 and .01 <= self.arm_seconds <= 5 and
                .01 <= self.release_seconds <= 5):
            raise ValidationError("detail_time")
        if not (0 < self.stable_log_delta <= 1 and 0 < self.local_log_contrast <= 4 and
                .05 <= self.surround_degrees <= 10):
            raise ValidationError("detail_contrast")
        if not (0 < self.minimum_relative_luminance <= 1 and
                0 < self.maximum_area_fraction <= .25):
            raise ValidationError("detail_area")
        if self.area_overflow_policy not in ("reject_batch", "ranked_pixels"):
            raise ValidationError("detail_overflow")
        if type(self.motion_propagation) is not bool:
            raise ValidationError("detail_motion")
        if not (.01 <= self.maximum_motion_degrees <= 1 and
                0 < self.motion_minimum_support <= 1 and
                0 <= self.motion_minimum_support_gain <= 1):
            raise ValidationError("detail_motion")
        if not 0 < self.max_added_log_rise_per_second <= 10:
            raise ValidationError("detail_motion")


def _box_mean(values: np.ndarray, radius_y: int, radius_x: int) -> np.ndarray:
    padded = np.pad(values.astype(np.float64), ((radius_y, radius_y), (radius_x, radius_x)),
                    mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    height, width = values.shape
    y1, x1 = 2 * radius_y + 1, 2 * radius_x + 1
    total = (integral[y1:y1 + height, x1:x1 + width] -
             integral[:height, x1:x1 + width] -
             integral[y1:y1 + height, :width] + integral[:height, :width])
    return total / (y1 * x1)


def _limited_mask(eligible: np.ndarray, score: np.ndarray, maximum: int,
                  preferred: np.ndarray | None = None) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    if indices.size <= maximum:
        return eligible
    result = np.zeros(eligible.size, dtype=bool)
    flat_score = score.ravel()
    if preferred is not None:
        preferred_indices = np.flatnonzero(eligible & preferred)
        keep = min(maximum, preferred_indices.size)
        if preferred_indices.size > keep:
            preferred_indices = preferred_indices[
                np.argpartition(flat_score[preferred_indices], -keep)[-keep:]]
        result[preferred_indices] = True
    remaining = maximum - int(result.sum())
    if remaining:
        candidates = indices[~result[indices]]
        selected = candidates[np.argpartition(flat_score[candidates], -remaining)[-remaining:]]
        result[selected] = True
    return result.reshape(eligible.shape)


def _advance_strength(previous: np.ndarray, target: np.ndarray, rise_step: float,
                      release_step: float, maximum: int) -> np.ndarray:
    """Move toward a target without overlapping release and arming budgets."""
    result = previous.copy()
    falling = target < previous
    result[falling] = np.maximum(target[falling], previous[falling] - release_step)
    rising = target > previous
    requested = np.zeros_like(previous)
    requested[rising] = np.minimum(target[rising], previous[rising] + rise_step) - previous[rising]
    available = max(0.0, maximum - float(result.sum()))
    requested_total = float(requested.sum())
    if requested_total:
        result += requested * min(1.0, available / requested_total)
    total = float(result.sum())
    if total > maximum:
        result *= maximum / total
    return result


def _translate(values: np.ndarray, shift_y: int, shift_x: int, fill) -> np.ndarray:
    """Translate an image-sized state array without wraparound."""
    result = np.full_like(values, fill)
    height, width = values.shape[:2]
    source_top, source_bottom = max(0, -shift_y), min(height, height - shift_y)
    source_left, source_right = max(0, -shift_x), min(width, width - shift_x)
    target_top, target_bottom = source_top + shift_y, source_bottom + shift_y
    target_left, target_right = source_left + shift_x, source_right + shift_x
    if source_top < source_bottom and source_left < source_right:
        result[target_top:target_bottom, target_left:target_right] = values[
            source_top:source_bottom, source_left:source_right]
    return result


def _motion_match(protected: np.ndarray, previous_log: np.ndarray,
                  current_log: np.ndarray, current_candidate: np.ndarray,
                  maximum_y: int, maximum_x: int, stable_delta: float,
                  minimum_support: float, minimum_gain: float) -> tuple[int, int, float, float]:
    """Find a small translation only when current detail strongly supports it."""
    count = int(protected.sum())
    if count < 4:
        return 0, 0, 0.0, 0.0
    rows, columns = np.nonzero(protected)
    reference = previous_log[rows, columns]
    height, width = protected.shape

    def support(shift_y: int, shift_x: int) -> tuple[float, float]:
        # Score only inherited pixels; off-screen pixels still count as failures.
        target_y, target_x = rows + shift_y, columns + shift_x
        inside = ((target_y >= 0) & (target_y < height) &
                  (target_x >= 0) & (target_x < width))
        target_y, target_x = target_y[inside], target_x[inside]
        difference = np.abs(reference[inside] - current_log[target_y, target_x])
        matched = (current_candidate[target_y, target_x] &
                   np.isfinite(reference[inside]) & (difference <= stable_delta))
        fraction = float(matched.sum() / count)
        error = (float(difference[matched].mean())
                 if np.any(matched) else math.inf)
        return fraction, error

    zero_support, zero_error = support(0, 0)
    best_shift = (0, 0)
    best_support = zero_support
    best_score = (zero_support, -zero_error, 0, 0, 0)
    for shift_y in range(-maximum_y, maximum_y + 1):
        for shift_x in range(-maximum_x, maximum_x + 1):
            if shift_x == 0 and shift_y == 0:
                continue
            fraction, error = support(shift_y, shift_x)
            score = (fraction, -error, -abs(shift_x) - abs(shift_y), shift_y, shift_x)
            if score > best_score:
                best_shift = (shift_y, shift_x)
                best_support = fraction
                best_score = score
    shift_y, shift_x = best_shift
    if (shift_x == 0 and shift_y == 0) or best_support < minimum_support or (
            best_support < zero_support + minimum_gain):
        return 0, 0, zero_support, best_support
    return shift_y, shift_x, zero_support, best_support


class StableDetailProtector:
    def __init__(self, profile: DisplayProfile,
                 config: StableDetailConfig = StableDetailConfig()):
        self.profile = profile
        self.config = config
        self.clock = FrameClock()
        self.previous_log = self.stable_age = self.strength = None
        self.envelope_previous_source_log = None
        self.envelope_previous_output_log = self.envelope_ceiling_log = None
        self.envelope_active = None
        self.last_output = self.last_trace = None

    def process(self, encoded: np.ndarray, source: np.ndarray, filtered: np.ndarray,
                timestamp: float, frame_id: int, *, discontinuity: bool = False) -> tuple[np.ndarray, dict]:
        if (encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3 or
                source.shape != encoded.shape or filtered.shape != encoded.shape):
            raise ValidationError("detail_shape")
        if not (np.isfinite(source).all() and np.isfinite(filtered).all() and
                np.all(source >= 0) and np.all(filtered >= 0) and
                np.all(filtered <= source + 1e-7)):
            raise ValidationError("detail_values")
        timing = self.clock.accept(frame_id, timestamp, encoded.shape, discontinuity)
        if timing.duplicate:
            return self.last_output.copy(), {**self.last_trace, "continuity": "duplicate"}

        source_y = np.sum(source * screen_reference.LUMA, axis=-1)
        filtered_y = np.sum(filtered * screen_reference.LUMA, axis=-1)
        source_log = np.log(self.profile.to_nits(source_y) + .5)
        horizontal, vertical = self.profile.visual_angle_degrees
        radius_x = max(1, round(self.config.surround_degrees * source_y.shape[1] / horizontal))
        radius_y = max(1, round(self.config.surround_degrees * source_y.shape[0] / vertical))
        surround = _box_mean(source_log, radius_y, radius_x)
        contrast = source_log - surround

        if timing.reset or self.previous_log is None:
            self.stable_age = np.zeros(source_y.shape, dtype=np.float64)
            self.strength = np.zeros(source_y.shape, dtype=np.float64)
            motion_shift = (0, 0)
            motion_zero_support = motion_best_support = 0.0
            motion_propagated = 0
        else:
            motion_shift = (0, 0)
            motion_zero_support = motion_best_support = 0.0
            motion_propagated = 0
            if self.config.motion_propagation and np.any(self.strength > 0):
                candidate = ((source_y >= self.config.minimum_relative_luminance) &
                             (contrast >= self.config.local_log_contrast))
                horizontal, vertical = self.profile.visual_angle_degrees
                maximum_x = max(1, math.ceil(
                    self.config.maximum_motion_degrees * source_y.shape[1] / horizontal))
                maximum_y = max(1, math.ceil(
                    self.config.maximum_motion_degrees * source_y.shape[0] / vertical))
                shift_y, shift_x, motion_zero_support, motion_best_support = _motion_match(
                    self.strength > 0, self.previous_log, source_log, candidate,
                    maximum_y, maximum_x, self.config.stable_log_delta,
                    self.config.motion_minimum_support,
                    self.config.motion_minimum_support_gain)
                if shift_x or shift_y:
                    moved_log = _translate(self.previous_log, shift_y, shift_x, np.nan)
                    matched = (candidate & np.isfinite(moved_log) &
                               (np.abs(moved_log - source_log) <= self.config.stable_log_delta))
                    self.previous_log = np.where(matched, moved_log, source_log)
                    self.stable_age = np.where(
                        matched, _translate(self.stable_age, shift_y, shift_x, 0.0), 0.0)
                    moved_strength = _translate(self.strength, shift_y, shift_x, 0.0)
                    self.strength = np.where(matched, moved_strength, 0.0)
                    motion_shift = (shift_x, shift_y)
                    motion_propagated = int((self.strength > 0).sum())
            stable = np.abs(source_log - self.previous_log) <= self.config.stable_log_delta
            self.stable_age = np.where(stable, self.stable_age + timing.dt, 0.0)
        eligible = ((self.stable_age >= self.config.stable_seconds) &
                    (source_y >= self.config.minimum_relative_luminance) &
                    (contrast >= self.config.local_log_contrast))
        maximum = max(1, int(self.config.maximum_area_fraction * eligible.size))
        previous_strength = self.strength.copy()
        one_code = screen_reference.LINEAR[1]
        undimmed = filtered_y >= source_y - one_code
        existing = eligible & (previous_strength > 0)
        new_candidates = eligible & (previous_strength == 0) & undimmed
        remaining = max(0, maximum - int(existing.sum()))
        overflow = int(new_candidates.sum()) > remaining
        if self.config.area_overflow_policy == "ranked_pixels":
            target = _limited_mask(existing | new_candidates, contrast, maximum, existing)
            target_strength = target.astype(np.float64)
            overflow_rejected = max(0, int(new_candidates.sum()) - remaining)
        else:
            target = existing.copy()
            overflow_rejected = int(new_candidates.sum()) if overflow else 0
            if not overflow:
                target |= new_candidates
            target_strength = target.astype(np.float64)
        if timing.reset:
            rise_step = release_step = 0.0
        else:
            rise_step = timing.dt / self.config.arm_seconds
            release_step = timing.dt / self.config.release_seconds
        self.strength = _advance_strength(
            self.strength, target_strength, rise_step, release_step, maximum)
        newly_armed = (previous_strength == 0) & (self.strength > 0)
        protection_ended = (previous_strength > 0) & (self.strength == 0)

        if np.any(self.strength > 0):
            mixed = filtered + self.strength[..., None] * (source - filtered)
            desired = np.clip(np.where(mixed <= .0031308, 12.92 * mixed,
                                       1.055 * mixed ** (1 / 2.4) - .055), 0, 1)
            quantized = np.minimum(encoded, np.rint(desired * 255).astype(np.uint8))
            output = screen_reference.LINEAR[quantized]
        else:
            output = filtered.copy()
        output_y = np.sum(output * screen_reference.LUMA, axis=-1)
        output_log = np.log(self.profile.to_nits(output_y) + .5)
        if timing.reset or self.envelope_ceiling_log is None:
            detail_envelope_overshoot = detail_realized_rate = 0.0
            envelope_realized_rate = 0.0
            active_envelope = self.strength > 0
        else:
            # Untouched pixels retain the base limiter's exact quantized output.
            # Previously protected pixels stay constrained until they rejoin it.
            active_envelope = self.envelope_active | (self.strength > 0)
            source_rise = np.maximum(source_log - self.envelope_previous_source_log, 0)
            proposed_ceiling = np.minimum(
                source_log, self.envelope_ceiling_log + source_rise +
                self.config.max_added_log_rise_per_second * timing.dt)
            needs_limit = active_envelope & (output_log > proposed_ceiling + 1e-12)
            if np.any(needs_limit):
                target_nits = np.maximum(np.exp(proposed_ceiling) - .5, 0)
                target_y = np.clip(
                    (target_nits - self.profile.black_luminance_nits) /
                    (self.profile.peak_luminance_nits - self.profile.black_luminance_nits), 0, 1)
                scale = np.minimum(1.0, np.divide(
                    target_y, output_y, out=np.ones_like(output_y), where=output_y > 0))
                limited = output * scale[..., None]
                encoded_limit = np.floor(np.clip(np.where(
                    limited <= .0031308, 12.92 * limited,
                    1.055 * limited ** (1 / 2.4) - .055), 0, 1) * 255 + 1e-12
                ).astype(np.uint8)
                limited_output = screen_reference.LINEAR[np.minimum(encoded, encoded_limit)]
                output = np.where(needs_limit[..., None], limited_output, output)
                output_y = np.sum(output * screen_reference.LUMA, axis=-1)
                output_log = np.log(self.profile.to_nits(output_y) + .5)
            detail_envelope_overshoot = float(
                np.maximum(output_log[active_envelope] - proposed_ceiling[active_envelope], 0)
                .max(initial=0))
            added = output_log - self.envelope_previous_output_log - source_rise
            detail_realized_rate = float(
                np.maximum(added, 0).max(initial=0) / timing.dt)
            envelope_realized_rate = float(
                np.maximum(added[active_envelope], 0).max(initial=0) / timing.dt)
        released = self.profile.to_nits(output_y) - self.profile.to_nits(filtered_y)
        trace = {
            "continuity": timing.continuity,
            "eligible_area_fraction": float(eligible.mean()),
            "target_area_fraction": float(target.mean()),
            "target_weighted_fraction": float(target_strength.mean()),
            "protected_area_fraction": float((self.strength > 0).mean()),
            "weighted_protection_fraction": float(self.strength.mean()),
            "protection_budget_fraction": maximum / eligible.size,
            "protection_budget_overshoot": float(
                max(0, self.strength.mean() - maximum / eligible.size)),
            "maximum_strength": float(self.strength.max(initial=0)),
            "mean_strength": float(self.strength.mean()),
            "maximum_strength_change": float(np.abs(self.strength - previous_strength).max(initial=0)),
            "max_released_dimming_nits": float(np.maximum(released, 0).max(initial=0)),
            "newly_armed_pixels": int(newly_armed.sum()),
            "protection_ended_pixels": int(protection_ended.sum()),
            "armed_while_dimmed_pixels": int((newly_armed & ~undimmed).sum()),
            "overflow_rejected_pixels": overflow_rejected,
            "radius_pixels": (radius_x, radius_y),
            "motion_shift_pixels": motion_shift,
            "motion_zero_support": motion_zero_support,
            "motion_best_support": motion_best_support,
            "motion_propagated_pixels": motion_propagated,
            "max_envelope_overshoot_log": detail_envelope_overshoot,
            "max_realized_added_log_rise_per_second": detail_realized_rate,
            "max_envelope_added_log_rise_per_second": envelope_realized_rate,
            "envelope_active_area_fraction": float(active_envelope.mean()),
        }
        self.previous_log = source_log.copy()
        self.envelope_previous_source_log = source_log.copy()
        self.envelope_previous_output_log = output_log.copy()
        self.envelope_ceiling_log = output_log.copy()
        self.envelope_active = active_envelope & (
            (self.strength > 0) | np.any(output != filtered, axis=-1))
        self.last_output, self.last_trace = output.copy(), trace.copy()
        return output, trace

    def snapshot(self) -> dict:
        return {"config": asdict(self.config)}
