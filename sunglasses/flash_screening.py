"""Conservative SDR flash-screening proxies for offline iteration.

The definitions are informed by WCAG's general-flash and red-flash terms, but
this implementation is deliberately labelled a screening proxy.  It uses a
rectangular approximation to a 10-degree field, omits the fine-pattern
exception, and has not been validated as a standards-conformance or medical
risk tool.  Inputs are normalized linear-light sRGB.
"""

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from .automatic import FrameClock
from .color import luminance, normalized_rgb
from .i18n import ValidationError
from .perceptual import DisplayProfile


@dataclass(frozen=True)
class FlashScreeningConfig:
    """Explicit thresholds for a conservative, standards-informed screen."""

    field_degrees: float = 10.0
    area_fraction: float = .25
    general_relative_delta: float = .10
    darker_relative_ceiling: float = .80
    saturated_red_ratio: float = .80
    red_uv_distance: float = .20
    pair_window_seconds: float = 1.0
    maximum_flashes_per_second: int = 3

    def __post_init__(self) -> None:
        values = (self.field_degrees, self.area_fraction, self.general_relative_delta,
                  self.darker_relative_ceiling, self.saturated_red_ratio,
                  self.red_uv_distance, self.pair_window_seconds)
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in values):
            raise ValidationError("flash_screening_finite")
        if not (.1 <= self.field_degrees <= 90 and 0 < self.area_fraction <= 1 and
                0 < self.general_relative_delta <= 1 and
                0 < self.darker_relative_ceiling <= 1 and
                0 < self.saturated_red_ratio <= 1 and
                0 < self.red_uv_distance <= 1 and
                .05 <= self.pair_window_seconds <= 5):
            raise ValidationError("flash_screening_range")
        if (type(self.maximum_flashes_per_second) is not int or
                not 0 <= self.maximum_flashes_per_second <= 60):
            raise ValidationError("flash_screening_count")


@dataclass(frozen=True)
class FlashScreeningFrame:
    frame_id: int
    timestamp: float
    continuity: str
    field_pixels: tuple[int, int]
    general_transition_area_fraction: float
    general_rise_transition_area_fraction: float
    general_fall_transition_area_fraction: float
    red_transition_area_fraction: float
    general_pair_area_fraction: float
    red_pair_area_fraction: float
    general_pair_field_fraction_max: float
    general_rise_pair_field_fraction_max: float
    general_fall_pair_field_fraction_max: float
    red_pair_field_fraction_max: float
    general_flash_event: bool
    general_rise_flash_event: bool
    general_fall_flash_event: bool
    red_flash_event: bool
    combined_flash_event: bool
    general_flashes_1s: int
    red_flashes_1s: int
    combined_flashes_1s: int
    screening_threshold_exceeded: bool


def _uv1976(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CIE 1976 UCS chromaticity from normalized linear-light sRGB."""
    red, green, blue = np.moveaxis(rgb.astype(np.float64, copy=False), -1, 0)
    x = .4124564 * red + .3575761 * green + .1804375 * blue
    y = .2126729 * red + .7151522 * green + .0721750 * blue
    z = .0193339 * red + .1191920 * green + .9503041 * blue
    denominator = x + 15 * y + 3 * z
    u = np.divide(4 * x, denominator, out=np.zeros_like(x), where=denominator > 0)
    v = np.divide(9 * y, denominator, out=np.zeros_like(y), where=denominator > 0)
    return u, v


def _red_ratio(rgb: np.ndarray) -> np.ndarray:
    total = rgb.sum(axis=-1, dtype=np.float64)
    return np.divide(rgb[..., 0], total, out=np.zeros_like(total), where=total > 0)


def _maximum_window_fraction(mask: np.ndarray, rows: int, columns: int) -> float:
    """Maximum covered fraction in any axis-aligned fixed angular field."""
    if not np.any(mask):
        return 0.0
    height, width = mask.shape
    rows, columns = min(rows, height), min(columns, width)
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = (integral[rows:, columns:] - integral[:-rows, columns:] -
              integral[rows:, :-columns] + integral[:-rows, :-columns])
    return float(totals.max(initial=0) / (rows * columns))


def _completed_pairs(transitions: np.ndarray, pending_sign: np.ndarray,
                     pending_time: np.ndarray, timestamp: float,
                     maximum_age: float) -> np.ndarray:
    active = transitions != 0
    recent = timestamp - pending_time <= maximum_age
    completed = active & recent & (pending_sign != 0) & (transitions == -pending_sign)
    pending_sign[completed] = 0
    pending_time[completed] = -np.inf
    arm = active & ~completed
    pending_sign[arm] = transitions[arm]
    pending_time[arm] = timestamp
    return completed


def _general_transitions(current: np.ndarray, extreme: np.ndarray,
                         direction: np.ndarray, config: FlashScreeningConfig) -> np.ndarray:
    transitions = np.zeros(current.shape, dtype=np.int8)
    original_direction = direction.copy()
    delta = current - extreme
    candidate = ((np.abs(delta) >= config.general_relative_delta) &
                 (np.minimum(current, extreme) < config.darker_relative_ceiling))
    initial = (original_direction == 0) & candidate
    transitions[initial] = np.sign(delta[initial]).astype(np.int8)
    direction[initial] = transitions[initial]
    extreme[initial] = current[initial]

    rising = original_direction > 0
    falling = original_direction < 0
    reverse_down = (rising &
                    (extreme - current >= config.general_relative_delta) &
                    (np.minimum(current, extreme) < config.darker_relative_ceiling))
    reverse_up = (falling &
                  (current - extreme >= config.general_relative_delta) &
                  (np.minimum(current, extreme) < config.darker_relative_ceiling))
    transitions[reverse_down], transitions[reverse_up] = -1, 1
    direction[reverse_down], direction[reverse_up] = -1, 1
    extreme[reverse_down | reverse_up] = current[reverse_down | reverse_up]
    keep_rising = rising & ~reverse_down & (current > extreme)
    keep_falling = falling & ~reverse_up & (current < extreme)
    extreme[keep_rising | keep_falling] = current[keep_rising | keep_falling]
    return transitions


def _red_transitions(current: np.ndarray, extreme_rgb: np.ndarray,
                     extreme_ratio: np.ndarray, direction: np.ndarray,
                     config: FlashScreeningConfig) -> np.ndarray:
    ratio = _red_ratio(current)
    old_u, old_v = _uv1976(extreme_rgb)
    new_u, new_v = _uv1976(current)
    distance = np.hypot(new_u - old_u, new_v - old_v)
    delta = ratio - extreme_ratio
    candidate = ((np.maximum(ratio, extreme_ratio) >= config.saturated_red_ratio) &
                 (distance > config.red_uv_distance) & (delta != 0))
    transitions = np.zeros(ratio.shape, dtype=np.int8)
    original_direction = direction.copy()
    initial = (original_direction == 0) & candidate
    transitions[initial] = np.sign(delta[initial]).astype(np.int8)
    direction[initial] = transitions[initial]
    extreme_ratio[initial] = ratio[initial]
    extreme_rgb[initial] = current[initial]

    rising = original_direction > 0
    falling = original_direction < 0
    reverse_down = rising & (delta < 0) & candidate
    reverse_up = falling & (delta > 0) & candidate
    transitions[reverse_down], transitions[reverse_up] = -1, 1
    direction[reverse_down], direction[reverse_up] = -1, 1
    reversed_pixels = reverse_down | reverse_up
    extreme_ratio[reversed_pixels] = ratio[reversed_pixels]
    extreme_rgb[reversed_pixels] = current[reversed_pixels]
    keep_rising = rising & ~reverse_down & (ratio > extreme_ratio)
    keep_falling = falling & ~reverse_up & (ratio < extreme_ratio)
    keep = keep_rising | keep_falling
    extreme_ratio[keep] = ratio[keep]
    extreme_rgb[keep] = current[keep]
    return transitions


class FlashScreeningAnalyzer:
    """Timestamp-aware opposing-transition and angular-area screening."""

    def __init__(self, profile: DisplayProfile,
                 config: FlashScreeningConfig = FlashScreeningConfig()):
        self.profile = profile
        self.config = config
        self.clock = FrameClock()
        self.last_measurement = None
        self.general_extreme = self.general_direction = None
        self.general_pending_sign = self.general_pending_time = None
        self.red_extreme_rgb = self.red_extreme_ratio = self.red_direction = None
        self.red_pending_sign = self.red_pending_time = None
        self.general_events = deque()
        self.red_events = deque()
        self.combined_events = deque()

    def _reset(self, rgb: np.ndarray) -> None:
        shape = rgb.shape[:2]
        self.general_extreme = luminance(rgb).astype(np.float64, copy=True)
        self.general_direction = np.zeros(shape, dtype=np.int8)
        self.general_pending_sign = np.zeros(shape, dtype=np.int8)
        self.general_pending_time = np.full(shape, -np.inf, dtype=np.float64)
        self.red_extreme_rgb = rgb.astype(np.float64, copy=True)
        self.red_extreme_ratio = _red_ratio(rgb)
        self.red_direction = np.zeros(shape, dtype=np.int8)
        self.red_pending_sign = np.zeros(shape, dtype=np.int8)
        self.red_pending_time = np.full(shape, -np.inf, dtype=np.float64)
        self.general_events.clear()
        self.red_events.clear()
        self.combined_events.clear()

    def process(self, frame, timestamp: float, frame_id: int, *,
                discontinuity: bool = False) -> FlashScreeningFrame:
        rgb = normalized_rgb(frame)
        if rgb.ndim != 3:
            raise ValidationError("flash_screening_shape")
        timing = self.clock.accept(frame_id, timestamp, rgb.shape, discontinuity)
        if timing.duplicate:
            return replace(self.last_measurement, continuity="duplicate")

        horizontal, vertical = self.profile.visual_angle_degrees
        field_columns = max(1, min(rgb.shape[1],
                                  math.ceil(self.config.field_degrees * rgb.shape[1] / horizontal)))
        field_rows = max(1, min(rgb.shape[0],
                               math.ceil(self.config.field_degrees * rgb.shape[0] / vertical)))
        if timing.reset:
            self._reset(rgb)
            general_transitions = red_transitions = np.zeros(rgb.shape[:2], dtype=np.int8)
            general_pairs = red_pairs = np.zeros(rgb.shape[:2], dtype=bool)
        else:
            general_transitions = _general_transitions(
                luminance(rgb), self.general_extreme, self.general_direction, self.config)
            red_transitions = _red_transitions(
                rgb, self.red_extreme_rgb, self.red_extreme_ratio,
                self.red_direction, self.config)
            general_pairs = _completed_pairs(
                general_transitions, self.general_pending_sign,
                self.general_pending_time, timestamp, self.config.pair_window_seconds)
            red_pairs = _completed_pairs(
                red_transitions, self.red_pending_sign,
                self.red_pending_time, timestamp, self.config.pair_window_seconds)

        general_field = _maximum_window_fraction(general_pairs, field_rows, field_columns)
        rise_field = _maximum_window_fraction(
            general_pairs & (general_transitions > 0), field_rows, field_columns)
        fall_field = _maximum_window_fraction(
            general_pairs & (general_transitions < 0), field_rows, field_columns)
        red_field = _maximum_window_fraction(red_pairs, field_rows, field_columns)
        general_event = general_field > self.config.area_fraction
        red_event = red_field > self.config.area_fraction
        combined_event = general_event or red_event
        for event, history in ((general_event, self.general_events),
                               (red_event, self.red_events),
                               (combined_event, self.combined_events)):
            if event:
                history.append(timestamp)
            while history and history[0] < timestamp - 1.0:
                history.popleft()
        counts = (len(self.general_events), len(self.red_events), len(self.combined_events))
        measurement = FlashScreeningFrame(
            frame_id=frame_id, timestamp=timestamp, continuity=timing.continuity,
            field_pixels=(field_columns, field_rows),
            general_transition_area_fraction=float((general_transitions != 0).mean()),
            general_rise_transition_area_fraction=float((general_transitions > 0).mean()),
            general_fall_transition_area_fraction=float((general_transitions < 0).mean()),
            red_transition_area_fraction=float((red_transitions != 0).mean()),
            general_pair_area_fraction=float(general_pairs.mean()),
            red_pair_area_fraction=float(red_pairs.mean()),
            general_pair_field_fraction_max=general_field,
            general_rise_pair_field_fraction_max=rise_field,
            general_fall_pair_field_fraction_max=fall_field,
            red_pair_field_fraction_max=red_field,
            general_flash_event=general_event, red_flash_event=red_event,
            general_rise_flash_event=rise_field > self.config.area_fraction,
            general_fall_flash_event=fall_field > self.config.area_fraction,
            combined_flash_event=combined_event,
            general_flashes_1s=counts[0], red_flashes_1s=counts[1],
            combined_flashes_1s=counts[2],
            screening_threshold_exceeded=(
                max(counts) > self.config.maximum_flashes_per_second))
        self.last_measurement = measurement
        return measurement


def summarize_flash_screening(measurements) -> dict:
    items = [item for item in measurements if item.continuity != "duplicate"]
    if not items:
        raise ValidationError("flash_screening_empty")
    return {
        "frames": len(items),
        "general_flash_events": sum(item.general_flash_event for item in items),
        "general_rise_flash_events": sum(item.general_rise_flash_event for item in items),
        "general_fall_flash_events": sum(item.general_fall_flash_event for item in items),
        "red_flash_events": sum(item.red_flash_event for item in items),
        "combined_flash_events": sum(item.combined_flash_event for item in items),
        "max_general_flashes_1s": max(item.general_flashes_1s for item in items),
        "max_red_flashes_1s": max(item.red_flashes_1s for item in items),
        "max_combined_flashes_1s": max(item.combined_flashes_1s for item in items),
        "max_general_pair_field_fraction": max(
            item.general_pair_field_fraction_max for item in items),
        "max_red_pair_field_fraction": max(
            item.red_pair_field_fraction_max for item in items),
        "screening_threshold_exceeded_frames": sum(
            item.screening_threshold_exceeded for item in items),
        "screening_threshold_exceeded": any(
            item.screening_threshold_exceeded for item in items),
        "discontinuities": sum(item.continuity == "discontinuity" for item in items),
    }


def compare_flash_screening(source, output) -> dict:
    """Compare exact-frame events, including the direction completing a pair.

    A mismatch is evidence to inspect, not proof of a new spatial event: shifted
    events and different regions require the paired numeric trace. Directional
    subsets need not sum to the general count (mixed-direction area can qualify).
    """
    if len(source) != len(output):
        raise ValidationError("flash_screening_pair")
    pairs = list(zip(source, output))
    if any(left.frame_id != right.frame_id or left.timestamp != right.timestamp or
           left.continuity != right.continuity for left, right in pairs):
        raise ValidationError("flash_screening_pair")
    pairs = [(left, right) for left, right in pairs if left.continuity != "duplicate"]
    result = {
        f"added_{kind}_event_frames": sum(
            getattr(right, f"{kind}_flash_event") and not getattr(left, f"{kind}_flash_event")
            for left, right in pairs)
        for kind in ("general", "general_rise", "general_fall", "red", "combined")
    }
    result.update({
        "removed_combined_event_frames": sum(
            left.combined_flash_event and not right.combined_flash_event for left, right in pairs),
        "output_only_threshold_frames": sum(
            right.screening_threshold_exceeded and not left.screening_threshold_exceeded
            for left, right in pairs),
        "max_combined_count_increase": max(
            (right.combined_flashes_1s - left.combined_flashes_1s for left, right in pairs),
            default=0),
    })
    return result
