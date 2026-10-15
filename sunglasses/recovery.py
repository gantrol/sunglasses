"""Causal screen-recovery candidates for offline, headless experiments.

These controllers are research candidates, not the product shader and not a
medical flicker model.  They keep the frozen screen detector and curve while
making the recovery policy explicit.  Inputs are the frozen 8-bit SDR capture
representation so quantization remains part of the experiment.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from .automatic import FrameClock
from . import demo_filter as screen_reference
from .i18n import ValidationError
from .perceptual import DisplayProfile


@dataclass(frozen=True)
class SourceHoldConfig:
    """Hold an attacked curve until the source returns near its pre-event level."""

    tentative_step_scale: float = 1.0
    return_load_margin: float = .003
    return_mean_margin: float = .005
    max_active_seconds: float = 5.0

    def __post_init__(self) -> None:
        values = (self.tentative_step_scale, self.return_load_margin,
                  self.return_mean_margin, self.max_active_seconds)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValidationError("recovery_hold_finite")
        if not 0 <= self.tentative_step_scale <= 1:
            raise ValidationError("recovery_tentative_scale")
        if not (0 <= self.return_load_margin <= .1 and 0 <= self.return_mean_margin <= .1):
            raise ValidationError("recovery_return_margin")
        if not .1 <= self.max_active_seconds <= 30:
            raise ValidationError("recovery_hold_time")


class SourceHoldController:
    """Keep compression stable while the source event itself remains present.

    A coherent one-way step may optionally receive only a tentative first
    attack.  A later opposing transition is no longer treated as an isolated
    step and receives the full detector demand.
    """

    def __init__(self, config: SourceHoldConfig = SourceHoldConfig()):
        self.config = config
        self.previous = None
        self.peak = self.broad = 0.0
        self.baseline_load = self.baseline_mean = 0.0
        self.active_age = 0.0
        self.active = False
        self.last_trace = None

    def _clear(self) -> None:
        self.peak = self.broad = self.active_age = 0.0
        self.active = False

    def step(self, load: float, mean: float, timestamp: float, *,
             coherent_step: bool = False, opposed_area: float = 0.0) -> tuple[float, dict]:
        values = (load, mean, timestamp, opposed_area)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValidationError("recovery_hold_input")
        if load < 0 or not 0 <= mean <= 1 or timestamp < 0 or not 0 <= opposed_area <= 1:
            raise ValidationError("recovery_hold_input")
        if type(coherent_step) is not bool:
            raise ValidationError("recovery_hold_input")

        demand = 0.0
        returned = expired = False
        tentative = False
        if self.previous is not None:
            previous_time, previous_load, previous_mean = self.previous
            dt = timestamp - previous_time
            if dt <= 0:
                raise ValidationError("recovery_hold_time_order")
            self.active_age += dt if self.active else 0.0
            demand = float(screen_reference.smooth(.015, .08, load - previous_load) *
                           screen_reference.smooth(.01, .08, mean - previous_mean))
            if demand > 0:
                if not self.active:
                    self.baseline_load, self.baseline_mean = previous_load, previous_mean
                    self.active_age = 0.0
                tentative = coherent_step and opposed_area < .10
                scale = self.config.tentative_step_scale if tentative else 1.0
                self.peak = max(self.peak, demand * scale)
                self.broad = max(self.broad, float(screen_reference.smooth(.08, .25, load)))
                self.active = True
            if self.active:
                returned = (load <= self.baseline_load + self.config.return_load_margin or
                            mean <= self.baseline_mean + self.config.return_mean_margin)
                expired = self.active_age >= self.config.max_active_seconds
                if returned or expired:
                    self._clear()
                else:
                    self.broad = max(self.broad, float(screen_reference.smooth(.08, .25, load)))
        self.previous = (timestamp, load, mean)
        gain = (self.peak * float(screen_reference.smooth(.01, .04, load))
                if self.active else 0.0)
        trace = {
            "demand": demand,
            "gain": gain,
            "broad": self.broad,
            "active": self.active,
            "active_age_seconds": self.active_age,
            "tentative_step": tentative,
            "opposed_area": opposed_area,
            "returned_to_source_baseline": returned,
            "expired": expired,
        }
        self.last_trace = trace
        return gain, trace


@dataclass(frozen=True)
class LogRiseLimitConfig:
    """Perceptual-domain recovery envelope; values are search parameters."""

    max_added_log_rise_per_second: float = .5
    contrast_floor_nits: float = .5
    cell_degrees: float = 2.0
    binary_search_steps: int = 18
    output_bits: int = 8
    max_amount_gradient_per_degree: float | None = None

    def __post_init__(self) -> None:
        scalars = (self.max_added_log_rise_per_second, self.contrast_floor_nits,
                   self.cell_degrees)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scalars):
            raise ValidationError("recovery_limit_finite")
        if not (0 < self.max_added_log_rise_per_second <= 10 and
                0 < self.contrast_floor_nits <= 100 and .1 <= self.cell_degrees <= 30):
            raise ValidationError("recovery_limit_range")
        if type(self.binary_search_steps) is not int or not 4 <= self.binary_search_steps <= 30:
            raise ValidationError("recovery_limit_steps")
        if type(self.output_bits) is not int or not 8 <= self.output_bits <= 16:
            raise ValidationError("recovery_limit_bits")
        if (self.max_amount_gradient_per_degree is not None and
                (not isinstance(self.max_amount_gradient_per_degree, (int, float)) or
                 not math.isfinite(self.max_amount_gradient_per_degree) or
                 not 0 < self.max_amount_gradient_per_degree <= 10)):
            raise ValidationError("recovery_limit_gradient")


def _partitions(shape: tuple[int, int], profile: DisplayProfile,
                config: LogRiseLimitConfig, tiled: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    if tiled:
        horizontal, vertical = profile.visual_angle_degrees
        rows = max(1, min(height, 64, math.ceil(vertical / config.cell_degrees)))
        columns = max(1, min(width, 96, math.ceil(horizontal / config.cell_degrees)))
    else:
        rows = columns = 1
    ys = np.arange(rows + 1) * height // rows
    xs = np.arange(columns + 1) * width // columns
    weights = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float64)
    return ys, xs, weights


def _means(values: np.ndarray, partitions) -> np.ndarray:
    ys, xs, weights = partitions
    reduced = np.add.reduceat(np.add.reduceat(values.astype(np.float64), ys[:-1], axis=0),
                              xs[:-1], axis=1)
    return reduced / weights


def _maxima(values: np.ndarray, partitions) -> np.ndarray:
    ys, xs, _ = partitions
    return np.maximum.reduceat(
        np.maximum.reduceat(values.astype(np.float64), ys[:-1], axis=0), xs[:-1], axis=1)


def apply_quantized_screen_curve(encoded: np.ndarray, linear: np.ndarray, y: np.ndarray,
                                 amount: float, broad: float, output_bits: int) -> np.ndarray:
    """Apply the frozen curve with an explicit output quantizer.

    Eight bits is bit-identical to ``demo_filter.apply`` after linear decoding.
    Higher depths are counterfactual research formats; the captured source is
    still the same 8-bit SDR input and unchanged pixels remain exact.
    """
    if (encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3 or
            linear.shape != encoded.shape or y.shape != encoded.shape[:2]):
        raise ValidationError("recovery_limit_shape")
    if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in (amount, broad)) or
            not all(0 <= value <= 1 for value in (amount, broad)) or
            type(output_bits) is not int or not 8 <= output_bits <= 16):
        raise ValidationError("recovery_limit_control")
    if output_bits == 8:
        encoded_output = screen_reference.apply(encoded, linear, y, amount, broad, 1.0)
        return screen_reference.LINEAR[encoded_output]
    if amount <= 0:
        return linear.copy()
    original = encoded.astype(np.float32) / 255
    extra = float(screen_reference.smooth(.5, 1, 1.0)) * broad
    knee, span = .65 - .20 * extra, .15 - .05 * extra
    eligible = y > knee
    mapped = knee + span * (1 - np.exp(-np.maximum(y - knee, 0) / span))
    factor = np.where(eligible, 1 + amount * (mapped / np.maximum(y, 1e-20) - 1), 1)
    corrected = np.maximum(linear * factor[..., None], 0)
    desired = np.minimum(original, np.where(corrected <= .0031308, 12.92 * corrected,
                                             1.055 * corrected ** (1 / 2.4) - .055))
    desired[~eligible] = original[~eligible]
    levels = float(2**output_bits - 1)
    quantized = np.minimum(original, np.rint(desired * levels) / levels)
    output = np.where(quantized <= .04045, quantized / 12.92,
                      ((quantized + .055) / 1.055) ** 2.4).astype(np.float32)
    output[~eligible] = linear[~eligible]
    return output


def _expand_tiles(values: np.ndarray, partitions, shape: tuple[int, int]) -> np.ndarray:
    ys, xs, _ = partitions
    expanded = np.empty(shape, dtype=np.float64)
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            expanded[ys[row]:ys[row + 1], xs[column]:xs[column + 1]] = values[row, column]
    return expanded


def _spatial_majorant(values: np.ndarray, profile: DisplayProfile,
                      max_gradient_per_degree: float) -> np.ndarray:
    """Smallest Manhattan-cone envelope with a bounded angular gradient.

    The envelope only raises recovery amounts.  It therefore cannot relax a
    temporal brightness ceiling already met by the unregularized tile map.
    """
    result = np.asarray(values, dtype=np.float64).copy()
    rows, columns = result.shape
    horizontal, vertical = profile.visual_angle_degrees
    horizontal_step = max_gradient_per_degree * horizontal / columns
    vertical_step = max_gradient_per_degree * vertical / rows
    for row in range(rows):
        for column in range(1, columns):
            result[row, column] = max(result[row, column],
                                      result[row, column - 1] - horizontal_step)
        for column in range(columns - 2, -1, -1):
            result[row, column] = max(result[row, column],
                                      result[row, column + 1] - horizontal_step)
    for column in range(columns):
        for row in range(1, rows):
            result[row, column] = max(result[row, column],
                                      result[row - 1, column] - vertical_step)
        for row in range(rows - 2, -1, -1):
            result[row, column] = max(result[row, column],
                                      result[row + 1, column] - vertical_step)
    return result


def apply_quantized_screen_curve_map(encoded: np.ndarray, linear: np.ndarray, y: np.ndarray,
                                     amount_map: np.ndarray, broad: float,
                                     output_bits: int) -> np.ndarray:
    """Apply the frozen shoulder with a hypothetical per-pixel amount map."""
    amounts = np.asarray(amount_map, dtype=np.float64)
    if (encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3 or
            linear.shape != encoded.shape or y.shape != encoded.shape[:2] or amounts.shape != y.shape):
        raise ValidationError("recovery_limit_shape")
    if (not np.isfinite(amounts).all() or np.any((amounts < 0) | (amounts > 1)) or
            not isinstance(broad, (int, float)) or not math.isfinite(broad) or not 0 <= broad <= 1 or
            type(output_bits) is not int or not 8 <= output_bits <= 16):
        raise ValidationError("recovery_limit_control")
    if np.all(amounts == amounts.flat[0]):
        return apply_quantized_screen_curve(encoded, linear, y, float(amounts.flat[0]),
                                             broad, output_bits)
    if not np.any(amounts > 0):
        return linear.copy()
    original = encoded.astype(np.float32) / 255
    knee, span = .65 - .20 * broad, .15 - .05 * broad
    eligible = y > knee
    mapped = knee + span * (1 - np.exp(-np.maximum(y - knee, 0) / span))
    factor = np.where(eligible, 1 + amounts * (mapped / np.maximum(y, 1e-20) - 1), 1)
    corrected = np.maximum(linear * factor[..., None], 0)
    desired = np.minimum(original, np.where(corrected <= .0031308, 12.92 * corrected,
                                             1.055 * corrected ** (1 / 2.4) - .055))
    desired[~eligible] = original[~eligible]
    if output_bits == 8:
        # Use the same decoded codebook as the uniform curve and detail output.
        # A second float64 sRGB decode can differ by an ULP after casting and
        # falsely keep the detail envelope active after it reaches the base.
        codes = np.minimum(encoded, np.rint(desired * 255).astype(np.uint8))
        output = screen_reference.LINEAR[codes]
        output[~eligible] = linear[~eligible]
        return output
    levels = float(2**output_bits - 1)
    quantized = np.minimum(original, np.rint(desired * levels) / levels)
    output = np.where(quantized <= .04045, quantized / 12.92,
                      ((quantized + .055) / 1.055) ** 2.4).astype(np.float32)
    output[~eligible] = linear[~eligible]
    return output


def _boundary_control_contrast(encoded: np.ndarray, linear: np.ndarray, y: np.ndarray,
                               amounts: np.ndarray, partitions, broad: float,
                               profile: DisplayProfile, config: LogRiseLimitConfig) -> tuple[float, float, float]:
    """Visibility-weight a tile-control discontinuity by local compressibility.

    This remains an uncalibrated engineering proxy.  It is zero when adjacent
    controls match or when the local pixels are below the frozen curve's knee.
    """
    horizontal_delta = np.abs(np.diff(amounts, axis=1))
    vertical_delta = np.abs(np.diff(amounts, axis=0))
    if not np.any(horizontal_delta > 0) and not np.any(vertical_delta > 0):
        return 0.0, 0.0, 0.0
    full = apply_quantized_screen_curve(encoded, linear, y, 1.0, broad, config.output_bits)
    full_y = np.sum(full * screen_reference.LUMA, axis=-1)
    capacity = (np.log(profile.to_nits(y) + config.contrast_floor_nits) -
                np.log(profile.to_nits(full_y) + config.contrast_floor_nits))
    ys, xs, _ = partitions
    samples = []
    for row in range(amounts.shape[0]):
        for column in range(1, amounts.shape[1]):
            boundary = xs[column]
            local_capacity = (capacity[ys[row]:ys[row + 1], boundary - 1] +
                              capacity[ys[row]:ys[row + 1], boundary]) / 2
            samples.append(horizontal_delta[row, column - 1] * local_capacity)
    for row in range(1, amounts.shape[0]):
        boundary = ys[row]
        for column in range(amounts.shape[1]):
            local_capacity = (capacity[boundary - 1, xs[column]:xs[column + 1]] +
                              capacity[boundary, xs[column]:xs[column + 1]]) / 2
            samples.append(vertical_delta[row - 1, column] * local_capacity)
    values = np.concatenate([np.asarray(sample, dtype=np.float64).ravel() for sample in samples])
    return (float(values.mean(dtype=np.float64)), float(np.quantile(values, .95)),
            float(values.max(initial=0)))


class LogRiseLimiter:
    """Constrain recovery in global or fixed-angular local log-luminance cells.

    The ceiling accumulates continuously, so an 8-bit output can eventually
    cross a quantization step.  Consequently the nominal rate bounds the
    accumulated envelope, not every quantized frame-to-frame jump; both are
    reported separately.
    """

    def __init__(self, profile: DisplayProfile, *, tiled: bool,
                 config: LogRiseLimitConfig = LogRiseLimitConfig()):
        if type(tiled) is not bool:
            raise ValidationError("recovery_limit_scope")
        self.profile = profile
        self.tiled = tiled
        self.config = config
        self.clock = FrameClock()
        self.partitions = self.previous_source_log = self.previous_output_log = None
        self.ceiling_log = None
        self.retained_broad = self.previous_amount = 0.0
        self.last_output = self.last_trace = None

    def _logs(self, y: np.ndarray) -> np.ndarray:
        relative = _means(y, self.partitions)
        nits = self.profile.to_nits(relative)
        return np.log(nits + self.config.contrast_floor_nits)

    def process(self, encoded: np.ndarray, linear: np.ndarray, y: np.ndarray,
                desired_amount: float, broad: float, timestamp: float, frame_id: int,
                *, discontinuity: bool = False) -> tuple[np.ndarray, dict]:
        if (encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3 or
                linear.shape != encoded.shape or y.shape != encoded.shape[:2]):
            raise ValidationError("recovery_limit_shape")
        controls = (desired_amount, broad)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in controls):
            raise ValidationError("recovery_limit_control")
        if not all(0 <= value <= 1 for value in controls):
            raise ValidationError("recovery_limit_control")
        timing = self.clock.accept(frame_id, timestamp, encoded.shape, discontinuity)
        if timing.duplicate:
            return self.last_output.copy(), {**self.last_trace, "continuity": "duplicate"}
        if timing.reset or self.partitions is None:
            self.partitions = _partitions(y.shape, self.profile, self.config, self.tiled)
            self.retained_broad = self.previous_amount = 0.0
        curve_broad = max(broad, self.retained_broad if self.previous_amount > 0 else 0.0)

        def evaluate(amount: float) -> tuple[np.ndarray, np.ndarray]:
            output = apply_quantized_screen_curve(encoded, linear, y, amount, curve_broad,
                                                   self.config.output_bits)
            output_y = np.sum(output * screen_reference.LUMA, axis=-1)
            return output, self._logs(output_y)

        source_log = self._logs(y)
        selected = desired_amount
        output, output_log = evaluate(selected)
        infeasible = False
        envelope_overshoot = 0.0
        realized_rate = 0.0
        if timing.reset:
            ceiling = output_log.copy()
        else:
            source_rise = np.maximum(source_log - self.previous_source_log, 0)
            proposed_ceiling = np.minimum(source_log, self.ceiling_log + source_rise +
                                          self.config.max_added_log_rise_per_second * timing.dt)

            def overshoot(values: np.ndarray) -> float:
                return float(np.maximum(values - proposed_ceiling, 0).max(initial=0))

            envelope_overshoot = overshoot(output_log)
            if envelope_overshoot > 1e-12:
                full_output, full_log = evaluate(1.0)
                if overshoot(full_log) > 1e-12:
                    selected, output, output_log, infeasible = 1.0, full_output, full_log, True
                    # The legal curve cannot meet this frame.  Rebase to the
                    # actual output rather than carrying an impossible debt.
                    ceiling = np.maximum(proposed_ceiling, output_log)
                else:
                    low, high = desired_amount, 1.0
                    for _ in range(self.config.binary_search_steps):
                        middle = (low + high) / 2
                        middle_output, middle_log = evaluate(middle)
                        if overshoot(middle_log) > 1e-12:
                            low = middle
                        else:
                            high, output, output_log = middle, middle_output, middle_log
                    selected = high
                    # Preserve the continuously accumulated budget across
                    # quantized output plateaus.
                    ceiling = proposed_ceiling
                envelope_overshoot = overshoot(output_log)
            else:
                # A new attack or a naturally slower recovery becomes the
                # baseline for any later algorithm-added rise.
                ceiling = np.minimum(proposed_ceiling, output_log)
            added = output_log - self.previous_output_log - source_rise
            realized_rate = float(np.maximum(added, 0).max(initial=0) / timing.dt)

        self.previous_source_log = source_log.copy()
        self.previous_output_log = output_log.copy()
        self.ceiling_log = ceiling.copy()
        self.previous_amount = selected
        self.retained_broad = curve_broad if selected > 0 else 0.0
        trace = {
            "continuity": timing.continuity,
            "desired_amount": desired_amount,
            "selected_amount": selected,
            "added_amount": selected - desired_amount,
            "curve_broad": curve_broad,
            "tile_count": int(source_log.size),
            "output_bits": self.config.output_bits,
            "infeasible": infeasible,
            "max_envelope_overshoot_log": envelope_overshoot,
            "max_realized_added_log_rise_per_second": realized_rate,
        }
        self.last_output, self.last_trace = output.copy(), trace.copy()
        return output, trace

    def snapshot(self) -> dict:
        return {"tiled": self.tiled, "config": asdict(self.config)}


class LocalLogRiseLimiter:
    """Use an independent recovery amount for each fixed-angular screen cell.

    This is an offline architectural candidate.  Piecewise-constant cells can
    introduce visible spatial boundaries.  An optional conservative spatial
    envelope bounds amount change per visual degree by raising neighboring
    amounts; it never lowers a tile's temporally required amount.
    """

    def __init__(self, profile: DisplayProfile,
                 config: LogRiseLimitConfig = LogRiseLimitConfig()):
        self.profile = profile
        self.config = config
        self.clock = FrameClock()
        self.partitions = self.previous_source_log = self.previous_output_log = None
        self.ceiling_log = self.previous_amounts = None
        self.retained_broad = 0.0
        self.last_output = self.last_trace = None

    def _logs(self, y: np.ndarray) -> np.ndarray:
        relative = _means(y, self.partitions)
        return np.log(self.profile.to_nits(relative) + self.config.contrast_floor_nits)

    def process(self, encoded: np.ndarray, linear: np.ndarray, y: np.ndarray,
                desired_amount: float, broad: float, timestamp: float, frame_id: int,
                *, discontinuity: bool = False) -> tuple[np.ndarray, dict]:
        if (encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3 or
                linear.shape != encoded.shape or y.shape != encoded.shape[:2]):
            raise ValidationError("recovery_limit_shape")
        if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (desired_amount, broad)) or
                not all(0 <= value <= 1 for value in (desired_amount, broad))):
            raise ValidationError("recovery_limit_control")
        timing = self.clock.accept(frame_id, timestamp, encoded.shape, discontinuity)
        if timing.duplicate:
            return self.last_output.copy(), {**self.last_trace, "continuity": "duplicate"}
        if timing.reset or self.partitions is None:
            self.partitions = _partitions(y.shape, self.profile, self.config, True)
            tile_shape = self.partitions[2].shape
            self.previous_amounts = np.zeros(tile_shape, dtype=np.float64)
            self.retained_broad = 0.0
        curve_broad = max(broad, self.retained_broad if np.any(self.previous_amounts > 0) else 0.0)

        def evaluate(amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            amount_map = _expand_tiles(amounts, self.partitions, y.shape)
            output = apply_quantized_screen_curve_map(encoded, linear, y, amount_map,
                                                       curve_broad, self.config.output_bits)
            output_y = np.sum(output * screen_reference.LUMA, axis=-1)
            return output, self._logs(output_y)

        source_log = self._logs(y)
        desired = np.full(source_log.shape, desired_amount, dtype=np.float64)
        selected = desired.copy()
        output, output_log = evaluate(selected)
        infeasible_mask = np.zeros(source_log.shape, dtype=bool)
        envelope_overshoot = 0.0
        realized_rate = 0.0
        proposed_ceiling = needs_limit = None
        if timing.reset:
            ceiling = output_log.copy()
        else:
            source_rise = np.maximum(source_log - self.previous_source_log, 0)
            proposed_ceiling = np.minimum(source_log, self.ceiling_log + source_rise +
                                          self.config.max_added_log_rise_per_second * timing.dt)
            needs_limit = output_log > proposed_ceiling + 1e-12
            if np.any(needs_limit):
                full_output, full_log = evaluate(np.ones_like(selected))
                infeasible_mask = needs_limit & (full_log > proposed_ceiling + 1e-12)
                solvable = needs_limit & ~infeasible_mask
                selected[infeasible_mask] = 1.0
                if np.any(solvable):
                    low, high = desired.copy(), np.ones_like(desired)
                    for _ in range(self.config.binary_search_steps):
                        middle = (low + high) / 2
                        middle_output, middle_log = evaluate(middle)
                        fails = middle_log > proposed_ceiling + 1e-12
                        low[solvable & fails] = middle[solvable & fails]
                        high[solvable & ~fails] = middle[solvable & ~fails]
                        output, output_log = middle_output, middle_log
                    selected[solvable] = high[solvable]
                output, output_log = evaluate(selected)
        required = selected.copy()
        if self.config.max_amount_gradient_per_degree is not None:
            selected = _spatial_majorant(
                required, self.profile, self.config.max_amount_gradient_per_degree)
            output, output_log = evaluate(selected)
        spatially_raised = selected > required + 1e-12
        if not timing.reset:
            ceiling = np.minimum(proposed_ceiling, output_log)
            ceiling[needs_limit & ~infeasible_mask] = proposed_ceiling[needs_limit & ~infeasible_mask]
            ceiling[infeasible_mask] = np.maximum(proposed_ceiling[infeasible_mask],
                                                   output_log[infeasible_mask])
            # Spatial regularization can leave the actual output well below a
            # continuously accumulated ceiling.  Do not bank that unused
            # brightness budget: releasing it when a highlight moves would
            # create a one-frame catch-up rise despite satisfying the older
            # envelope numerically.
            ceiling[spatially_raised] = output_log[spatially_raised]
            envelope_overshoot = float(np.maximum(output_log - proposed_ceiling, 0).max(initial=0))
            added = output_log - self.previous_output_log - source_rise
            realized_rate = float(np.maximum(added, 0).max(initial=0) / timing.dt)

        horizontal, vertical = self.profile.visual_angle_degrees
        rows, columns = selected.shape
        horizontal_deltas = np.abs(np.diff(selected, axis=1))
        vertical_deltas = np.abs(np.diff(selected, axis=0))
        neighbor_deltas = [horizontal_deltas.max(initial=0), vertical_deltas.max(initial=0)]
        neighbor_gradients = [
            horizontal_deltas.max(initial=0) / (horizontal / columns),
            vertical_deltas.max(initial=0) / (vertical / rows),
        ]
        boundary_mean, boundary_p95, boundary_max = _boundary_control_contrast(
            encoded, linear, y, selected, self.partitions, curve_broad, self.profile, self.config)
        spatial_credit = (float(np.maximum(ceiling[spatially_raised] - output_log[spatially_raised], 0)
                                .max(initial=0)) if np.any(spatially_raised) else 0.0)
        added_amounts = np.maximum(selected - desired, 0)
        self.previous_source_log = source_log.copy()
        self.previous_output_log = output_log.copy()
        self.ceiling_log = ceiling.copy()
        self.previous_amounts = selected.copy()
        self.retained_broad = curve_broad if np.any(selected > 0) else 0.0
        trace = {
            "continuity": timing.continuity,
            "desired_amount": desired_amount,
            "selected_amount": float(selected.max(initial=0)),
            "selected_amount_mean": float(selected.mean()),
            "added_amount": float(added_amounts.max(initial=0)),
            "added_amount_mean": float(added_amounts.mean()),
            "required_amount": float(required.max(initial=0)),
            "required_amount_mean": float(required.mean()),
            "spatially_added_amount_mean": float(np.maximum(selected - required, 0).mean()),
            "spatially_regularized_tiles": int(spatially_raised.sum()),
            "max_spatial_ceiling_credit_log": spatial_credit,
            "curve_broad": curve_broad,
            "tile_count": int(source_log.size),
            "extended_tiles": int((added_amounts > 1e-9).sum()),
            "infeasible": bool(np.any(infeasible_mask)),
            "infeasible_tiles": int(infeasible_mask.sum()),
            "max_envelope_overshoot_log": envelope_overshoot,
            "max_realized_added_log_rise_per_second": realized_rate,
            "max_neighbor_amount_delta": float(max(neighbor_deltas)),
            "max_neighbor_amount_gradient_per_degree": float(max(neighbor_gradients)),
            "boundary_control_contrast_mean": boundary_mean,
            "boundary_control_contrast_p95": boundary_p95,
            "boundary_control_contrast_max": boundary_max,
            "output_bits": self.config.output_bits,
        }
        self.last_output, self.last_trace = output.copy(), trace.copy()
        return output, trace

    def snapshot(self) -> dict:
        return {"tiled": True, "local_amount_map": True, "config": asdict(self.config)}


class PixelLogRiseLimiter:
    """Select one amount per angular cell from pixelwise temporal ceilings.

    Unlike ``LocalLogRiseLimiter``, a cell mean cannot hide a bright pixel.
    Unused ceiling credit is capped to the next output quantization step so a
    constraint imposed by another pixel cannot bank a large catch-up rise.
    """

    def __init__(self, profile: DisplayProfile,
                 config: LogRiseLimitConfig = LogRiseLimitConfig()):
        self.profile = profile
        self.config = config
        self.clock = FrameClock()
        self.partitions = self.previous_source_log = self.previous_output_log = None
        self.ceiling_log = self.previous_amounts = None
        self.retained_broad = 0.0
        self.last_output = self.last_trace = None

    def _logs(self, y: np.ndarray) -> np.ndarray:
        return np.log(self.profile.to_nits(y) + self.config.contrast_floor_nits)

    def _quantization_credit(self, encoded: np.ndarray, output: np.ndarray,
                             output_log: np.ndarray) -> np.ndarray:
        levels = float(2**self.config.output_bits - 1)
        current = np.clip(np.where(output <= .0031308, 12.92 * output,
                                   1.055 * output ** (1 / 2.4) - .055), 0, 1)
        original = encoded.astype(np.float64) / 255
        codes = np.rint(current * levels)
        next_encoded = np.minimum(original, np.maximum(current, (codes + 1) / levels))
        next_linear = np.where(next_encoded <= .04045, next_encoded / 12.92,
                               ((next_encoded + .055) / 1.055) ** 2.4)
        next_y = np.sum(next_linear * screen_reference.LUMA, axis=-1)
        return np.maximum(self._logs(next_y) - output_log, 0)

    def process(self, encoded: np.ndarray, linear: np.ndarray, y: np.ndarray,
                desired_amount: float, broad: float, timestamp: float, frame_id: int,
                *, discontinuity: bool = False) -> tuple[np.ndarray, dict]:
        if (encoded.dtype != np.uint8 or encoded.ndim != 3 or encoded.shape[-1] != 3 or
                linear.shape != encoded.shape or y.shape != encoded.shape[:2]):
            raise ValidationError("recovery_limit_shape")
        if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (desired_amount, broad)) or
                not all(0 <= value <= 1 for value in (desired_amount, broad))):
            raise ValidationError("recovery_limit_control")
        timing = self.clock.accept(frame_id, timestamp, encoded.shape, discontinuity)
        if timing.duplicate:
            return self.last_output.copy(), {**self.last_trace, "continuity": "duplicate"}
        if timing.reset or self.partitions is None:
            self.partitions = _partitions(y.shape, self.profile, self.config, True)
            self.previous_amounts = np.zeros(self.partitions[2].shape, dtype=np.float64)
            self.retained_broad = 0.0
        curve_broad = max(broad, self.retained_broad if np.any(self.previous_amounts > 0) else 0.0)

        def evaluate(amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            amount_map = _expand_tiles(amounts, self.partitions, y.shape)
            output = apply_quantized_screen_curve_map(encoded, linear, y, amount_map,
                                                       curve_broad, self.config.output_bits)
            output_y = np.sum(output * screen_reference.LUMA, axis=-1)
            return output, self._logs(output_y)

        source_log = self._logs(y)
        tile_shape = self.partitions[2].shape
        desired = np.full(tile_shape, desired_amount, dtype=np.float64)
        selected = desired.copy()
        output, output_log = evaluate(selected)
        infeasible_tiles = np.zeros(tile_shape, dtype=bool)
        proposed_ceiling = source_rise = None
        if timing.reset:
            ceiling = output_log.copy()
            realized_rate = envelope_overshoot = 0.0
        else:
            source_rise = np.maximum(source_log - self.previous_source_log, 0)
            proposed_ceiling = np.minimum(
                source_log, self.ceiling_log + source_rise +
                self.config.max_added_log_rise_per_second * timing.dt)

            def tile_overshoot(values: np.ndarray) -> np.ndarray:
                return _maxima(np.maximum(values - proposed_ceiling, 0), self.partitions)

            needs_limit = tile_overshoot(output_log) > 1e-12
            if np.any(needs_limit):
                _, full_log = evaluate(np.ones_like(selected))
                infeasible_tiles = needs_limit & (tile_overshoot(full_log) > 1e-12)
                solvable = needs_limit & ~infeasible_tiles
                selected[infeasible_tiles] = 1.0
                if np.any(solvable):
                    low, high = desired.copy(), np.ones_like(desired)
                    for _ in range(self.config.binary_search_steps):
                        middle = (low + high) / 2
                        _, middle_log = evaluate(middle)
                        fails = tile_overshoot(middle_log) > 1e-12
                        low[solvable & fails] = middle[solvable & fails]
                        high[solvable & ~fails] = middle[solvable & ~fails]
                    selected[solvable] = high[solvable]
                output, output_log = evaluate(selected)

        required = selected.copy()
        if self.config.max_amount_gradient_per_degree is not None:
            selected = _spatial_majorant(
                required, self.profile, self.config.max_amount_gradient_per_degree)
            output, output_log = evaluate(selected)
        spatially_raised = selected > required + 1e-12
        if not timing.reset:
            credit = self._quantization_credit(encoded, output, output_log)
            ceiling = np.minimum(proposed_ceiling, output_log + credit)
            spatial_mask = _expand_tiles(spatially_raised, self.partitions, y.shape).astype(bool)
            ceiling[spatial_mask] = output_log[spatial_mask]
            infeasible_pixels = output_log > proposed_ceiling + 1e-12
            ceiling[infeasible_pixels] = output_log[infeasible_pixels]
            envelope_overshoot = float(
                np.maximum(output_log - proposed_ceiling, 0).max(initial=0))
            added = output_log - self.previous_output_log - source_rise
            realized_rate = float(np.maximum(added, 0).max(initial=0) / timing.dt)
        else:
            credit = np.zeros_like(output_log)
            spatial_mask = np.zeros_like(output_log, dtype=bool)

        horizontal, vertical = self.profile.visual_angle_degrees
        rows, columns = selected.shape
        horizontal_deltas = np.abs(np.diff(selected, axis=1))
        vertical_deltas = np.abs(np.diff(selected, axis=0))
        neighbor_deltas = [horizontal_deltas.max(initial=0), vertical_deltas.max(initial=0)]
        neighbor_gradients = [
            horizontal_deltas.max(initial=0) / (horizontal / columns),
            vertical_deltas.max(initial=0) / (vertical / rows),
        ]
        boundary_mean, boundary_p95, boundary_max = _boundary_control_contrast(
            encoded, linear, y, selected, self.partitions, curve_broad, self.profile, self.config)
        added_amounts = np.maximum(selected - desired, 0)
        unused_credit = np.maximum(ceiling - output_log, 0)
        self.previous_source_log = source_log.copy()
        self.previous_output_log = output_log.copy()
        self.ceiling_log = ceiling.copy()
        self.previous_amounts = selected.copy()
        self.retained_broad = curve_broad if np.any(selected > 0) else 0.0
        trace = {
            "continuity": timing.continuity,
            "desired_amount": desired_amount,
            "selected_amount": float(selected.max(initial=0)),
            "selected_amount_mean": float(selected.mean()),
            "added_amount": float(added_amounts.max(initial=0)),
            "added_amount_mean": float(added_amounts.mean()),
            "required_amount": float(required.max(initial=0)),
            "required_amount_mean": float(required.mean()),
            "spatially_added_amount_mean": float(np.maximum(selected - required, 0).mean()),
            "spatially_regularized_tiles": int(spatially_raised.sum()),
            "max_spatial_ceiling_credit_log": float(
                unused_credit[spatial_mask].max(initial=0) if np.any(spatial_mask) else 0.0),
            "max_unused_ceiling_credit_log": float(unused_credit.max(initial=0)),
            "max_quantization_credit_log": float(credit.max(initial=0)),
            "curve_broad": curve_broad,
            "tile_count": int(selected.size),
            "extended_tiles": int((added_amounts > 1e-9).sum()),
            "infeasible": bool(np.any(infeasible_tiles)),
            "infeasible_tiles": int(infeasible_tiles.sum()),
            "max_envelope_overshoot_log": envelope_overshoot,
            "max_realized_added_log_rise_per_second": realized_rate,
            "max_neighbor_amount_delta": float(max(neighbor_deltas)),
            "max_neighbor_amount_gradient_per_degree": float(max(neighbor_gradients)),
            "boundary_control_contrast_mean": boundary_mean,
            "boundary_control_contrast_p95": boundary_p95,
            "boundary_control_contrast_max": boundary_max,
            "output_bits": self.config.output_bits,
            "constraint_domain": "pixel_log_luminance",
        }
        self.last_output, self.last_trace = output.copy(), trace.copy()
        return output, trace

    def snapshot(self) -> dict:
        return {"tiled": True, "pixel_envelope": True, "config": asdict(self.config)}
