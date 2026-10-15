"""Offline temporal-vision proxies and display-layer change attribution.

The measurements in this module are engineering diagnostics.  They are not a
medical risk score, a WCAG/ITU conformance test, or a substitute for measuring
the photons emitted by a display.  Inputs are normalized linear-light sRGB.
"""

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from .automatic import FrameClock
from .color import luminance, normalized_rgb
from .i18n import ValidationError


@dataclass(frozen=True)
class DisplayProfile:
    """Explicit SDR display and viewing assumptions used by the proxies."""

    peak_luminance_nits: float
    black_luminance_nits: float
    diagonal_inches: float
    resolution: tuple[int, int]
    viewing_distance_m: float

    def __post_init__(self) -> None:
        numeric = (self.peak_luminance_nits, self.black_luminance_nits,
                   self.diagonal_inches, self.viewing_distance_m)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise ValidationError("perceptual_profile_finite")
        if not (0 <= self.black_luminance_nits < self.peak_luminance_nits <= 10_000):
            raise ValidationError("perceptual_profile_luminance")
        if not (.1 <= self.diagonal_inches <= 200 and .05 <= self.viewing_distance_m <= 10):
            raise ValidationError("perceptual_profile_geometry")
        if (not isinstance(self.resolution, tuple) or len(self.resolution) != 2 or
                any(type(value) is not int or not 16 <= value <= 32_768 for value in self.resolution)):
            raise ValidationError("perceptual_profile_resolution")

    @property
    def physical_size_m(self) -> tuple[float, float]:
        width_pixels, height_pixels = self.resolution
        diagonal_m = self.diagonal_inches * .0254
        scale = diagonal_m / math.hypot(width_pixels, height_pixels)
        return width_pixels * scale, height_pixels * scale

    @property
    def visual_angle_degrees(self) -> tuple[float, float]:
        width_m, height_m = self.physical_size_m
        angle = lambda size: math.degrees(2 * math.atan(size / (2 * self.viewing_distance_m)))
        return angle(width_m), angle(height_m)

    @property
    def pixels_per_degree(self) -> tuple[float, float]:
        width_degrees, height_degrees = self.visual_angle_degrees
        return self.resolution[0] / width_degrees, self.resolution[1] / height_degrees

    def to_nits(self, relative_luminance):
        values = np.asarray(relative_luminance)
        return self.black_luminance_nits + values * (self.peak_luminance_nits - self.black_luminance_nits)


@dataclass(frozen=True)
class PerceptualConfig:
    """Search parameters for interpretable proxies, not validated thresholds."""

    adaptation_seconds: float = 1.0
    energy_seconds: float = .25
    contrast_floor_nits: float = .5
    cell_degrees: float = 2.0
    temporal_bands_hz: tuple[float, ...] = (2.0, 8.0, 15.0)
    changed_log_contrast: float = math.log(1.05)
    transition_log_contrast: float = math.log(1.10)
    opposition_window_seconds: float = .5
    event_area_fraction: float = .10
    coherent_area_fraction: float = .50
    coherent_direction_fraction: float = .90
    highlight_luminance_nits: float = 100.0
    red_change_threshold: float = .10

    def __post_init__(self) -> None:
        scalars = (self.adaptation_seconds, self.energy_seconds, self.contrast_floor_nits,
                   self.cell_degrees, self.changed_log_contrast, self.transition_log_contrast,
                   self.opposition_window_seconds, self.event_area_fraction,
                   self.coherent_area_fraction, self.coherent_direction_fraction,
                   self.highlight_luminance_nits, self.red_change_threshold)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scalars):
            raise ValidationError("perceptual_config_finite")
        if not (.01 <= self.adaptation_seconds <= 30 and .01 <= self.energy_seconds <= 10):
            raise ValidationError("perceptual_config_time")
        if not (0 < self.contrast_floor_nits <= 100 and .1 <= self.cell_degrees <= 30):
            raise ValidationError("perceptual_config_scale")
        if (not isinstance(self.temporal_bands_hz, tuple) or not self.temporal_bands_hz or
                len(self.temporal_bands_hz) > 8 or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                    for value in self.temporal_bands_hz) or
                tuple(sorted(self.temporal_bands_hz)) != self.temporal_bands_hz or
                len(set(self.temporal_bands_hz)) != len(self.temporal_bands_hz)):
            raise ValidationError("perceptual_config_bands")
        if not (0 < self.changed_log_contrast <= self.transition_log_contrast <= 4):
            raise ValidationError("perceptual_config_contrast")
        if not (.01 <= self.opposition_window_seconds <= 5):
            raise ValidationError("perceptual_config_opposition_time")
        if not all(0 < value <= 1 for value in (self.event_area_fraction, self.coherent_area_fraction,
                                                self.coherent_direction_fraction)):
            raise ValidationError("perceptual_config_area")
        if not (0 < self.highlight_luminance_nits <= 10_000 and 0 < self.red_change_threshold <= 1):
            raise ValidationError("perceptual_config_threshold")


@dataclass(frozen=True)
class FrameMeasurement:
    frame_id: int
    timestamp: float
    continuity: str
    mean_nits: float
    p99_nits: float
    peak_nits: float
    highlight_area_fraction: float
    adapted_contrast_rms: float
    adapted_contrast_p95: float
    mean_abs_log_change: float
    p95_abs_log_change: float
    changed_area_fraction: float
    opposed_area_fraction: float
    direction_coherence: float
    coherent_step: bool
    red_change_mean: float
    red_changed_area_fraction: float
    band_rms: tuple[float, ...]
    band_p95: tuple[float, ...]
    temporal_energy_mean: float
    temporal_energy_p95: float
    opposing_transitions_1s: int
    nyquist_hz: float | None


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    flat_values = np.asarray(values, dtype=np.float64).ravel()
    flat_weights = np.asarray(weights, dtype=np.float64).ravel()
    if flat_values.size != flat_weights.size or not flat_values.size or flat_weights.sum() <= 0:
        raise ValueError("invalid weighted quantile input")
    order = np.argsort(flat_values, kind="stable")
    cumulative = np.cumsum(flat_weights[order])
    index = min(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"), len(order) - 1)
    return float(flat_values[order[index]])


def _grid(profile: DisplayProfile, shape: tuple[int, int], cell_degrees: float):
    height, width = shape
    horizontal, vertical = profile.visual_angle_degrees
    rows = max(1, min(height, 64, math.ceil(vertical / cell_degrees)))
    cols = max(1, min(width, 96, math.ceil(horizontal / cell_degrees)))
    ys = np.arange(rows + 1) * height // rows
    xs = np.arange(cols + 1) * width // cols
    weights = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float64)
    return ys, xs, weights


def _block_means(values: np.ndarray, ys: np.ndarray, xs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    reduced = np.add.reduceat(np.add.reduceat(values.astype(np.float64), ys[:-1], axis=0), xs[:-1], axis=1)
    return reduced / weights


class PerceptualAnalyzer:
    """Causal, timestamp-aware measurements for one full-frame image stream."""

    def __init__(self, profile: DisplayProfile, config: PerceptualConfig = PerceptualConfig()):
        self.profile = profile
        self.config = config
        self.clock = FrameClock()
        self.last_measurement = None
        self.previous_log = self.previous_red = None
        self.last_sign = self.last_sign_time = None
        self.adaptation = None
        self.fast = self.slow = self.energy = None
        self.ys = self.xs = self.tile_weights = None
        self.opposition_times = deque()

    def _reset(self, log_luminance: np.ndarray, red: np.ndarray, tiles: np.ndarray) -> None:
        self.previous_log = log_luminance.copy()
        self.previous_red = red.copy()
        self.last_sign = np.zeros(log_luminance.shape, dtype=np.int8)
        self.last_sign_time = np.full(log_luminance.shape, -np.inf, dtype=np.float32)
        self.adaptation = tiles.copy()
        self.fast = [tiles.copy() for _ in self.config.temporal_bands_hz]
        self.slow = [tiles.copy() for _ in self.config.temporal_bands_hz]
        self.energy = [np.zeros_like(tiles) for _ in self.config.temporal_bands_hz]
        self.opposition_times.clear()

    def process(self, frame, timestamp: float, frame_id: int, *, discontinuity: bool = False) -> FrameMeasurement:
        rgb = normalized_rgb(frame)
        if rgb.ndim != 3:
            raise ValidationError("perceptual_shape")
        timing = self.clock.accept(frame_id, timestamp, rgb.shape, discontinuity)
        if timing.duplicate:
            return replace(self.last_measurement, continuity="duplicate")

        relative_y = luminance(rgb)
        nits = self.profile.to_nits(relative_y).astype(np.float64, copy=False)
        log_luminance = np.log(nits + self.config.contrast_floor_nits)
        red = np.maximum(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]), 0).astype(np.float64)
        if timing.reset or self.ys is None or self.previous_log.shape != log_luminance.shape:
            self.ys, self.xs, self.tile_weights = _grid(self.profile, relative_y.shape, self.config.cell_degrees)
        tiles = _block_means(log_luminance, self.ys, self.xs, self.tile_weights)

        zeros = tuple(0.0 for _ in self.config.temporal_bands_hz)
        if timing.reset:
            self._reset(log_luminance, red, tiles)
            values = dict(adapted_contrast_rms=0.0, adapted_contrast_p95=0.0,
                          mean_abs_log_change=0.0, p95_abs_log_change=0.0,
                          changed_area_fraction=0.0, opposed_area_fraction=0.0,
                          direction_coherence=0.0, coherent_step=False,
                          red_change_mean=0.0, red_changed_area_fraction=0.0,
                          band_rms=zeros, band_p95=zeros,
                          temporal_energy_mean=0.0, temporal_energy_p95=0.0,
                          opposing_transitions_1s=0, nyquist_hz=None)
        else:
            delta = log_luminance - self.previous_log
            absolute_delta = np.abs(delta)
            changed = absolute_delta >= self.config.changed_log_contrast
            significant = absolute_delta >= self.config.transition_log_contrast
            signs = np.sign(delta).astype(np.int8)
            recent = timestamp - self.last_sign_time <= self.config.opposition_window_seconds
            opposed = significant & recent & (signs == -self.last_sign)
            opposed_area = float(opposed.mean())
            self.last_sign[significant] = signs[significant]
            self.last_sign_time[significant] = timestamp
            if opposed_area >= self.config.event_area_fraction:
                self.opposition_times.append(timestamp)
            while self.opposition_times and self.opposition_times[0] < timestamp - 1.0:
                self.opposition_times.popleft()

            mean_absolute = float(absolute_delta.mean(dtype=np.float64))
            coherence = abs(float(delta.mean(dtype=np.float64))) / mean_absolute if mean_absolute else 0.0
            red_delta = np.abs(red - self.previous_red)
            contrast = tiles - self.adaptation
            adapt_alpha = 1 - math.exp(-timing.dt / self.config.adaptation_seconds)
            self.adaptation += adapt_alpha * contrast

            band_rms, band_p95 = [], []
            energy_alpha = 1 - math.exp(-timing.dt / self.config.energy_seconds)
            for index, frequency in enumerate(self.config.temporal_bands_hz):
                fast_tau = 1 / (2 * math.pi * frequency * math.sqrt(2))
                slow_tau = math.sqrt(2) / (2 * math.pi * frequency)
                fast_alpha = 1 - math.exp(-timing.dt / fast_tau)
                slow_alpha = 1 - math.exp(-timing.dt / slow_tau)
                self.fast[index] += fast_alpha * (tiles - self.fast[index])
                self.slow[index] += slow_alpha * (tiles - self.slow[index])
                band = self.fast[index] - self.slow[index]
                self.energy[index] += energy_alpha * (band * band - self.energy[index])
                band_rms.append(math.sqrt(float(np.average(band * band, weights=self.tile_weights))))
                band_p95.append(_weighted_quantile(np.abs(band), self.tile_weights, .95))
            combined_energy = np.mean(self.energy, axis=0)
            changed_area = float(changed.mean())
            coherent = (changed_area >= self.config.coherent_area_fraction and
                        coherence >= self.config.coherent_direction_fraction and
                        opposed_area < self.config.event_area_fraction)
            values = dict(
                adapted_contrast_rms=math.sqrt(float(np.average(contrast * contrast, weights=self.tile_weights))),
                adapted_contrast_p95=_weighted_quantile(np.abs(contrast), self.tile_weights, .95),
                mean_abs_log_change=mean_absolute,
                p95_abs_log_change=float(np.quantile(absolute_delta, .95)),
                changed_area_fraction=changed_area,
                opposed_area_fraction=opposed_area,
                direction_coherence=coherence,
                coherent_step=coherent,
                red_change_mean=float(red_delta.mean(dtype=np.float64)),
                red_changed_area_fraction=float((red_delta >= self.config.red_change_threshold).mean()),
                band_rms=tuple(band_rms), band_p95=tuple(band_p95),
                temporal_energy_mean=math.sqrt(float(np.average(combined_energy, weights=self.tile_weights))),
                temporal_energy_p95=math.sqrt(_weighted_quantile(combined_energy, self.tile_weights, .95)),
                opposing_transitions_1s=len(self.opposition_times), nyquist_hz=.5 / timing.dt)
            self.previous_log = log_luminance.copy()
            self.previous_red = red.copy()

        measurement = FrameMeasurement(
            frame_id=frame_id, timestamp=timestamp, continuity=timing.continuity,
            mean_nits=float(nits.mean(dtype=np.float64)), p99_nits=float(np.quantile(nits, .99)),
            peak_nits=float(nits.max()),
            highlight_area_fraction=float((nits >= self.config.highlight_luminance_nits).mean()), **values)
        self.last_measurement = measurement
        return measurement


def _sample_weights(measurements: list[FrameMeasurement]) -> np.ndarray:
    if len(measurements) == 1:
        return np.ones(1)
    times = np.array([item.timestamp for item in measurements], dtype=np.float64)
    intervals = np.diff(times)
    positive = intervals[intervals > 0]
    tail = float(np.median(positive)) if positive.size else 1.0
    return np.r_[np.maximum(intervals, 0), tail]


def summarize_measurements(measurements) -> dict:
    """Summarize unique samples; percentiles are display-time weighted."""
    items = [item for item in measurements if item.continuity != "duplicate"]
    if not items:
        raise ValidationError("perceptual_empty")
    weights = _sample_weights(items)
    value = lambda name: np.array([getattr(item, name) for item in items], dtype=np.float64)
    return {
        "frames": len(items),
        "duration_seconds": max(0.0, items[-1].timestamp - items[0].timestamp),
        "mean_luminance_nits": float(np.average(value("mean_nits"), weights=weights)),
        "peak_luminance_nits": float(value("peak_nits").max()),
        "highlight_area_fraction_p95": _weighted_quantile(value("highlight_area_fraction"), weights, .95),
        "adapted_contrast_p95": _weighted_quantile(value("adapted_contrast_p95"), weights, .95),
        "local_log_change_p95": _weighted_quantile(value("p95_abs_log_change"), weights, .95),
        "changed_area_fraction_p95": _weighted_quantile(value("changed_area_fraction"), weights, .95),
        "max_opposed_area_fraction": float(value("opposed_area_fraction").max()),
        "max_opposing_transitions_1s": int(value("opposing_transitions_1s").max()),
        "coherent_step_frames": sum(item.coherent_step for item in items),
        "red_change_mean_p95": _weighted_quantile(value("red_change_mean"), weights, .95),
        "red_changed_area_fraction_max": float(value("red_changed_area_fraction").max()),
        "temporal_energy_p95": _weighted_quantile(value("temporal_energy_p95"), weights, .95),
        "temporal_energy_max": float(value("temporal_energy_p95").max()),
        "minimum_nyquist_hz": min((item.nyquist_hz for item in items if item.nyquist_hz is not None), default=None),
        "discontinuities": sum(item.continuity == "discontinuity" for item in items),
    }


@dataclass(frozen=True)
class PairedMeasurement:
    frame_id: int
    timestamp: float
    continuity: str
    source: FrameMeasurement
    output: FrameMeasurement
    added_rise_nits_mean: float
    added_rise_nits_p95: float
    added_rise_nits_max: float
    added_rise_area_over_1_nit: float
    added_fall_nits_mean: float
    added_fall_nits_p95: float
    added_fall_nits_max: float
    added_fall_area_over_1_nit: float
    temporal_residual_nits_mean: float
    temporal_residual_nits_p95: float
    temporal_residual_nits_max: float
    spatial_residual_log_gradient_mean: float
    spatial_residual_log_gradient_p95: float
    spatial_residual_log_gradient_max: float
    same_frame_brightening_nits_max: float
    same_frame_dimming_nits_mean: float
    same_frame_dimming_nits_p95: float
    same_frame_dimming_nits_max: float


class PairedPerceptualAnalyzer:
    """Compare source and output while separating filter-added transitions."""

    def __init__(self, profile: DisplayProfile, config: PerceptualConfig = PerceptualConfig()):
        self.profile = profile
        self.source = PerceptualAnalyzer(profile, config)
        self.output = PerceptualAnalyzer(profile, config)
        self.previous_source = self.previous_output = self.last_measurement = None

    def process(self, source_frame, output_frame, timestamp: float, frame_id: int,
                *, discontinuity: bool = False) -> PairedMeasurement:
        source_rgb, output_rgb = normalized_rgb(source_frame), normalized_rgb(output_frame)
        if source_rgb.shape != output_rgb.shape or source_rgb.ndim != 3:
            raise ValidationError("perceptual_pair_shape")
        source_measurement = self.source.process(source_rgb, timestamp, frame_id, discontinuity=discontinuity)
        output_measurement = self.output.process(output_rgb, timestamp, frame_id, discontinuity=discontinuity)
        if source_measurement.continuity == "duplicate":
            return replace(self.last_measurement, continuity="duplicate",
                           source=source_measurement, output=output_measurement)

        source_nits = self.profile.to_nits(luminance(source_rgb)).astype(np.float64, copy=False)
        output_nits = self.profile.to_nits(luminance(output_rgb)).astype(np.float64, copy=False)
        if source_measurement.continuity in ("first", "discontinuity"):
            added_rise = added_fall = residual = np.zeros_like(source_nits)
        else:
            source_delta = source_nits - self.previous_source
            output_delta = output_nits - self.previous_output
            added_rise = np.maximum(output_delta - np.maximum(source_delta, 0), 0)
            added_fall = np.maximum(-output_delta - np.maximum(-source_delta, 0), 0)
            residual = np.abs(output_delta - source_delta)
        brightening = np.maximum(output_nits - source_nits, 0)
        dimming = np.maximum(source_nits - output_nits, 0)
        attenuation = (np.log(source_nits + self.source.config.contrast_floor_nits) -
                       np.log(output_nits + self.source.config.contrast_floor_nits))
        spatial_edges = np.concatenate((np.abs(np.diff(attenuation, axis=0)).ravel(),
                                        np.abs(np.diff(attenuation, axis=1)).ravel()))
        spatial_mean = float(spatial_edges.mean(dtype=np.float64)) if spatial_edges.size else 0.0
        spatial_p95 = float(np.quantile(spatial_edges, .95)) if spatial_edges.size else 0.0
        spatial_max = float(spatial_edges.max()) if spatial_edges.size else 0.0
        measurement = PairedMeasurement(
            frame_id=frame_id, timestamp=timestamp, continuity=source_measurement.continuity,
            source=source_measurement, output=output_measurement,
            added_rise_nits_mean=float(added_rise.mean(dtype=np.float64)),
            added_rise_nits_p95=float(np.quantile(added_rise, .95)),
            added_rise_nits_max=float(added_rise.max()),
            added_rise_area_over_1_nit=float((added_rise > 1).mean()),
            added_fall_nits_mean=float(added_fall.mean(dtype=np.float64)),
            added_fall_nits_p95=float(np.quantile(added_fall, .95)),
            added_fall_nits_max=float(added_fall.max()),
            added_fall_area_over_1_nit=float((added_fall > 1).mean()),
            temporal_residual_nits_mean=float(residual.mean(dtype=np.float64)),
            temporal_residual_nits_p95=float(np.quantile(residual, .95)),
            temporal_residual_nits_max=float(residual.max()),
            spatial_residual_log_gradient_mean=spatial_mean,
            spatial_residual_log_gradient_p95=spatial_p95,
            spatial_residual_log_gradient_max=spatial_max,
            same_frame_brightening_nits_max=float(brightening.max()),
            same_frame_dimming_nits_mean=float(dimming.mean(dtype=np.float64)),
            same_frame_dimming_nits_p95=float(np.quantile(dimming, .95)),
            same_frame_dimming_nits_max=float(dimming.max()))
        self.previous_source = source_nits.copy()
        self.previous_output = output_nits.copy()
        self.last_measurement = measurement
        return measurement


def summarize_paired(measurements) -> dict:
    items = [item for item in measurements if item.continuity != "duplicate"]
    if not items:
        raise ValidationError("perceptual_empty")
    weights = _sample_weights([item.source for item in items])
    value = lambda name: np.array([getattr(item, name) for item in items], dtype=np.float64)
    return {
        "added_rise_nits_mean_p95": _weighted_quantile(value("added_rise_nits_mean"), weights, .95),
        "added_rise_nits_p95_max": float(value("added_rise_nits_p95").max()),
        "added_rise_nits_max": float(value("added_rise_nits_max").max()),
        "added_rise_area_over_1_nit_max": float(value("added_rise_area_over_1_nit").max()),
        "added_fall_nits_mean_p95": _weighted_quantile(value("added_fall_nits_mean"), weights, .95),
        "added_fall_nits_p95_max": float(value("added_fall_nits_p95").max()),
        "added_fall_nits_max": float(value("added_fall_nits_max").max()),
        "added_fall_area_over_1_nit_max": float(value("added_fall_area_over_1_nit").max()),
        "temporal_residual_nits_mean_p95": _weighted_quantile(value("temporal_residual_nits_mean"), weights, .95),
        "temporal_residual_nits_p95_max": float(value("temporal_residual_nits_p95").max()),
        "temporal_residual_nits_max": float(value("temporal_residual_nits_max").max()),
        "spatial_residual_log_gradient_mean_p95": _weighted_quantile(
            value("spatial_residual_log_gradient_mean"), weights, .95),
        "spatial_residual_log_gradient_p95_max": float(
            value("spatial_residual_log_gradient_p95").max()),
        "spatial_residual_log_gradient_max": float(value("spatial_residual_log_gradient_max").max()),
        "same_frame_brightening_nits_max": float(value("same_frame_brightening_nits_max").max()),
        "same_frame_dimming_nits_mean_p95": _weighted_quantile(value("same_frame_dimming_nits_mean"), weights, .95),
        "same_frame_dimming_nits_max": float(value("same_frame_dimming_nits_max").max()),
        "mean_dimming_over_1_nit_seconds": float(weights[value("same_frame_dimming_nits_mean") > 1].sum()),
    }


@dataclass(frozen=True)
class LayerSample:
    """Synchronized scalar evidence in one photometric scale."""

    timestamp: float
    source: float
    algorithm: float | None = None
    composed: float | None = None
    measured: float | None = None
    capture_discontinuity: bool = False
    overlay_cloaked: bool | None = None
    present_ok: bool | None = None


@dataclass(frozen=True)
class LayerAttribution:
    start_timestamp: float
    end_timestamp: float
    observed_change: float
    origin: str
    origin_contribution: float
    confidence: float
    components: tuple[tuple[str, float], ...]
    evidence: tuple[str, ...]


def attribute_layer_changes(samples, *, change_threshold: float,
                            mixed_confidence: float = .60) -> list[LayerAttribution]:
    """Attribute transitions using a telescoping input-to-photons decomposition."""
    items = list(samples)
    if (not math.isfinite(change_threshold) or change_threshold <= 0 or
            not math.isfinite(mixed_confidence) or not .25 <= mixed_confidence <= 1):
        raise ValidationError("layer_threshold")
    for index, sample in enumerate(items):
        numeric = [sample.timestamp, sample.source, *[value for value in
                   (sample.algorithm, sample.composed, sample.measured) if value is not None]]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise ValidationError("layer_value")
        if index and sample.timestamp <= items[index - 1].timestamp:
            raise ValidationError("layer_time")
        if sample.composed is not None and sample.algorithm is None:
            raise ValidationError("layer_order")
        if sample.measured is not None and sample.composed is None:
            raise ValidationError("layer_order")

    results = []
    names = ("source_content", "algorithm_state", "composition_or_overlay", "display_path")
    fields = ("source", "algorithm", "composed", "measured")
    for previous, current in zip(items, items[1:]):
        if current.capture_discontinuity:
            results.append(LayerAttribution(previous.timestamp, current.timestamp, 0.0,
                           "capture_discontinuity", 0.0, 0.0, (), ("history_reset",)))
            continue
        common = []
        for field in fields:
            old, new = getattr(previous, field), getattr(current, field)
            if old is None or new is None:
                break
            common.append(float(new - old))
        components = [common[0]] if common else []
        components.extend(common[index] - common[index - 1] for index in range(1, len(common)))
        if not components or max(map(abs, components)) < change_threshold:
            continue
        dominant = int(np.argmax(np.abs(components)))
        total = sum(map(abs, components))
        confidence = abs(components[dominant]) / total if total else 0.0
        origin = names[dominant] if confidence >= mixed_confidence else "mixed"
        evidence = []
        if dominant == 2 and previous.overlay_cloaked != current.overlay_cloaked:
            evidence.append("overlay_cloak_changed")
        if current.present_ok is False:
            evidence.append("present_failure")
        if len(common) < len(fields):
            evidence.append(f"unobserved_after_{fields[len(common) - 1]}")
        if abs(common[-1]) < change_threshold:
            evidence.append("downstream_cancelled")
        results.append(LayerAttribution(
            start_timestamp=previous.timestamp, end_timestamp=current.timestamp,
            observed_change=common[-1], origin=origin,
            origin_contribution=components[dominant], confidence=confidence,
            components=tuple(zip(names[:len(components)], components)), evidence=tuple(evidence)))
    return results
