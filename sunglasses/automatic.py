"""Causal automatic highlight control. No capture, playback or safety classification."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .color import luminance, normalized_rgb
from .filter import FilterConfig, compress_highlights
from .i18n import ValidationError

LIGHT = FilterConfig(.65, .90)
STRONG = FilterConfig(.20, .55)
HIGHLIGHT = FilterConfig(.65, .80)
LOAD_START, LOAD_FULL = .01, .04
TIME_EPSILON = 1e-6


@dataclass(frozen=True)
class AutoConfig:
    strength: float = 1.0
    hold_seconds: float = .20
    release_seconds: float = .50
    repeat_enabled: bool = True
    policy: str = "highlight"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.strength, self.hold_seconds, self.release_seconds)):
            raise ValidationError("parameters_finite")
        if not 0 <= self.strength <= 1:
            raise ValidationError("strength_range")
        if not 0 <= self.hold_seconds <= 5 or not .01 <= self.release_seconds <= 10:
            raise ValidationError("auto_time")
        if type(self.repeat_enabled) is not bool:
            raise ValidationError("auto_repeat")
        if self.policy not in ("highlight", "legacy"):
            raise ValidationError("auto_policy")


@dataclass(frozen=True)
class Timing:
    dt: float
    first: bool
    reset: bool
    duplicate: bool
    continuity: str


class FrameClock:
    """Timestamp validation independent of the CPU/GPU image processing backend."""

    def __init__(self):
        self.last = None
        self.intervals = deque(maxlen=8)

    def accept(self, frame_id: int, timestamp: float, shape: tuple, discontinuity=False) -> Timing:
        if type(frame_id) is not int or not 0 <= frame_id < 2**64 or not math.isfinite(timestamp) or timestamp < 0:
            raise ValidationError("frame_time")
        current = (frame_id, timestamp, shape)
        if self.last is None:
            self.last = current
            return Timing(0, True, True, False, "first")
        old_id, old_time, old_shape = self.last
        if current == self.last and not discontinuity:
            return Timing(0, False, False, True, "duplicate")
        dt = timestamp - old_time
        reset = (discontinuity or shape != old_shape or dt <= 0 or frame_id != old_id + 1)
        if not reset and len(self.intervals) >= 3:
            reset = dt > 1.5 * float(np.median(self.intervals)) + TIME_EPSILON
        self.last = current
        if reset:
            self.intervals.clear()
            return Timing(0, False, True, False, "discontinuity")
        self.intervals.append(dt)
        return Timing(dt, False, False, False, "continuous" if len(self.intervals) >= 3 else "warmup")


def smooth(lo, hi, values):
    z = np.clip((np.asarray(values) - lo) / (hi - lo), 0, 1)
    return z * z * (3 - 2 * z)


def grid_shape(height, width):
    return min(height, max(1, min(64, math.floor(32 * height / width + .5)))), min(32, width)


def features(y, previous=None):
    """Exact integer partition; positive/negative differences precede reduction."""
    height, width = y.shape
    rows, cols = grid_shape(height, width)
    ys = np.arange(rows + 1) * height // rows
    xs = np.arange(cols + 1) * width // cols
    counts = np.diff(ys)[:, None] * np.diff(xs)[None, :]

    def average(values):
        reduced = np.add.reduceat(np.add.reduceat(values.astype(np.float64), ys[:-1], axis=0), xs[:-1], axis=1)
        return (reduced / counts).reshape(-1)

    difference = y - previous if previous is not None else np.zeros_like(y)
    return np.stack((average(y), average(y > .65), average(np.maximum(y - .65, 0)),
                     average(np.maximum(difference, 0)), average(np.maximum(-difference, 0))), axis=1)


class TemporalControl:
    """Finite history per block; time represented as ages, not large float timestamps."""

    def __init__(self, config=AutoConfig()):
        self.config = config
        self.amount = 0.0
        self.hold = 0.0
        self.latch = False
        self.anchor = self.extreme = self.direction = None
        self.turns = []

    def update(self, stats, timing, shape=None):
        if self.config.policy == "highlight":
            # Weight the integer partitions by their pixel counts, including odd sizes.
            weights = None
            if shape is not None:
                height, width = shape
                rows, cols = grid_shape(height, width)
                weights = (np.diff(np.arange(rows + 1) * height // rows)[:, None] *
                           np.diff(np.arange(cols + 1) * width // cols)[None, :]).ravel()
            area = (float(np.rint(stats[:, 1] * weights).sum() / weights.sum()) if weights is not None
                    else float(stats[:, 1].mean()))
            # Continuous highlight load: a white pixel counts as one, a pixel
            # just above the knee counts almost zero. Raw bright-area counting
            # would jump when a large near-threshold region crosses the knee.
            load = float(np.average(stats[:, 2], weights=weights) / (1 - HIGHLIGHT.knee))
            self.amount = float(smooth(LOAD_START, LOAD_FULL, load))
            # No hold or history tail: a frame below the gate is exactly unchanged.
            self.hold, self.latch = 0.0, self.amount > 0
            return dict(amount=self.amount, bright=self.amount, rise=0.0, repeat=0.0,
                        latch=self.latch, hold_remaining=0.0, reason_bits=int(self.latch),
                        bright_area=area, highlight_load=load)
        means = stats[:, 0]
        if timing.reset:
            self.anchor = means.copy()
            self.extreme = means.copy()
            self.direction = np.zeros(len(means), dtype=np.int8)
            self.turns = [deque(maxlen=8) for _ in means]
            self.latch = False
            self.hold = 0.0 if timing.first else self.config.hold_seconds
            if timing.first:
                self.amount = 0.0
        for j, value in enumerate(means):
            ages = self.turns[j]
            for k in range(len(ages)):
                ages[k] += timing.dt
            while ages and ages[0] > .75 + TIME_EPSILON:
                ages.popleft()
            direction = self.direction[j]
            if direction == 0:
                if abs(value - self.anchor[j]) >= .08:
                    self.direction[j] = 1 if value > self.anchor[j] else -1
                    self.extreme[j] = value
            elif direction > 0:
                self.extreme[j] = max(self.extreme[j], value)
                if self.extreme[j] - value >= .08:
                    ages.append(0.0)
                    self.direction[j], self.extreme[j] = -1, value
            else:
                self.extreme[j] = min(self.extreme[j], value)
                if value - self.extreme[j] >= .08:
                    ages.append(0.0)
                    self.direction[j], self.extreme[j] = 1, value
        bright = float(smooth(.01, .05, stats[:, 2]).max())
        rise = float((smooth(.02, .15, stats[:, 3]) * smooth(.20, .45, means)).max())
        repeat = float(smooth(2, 4, np.array([len(t) for t in self.turns])).max()) if self.config.repeat_enabled else 0.0
        demand = max(bright, rise, repeat)
        if not self.latch and demand >= .60:
            self.latch = True
        if self.latch and demand <= .30:
            self.latch = False
        target = demand if self.latch else 0.0
        # Subtract elapsed time before refreshing the hold on a new attack.
        release_dt = max(0.0, timing.dt - self.hold)
        self.hold = max(0.0, self.hold - timing.dt)
        if self.latch and target >= self.amount:
            self.amount, self.hold = target, self.config.hold_seconds
        elif self.hold <= TIME_EPSILON:
            self.amount = target + (self.amount - target) * math.exp(-release_dt / self.config.release_seconds)
        if not self.latch and self.hold <= TIME_EPSILON and self.amount < .001:
            self.amount = 0.0
        reasons = int(bright > 0) | (int(rise > 0) << 1) | (int(repeat > 0) << 2)
        return dict(amount=self.amount, bright=bright, rise=rise, repeat=repeat,
                    latch=self.latch, hold_remaining=self.hold, reason_bits=reasons)


def apply_auto(rgb, amount, strength=1.0, *, policy="highlight"):
    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValidationError("auto_amount")
    config = AutoConfig(strength=strength, policy=policy)
    rgb = normalized_rgb(rgb)
    if config.strength == 0:
        return rgb.copy()
    if policy == "highlight":
        if amount == 0:
            return rgb.copy()
        compressed = compress_highlights(rgb, HIGHLIGHT)
        return rgb + (config.strength * amount) * (compressed - rgb)
    light = compress_highlights(rgb, LIGHT)
    strong = compress_highlights(rgb, STRONG)
    # Difference form preserves unchanged dark pixels exactly, even during recovery.
    return np.minimum(rgb, np.maximum(0, rgb + config.strength * ((light - rgb) + amount * (strong - light))))


class AutoFilter:
    def __init__(self, config=AutoConfig()):
        self.config = config
        self.clock = FrameClock()
        self.control = TemporalControl(config)
        self.previous_y = None
        self.last_output = self.last_trace = None

    def process(self, frame, timestamp, frame_id, *, discontinuity=False):
        rgb = normalized_rgb(frame)
        if rgb.ndim != 3:
            raise ValidationError("auto_shape")
        timing = self.clock.accept(frame_id, timestamp, rgb.shape, discontinuity)
        if timing.duplicate:
            return self.last_output.copy(), {**self.last_trace, "continuity": "duplicate"}
        y = luminance(rgb)
        stats = features(y, None if timing.reset else self.previous_y)
        trace = self.control.update(stats, timing, y.shape)
        output = apply_auto(rgb, trace["amount"], self.config.strength, policy=self.config.policy)
        trace.update(frame_id=frame_id, timestamp=timestamp, continuity=timing.continuity)
        self.previous_y = y.copy()
        self.last_output, self.last_trace = output.copy(), trace.copy()
        return output, trace
