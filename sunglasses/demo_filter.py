"""Frozen CPU reference for offline screen-burst demonstrations.

This is the screen.hlsl algorithm validated on 2026-10-12, not AutoFilter.
Changing the shader requires a fresh equivalence check before updating SHA256.
"""
from dataclasses import asdict, dataclass

import numpy as np

SHADER_SHA256 = "2d04a54e853ed74208b12d421301698bf081d9aea72dd9d2a844c7c11cdcb234"
LUMA = np.array([.2126, .7152, .0722], np.float32)
_ENCODED = np.arange(256, dtype=np.float32) / 255
LINEAR = np.where(_ENCODED <= .04045, _ENCODED / 12.92, ((_ENCODED + .055) / 1.055) ** 2.4)
STRENGTHS = (0., .35, .5, 1.)


def smooth(lo, hi, value):
    x = np.clip((value - lo) / (hi - lo), 0, 1)
    return x * x * (3 - 2 * x)


@dataclass
class Controller:
    previous: tuple[float, float, float] | None = None
    peak: float = 0.
    age: float = 0.
    broad: float = 0.

    def snapshot(self):
        return asdict(self)

    def step(self, load, mean, t):
        demand = 0.
        if self.previous is not None:
            pt, pl, pm = self.previous
            self.age = min(.32, self.age + float(np.float32(min(1., t - pt))))
            if self.age >= .32:
                self.broad = 0.
            demand = float(smooth(.015, .08, load - pl) * smooth(.01, .08, mean - pm))
            if demand > 0:
                self.peak = max(demand, self.peak * (1 - smooth(.04, .32, self.age)))
                self.age = 0.
            if self.age >= .32:
                self.peak = 0.
            self.broad = (0. if self.age >= .32 or self.peak <= 0 else
                          max(self.broad, float(smooth(.08, .25, load))))
        self.previous = (t, load, mean)
        return float(self.peak * (1 - smooth(.04, .32, self.age)) * smooth(.01, .04, load))


def stats(frame):
    linear = LINEAR[frame]
    y = np.sum(linear * LUMA, axis=-1)
    return linear, y, float(np.maximum(y - .65, 0).mean() / .35), float(y.mean())


def apply(frame, linear, y, gain, broad, strength):
    original = frame.astype(np.float32) / 255
    amount = min(2 * strength, 1) * gain
    if amount <= 0:
        return frame.copy()
    extra = smooth(.5, 1, strength) * broad
    knee, span = .65 - .20 * extra, .15 - .05 * extra
    eligible = y > knee
    mapped = knee + span * (1 - np.exp(-np.maximum(y - knee, 0) / span))
    factor = np.where(eligible, 1 + amount * (mapped / np.maximum(y, 1e-20) - 1), 1)
    corrected = np.maximum(linear * factor[..., None], 0)
    desired = np.minimum(original, np.where(corrected <= .0031308, 12.92 * corrected,
                                            1.055 * corrected ** (1 / 2.4) - .055))
    desired[~eligible] = original[~eligible]
    return np.rint(desired * 255).astype(np.uint8)
