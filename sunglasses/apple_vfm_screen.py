"""Offline numeric reference for the optional Apple VFM desktop adapter.

Not a capture implementation. The MATLAB core is sampled at 60 Hz using the
last observed APL before each tick. Exact-tick captures supply that tick; late
captures cannot rewrite processed history. Gaps over .5 s reset uncertainty.
"""
import math

import numpy as np

from .apple_vfm import AppleVFMConfig, AppleVFMReference, apply_matlab_mitigation

ALGORITHM = "apple_matlab_vfm_60hz_v1"


class AppleVFMScreen:
    def __init__(self):
        self.reference = AppleVFMReference(AppleVFMConfig(60, 100, 1265.625))
        self.encoded = self.linear = self.measurement = self.metadata = None
        self.last_apl = self.last_time = self.next_tick = self.elapsed = 0.

    def _advance(self, timestamp, new_apl=None, reset=False):
        restart = reset or self.measurement is None or timestamp - self.last_time > .5
        now = timestamp if restart else max(timestamp, self.last_time)
        if restart:
            self.reference.reset()
            self.next_tick = now
            old_samples, current_sample = 0, 1
        else:
            old_samples = max(0, math.ceil((now - self.next_tick) * 60 - 1e-7))
            current_sample = int(abs(now - (self.next_tick + old_samples / 60)) < 1e-7)
            if old_samples > 30:
                raise ValueError("apple_vfm_clock")
        for _ in range(old_samples):
            self.measurement = self.reference.process_apl(self.last_apl)
        if new_apl is not None:
            self.last_apl = new_apl
        if current_sample:
            self.measurement = self.reference.process_apl(self.last_apl)
        self.next_tick += (old_samples + current_sample) / 60
        self.last_time = now

    @staticmethod
    def _strength(strength):
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("apple_vfm_strength")

    def _output(self, strength):
        self._strength(strength)
        if self.measurement is None or self.linear is None or self.encoded is None:
            raise ValueError("apple_vfm_no_frame")
        mitigation = self.measurement.applied_mitigation * strength
        if mitigation == 0:
            return self.encoded.astype(np.float32) / 255
        linear = apply_matlab_mitigation(
            self.linear, mitigation, self.measurement.output_adaptation_luminance_nits / 100)
        return np.maximum(linear, 0) ** (1 / 2.2)

    def process(self, encoded, timestamp, frame_id, strength=1.):
        self._strength(strength)
        if (not isinstance(encoded, np.ndarray) or encoded.dtype != np.uint8 or
                encoded.ndim != 3 or encoded.shape[-1] != 3 or min(encoded.shape[:2]) < 16 or
                not math.isfinite(timestamp) or timestamp < 0 or type(frame_id) is not int or frame_id < 0):
            raise ValueError("apple_vfm_frame")
        metadata = (frame_id, timestamp, encoded.shape)
        if metadata == self.metadata:
            return self._output(strength)
        reset = (self.metadata is None or encoded.shape != self.metadata[2] or
                 timestamp <= self.metadata[1] or frame_id != self.metadata[0] + 1)
        self.encoded = encoded.copy()
        self.linear = (encoded.astype(np.float32) / 255) ** 2.2
        apl = float(np.sum(self.linear.astype(np.float64) * [.299, .587, .114], axis=-1).mean())
        self._advance(timestamp, apl, reset)
        self.metadata = metadata
        self.elapsed = 0.
        return self._output(strength)

    def refresh(self, elapsed, strength=1.):
        self._strength(strength)
        if self.metadata is None or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("apple_vfm_refresh")
        self.elapsed = max(self.elapsed, min(86400., elapsed))
        self._advance(self.metadata[1] + self.elapsed)
        return self._output(strength)
