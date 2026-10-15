"""Display-timeline model of the capture -> mask -> composition loop.

The frozen screen algorithm (``demo_filter`` / ``screen.hlsl``) is evaluated as
it is presented, not only against the frame that produced its mask:

* the mask computed from a capture becomes visible ``lag`` display ticks later;
* the clean model delivers only changed images; overlay_dirty also delivers
  frames when the visible mask changes. These are model assumptions, not WGC
  guarantees (the 2026-10-14 static-screen recordings still received frames);
* without new frames the native loop wakes every 50 ms and redraws the cached
  mask at its true age until 0.32 s (``capture.cpp`` refreshMask);
* ``feedback="full"`` models a failed capture exclusion, i.e. the analysed frame
  is the composited screen including the mask.

The model is an offline engineering hypothesis generator.  It does not replace
DWM, WGC or photometric measurements and makes no claims about real latency.
"""

from dataclasses import dataclass

import numpy as np

from .demo_filter import LINEAR, LUMA, Controller, smooth
from .presentation_scenes import Scene

IDLE_WAIT = .05
EXPIRY = .32


@dataclass(frozen=True)
class Timing:
    name: str
    lag: int                  # display ticks from capture to visible mask
    feedback: str = "none"    # "none" | "full"
    overlay_dirty: bool = False  # mask presents also produce (excluded) capture frames
    capture_hz: float | None = None  # optional periodic delivery; not an OS guarantee

    def label(self) -> str:
        return self.name


TIMINGS = (
    Timing("clean-lag1", 1), Timing("clean-lag2", 2), Timing("clean-lag3", 3),
    Timing("dirty-lag2", 2, overlay_dirty=True),
    Timing("feedback-lag1", 1, "full"), Timing("feedback-lag2", 2, "full"), Timing("feedback-lag3", 3, "full"),
)


def _encode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - .055)


def overlay(captured: np.ndarray, amount: float, broad: float, strength: float):
    """Premultiplied 8-bit (color, alpha) exactly following screen.hlsl Overlay."""
    h, w = captured.shape[:2]
    if amount <= 0:
        return np.zeros((h, w, 3), np.float32), np.zeros((h, w, 1), np.float32)
    linear = LINEAR[captured]
    value = linear @ LUMA
    extra = float(smooth(.5, 1, strength)) * broad
    knee, span = .65 - .20 * extra, .15 - .05 * extra
    eligible = value > knee
    mapped = knee + span * (1 - np.exp(-np.maximum(value - knee, 0) / span))
    factor = 1 + amount * (mapped / np.maximum(value, 1e-20) - 1)
    rgb = linear * factor[..., None]
    original = captured.astype(np.float32) / 255
    desired = np.minimum(original, _encode(rgb))
    ratio = np.where(original > 0, desired / np.maximum(original, 1e-20), 1)
    alpha = np.clip(1 - ratio.min(axis=-1, keepdims=True), 0, 1)
    alpha = np.where(eligible[..., None] & (alpha >= 1e-6), alpha, 0)
    color = np.clip(desired - (1 - alpha) * original, 0, alpha)
    # B8G8R8A8 UNORM swap chain.
    return np.rint(color * 255) / 255, np.rint(alpha * 255) / 255


def _stats(frame):
    y = LINEAR[frame] @ LUMA
    return float(np.maximum(y - .65, 0).mean() / .35), float(y.mean())


def simulate(scene: Scene, hz: float, timing: Timing, strength: float):
    """Return per-tick source and composite frames plus loop telemetry."""
    if timing.capture_hz is not None and (not np.isfinite(timing.capture_hz) or not 0 < timing.capture_hz <= hz):
        raise ValueError("Periodic capture rate must be positive and no greater than display rate")
    ticks = int(round(scene.duration * hz))
    dt = 1 / hz
    controller = Controller()
    cached = None                 # last processed captured frame
    state = (0., 0., 0., 0.)      # peak, age, load, broad at the cached capture
    last_received = None          # tick of the last processed capture
    next_idle = None
    expired = True
    pending: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    h, w = scene.render(0.).shape[:2]
    visible = (np.zeros((h, w, 3), np.float32), np.zeros((h, w, 1), np.float32))
    previous_capture = None
    sources = np.empty((ticks, h, w, 3), np.uint8)
    outputs = np.empty_like(sources)
    log = {"captured": np.zeros(ticks, bool), "refreshed": np.zeros(ticks, bool),
           "amount": np.zeros(ticks, np.float32), "alpha_mean": np.zeros(ticks, np.float32)}
    visible_changed = False
    next_capture = 0.

    def mask_for(elapsed):
        if cached is None:
            raise RuntimeError("A captured frame is required before generating a mask")
        peak, age, load, broad = state
        envelope = 1 - float(smooth(.04, .32, age + elapsed))
        amount = min(2 * strength, 1) * peak * envelope * float(smooth(.01, .04, load))
        return amount, overlay(cached, amount, broad, strength)

    for n in range(ticks):
        t = n * dt
        source = scene.render(t)
        sources[n] = source
        while pending and pending[0][0] <= n:
            new = pending.pop(0)[1]
            visible_changed = not (np.array_equal(new[1], visible[1]) and np.array_equal(new[0], visible[0]))
            visible = new
        color, alpha = visible
        composite = np.clip(np.rint((color + (1 - alpha) * (source / 255.)) * 255), 0, 255).astype(np.uint8)
        outputs[n] = composite
        log["alpha_mean"][n] = float(alpha.mean())

        in_gap = any(a <= t < b for a, b in scene.capture_gaps)
        captured_image = composite if timing.feedback == "full" else source
        dirty = previous_capture is None or not np.array_equal(captured_image, previous_capture)
        if timing.overlay_dirty and visible_changed:
            dirty = True
        if timing.capture_hz is not None:
            dirty = t >= next_capture - 1e-9
            if dirty:
                next_capture = (np.floor(t * timing.capture_hz + 1e-9) + 1) / timing.capture_hz
        visible_changed = False
        if not in_gap:
            previous_capture = captured_image
        if dirty and not in_gap:
            load, mean = _stats(captured_image)
            controller.step(load, mean, t)
            cached = captured_image.copy()
            state = (controller.peak, controller.age, load, controller.broad)
            last_received, expired = n, False
            amount, mask = mask_for(0.)
            pending.append((n + timing.lag, mask))
            log["captured"][n] = True
            log["amount"][n] = amount
            next_idle = t + IDLE_WAIT
        elif cached is not None and next_idle is not None and t >= next_idle - 1e-9:
            next_idle = t + IDLE_WAIT
            if not expired and last_received is not None:
                elapsed = (n - last_received) * dt
                amount, mask = mask_for(elapsed)
                pending.append((n + timing.lag, mask))
                log["refreshed"][n] = True
                log["amount"][n] = amount
                expired = elapsed >= EXPIRY
    return sources, outputs, log
