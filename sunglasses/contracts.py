"""Behaviour-contract checks on simulated (or captured) source/output pairs.

Traces are ROI means of linear relative luminance.  The filter's own temporal
behaviour on a region is its attenuation ``a(t) = log2(Y_source / Y_output)``
in stops.  Legs of ``a`` are classified against legs of the source:

* an attenuation onset must start near a source brightening;
* a release must coincide with a source darkening or be slower than
  ``RELEASE_RATE`` (adaptation); a faster unexplained release is an added
  flash (dim -> bright rebound, judged a flash by the user on 2026-10-13);
* regions whose source never changes must keep ``a`` constant.

All constants are engineering starting values pending subjective calibration;
they are not visibility or safety thresholds.
"""

from dataclasses import dataclass

import numpy as np

from .demo_filter import LINEAR, LUMA

LEG_STOPS = .05          # attenuation hysteresis
SOURCE_LEG_STOPS = .10   # source change hysteresis
SOURCE_RATE = 1.0        # stops per second; faster source movement can explain filter legs
STATIC_EPS = .02         # allowed peak-to-peak attenuation drift on unchanged regions
RELEASE_RATE = .30       # stops per second; slower releases count as adaptation
MATCH_WINDOW = .15       # seconds between a source change and the explained filter leg
STALE_TICKS = 3          # tolerated ticks of mask on new content after a change
DARK_FLOOR = .02         # ROI luminance below which attenuation is not defined


@dataclass(frozen=True)
class Leg:
    start: float
    end: float
    direction: int
    amplitude: float

    @property
    def rate(self) -> float:
        return self.amplitude / max(self.end - self.start, 1e-9)


def legs(values: np.ndarray, times: np.ndarray, threshold: float) -> list[Leg]:
    """Hysteresis turning points; NaN breaks a leg (unobserved/undefined)."""
    result: list[Leg] = []
    anchor = extreme = None
    direction = 0

    def finish():
        if direction and anchor is not None and extreme is not None:
            amplitude = abs(float(values[extreme] - values[anchor]))
            if amplitude >= threshold:
                # Departure: last sample still within 20% of the amplitude from the anchor level.
                span = np.abs(values[anchor:extreme + 1] - values[anchor]) <= .2 * amplitude
                departure = anchor + int(np.flatnonzero(span)[-1])
                result.append(Leg(float(times[departure]), float(times[extreme]), direction, amplitude))

    for i, value in enumerate(values):
        if not np.isfinite(value):
            finish()
            anchor = extreme = None
            direction = 0
            continue
        if anchor is None:
            anchor = extreme = i
            continue
        if not direction:
            if abs(value - values[anchor]) >= threshold:
                direction = 1 if value > values[anchor] else -1
                extreme = i
        elif direction * (value - values[extreme]) > 0:
            extreme = i
        elif direction * (value - values[extreme]) <= -threshold:
            finish()
            anchor, extreme, direction = extreme, i, -direction
    finish()
    return result


def roi_luminance(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = LINEAR[frames[:, mask]] @ LUMA
    return y.mean(axis=1)


def roi_code(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Encoded luma proxy (0-255) as used for the field recording analysis."""
    rgb = frames[:, mask].astype(np.float32)
    return (rgb @ np.array([.2126, .7152, .0722], np.float32)).mean(axis=1)


def source_motion(times, ys):
    """Per-tick source direction: +1/-1 where log2 luminance moves faster than SOURCE_RATE."""
    log_y = np.log2(np.maximum(ys, 1e-4))
    rate = np.diff(log_y, prepend=log_y[0]) / np.diff(times, prepend=times[0] - (times[1] - times[0]))
    return np.where(rate >= SOURCE_RATE, 1, np.where(rate <= -SOURCE_RATE, -1, 0))


def _near(leg: Leg, times, motion, direction: int) -> bool:
    window = (times >= leg.start - MATCH_WINDOW) & (times <= leg.start + MATCH_WINDOW)
    return bool(np.any(motion[window] == direction))


def attenuation_report(times, ys, yo):
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(ys > DARK_FLOOR, np.log2(ys / np.maximum(yo, 1e-9)), np.nan)
    src = legs(np.log2(np.maximum(ys, 1e-4)), times, SOURCE_LEG_STOPS)
    motion = source_motion(times, ys)
    att = legs(a, times, LEG_STOPS)
    spontaneous = [g for g in att if g.direction > 0 and not _near(g, times, motion, 1)]
    unexplained = [g for g in att if g.direction < 0 and not _near(g, times, motion, -1)]
    fast = [g for g in unexplained if g.rate > RELEASE_RATE]
    slow = [g for g in unexplained if g.rate <= RELEASE_RATE]
    return a, src, att, spontaneous, fast, slow, motion


def stale_and_brightened(sources: np.ndarray, outputs: np.ndarray):
    """Darkening of pixels no strength may touch (Y<=.45) can only come from a stale mask."""
    ys = LINEAR[sources] @ LUMA
    diff = outputs.astype(np.int16) - sources.astype(np.int16)
    stale = ((diff.min(axis=-1) < -3) & (ys <= .45)).mean(axis=(1, 2))
    bright = (diff.max(axis=-1) > 2).mean(axis=(1, 2))
    return stale, bright


def _runs(flags: np.ndarray) -> int:
    longest = current = 0
    for f in flags:
        current = current + 1 if f else 0
        longest = max(longest, current)
    return longest


def evaluate(scene, sources, outputs, hz) -> dict:
    times = np.arange(len(sources)) / hz
    report: dict = {"scene": scene.id, "contracts": list(scene.contracts), "violations": [], "rois": {}}
    v = report["violations"]
    for name, mask in scene.rois.items():
        ys, yo = roi_luminance(sources, mask), roi_luminance(outputs, mask)
        a, src, att, spont, fast, slow, motion = attenuation_report(times, ys, yo)
        finite = a[np.isfinite(a)]
        roi = {"max_attenuation_stops": float(finite.max()) if finite.size else None,
               "source_legs": len(src), "attenuation_legs": len(att),
               "spontaneous_onsets": len(spont), "fast_releases": len(fast), "slow_releases": len(slow),
               "fast_release_max_rate": max((g.rate for g in fast), default=0.),
               "first_fast_release_at": fast[0].start if fast else None}
        if name in scene.unchanged:
            settled = a[times > .1]
            settled = settled[np.isfinite(settled)]
            ptp = float(np.ptp(settled)) if settled.size else 0.
            roi["unchanged_ptp_stops"] = ptp
            if ptp > STATIC_EPS:
                v.append(f"{name}: unchanged region attenuation varied {ptp:.3f} stops")
        if spont:
            v.append(f"{name}: {len(spont)} attenuation onsets without a source brightening")
        if fast:
            v.append(f"{name}: {len(fast)} fast releases (dim->bright rebound), max {roi['fast_release_max_rate']:.1f} stops/s")
        moving = np.flatnonzero(motion != 0)
        if "S1" in scene.contracts and moving.size:
            tail = times >= times[moving[-1]] + 3.
            redim = [g for g in legs(np.where(tail, a, np.nan), times, LEG_STOPS) if g.direction > 0]
            if redim:
                v.append(f"{name}: {len(redim)} re-dimming legs 3 s after the last source change")
        # Benefit for flash-type content: per source brightening of >=1 stop.
        flashes = [g for g in src if g.direction > 0 and g.amplitude >= 1.]
        if flashes and any(c in scene.contracts for c in ("S5", "S6", "S7")):
            peaks, means, leaks = [], [], []
            for g in flashes:
                near = (times >= g.end - .1) & (times <= g.end + .25)
                bright = np.flatnonzero(near & (ys > .5 * ys[near].max()))
                peaks.append(float(np.log2(ys[bright].max() / max(yo[bright].max(), 1e-9))))
                means.append(float(np.log2(ys[bright].sum() / max(yo[bright].sum(), 1e-9))))
                leaks.append(_runs(np.nan_to_num(a[bright], nan=0) < LEG_STOPS))
            roi["flash_peak_reduction_stops"] = [round(x, 3) for x in peaks]
            roi["flash_mean_reduction_stops"] = [round(x, 3) for x in means]
            roi["flash_leak_ticks"] = leaks
            src_out = legs(np.log2(np.maximum(yo, 1e-4)), times, SOURCE_LEG_STOPS)
            roi["output_legs"] = len(src_out)
            if len(src_out) > len(src):
                v.append(f"{name}: output has more brightness legs ({len(src_out)}) than source ({len(src)})")
        report["rois"][name] = roi
    stale, bright = stale_and_brightened(sources, outputs)
    report["stale_mask_max_fraction"] = float(stale.max())
    report["stale_mask_longest_ticks"] = _runs(stale > .001)
    report["brightened_max_fraction"] = float(bright.max())
    if report["stale_mask_longest_ticks"] > STALE_TICKS:
        v.append(f"stale mask on new content for {report['stale_mask_longest_ticks']} ticks")
    if report["brightened_max_fraction"] > .001:
        v.append(f"output brighter than source on {report['brightened_max_fraction']:.1%} of pixels")
    report["status"] = "fail" if v else "pass"
    return report


def field_symptom(code_trace: np.ndarray, hz: float, switch: float, record_fps: float = 30.):
    """Describe valleys like the 2026-10-13 recording analysis (4-code hysteresis at 30 fps)."""
    step = max(1, int(round(hz / record_fps)))
    sampled = code_trace[::step]
    times = np.arange(len(sampled)) * step / hz
    window = (times >= switch) & (times <= switch + 3)
    lg = legs(sampled[window], times[window], 4.)
    valleys = [g.end for g in lg if g.direction < 0]
    final = float(sampled[-1])
    deviating = np.flatnonzero(window & (np.abs(sampled - final) > 1))
    return {"valleys_in_3s": len(valleys),
            "median_valley_spacing_s": float(np.median(np.diff(valleys))) if len(valleys) > 1 else None,
            "min_code": float(sampled[window].min()) if window.any() else None,
            "settled_after_s": float(times[deviating[-1]] - switch) if deviating.size else 0.,
            "valley_times_s": [round(t - switch, 3) for t in valleys][:12]}
