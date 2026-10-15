"""Source-paired measurements, separate from the frozen behaviour-contract verdict.

Synthetic samples describe held display ticks, so integrals use exact tick
durations. Recording samples have gaps and must not use these integrals.
Relative SDR luminance is not measured panel luminance or a comfort score.
"""

import numpy as np

from ..contracts import LEG_STOPS, roi_luminance
from ..demo_filter import LINEAR, LUMA


def sampling(times, expected_hz=None):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("At least two finite, strictly increasing timestamps are required")
    gaps = np.diff(times)
    report = {"samples": len(times), "span_seconds": float(times[-1] - times[0]),
              "observed_mean_hz": float((len(times) - 1) / (times[-1] - times[0])),
              "maximum_gap_seconds": float(gaps.max()), "median_gap_seconds": float(np.median(gaps))}
    if expected_hz is not None:
        if not np.isfinite(expected_hz) or expected_hz <= 0:
            raise ValueError("Expected sample rate must be positive and finite")
        # A gap count, not a claim to know the missing frame contents or cause.
        report.update(expected_hz=float(expected_hz), gaps_over_1_5_periods=int(np.count_nonzero(gaps > 1.5 / expected_hz)))
    return report


def runs(flags):
    flags = np.asarray(flags, dtype=bool)
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def excursions(source, output, hz):
    """Describe source-defined brightness excursions, not medical 'flash' events.

    Threshold = max(.1, midpoint of this ROI's source min/max); ignore source
    ranges <= .05. Threshold selection never depends on the candidate output.
    Both left/right censored intervals are marked; no global effectiveness pass.
    """
    source, output = np.asarray(source, dtype=np.float64), np.asarray(output, dtype=np.float64)
    if (source.ndim != 1 or source.shape != output.shape or not len(source) or
            not np.isfinite(source).all() or not np.isfinite(output).all() or
            np.any(source < 0) or np.any(output < 0) or not np.isfinite(hz) or hz <= 0):
        raise ValueError("Invalid paired luminance trace or rate")
    threshold = max(.1, float((source.min() + source.max()) / 2))
    events = []
    if np.ptp(source) > .05:
        for start, end in runs(source > threshold):
            ys, yo = source[start:end], output[start:end]
            reduced = yo < ys * 2 ** -LEG_STOPS
            indices = np.flatnonzero(reduced)
            src_excess = float(np.maximum(ys - threshold, 0).sum() / hz)
            out_excess = float(np.maximum(yo - threshold, 0).sum() / hz)
            events.append({"start_seconds": start / hz, "end_seconds": end / hz,
                           "left_censored": start == 0, "right_censored": end == len(source),
                           "source_peak": float(ys.max()), "output_peak": float(yo.max()),
                           "peak_reduction_fraction": float(1 - yo.max() / ys.max()),
                           "source_excess_integral": src_excess, "output_excess_integral": out_excess,
                           "excess_reduction_fraction": 1 - out_excess / src_excess,
                           "initial_unreduced_samples": int(indices[0]) if len(indices) else end - start,
                           "first_reduction_seconds": float(indices[0] / hz) if len(indices) else None,
                           "unreduced_seconds": float(np.count_nonzero(~reduced) / hz),
                           "reduction_observed": bool(len(indices))})
    return {"source_threshold": threshold, "event_count": len(events),
            "events_with_reduction": sum(event["reduction_observed"] for event in events), "events": events}


def paired(scene, sources, outputs, hz):
    if (sources.dtype != np.uint8 or outputs.dtype != np.uint8 or sources.shape != outputs.shape or
            sources.ndim != 4 or sources.shape[-1] != 3 or len(sources) < 2):
        raise ValueError("Expected equal uint8 RGB frame sequences")
    times = np.arange(len(sources), dtype=np.float64) / hz
    quality = sampling(times, hz)
    rois, traces = {}, {"time_seconds": times}
    for name, mask in scene.rois.items():
        if mask.shape != sources.shape[1:3] or mask.dtype != bool or not mask.any():
            raise ValueError(f"Invalid ROI: {name}")
        ys, yo = roi_luminance(sources, mask), roi_luminance(outputs, mask)
        event = excursions(ys, yo, hz)
        delta = yo - ys
        item = {**event, "mean_luminance_difference": float(delta.mean()),
                "maximum_luminance_decrease": float(np.maximum(-delta, 0).max()),
                "maximum_luminance_increase": float(np.maximum(delta, 0).max()),
                "maximum_absolute_difference": float(np.abs(delta).max())}
        if name in scene.unchanged:
            item["declared_unchanged"] = True
            item["source_pixels_unchanged"] = bool(np.all(sources[:, mask] == sources[0, mask]))
        rois[name] = item
        traces[name + "__source"] = ys
        traces[name + "__output"] = yo
    # Per-frame reduction bounds memory; average ROI means cannot reveal damaged
    # dark text when a bright background dominates the region's average.
    dark_changed = brightened = changed = total_dark = total_pixels = 0
    dark_max_change = 0
    for source, output in zip(sources, outputs):
        dark = (LINEAR[source] @ LUMA) <= .45
        difference = output.astype(np.int16) - source.astype(np.int16)
        absolute = np.max(np.abs(difference), axis=-1)
        dark_changed += int(np.count_nonzero(dark & (absolute > 3)))
        total_dark += int(np.count_nonzero(dark))
        if dark.any():
            dark_max_change = max(dark_max_change, int(absolute[dark].max()))
        brightened += int(np.count_nonzero(difference.max(axis=-1) > 2))
        changed += int(np.count_nonzero(absolute))
        total_pixels += dark.size
    report = {"sampling": quality, "rois": rois, "identity": changed == 0,
              "dark_pixel_changed_fraction": dark_changed / total_dark if total_dark else None,
              "dark_pixel_max_code_change": dark_max_change if total_dark else None,
              "brightened_pixel_fraction": brightened / total_pixels,
              "changed_pixel_fraction": changed / total_pixels,
              "limits": ["Brightness excursions are engineering measurements, not a flash safety classifier.",
                         "No overall effectiveness verdict; identity can satisfy contracts with zero reduction.",
                         "Reduced-grid dark-pixel changes do not measure text legibility."]}
    return report, traces
