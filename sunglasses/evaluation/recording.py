"""Offline analysis of the existing run_check black/white fixture.

Reads original video/logs; writes only to a new, separate evidence directory.
This is not an arbitrary-game classifier or a physical display measurement.
"""

import json
import subprocess

import numpy as np

from ..demo_filter import LINEAR
from .evidence import checked, digest, implementation_hashes, new_directory, write_json
from .metrics import runs, sampling


def read_video(path, error_log):
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate:frame=best_effort_timestamp_time", "-of", "json", str(path)],
        text=True, timeout=60))
    stream, = probe["streams"]
    width, height = stream["width"], stream["height"]
    if (width, height) != (800, 400):
        raise ValueError("This fixture analyzer expects the run_check 800x400 crop")
    times = np.asarray([float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]])
    quality = sampling(times, 120.)
    codes, luminance = [], []
    with error_log.open("wb") as errors:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                    "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgra", "-"],
                                   stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                raw = process.stdout.read(width * height * 4)
                if not raw:
                    break
                if len(raw) != width * height * 4:
                    raise ValueError("Truncated decoded frame")
                pixels = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
                # Compute linear light per pixel before averaging; do not decode
                # the mean code as if a mixed ROI were a uniform gray patch.
                code = linear = 0.
                for channel, weight in enumerate((.0722, .7152, .2126)):
                    values = pixels[..., channel]
                    code += weight * float(values.mean())
                    linear += weight * float(LINEAR[values].mean())
                codes.append(code)
                luminance.append(linear)
            if process.wait(timeout=10):
                raise RuntimeError("FFmpeg decode failed; inspect decode.log")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    if len(codes) != len(times):
        raise ValueError("Decoded frames and encoded PTS do not match one-to-one")
    return times, np.asarray(codes), np.asarray(luminance), {"stream": stream, **quality}


def white_intervals(times, codes):
    sampling(times)
    if len(times) != len(codes) or not np.isfinite(codes).all():
        raise ValueError("Invalid code trace")
    result = []
    for start, end in runs(codes > 128):
        segment, relative = codes[start:end], times[start:end] - times[start]
        dim = segment < 254.5
        indices = np.flatnonzero(dim)
        steady = segment[relative >= .5]
        result.append({"start_frame": start, "end_frame_exclusive": end, "samples": end - start,
                       "start_pts": float(times[start]), "left_censored": start == 0,
                       "right_censored": end == len(codes),
                       "minimum_code": float(segment.min()), "maximum_code": float(segment.max()),
                       "dim_intervals": len(runs(dim)),
                       "initial_full_white_samples": int(indices[0]) if len(indices) else None,
                       "first_dim_seconds": float(relative[indices[0]]) if len(indices) else None,
                       "last_dim_seconds": float(relative[indices[-1]]) if len(indices) else None,
                       "steady_samples": len(steady),
                       "steady_peak_to_peak": float(np.ptp(steady)) if len(steady) else None})
    return result


def telemetry(timeline, records):
    frequency, events = timeline["qpc_frequency"], timeline["events"]
    if not np.isfinite(frequency) or frequency <= 0 or len(events) < 2:
        raise ValueError("Invalid timeline")
    ticks = np.asarray([event["qpc"] for event in events], dtype=np.float64)
    if not np.isfinite(ticks).all() or np.any(np.diff(ticks) < 0):
        raise ValueError("Non-monotonic timeline")
    start, end = events[0]["qpc"], events[-1]["qpc"]
    gpu, exclusion, seen = [], [], set()
    for record in records:
        if not start <= record.get("qpc", -1) <= end:
            continue
        # Rotated/copied logs may overlap; count identical rows once.
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if record.get("type") == "gpu":
            gpu.append(record)
        elif record.get("type") == "event" and record.get("event", "").startswith("capture_exclusion_"):
            exclusion.append(record)
    holds, strength = [], None
    for index, event in enumerate(events):
        label = event["label"]
        if label.startswith("strength "):
            strength = int(label.split()[-1])
        if label != "white":
            continue
        black = next((item for item in events[index + 1:] if item["label"] == "black"), None)
        if black is None or black["qpc"] - event["qpc"] <= .5 * frequency:
            continue
        lower = event["qpc"] + .5 * frequency
        samples = [row for row in gpu if lower <= row["qpc"] < black["qpc"] and not row.get("refresh", False)]
        means = np.asarray([row["mean_luminance"] for row in samples], dtype=np.float64)
        if not np.isfinite(means).all():
            raise ValueError("Non-finite telemetry luminance")
        holds.append({"strength_percent": strength, "window_seconds": [(lower - start) / frequency,
                      (black["qpc"] - start) / frequency], "samples": len(means),
                      "capture_mean_min": float(means.min()) if len(means) else None,
                      "capture_mean_max": float(means.max()) if len(means) else None,
                      "capture_mean_std": float(means.std()) if len(means) else None,
                      "repeated_triggers": sum(bool(row.get("triggered")) for row in samples)})
    mode = timeline.get("recording_mode")
    if mode is None:
        mode = next((event["label"].split()[-1] for event in events
                     if event["label"] in ("recording mode visible", "recording mode hidden")), None)
    return {"recording_mode": mode, "sequence_completed": timeline.get("sequence_completed"),
            "recorded_exe_sha256": timeline.get("exe_sha256"), "holds": holds, "exclusion_events": exclusion}


def findings(metadata, intervals=None):
    """Explicitly withhold output claims for hidden, missing or incomplete data."""
    complete = (metadata["sequence_completed"] is True and len(metadata["holds"]) == 2 and
                all(hold["samples"] >= 2 for hold in metadata["holds"]))
    result = {"sequence": "complete" if complete else "incomplete_or_legacy_unknown",
              "capture_stability": "insufficient_evidence", "output_effect": "insufficient_evidence",
              "output_stability": "insufficient_evidence", "overall_effectiveness": "not_assessed"}
    if complete:
        stable = all(hold["capture_mean_max"] - hold["capture_mean_min"] <= .002 and
                     hold["repeated_triggers"] == 0 for hold in metadata["holds"])
        result["capture_stability"] = "no_repeated_trigger_or_mean_variation_observed" if stable else "variation_observed"
    if metadata["recording_mode"] == "hidden":
        result["output_effect"] = result["output_stability"] = "not_observable_in_hidden_recording"
    elif metadata["recording_mode"] == "visible" and complete and intervals is not None:
        usable = len(intervals) == 2 and all(not row["left_censored"] and not row["right_censored"] and
                                          row["steady_samples"] >= 2 for row in intervals)
        if usable:
            result["output_effect"] = "dimming_observed" if all(row["dim_intervals"] for row in intervals) else "dimming_missing_in_a_hold"
            result["output_stability"] = ("no_variation_observed_in_settled_samples" if
                all(row["steady_peak_to_peak"] <= 1 for row in intervals) else "settled_variation_observed")
    return result


def analyze_run(run, out, metadata_only=False):
    run = checked(run)
    paths = [run / "timeline.json", *sorted(run.glob("screen-diagnostics*.jsonl"))]
    for name in ("provenance.json", "composite.mkv"):
        if (run / name).is_file():
            paths.append(run / name)
    hashes = {path.name: digest(path) for path in paths}
    out = new_directory(out, source=run)
    timeline = json.loads((run / "timeline.json").read_text(encoding="utf-8"))

    def rows():
        for path in paths:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)

    report = {"schema": 1, "kind": "runtime-black-white-recording", "input_sha256": hashes,
              "source_sha256": implementation_hashes(), **telemetry(timeline, rows()),
              "limits": ["Known run_check black/white fixture only, not arbitrary desktop or game content.",
                         "Video intervals use observed pixels; their alignment to QPC is not calibrated.",
                         "PTS samples are not physical display frames; gaps cannot be reconstructed.",
                         "Hidden recordings omit the filter; telemetry alone cannot prove final output.",
                         "Code values and relative sRGB luminance are not measured nits."]}
    intervals = None
    if not metadata_only and (run / "composite.mkv").is_file():
        times, codes, linear, quality = read_video(run / "composite.mkv", out / "decode.log")
        intervals = white_intervals(times, codes)
        report.update(video=quality, composite_intervals=intervals)
        if len(intervals) == len(report["holds"]) == 2:
            for row, hold in zip(intervals, report["holds"]):
                row["sequence_strength_percent"] = hold["strength_percent"]
        np.savetxt(out / "trace.csv", np.column_stack((times, codes, linear)), delimiter=",",
                   header="pts_seconds,mean_encoded_luma,mean_linear_luminance", comments="", fmt="%.9g")
        report["decoder_version"] = subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=10).splitlines()[0]
    report["findings"] = findings(report, intervals)
    if {path.name: digest(path) for path in paths} != hashes:
        raise ValueError("Original evidence changed during analysis; report not finalized")
    write_json(out / "summary.json", report)
    lines = ["# Recorded fixture measurements", "", "| Finding | Observation |", "| --- | --- |"]
    lines += [f"| {key} | {value} |" for key, value in report["findings"].items()]
    lines += ["", "Limits:", ""] + ["- " + value for value in report["limits"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
