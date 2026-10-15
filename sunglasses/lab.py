"""Bounded, offline synthetic evaluation with static previews only."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform

import numpy as np
import PIL

from .color import luminance
from .filter import FilterConfig, compress_highlights
from .fixtures import CASE_BY_ID, SequenceSpec, frames
from .image_io import save_sdr_png
from .metrics import matched_gains, measure
from .i18n import ValidationError
from .jobs import JobContext
from .provenance import source_hashes


def run_suite(output: Path, case_ids: list[str], spec: SequenceSpec, config: FilterConfig,
              threshold: float = .5, *, job: JobContext = JobContext()) -> dict:
    if not 0 < threshold < 1:
        raise ValidationError("threshold")
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(c not in CASE_BY_ID for c in case_ids):
        raise ValidationError("case_selection")
    # A run owns a new directory; no recursive deletion or overwrite option.
    job.check()
    output.mkdir(parents=True, exist_ok=False)
    previews = output / "previews"
    previews.mkdir()
    report = {
        "schema_version": 1, "kind": "synthetic_offline_reference",
        "color_space": "linear-sRGB", "config": asdict(config), "sequence": asdict(spec),
        "actual_duration_seconds": spec.frame_count / spec.fps,
        "diagnostic_threshold_y": threshold,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "pillow_version": PIL.__version__,
        "source_sha256": source_hashes("color.py", "filter.py", "fixtures.py", "metrics.py", "lab.py", "image_io.py"),
        "limitations": ["Synthetic signals only; no game compatibility or comfort result.",
                        "Luminance units are relative, not nits or medical thresholds.",
                        "Matched constant gains use the full clip, not a real-time strategy.",
                        "This static suite does not run the automatic temporal controller."],
        "cases": [],
    }
    csv_rows = []
    for case_index, case_id in enumerate(case_ids):
        job.advance(case_index * spec.frame_count, len(case_ids) * spec.frame_count, case_id)
        source = np.empty((spec.frame_count, spec.height, spec.width), dtype=np.float32)
        filtered = np.empty_like(source)
        sample = None
        sample_index, sample_peak = 0, -1.0
        for index, (_, frame) in enumerate(frames(case_id, spec)):
            job.advance(case_index * spec.frame_count + index, len(case_ids) * spec.frame_count, case_id)
            source[index] = luminance(frame)
            filtered[index] = luminance(compress_highlights(frame, config))
            peak = float(source[index].max())
            if peak > sample_peak:
                sample, sample_index, sample_peak = frame.copy(), index, peak
        gains = matched_gains(source, filtered, threshold)
        metrics = {}
        for method in ("original", "soft_knee", *gains):
            job.check()
            values = source if method == "original" else filtered if method == "soft_knee" else source * gains[method]
            metrics[method] = measure(values, source, config.knee, threshold, 1 / spec.fps)
            csv_rows.append({"case": case_id, "method": method, "constant_gain": gains.get(method), **metrics[method]})
        assert sample is not None
        # Static, labeled-by-report comparison; no GIF, video, browser launch or playback.
        comparison = np.concatenate((sample, sample * gains["dim_peak"], compress_highlights(sample, config)), axis=1)
        save_sdr_png(previews / f"{case_id}.png", comparison)
        case = CASE_BY_ID[case_id]
        report["cases"].append({"id": case_id, "description": case.description,
                                "limitation": case.limitation, "baseline_gains": gains,
                                "preview_frame": sample_index, "metrics": metrics})
    job.check()
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# Synthetic SDR experiment", "", *[f"- {item}" for item in report["limitations"]], "",
             "Static preview order: original / constant dimming matched to clip peak / soft knee.", "",
             "| Case | Original peak Y | Soft peak Y | Soft dark retention | Peak-matched dim dark retention |",
             "| --- | ---: | ---: | ---: | ---: |"]
    def number(value):
        return "n/a" if value is None else f"{value:.5f}"
    for case in report["cases"]:
        m = case["metrics"]
        lines.append(f"| {case['id']} | {number(m['original']['peak_y'])} | {number(m['soft_knee']['peak_y'])} | "
                     f"{number(m['soft_knee']['dark_luminance_retention'])} | {number(m['dim_peak']['dark_luminance_retention'])} |")
    for case in report["cases"]:
        lines += ["", f"## {case['id']}", "", case["description"], "", case["limitation"], "",
                  f"[Static comparison](previews/{case['id']}.png)"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
