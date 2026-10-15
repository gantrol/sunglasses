"""Offline automatic-mode evaluation; bounded arrays and explicit static output only."""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import platform

import numpy as np

from .automatic import AutoConfig, AutoFilter, LIGHT, STRONG, HIGHLIGHT, LOAD_START, LOAD_FULL
from .color import luminance
from .filter import compress_highlights
from .fixtures import AUTO_CASES, CASE_BY_ID, SequenceSpec, frames
from .image_io import save_sdr_png
from .i18n import ValidationError
from .metrics import measure
from .jobs import JobContext
from .provenance import source_hashes


def run_auto_suite(output: Path, case_ids: list[str], spec: SequenceSpec,
                   config: AutoConfig, threshold=.5, *, job: JobContext = JobContext()):
    if not 0 < threshold < 1:
        raise ValidationError("threshold")
    known = {case.id for case in AUTO_CASES}
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(c not in known for c in case_ids):
        raise ValidationError("case_selection")
    job.check()
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    (output / "previews").mkdir()
    report = {
        "schema_version": 1, "kind": "synthetic_automatic_reference",
        "color_space": "linear-sRGB", "config": asdict(config), "sequence": asdict(spec),
        "diagnostic_threshold_y": threshold,
        "highlight_curve": asdict(HIGHLIGHT), "highlight_load_start": LOAD_START, "highlight_load_full": LOAD_FULL,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": source_hashes("automatic.py", "color.py", "filter.py", "fixtures.py", "metrics.py", "auto_lab.py"),
        "limitations": ["Synthetic numeric results; no capture, display, latency or comfort validation.",
                        "Saturated red and equal-luminance color changes are not effectively reduced.",
                        ("Highlight-only mode ignores repeat/hold/release; without_repeat is an identical control."
                         if config.policy == "highlight" else "Legacy block-mean reversals can miss fine patterns or react to normal motion/HUD."),
                        "Highlight-load gating can change other bright pixels when a bright region appears or disappears; temporal quality remains unvalidated.",
                        "RGB frames are never averaged over time.",
                        "Event annotations are provided only for the isolated-flash fixtures; other cases are diagnostics."],
        "cases": [],
    }
    rows = []
    methods = ("original", "fixed_light", "fixed_strong", "automatic", "without_repeat")
    for case_index, case_id in enumerate(case_ids):
        job.advance(case_index * spec.frame_count, len(case_ids) * spec.frame_count, case_id)
        # At most 5 x 20M float32 luminance samples, one case at a time.
        values = {name: np.empty((spec.frame_count, spec.height, spec.width), np.float32) for name in methods}
        controllers = {"automatic": AutoFilter(config), "without_repeat": AutoFilter(replace(config, repeat_enabled=False))}
        traces = []
        preview = None
        peak, sample_index = -1.0, 0
        event_index = (spec.frame_count // 10 if case_id == "midtone_recovery" else
                       spec.frame_count // 2 if case_id == "single_white_flash" else None)
        for index, (timestamp, frame) in enumerate(frames(case_id, spec)):
            job.advance(case_index * spec.frame_count + index, len(case_ids) * spec.frame_count, case_id)
            filtered = {"original": frame,
                        "fixed_light": compress_highlights(frame, replace(LIGHT, strength=config.strength)),
                        "fixed_strong": compress_highlights(frame, replace(STRONG, strength=config.strength))}
            for name, controller in controllers.items():
                filtered[name], trace = controller.process(frame, timestamp, index)
                traces.append({"method": name, **trace, "event": "white_flash" if index == event_index else ""})
            for name in methods:
                values[name][index] = luminance(filtered[name])
            current = float(values["original"][index].max())
            if current > peak:
                peak, sample_index = current, index
                preview = np.concatenate([filtered[name] for name in ("original", "fixed_strong", "automatic")], axis=1)
        metrics = {}
        source = values["original"]
        ordinary = np.ones(spec.frame_count, dtype=bool)
        if event_index is not None:
            ordinary[event_index] = False
        for name in methods:
            job.check()
            metrics[name] = measure(values[name], source, STRONG.knee, threshold, 1 / spec.fps)
            metrics[name]["outside_event_mean_abs_y_error"] = (
                float(np.abs(values[name][ordinary] - source[ordinary]).mean(dtype=np.float64))
                if event_index is not None else None)
            rows.append({"case": case_id, "method": name, **metrics[name]})
        automatic = [t for t in traces if t["method"] == "automatic"]
        recovery = next((t["timestamp"] - event_index / spec.fps for t in automatic
                         if event_index is not None and t["frame_id"] > event_index and t["amount"] <= .05), None)
        case = CASE_BY_ID[case_id]
        findings = []
        if metrics["automatic"]["max_pixel_delta"] > metrics["original"]["max_pixel_delta"] + 1e-6:
            findings.append("Maximum adjacent-frame luminance step increased; automatic control can introduce a larger transition.")
        if case_id == "moving_pattern" and max(t["amount"] for t in automatic) > .05:
            findings.append("Negative motion control triggered: evidence is not specific to flashing effects.")
        if config.policy == "legacy" and case_id == "fine_antiphase" and max(t["repeat"] for t in automatic) == 0:
            findings.append("Fine-pattern swaps missed by the block-mean reversal branch at these dimensions.")
        report["cases"].append({"id": case_id, "description": case.description, "limitation": case.limitation,
                                "findings": findings,
                                "preview_frame": sample_index, "event_frame": event_index,
                                "recovery_to_5_percent_seconds": recovery, "metrics": metrics,
                                "max_repeat_demand": max(t["repeat"] for t in automatic),
                                "frames_with_amount_over_05": sum(t["amount"] > .05 for t in automatic),
                                "final_amount": automatic[-1]["amount"]})
        with (output / "traces" / f"{case_id}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
            writer.writeheader()
            writer.writerows(traces)
        save_sdr_png(output / "previews" / f"{case_id}.png", preview)
        del values, source, filtered
    job.check()
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Automatic SDR experiment", "", *[f"- {s}" for s in report["limitations"]], "",
             "Static preview order: original / fixed strong / automatic.", "",
             "| Case | Source peak Y | Auto peak Y | Source max step | Auto max step | Max repeat demand | Final amount | Recovery to 5% (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["cases"]:
        m = case["metrics"]
        recovery = case["recovery_to_5_percent_seconds"]
        lines.append(f"| {case['id']} | {m['original']['peak_y']:.5f} | {m['automatic']['peak_y']:.5f} | "
                     f"{m['original']['max_pixel_delta']:.5f} | {m['automatic']['max_pixel_delta']:.5f} | "
                     f"{case['max_repeat_demand']:.3f} | {case['final_amount']:.5f} | "
                     + (f"{recovery:.3f}" if recovery is not None else "n/a") + " |")
    for case in report["cases"]:
        lines += ["", f"## {case['id']}", "", case["description"], "", case["limitation"], "",
                  *[f"- {finding}" for finding in case["findings"]], "",
                  f"[Control trace](traces/{case['id']}.csv) · [Static comparison](previews/{case['id']}.png)"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
