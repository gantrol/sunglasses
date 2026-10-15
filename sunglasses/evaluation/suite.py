"""Reproducible off/50%/100% comparisons around the frozen screen reference.

The original contract_suite remains available for historical reproduction.
This runner keeps benefit measurements distinct from those contract verdicts.
Jobs run sequentially to bound resident frame arrays; no UI or native build.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np

from .. import contract_suite, demo_filter
from ..contracts import evaluate
from ..presentation_scenes import PROFILES, build
from ..presentation_sim import TIMINGS, Timing, simulate
from .evidence import digest, implementation_hashes, new_directory, write_json
from .metrics import paired

TIMING_OPTIONS = {item.name: item for item in TIMINGS}
TIMING_OPTIONS["periodic60-lag2"] = Timing("periodic60-lag2", 2, capture_hz=60.)
DEFAULT_SCENES = ("D1_static_light", "D3_game_to_light", "D8_scroll", "D12_window_video_flash",
                  "G1_single_flash", "G3_strobe", "G7_antiphase")


def check_reference():
    path = Path(__file__).resolve().parents[2] / "shaders/screen.hlsl"
    if digest(path) != demo_filter.SHADER_SHA256:
        raise ValueError("Frozen CPU reference no longer matches the recorded screen shader hash")


def run_checks(out):
    out = new_directory(out)
    checks = contract_suite.self_check()
    report = {"schema": 1, "kind": "evaluator-controls", "source_sha256": implementation_hashes(),
              "checks": checks, "passed": all(item["ok"] for item in checks)}
    write_json(out / "summary.json", report)
    return report


def run_suite(out, profiles=("portrait",), timings=("clean-lag2",), scenes=DEFAULT_SCENES):
    check_reference()
    if not profiles or not timings:
        raise ValueError("At least one profile and timing are required")
    if any(len(values) != len(set(values)) for values in (profiles, timings, scenes or ())):
        raise ValueError("Duplicate profile, timing or scene selections would overwrite per-case evidence")
    # Validate all selection before creating a partial evidence folder.
    selected = []
    for key in profiles:
        if key not in PROFILES:
            raise ValueError(f"Unknown profile: {key}")
        available = {scene.id: scene for scene in build(PROFILES[key])}
        for sid in scenes or available:
            if sid not in available:
                raise ValueError(f"Unknown scene: {sid}")
            selected.append((key, available[sid]))
    if any(name not in TIMING_OPTIONS for name in timings):
        raise ValueError("Unknown timing model")
    out = new_directory(out)
    (out / "traces").mkdir()
    hashes = implementation_hashes()
    checks = contract_suite.self_check()
    write_json(out / "self_check.json", checks)
    if not all(item["ok"] for item in checks):
        raise ValueError("Evaluator controls failed; candidate comparison was not run")
    reports = []
    count = len(selected) * len(timings) * 3
    for key, scene in selected:
        profile = PROFILES[key]
        hz = profile.refresh_hz
        source_hash = None
        for timing_name in timings:
            timing = TIMING_OPTIONS[timing_name]
            for strength in (0., .5, 1.):
                if strength == 0:
                    sources = np.stack([scene.render(i / hz) for i in range(round(scene.duration * hz))])
                    outputs = sources
                    captures = refreshes = 0
                else:
                    sources, outputs, log = simulate(scene, hz, timing, strength)
                    captures, refreshes = int(log["captured"].sum()), int(log["refreshed"].sum())
                source_sha = hashlib.sha256(memoryview(sources)).hexdigest()
                if source_hash is not None and source_hash != source_sha:
                    raise ValueError("Candidates did not receive identical source frames")
                source_hash = source_sha
                contract = evaluate(scene, sources, outputs, hz)
                measurement, traces = paired(scene, sources, outputs, hz)
                token = f"{key}--{scene.id}--{timing_name}--{int(strength * 100)}"
                np.savez_compressed(out / "traces" / (token + ".npz"), **traces)
                report = {"profile": key, "scene": scene.id, "timing": timing_name, "strength": strength,
                          "source_frames_sha256": source_sha, "captures": captures, "refreshes": refreshes,
                          "contracts": contract, "measurements": measurement,
                          "trace": "traces/" + token + ".npz"}
                write_json(out / (token + ".json"), report)
                reports.append(report)
                print(f"[{len(reports)}/{count}] {token}: contracts={contract['status']}, "
                      f"identity={measurement['identity']}", flush=True)
                del sources, outputs, traces
    if implementation_hashes() != hashes:
        raise ValueError("Evaluation implementation changed during the run; results are not finalized")
    summary = {"schema": 1, "kind": "synthetic-comparison", "source_sha256": hashes,
               "scope": "offline reduced-grid model of sudden highlights; no GPU/capture/display/Apple VFM",
               "profiles": {key: vars(PROFILES[key]) for key in profiles},
               "timings": [vars(TIMING_OPTIONS[name]) for name in timings],
               "measurement_policy": {"transfer": "sRGB", "luminance": "relative, Rec.709 weights",
                   "excursion_threshold": "max(0.1, midpoint of source min/max); source range > 0.05",
                   "first_reduction": "attenuation > 0.05 stop inside each source excursion",
                   "integral": "sum of excess over threshold times exact simulated tick duration",
                   "overall_effectiveness": "not_assessed; contract pass alone is insufficient"},
               "reports": reports}
    write_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile", "scene", "timing", "strength", "contract_status", "identity",
                         "dark_pixel_changed_fraction", "roi", "source_events", "events_with_reduction",
                         "initial_unreduced_samples_max", "peak_reduction_max", "violations"])
        for report in reports:
            measurement = report["measurements"]
            for roi, value in measurement["rois"].items():
                events = value["events"]
                writer.writerow([report["profile"], report["scene"], report["timing"], report["strength"],
                    report["contracts"]["status"], measurement["identity"], measurement["dark_pixel_changed_fraction"],
                    roi, len(events), value["events_with_reduction"],
                    max((event["initial_unreduced_samples"] for event in events), default=None),
                    max((event["peak_reduction_fraction"] for event in events), default=None),
                    " | ".join(report["contracts"]["violations"])])
    lines = ["# Synthetic comparison", "", summary["scope"], "",
             "Contract pass does not establish benefit. Off/identity is a negative benefit control.", "",
             "| Strength | Runs | Contract pass | Contract fail | Identity output |", "| --- | --- | --- | --- | --- |"]
    for strength in (0., .5, 1.):
        rows = [row for row in reports if row["strength"] == strength]
        passed = sum(row["contracts"]["status"] == "pass" for row in rows)
        lines.append(f"| {strength:.0%} | {len(rows)} | {passed} | {len(rows) - passed} | "
                     f"{sum(row['measurements']['identity'] for row in rows)} |")
    lines += ["", "Per-case measurements and failures: summary.csv and summary.json.",
              "Linear relative ROI traces with actual simulated times: traces/*.npz.",
              "No overall effectiveness, comfort, physical timing or game compatibility verdict."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
