"""Run the behaviour-contract suite offline and write a local evidence folder.

Usage: python -m sunglasses.contract_suite --out docs/records/research/<run>
No UI, capture, GPU or playback.  Synchronous file writes are limited to the
final JSON/CSV summaries of this offline command.
"""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import contracts, demo_filter, presentation_scenes, presentation_sim
from .contracts import evaluate, field_symptom, roi_code
from .presentation_scenes import PROFILES, build
from .presentation_sim import TIMINGS, simulate

SUITE_TIMINGS = ("clean-lag2", "dirty-lag2", "feedback-lag3")
STRENGTHS = (.5, 1.)


def _job(args):
    profile_key, scene_id, timing_name, strength = args
    profile = PROFILES[profile_key]
    scene = next(s for s in build(profile) if s.id == scene_id)
    timing = next(t for t in TIMINGS if t.name == timing_name)
    sources, outputs, log = simulate(scene, profile.refresh_hz, timing, strength)
    report = evaluate(scene, sources, outputs, profile.refresh_hz)
    report.update(profile=profile_key, timing=timing_name, strength=strength,
                  captures=int(log["captured"].sum()), refreshes=int(log["refreshed"].sum()))
    if "switch" in scene.meta and "blank" in scene.rois:
        code = roi_code(outputs, scene.rois["blank"])
        report["field_symptom_30fps"] = field_symptom(code, profile.refresh_hz, scene.meta["switch"])
        report["field_symptom_display_rate"] = field_symptom(code, profile.refresh_hz, scene.meta["switch"],
                                                             record_fps=profile.refresh_hz)
        report["trace_blank_code"] = [round(float(x), 2) for x in code]
    return report


def _attenuate(frames, stops):
    """Darken every pixel above the Y=.45 knee by a per-frame number of stops (pure fault injection)."""
    linear = demo_filter.LINEAR[frames]
    y = linear @ demo_filter.LUMA
    gain = np.where(y > .45, 2.0 ** -np.asarray(stops, np.float32)[:, None, None], 1.)
    out = linear * gain[..., None]
    enc = np.where(out <= .0031308, 12.92 * out, 1.055 * np.power(np.maximum(out, 0), 1 / 2.4) - .055)
    return np.clip(np.rint(enc * 255), 0, 255).astype(np.uint8)


def self_check() -> list[dict]:
    """The evaluator must accept identity and adaptation, and reject injected faults."""
    profile = PROFILES["main"]
    hz = profile.refresh_hz
    scenes = {s.id: s for s in build(profile)}
    results = []

    def run(label, scene_id, make_output, expect):
        scene = scenes[scene_id]
        n = int(round(scene.duration * hz))
        sources = np.stack([scene.render(i / hz) for i in range(n)])
        times = np.arange(n) / hz
        outputs = make_output(sources, times)
        status = evaluate(scene, sources, outputs, hz)
        results.append({"control": label, "scene": scene_id, "expected": expect, "got": status["status"],
                        "violations": status["violations"], "ok": status["status"] == expect})

    for sid in ("D1_static_light", "D3_game_to_light", "D4_light_to_game", "D11_carousel", "G3_strobe"):
        run("identity", sid, lambda s, t: s.copy(), "pass")
    # 10 Hz on/off 0.86-stop dimming on a static page (field-like oscillation).
    run("oscillation_10hz", "D1_static_light", lambda s, t: _attenuate(s, .86 * ((t * 10) % 1 < .5)), "fail")
    # One dip at the switch, released at ~2.7 stops/s (current envelope shape).
    run("fast_rebound", "D3_game_to_light",
        lambda s, t: _attenuate(s, np.where(t >= 2.02, .86 * np.clip(1 - (t - 2.06) / .28, 0, 1), 0)), "fail")
    # Same onset, held, then released at 0.2 stops/s: adaptation, allowed.
    run("slow_adaptation", "D3_game_to_light",
        lambda s, t: _attenuate(s, np.where(t >= 2.02, .86 * np.clip(1 - np.maximum(t - 3, 0) * .2 / .86, 0, 1), 0)),
        "pass")
    # Dimming that starts with no source change.
    run("spontaneous_onset", "D1_static_light", lambda s, t: _attenuate(s, np.where(t > 5, .3, 0)), "fail")

    # Stale mask: keep darkening the old bright layout for 10 ticks after switching to the dark game.
    def stale(s, t):
        out = s.copy()
        hit = (t >= 2) & (t < 2 + 10 / hz)
        out[hit] = (out[hit] * .5).astype(np.uint8)
        return out
    run("stale_mask", "D4_light_to_game", stale, "fail")
    return results


def _hashes():
    files = [Path(m.__file__) for m in (contracts, demo_filter, presentation_scenes, presentation_sim)]
    files.append(Path(__file__))
    shader = Path(__file__).resolve().parents[1] / "shaders" / "screen.hlsl"
    files.append(shader)
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    checks = self_check()
    (args.out / "self_check.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    for c in checks:
        print(f"self-check {c['control']:<18} {c['scene']:<20} expected {c['expected']:<4} got {c['got']}"
              + ("" if c["ok"] else "  <-- MISMATCH " + "; ".join(c["violations"])), flush=True)
    if not all(c["ok"] for c in checks):
        print("Evaluator self-check failed; algorithm suite not run.")
        return 1
    jobs = []
    for key, profile in PROFILES.items():
        for scene in build(profile):
            names = [t.name for t in TIMINGS] if "switch" in scene.meta else SUITE_TIMINGS
            for timing in names:
                for strength in STRENGTHS:
                    jobs.append((key, scene.id, timing, strength))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(_job, jobs, chunksize=1))
    traces = {f"{r['profile']}|{r['scene']}|{r['timing']}|{r['strength']}": r.pop("trace_blank_code")
              for r in reports if "trace_blank_code" in r}
    meta = {"schema": 1, "source_sha256": _hashes(), "profiles": {k: vars(v) for k, v in PROFILES.items()},
            "timings": [vars(t) for t in TIMINGS], "constants": {
                k: getattr(contracts, k) for k in ("LEG_STOPS", "SOURCE_LEG_STOPS", "STATIC_EPS", "RELEASE_RATE",
                                                   "MATCH_WINDOW", "STALE_TICKS", "DARK_FLOOR")},
            "scope": "offline grid simulation of the frozen screen.hlsl CPU reference; no DWM/WGC/GPU/display"}
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, "reports": reports}, indent=1), encoding="utf-8")
    (args.out / "d3_traces.json").write_text(json.dumps(traces), encoding="utf-8")
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile", "scene", "timing", "strength", "status", "violations"])
        for r in reports:
            writer.writerow([r["profile"], r["scene"], r["timing"], r["strength"], r["status"],
                             " | ".join(r["violations"])])
    passed = sum(r["status"] == "pass" for r in reports)
    print(f"{passed}/{len(reports)} runs pass; evidence in {args.out}")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
