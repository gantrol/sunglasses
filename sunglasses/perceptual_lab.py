"""Reusable synthetic scorecard for temporal-vision engineering proxies.

The suite writes numeric data only.  It does not render or play flashing media,
start the application, capture a desktop, or claim standards conformance.
"""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform

import numpy as np

from . import demo_filter as screen_reference
from .apple_vfm import AppleVFMConfig, AppleVFMReference, KERNEL_SHA256, UPSTREAM_COMMIT
from .automatic import AutoConfig, AutoFilter, STRONG
from .color import linear_to_srgb
from .filter import compress_highlights
from .flash_screening import (
    FlashScreeningAnalyzer,
    FlashScreeningConfig,
    compare_flash_screening,
    summarize_flash_screening,
)
from .fixtures import AUTO_CASES, CASE_BY_ID, SequenceSpec, frames
from .i18n import ValidationError
from .jobs import JobContext
from .perceptual import (
    DisplayProfile,
    PairedPerceptualAnalyzer,
    PerceptualAnalyzer,
    PerceptualConfig,
    summarize_measurements,
    summarize_paired,
)
from .provenance import source_hashes
from .recovery import (
    LogRiseLimitConfig,
    LogRiseLimiter,
    SourceHoldConfig,
    SourceHoldController,
)


METHODS = ("original", "fixed_strong", "highlight_auto", "legacy_temporal",
           "screen_burst_100", "screen_source_hold", "screen_step_hold_35",
           "screen_global_log_limit", "screen_tile_log_limit", "apple_matlab_vfm")
SCREEN_METHODS = ("screen_burst_100", "screen_source_hold", "screen_step_hold_35",
                  "screen_global_log_limit", "screen_tile_log_limit")
FREQUENCY_SWEEP_HZ = (.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)


def _flat_frame(case_id, method, measurement, paired=None, screening=None):
    row = {"case": case_id, "method": method, **asdict(measurement)}
    bands = row.pop("band_rms"), row.pop("band_p95")
    for label, values in zip(("band_rms", "band_p95"), bands):
        for index, value in enumerate(values):
            row[f"{label}_{index}"] = value
    if paired is not None:
        pair = asdict(paired)
        pair.pop("source")
        pair.pop("output")
        for key in ("frame_id", "timestamp", "continuity"):
            pair.pop(key)
        row.update(pair)
    else:
        for key in ("added_rise_nits_mean", "added_rise_nits_p95", "added_rise_nits_max",
                    "added_rise_area_over_1_nit", "added_fall_nits_mean",
                    "added_fall_nits_p95", "added_fall_nits_max",
                    "added_fall_area_over_1_nit", "temporal_residual_nits_mean",
                    "temporal_residual_nits_p95", "temporal_residual_nits_max",
                    "spatial_residual_log_gradient_mean",
                    "spatial_residual_log_gradient_p95", "spatial_residual_log_gradient_max",
                    "same_frame_brightening_nits_max",
                    "same_frame_dimming_nits_mean", "same_frame_dimming_nits_p95",
                    "same_frame_dimming_nits_max"):
            row[key] = 0.0
    if screening is not None:
        screen = asdict(screening)
        for key in ("frame_id", "timestamp", "continuity"):
            screen.pop(key)
        row.update({f"screening_{key}": value for key, value in screen.items()})
    return row


def _findings(case_id, summaries, paired, diagnostics, screenings, screening_comparisons):
    source = summaries["original"]
    findings = []
    for method in METHODS[1:]:
        if summaries[method]["temporal_energy_p95"] > source["temporal_energy_p95"] * 1.01 + 1e-9:
            findings.append(f"{method}: temporal-band proxy increased relative to the unfiltered source.")
        if paired[method]["added_rise_nits_p95_max"] > 1:
            findings.append(f"{method}: at least one local output rise exceeded the simultaneous source rise by over 1 nit.")
        if screening_comparisons[method]["added_combined_event_frames"]:
            findings.append(
                f"{method}: standards-informed screening reports output-only event timing; inspect the paired trace.")
        if (screenings[method]["screening_threshold_exceeded"] and
                screening_comparisons[method]["output_only_threshold_frames"]):
            findings.append(
                f"{method}: output exceeds the conservative rolling screen where its paired source does not.")
    if case_id in ("antiphase_blocks", "fine_antiphase"):
        findings.append("Local opposing-transition evidence remains visible even though whole-frame mean luminance is constant.")
        if diagnostics["apple_matlab_vfm"]["max_risk"] == 0:
            findings.append("The whole-frame Apple MATLAB reference does not respond to this equal-mean spatial reversal.")
    if case_id == "hard_cut" and source["max_opposing_transitions_1s"] == 0:
        findings.append("The isolated coherent cut is reported separately and is not counted as an opposing transition.")
        if diagnostics["apple_matlab_vfm"]["max_unmasked_risk"] > 0 and diagnostics["apple_matlab_vfm"]["max_risk"] == 0:
            findings.append("The Apple MATLAB normalized-response gate masks this isolated step in the reference trace.")
        current_rise = paired["screen_burst_100"]["added_rise_nits_p95_max"]
        local_rise = paired["screen_tile_log_limit"]["added_rise_nits_p95_max"]
        findings.append(f"The angular-cell recovery candidate changes max added rise from {current_rise:.3f} to {local_rise:.3f} assumed nits.")
        findings.append("Source-hold candidates avoid timer recovery by keeping a stable cut dim; this is a fidelity cost, not a free fix.")
    if case_id == "white_pulses":
        findings.append("The 35% tentative-step candidate weakens the first coherent rise, then permits full attacks after opposing evidence.")
    if case_id == "saturated_red":
        retained = summaries["highlight_auto"]["red_change_mean_p95"] / max(source["red_change_mean_p95"], 1e-12)
        findings.append(f"highlight_auto retains {retained:.3f} of the red-change proxy in this fixture.")
    if case_id == "moving_pattern":
        findings.append("The opposing-transition proxy also reacts to ordinary pattern motion; it is evidence, not a flash label.")
    if paired["apple_matlab_vfm"]["same_frame_brightening_nits_max"] > 1:
        findings.append("The Apple MATLAB luminance+contrast mapping brightens at least one pixel by over 1 assumed nit.")
    return findings


def _frequency_sweep(spec: SequenceSpec, profile: DisplayProfile,
                     config: PerceptualConfig, apple_config: AppleVFMConfig,
                     screening_config: FlashScreeningConfig):
    """Finite strong square-wave probes; not a psychophysical threshold test."""
    rows = []
    duration, lead = 3.0, .5
    for frequency in FREQUENCY_SWEEP_HZ:
        if frequency > spec.fps / 2:
            continue
        apple = AppleVFMReference(apple_config)
        analyzer = PerceptualAnalyzer(profile, config)
        screener = FlashScreeningAnalyzer(profile, screening_config)
        screen = screen_reference.Controller()
        measurements, screening_frames, apple_frames, screen_gains = [], [], [], []
        for index in range(round(duration * spec.fps)):
            timestamp = index / spec.fps
            phase_time = timestamp - lead
            high = phase_time >= 0 and int(phase_time * 2 * frequency) % 2 == 0
            frame = np.full((16, 16, 3), 1.0 if high else .04, dtype=np.float32)
            encoded = np.rint(linear_to_srgb(frame) * 255).astype(np.uint8)
            screen_input, _, load, mean = screen_reference.stats(encoded)
            screen_gains.append(screen.step(load, mean, timestamp))
            measurements.append(analyzer.process(screen_input, timestamp, index))
            screening_frames.append(screener.process(screen_input, timestamp, index))
            _, apple_frame = apple.process_linear_rgb(frame)
            apple_frames.append(apple_frame)
        summary = summarize_measurements(measurements)
        screening = summarize_flash_screening(screening_frames)
        rows.append({
            "frequency_hz": frequency,
            "cycles_after_lead": (duration - lead) * frequency,
            "apple_max_unmasked_risk": max(item.unmasked_risk for item in apple_frames),
            "apple_max_risk": max(item.risk for item in apple_frames),
            "apple_first_risk_frame": next((item.frame_index for item in apple_frames if item.risk > 0), None),
            "apple_max_normalized_step_energy": max(item.normalized_step_energy for item in apple_frames),
            "screen_max_gain": max(screen_gains),
            "screen_frames_with_gain": sum(gain > 0 for gain in screen_gains),
            "opposing_transitions_1s": summary["max_opposing_transitions_1s"],
            "temporal_energy_p95": summary["temporal_energy_p95"],
            "screening_general_flashes_1s": screening["max_general_flashes_1s"],
            "screening_threshold_exceeded": screening["screening_threshold_exceeded"],
        })
    return rows


def run_perceptual_suite(output: Path, case_ids: list[str], spec: SequenceSpec,
                         profile: DisplayProfile, config: PerceptualConfig = PerceptualConfig(),
                         *, job: JobContext = JobContext()):
    """Run a bounded offline comparison and write JSON/CSV/Markdown evidence."""
    known = {case.id for case in AUTO_CASES}
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(case not in known for case in case_ids):
        raise ValidationError("case_selection")
    job.check()
    output.mkdir(parents=True, exist_ok=False)
    traces = output / "traces"
    traces.mkdir()
    controls = output / "controls"
    controls.mkdir()
    apple_config = AppleVFMConfig(spec.fps, profile.peak_luminance_nits,
                                  profile.visual_angle_degrees[0] * profile.visual_angle_degrees[1])
    hold_config = SourceHoldConfig()
    step_hold_config = SourceHoldConfig(tentative_step_scale=.35)
    limit_config = LogRiseLimitConfig()
    screening_config = FlashScreeningConfig()
    frequency_sweep = _frequency_sweep(
        spec, profile, config, apple_config, screening_config)
    report = {
        "schema_version": 2,
        "kind": "synthetic_perceptual_proxy_scorecard",
        "color_space": "normalized linear-sRGB interpreted through the explicit SDR display profile",
        "sequence": asdict(spec),
        "display_profile": asdict(profile),
        "profile_geometry": {
            "visual_angle_degrees": profile.visual_angle_degrees,
            "pixels_per_degree": profile.pixels_per_degree,
        },
        "config": asdict(config),
        "flash_screening": {
            "config": asdict(screening_config),
            "semantics": (
                "conservative SDR screening proxy; rectangular angular field, no fine-pattern exception"),
        },
        "methods": list(METHODS),
        "screen_reference_shader_sha256": screen_reference.SHADER_SHA256,
        "apple_reference": {"upstream_commit": UPSTREAM_COMMIT, "kernel_sha256": KERNEL_SHA256,
                            "config": asdict(apple_config), "semantics": "MATLAB temporal core"},
        "recovery_candidates": {
            "source_hold": asdict(hold_config),
            "step_hold_35": asdict(step_hold_config),
            "log_rise_limit": asdict(limit_config),
            "semantics": "causal 8-bit SDR screen-reference candidates",
        },
        "frequency_sweep": {
            "signal": "0.04/1.0 linear full-frame square wave after a 0.5-second dark lead-in",
            "duration_seconds": 3.0,
            "rows": frequency_sweep,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": source_hashes("perceptual.py", "perceptual_lab.py", "flash_screening.py",
                                        "apple_vfm.py", "recovery.py",
                                        "automatic.py", "demo_filter.py", "filter.py", "fixtures.py", "color.py",
                                        "data/apple_ufm_kernels_6e9604d.txt",
                                        "data/APPLE_VIDEO_FLASHING_REDUCTION_NOTICE.md"),
        "limitations": [
            "Engineering proxies only: not a medical score or WCAG/ITU conformance result.",
            "The standards-informed screen approximates a 10-degree field with an axis-aligned rectangle and deliberately omits the fine-pattern exception.",
            "Output-only flash events are exact-frame mismatches; temporal shifts require trace review rather than automatic blame.",
            "The nits and visual-angle results are assumptions until the actual display and viewing distance are measured.",
            "Temporal filter bands are search features, not a validated implementation of Apple's UFM/VFM model.",
            "Local opposing changes include motion and camera cuts; RGB alone cannot identify game semantics.",
            "The scorecard stops at same-frame pixel output and does not observe DWM composition or emitted light.",
            "screen_burst_100 is the frozen 8-bit SDR CPU reference for one validated shader hash, not a live GPU/capture result.",
            "apple_matlab_vfm ports the pinned MATLAB temporal core and official kernels; it is not the shipping Apple feature.",
            "The Apple image helper receives already-linear project fixtures; upstream file decoding uses a power-2.2 approximation.",
            "A 1e-6-nit numeric floor makes all-black input finite and is an explicit deviation from MATLAB's zero offset.",
            "Recovery-candidate parameters are predeclared search values, not visibility or safety thresholds.",
            "Source hold trades recovery flicker for persistent dimming and would require an explicit capture-loss exit in production.",
            "The log-rise ceiling uses fixed screen cells, not gaze tracking; 8-bit quantization can exceed its nominal per-frame rate.",
            "Tentative step handling weakens first-pulse suppression because a causal controller cannot see the future reversal.",
            "The finite square-wave sweep measures sampled implementations, not a human contrast-sensitivity threshold.",
            "Full-reference perceptual quality and readability require separate evaluation.",
        ],
        "cases": [],
    }
    metrics_rows, total = [], len(case_ids) * spec.frame_count
    for case_index, case_id in enumerate(case_ids):
        analyzers = {name: PerceptualAnalyzer(profile, config) for name in METHODS}
        pairs = {name: PairedPerceptualAnalyzer(profile, config) for name in METHODS[1:]}
        screeners = {name: FlashScreeningAnalyzer(profile, screening_config) for name in METHODS}
        source_screeners = {name: FlashScreeningAnalyzer(profile, screening_config)
                            for name in METHODS}
        controllers = {
            "highlight_auto": AutoFilter(),
            "legacy_temporal": AutoFilter(AutoConfig(policy="legacy")),
            "screen_burst_100": screen_reference.Controller(),
            "screen_source_hold": SourceHoldController(hold_config),
            "screen_step_hold_35": SourceHoldController(step_hold_config),
            "screen_global_log_limit": LogRiseLimiter(profile, tiled=False, config=limit_config),
            "screen_tile_log_limit": LogRiseLimiter(profile, tiled=True, config=limit_config),
            "apple_matlab_vfm": AppleVFMReference(apple_config),
        }
        decision_analyzer = PerceptualAnalyzer(profile, config)
        measurements = {name: [] for name in METHODS}
        paired_measurements = {name: [] for name in METHODS[1:]}
        screening_measurements = {name: [] for name in METHODS}
        source_screening_measurements = {name: [] for name in METHODS}
        rows, control_rows, apple_frames = [], [], []
        for index, (timestamp, frame) in enumerate(frames(case_id, spec)):
            job.advance(case_index * spec.frame_count + index, total, case_id)
            encoded = np.rint(linear_to_srgb(frame) * 255).astype(np.uint8)
            screen_input, screen_y, screen_load, screen_mean = screen_reference.stats(encoded)
            decision = decision_analyzer.process(screen_input, timestamp, index)
            screen_gain = controllers["screen_burst_100"].step(screen_load, screen_mean, timestamp)
            screen_output = screen_reference.apply(encoded, screen_input, screen_y, screen_gain,
                                                     controllers["screen_burst_100"].broad, 1.0)
            hold_gain, hold_trace = controllers["screen_source_hold"].step(
                screen_load, screen_mean, timestamp, coherent_step=decision.coherent_step,
                opposed_area=decision.opposed_area_fraction)
            hold_output = screen_reference.apply(encoded, screen_input, screen_y, hold_gain,
                                                  controllers["screen_source_hold"].broad, 1.0)
            step_gain, step_trace = controllers["screen_step_hold_35"].step(
                screen_load, screen_mean, timestamp, coherent_step=decision.coherent_step,
                opposed_area=decision.opposed_area_fraction)
            step_output = screen_reference.apply(encoded, screen_input, screen_y, step_gain,
                                                  controllers["screen_step_hold_35"].broad, 1.0)
            global_output, global_trace = controllers["screen_global_log_limit"].process(
                encoded, screen_input, screen_y, screen_gain,
                controllers["screen_burst_100"].broad, timestamp, index)
            tile_output, tile_trace = controllers["screen_tile_log_limit"].process(
                encoded, screen_input, screen_y, screen_gain,
                controllers["screen_burst_100"].broad, timestamp, index)
            highlight_output, highlight_trace = controllers["highlight_auto"].process(frame, timestamp, index)
            legacy_output, legacy_trace = controllers["legacy_temporal"].process(frame, timestamp, index)
            apple_output, apple_trace = controllers["apple_matlab_vfm"].process_linear_rgb(frame)
            apple_frames.append(apple_trace)
            outputs = {
                "original": frame,
                "fixed_strong": compress_highlights(frame, STRONG),
                "highlight_auto": highlight_output,
                "legacy_temporal": legacy_output,
                "screen_burst_100": screen_reference.LINEAR[screen_output],
                "screen_source_hold": screen_reference.LINEAR[hold_output],
                "screen_step_hold_35": screen_reference.LINEAR[step_output],
                "screen_global_log_limit": global_output,
                "screen_tile_log_limit": tile_output,
                "apple_matlab_vfm": apple_output,
            }
            pair_sources = {name: frame for name in METHODS[1:]}
            for name in SCREEN_METHODS:
                pair_sources[name] = screen_input
            paired_by_method = {}
            for method in METHODS[1:]:
                paired_by_method[method] = pairs[method].process(pair_sources[method], outputs[method], timestamp, index)
                paired_measurements[method].append(paired_by_method[method])
            for method in METHODS:
                measurement = analyzers[method].process(outputs[method], timestamp, index)
                measurements[method].append(measurement)
                source_frame = pair_sources.get(method, frame)
                source_screening_measurements[method].append(
                    source_screeners[method].process(source_frame, timestamp, index))
                screening = screeners[method].process(outputs[method], timestamp, index)
                screening_measurements[method].append(screening)
                rows.append(_flat_frame(
                    case_id, method, measurement, paired_by_method.get(method), screening))
            control_rows.append({
                "frame_id": index, "timestamp": timestamp,
                "screen_gain": screen_gain, "screen_broad": controllers["screen_burst_100"].broad,
                "hold_gain": hold_gain, "hold_active": int(hold_trace["active"]),
                "hold_expired": int(hold_trace["expired"]),
                "step_hold_gain": step_gain, "step_hold_active": int(step_trace["active"]),
                "step_hold_tentative": int(step_trace["tentative_step"]),
                "global_limit_amount": global_trace["selected_amount"],
                "global_limit_added_amount": global_trace["added_amount"],
                "global_limit_infeasible": int(global_trace["infeasible"]),
                "global_limit_realized_log_rate": global_trace["max_realized_added_log_rise_per_second"],
                "tile_limit_amount": tile_trace["selected_amount"],
                "tile_limit_added_amount": tile_trace["added_amount"],
                "tile_limit_infeasible": int(tile_trace["infeasible"]),
                "tile_limit_realized_log_rate": tile_trace["max_realized_added_log_rise_per_second"],
                "tile_limit_envelope_overshoot": tile_trace["max_envelope_overshoot_log"],
                "highlight_amount": highlight_trace["amount"], "legacy_amount": legacy_trace["amount"],
                "apple_apl_relative": apple_trace.apl_relative,
                "apple_adaptation_nits": apple_trace.adaptation_luminance_nits,
                "apple_contrast": apple_trace.contrast, "apple_response_energy": apple_trace.response_energy,
                "apple_normalized_step_energy": apple_trace.normalized_step_energy,
                "apple_unmasked_risk": apple_trace.unmasked_risk, "apple_risk": apple_trace.risk,
                "apple_applied_mitigation": apple_trace.applied_mitigation,
                "apple_next_mitigation": apple_trace.next_mitigation,
                "apple_adaptation_floor_applied": int(apple_trace.adaptation_floor_applied),
            })
        summaries = {name: summarize_measurements(values) for name, values in measurements.items()}
        paired_summaries = {name: summarize_paired(values) for name, values in paired_measurements.items()}
        screening_summaries = {
            name: summarize_flash_screening(values)
            for name, values in screening_measurements.items()}
        screening_comparisons = {
            name: compare_flash_screening(source_screening_measurements[name], values)
            for name, values in screening_measurements.items()}
        diagnostics = {
            "screen_burst_100": {
                "max_gain": max(row["screen_gain"] for row in control_rows),
                "frames_with_gain": sum(row["screen_gain"] > 0 for row in control_rows),
            },
            "screen_source_hold": {
                "max_gain": max(row["hold_gain"] for row in control_rows),
                "frames_with_gain": sum(row["hold_gain"] > 0 for row in control_rows),
                "active_frames": sum(row["hold_active"] for row in control_rows),
                "expired_frames": sum(row["hold_expired"] for row in control_rows),
            },
            "screen_step_hold_35": {
                "max_gain": max(row["step_hold_gain"] for row in control_rows),
                "frames_with_gain": sum(row["step_hold_gain"] > 0 for row in control_rows),
                "active_frames": sum(row["step_hold_active"] for row in control_rows),
                "tentative_step_frames": sum(row["step_hold_tentative"] for row in control_rows),
            },
            "screen_global_log_limit": {
                "max_amount": max(row["global_limit_amount"] for row in control_rows),
                "extended_frames": sum(row["global_limit_added_amount"] > 1e-9 for row in control_rows),
                "infeasible_frames": sum(row["global_limit_infeasible"] for row in control_rows),
                "max_realized_added_log_rise_per_second": max(
                    row["global_limit_realized_log_rate"] for row in control_rows),
            },
            "screen_tile_log_limit": {
                "max_amount": max(row["tile_limit_amount"] for row in control_rows),
                "extended_frames": sum(row["tile_limit_added_amount"] > 1e-9 for row in control_rows),
                "infeasible_frames": sum(row["tile_limit_infeasible"] for row in control_rows),
                "max_realized_added_log_rise_per_second": max(
                    row["tile_limit_realized_log_rate"] for row in control_rows),
                "max_envelope_overshoot_log": max(
                    row["tile_limit_envelope_overshoot"] for row in control_rows),
                "tile_count": tile_trace["tile_count"],
            },
            "highlight_auto": {"max_amount": max(row["highlight_amount"] for row in control_rows)},
            "legacy_temporal": {"max_amount": max(row["legacy_amount"] for row in control_rows)},
            "apple_matlab_vfm": {
                "max_unmasked_risk": max(item.unmasked_risk for item in apple_frames),
                "max_risk": max(item.risk for item in apple_frames),
                "max_mitigation": max(item.next_mitigation for item in apple_frames),
                "first_risk_frame": next((item.frame_index for item in apple_frames if item.risk > 0), None),
                "frames_with_applied_mitigation": sum(item.applied_mitigation > 0 for item in apple_frames),
                "max_normalized_step_energy": max(item.normalized_step_energy for item in apple_frames),
                "adaptation_floor_frames": sum(item.adaptation_floor_applied for item in apple_frames),
            },
        }
        case = CASE_BY_ID[case_id]
        report["cases"].append({
            "id": case_id,
            "description": case.description,
            "fixture_limitation": case.limitation,
            "measurements": summaries,
            "source_aware_comparison": paired_summaries,
            "flash_screening": screening_summaries,
            "source_aware_flash_screening": screening_comparisons,
            "algorithm_diagnostics": diagnostics,
            "findings": _findings(
                case_id, summaries, paired_summaries, diagnostics,
                screening_summaries, screening_comparisons),
        })
        for method in METHODS:
            metrics_rows.append({"case": case_id, "method": method, **summaries[method],
                                 **{f"screening_{key}": value
                                    for key, value in screening_summaries[method].items()},
                                 **{f"source_aware_screening_{key}": value
                                    for key, value in screening_comparisons[method].items()},
                                 **({f"source_aware_{key}": value for key, value in paired_summaries[method].items()}
                                    if method in paired_summaries else {})})
        with (traces / f"{case_id}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (controls / f"{case_id}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(control_rows[0]))
            writer.writeheader()
            writer.writerows(control_rows)
    job.check()
    # Keep the in-memory return value identical to its JSON representation;
    # dataclass tuple fields otherwise become lists only after serialization.
    report = json.loads(json.dumps(report, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fieldnames = list(dict.fromkeys(key for row in metrics_rows for key in row))
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    with (output / "frequency_sweep.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frequency_sweep[0]))
        writer.writeheader()
        writer.writerows(frequency_sweep)

    lines = ["# Synthetic temporal-vision proxy scorecard", "",
             *[f"- {item}" for item in report["limitations"]], "",
             "| Case | Method | Peak nits | Temporal energy P95 | Max opposed area | Max opposing transitions/s | Screen flashes/s | Red change P95 | Added rise max (nits) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in report["cases"]:
        for method in METHODS:
            summary = case["measurements"][method]
            screening = case["flash_screening"][method]
            paired = case["source_aware_comparison"].get(method, {})
            lines.append(f"| {case['id']} | {method} | {summary['peak_luminance_nits']:.3f} | "
                         f"{summary['temporal_energy_p95']:.6f} | {summary['max_opposed_area_fraction']:.3f} | "
                         f"{summary['max_opposing_transitions_1s']} | "
                         f"{screening['max_combined_flashes_1s']} | "
                         f"{summary['red_change_mean_p95']:.6f} | "
                         f"{paired.get('added_rise_nits_p95_max', 0):.3f} |")
    for case in report["cases"]:
        lines += ["", f"## {case['id']}", "", *[f"- {item}" for item in case["findings"]], "",
                  f"[Frame trace](traces/{case['id']}.csv) · [Control trace](controls/{case['id']}.csv)"]
    lines += ["", "## Strong square-wave frequency sweep", "",
              "| Frequency (Hz) | Apple max risk | Apple unmasked | Screen active frames | Opposing transitions/s | Screen flashes/s | Temporal energy P95 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in frequency_sweep:
        lines.append(f"| {row['frequency_hz']:.1f} | {row['apple_max_risk']:.3f} | "
                     f"{row['apple_max_unmasked_risk']:.3f} | {row['screen_frames_with_gain']} | "
                     f"{row['opposing_transitions_1s']} | "
                     f"{row['screening_general_flashes_1s']} | "
                     f"{row['temporal_energy_p95']:.6f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
