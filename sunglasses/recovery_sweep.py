"""Bounded, headless ablation of local recovery scale, rate and quantization.

The tuning and holdout partitions are declared before any result is computed.
This module never renders media, captures a display or selects a production
default.  Its Pareto front is an engineering search result, not a comfort or
safety threshold.
"""

from dataclasses import asdict, dataclass
import csv
from itertools import product
import json
import math
from pathlib import Path
import platform

import numpy as np

from . import demo_filter as screen_reference
from .color import linear_to_srgb
from .fixtures import CASE_BY_ID, SequenceSpec, frames
from .i18n import ValidationError
from .jobs import JobContext
from .perceptual import (
    DisplayProfile,
    PairedPerceptualAnalyzer,
    PerceptualConfig,
    summarize_measurements,
    summarize_paired,
)
from .provenance import source_hashes
from .recovery import LocalLogRiseLimiter, LogRiseLimitConfig, LogRiseLimiter


CUSTOM_CASES = {
    "partial_bright_step": "Forty percent of the frame becomes bright and remains bright.",
    "contracting_highlight": "A large bright disk contracts while the background stays dark.",
    "redistributed_midtones": "Compressible white area becomes an equal-mean protected midtone.",
}
KNOWN_CASES = set(CASE_BY_ID) | set(CUSTOM_CASES)


@dataclass(frozen=True)
class RecoverySweepDesign:
    limiter_modes: tuple[str, ...] = ("scalar_worst", "local_map", "local_cone")
    cell_degrees: tuple[float, ...] = (1.0, 2.0, 4.0)
    log_rates_per_second: tuple[float, ...] = (.25, .5, 1.0)
    output_bits: tuple[int, ...] = (8, 10, 12)
    spatial_gradients_per_degree: tuple[float, ...] = (.05,)
    tuning_cases: tuple[str, ...] = ("hard_cut", "localized_burst", "white_pulses")
    holdout_cases: tuple[str, ...] = (
        "partial_bright_step",
        "contracting_highlight",
        "redistributed_midtones",
        "single_white_flash",
        "slow_ramp",
    )

    def __post_init__(self) -> None:
        if (not isinstance(self.limiter_modes, tuple) or not self.limiter_modes or
                len(set(self.limiter_modes)) != len(self.limiter_modes) or
                any(value not in ("scalar_worst", "local_map", "local_cone")
                    for value in self.limiter_modes)):
            raise ValidationError("recovery_sweep_modes")
        numeric_groups = (self.cell_degrees, self.log_rates_per_second,
                          self.spatial_gradients_per_degree)
        if any(not isinstance(group, tuple) or not group for group in numeric_groups):
            raise ValidationError("recovery_sweep_values")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for group in numeric_groups for value in group):
            raise ValidationError("recovery_sweep_values")
        if tuple(sorted(set(self.cell_degrees))) != self.cell_degrees:
            raise ValidationError("recovery_sweep_values")
        if tuple(sorted(set(self.log_rates_per_second))) != self.log_rates_per_second:
            raise ValidationError("recovery_sweep_values")
        if (not isinstance(self.output_bits, tuple) or not self.output_bits or
                any(type(value) is not int or not 8 <= value <= 16 for value in self.output_bits) or
                tuple(sorted(set(self.output_bits))) != self.output_bits):
            raise ValidationError("recovery_sweep_bits")
        cases = (*self.tuning_cases, *self.holdout_cases)
        if (not self.tuning_cases or not self.holdout_cases or len(set(cases)) != len(cases) or
                any(case not in KNOWN_CASES for case in cases)):
            raise ValidationError("recovery_sweep_cases")


def _custom_frames(case_id: str, spec: SequenceSpec):
    yy, xx = np.mgrid[:spec.height, :spec.width]
    x = xx / spec.width
    y = yy / spec.height
    for index in range(spec.frame_count):
        phase = index / spec.frame_count
        frame = np.full((spec.height, spec.width, 3), .04, dtype=np.float32)
        if case_id == "partial_bright_step":
            if phase >= .5:
                frame[:, x[0] < .40] = 1.0
        elif case_id == "contracting_highlight":
            if .25 <= phase < .70:
                progress = (phase - .25) / .45
                radius = .27 - .16 * progress
                disk = (x - .67) ** 2 + (y - .42) ** 2 < radius**2
                frame[disk] = (1.0, .94, .82)
        elif case_id == "redistributed_midtones":
            if .25 <= phase < .40:
                frame[:] = 0
                frame[:, x[0] < .25] = 1.0
            elif phase >= .40:
                frame[:] = .25
        else:
            raise ValidationError("recovery_sweep_cases")
        yield index / spec.fps, frame


def _case_frames(case_id: str, spec: SequenceSpec):
    return _custom_frames(case_id, spec) if case_id in CUSTOM_CASES else frames(case_id, spec)


def _candidate_id(mode: str, cell_degrees: float, rate: float, bits: int,
                  gradient: float | None) -> str:
    cell = str(cell_degrees).replace(".", "p")
    speed = str(rate).replace(".", "p")
    suffix = "" if gradient is None else f"_g{str(gradient).replace('.', 'p')}"
    return f"{mode}_cell_{cell}_rate_{speed}_q{bits}{suffix}"


def _prepare(case_id: str, spec: SequenceSpec):
    controller = screen_reference.Controller()
    prepared = []
    for index, (timestamp, frame) in enumerate(_case_frames(case_id, spec)):
        encoded = np.rint(linear_to_srgb(frame) * 255).astype(np.uint8)
        linear, y, load, mean = screen_reference.stats(encoded)
        gain = controller.step(load, mean, timestamp)
        baseline = screen_reference.LINEAR[
            screen_reference.apply(encoded, linear, y, gain, controller.broad, 1.0)]
        prepared.append((index, timestamp, encoded, linear, y, gain, controller.broad, baseline))
    return prepared


def _metrics(case_id: str, split: str, candidate_id: str, prepared, profile: DisplayProfile,
             perceptual_config: PerceptualConfig,
             limiter: LogRiseLimiter | LocalLogRiseLimiter | None,
             spec: SequenceSpec, audit_rows: list[dict], *, job: JobContext,
             progress: list[int], total: int, parameters: dict):
    analyzer = PairedPerceptualAnalyzer(profile, perceptual_config)
    paired, traces = [], []
    for index, timestamp, encoded, linear, y, gain, broad, baseline in prepared:
        if limiter is None:
            output = baseline
            trace = {
                "desired_amount": gain,
                "selected_amount": gain,
                "selected_amount_mean": gain,
                "added_amount": 0.0,
                "added_amount_mean": 0.0,
                "extended_tiles": 0,
                "infeasible": False,
                "infeasible_tiles": 0,
                "max_envelope_overshoot_log": 0.0,
                "max_realized_added_log_rise_per_second": None,
                "max_neighbor_amount_delta": 0.0,
                "max_neighbor_amount_gradient_per_degree": 0.0,
                "boundary_control_contrast_mean": 0.0,
                "boundary_control_contrast_p95": 0.0,
                "boundary_control_contrast_max": 0.0,
                "max_spatial_ceiling_credit_log": 0.0,
            }
        else:
            output, trace = limiter.process(encoded, linear, y, gain, broad, timestamp, index)
        measurement = analyzer.process(linear, output, timestamp, index)
        paired.append(measurement)
        traces.append(trace)
        if trace["added_amount"] > 1e-9 or trace["infeasible"]:
            audit_rows.append({
                "split": split,
                "case": case_id,
                "candidate": candidate_id,
                "frame_id": index,
                "timestamp": timestamp,
                "desired_amount": trace["desired_amount"],
                "selected_amount": trace["selected_amount"],
                "selected_amount_mean": trace.get("selected_amount_mean", trace["selected_amount"]),
                "added_amount": trace["added_amount"],
                "added_amount_mean": trace.get("added_amount_mean", trace["added_amount"]),
                "extended_tiles": trace.get(
                    "extended_tiles", trace.get("tile_count", 1) if trace["added_amount"] > 1e-9 else 0),
                "infeasible": int(trace["infeasible"]),
                "infeasible_tiles": trace.get(
                    "infeasible_tiles", trace.get("tile_count", 1) if trace["infeasible"] else 0),
                "envelope_overshoot_log": trace["max_envelope_overshoot_log"],
                "realized_added_log_rate": trace["max_realized_added_log_rise_per_second"],
                "max_neighbor_amount_delta": trace.get("max_neighbor_amount_delta", 0.0),
                "max_neighbor_amount_gradient_per_degree": trace.get(
                    "max_neighbor_amount_gradient_per_degree", 0.0),
                "boundary_control_contrast_max": trace.get("boundary_control_contrast_max", 0.0),
                "spatial_ceiling_credit_log": trace.get("max_spatial_ceiling_credit_log", 0.0),
            })
        progress[0] += 1
        job.advance(progress[0], total, f"{split}:{case_id}:{candidate_id}")
    pair_summary = summarize_paired(paired)
    output_summary = summarize_measurements([item.output for item in paired])
    realized_rates = [trace["max_realized_added_log_rise_per_second"] for trace in traces
                      if trace["max_realized_added_log_rise_per_second"] is not None]
    feasible_rates = [trace["max_realized_added_log_rise_per_second"] for trace in traces
                      if (trace["max_realized_added_log_rise_per_second"] is not None and
                          not trace["infeasible"])]
    return {
        "split": split,
        "case": case_id,
        "candidate": candidate_id,
        **parameters,
        "peak_luminance_nits": output_summary["peak_luminance_nits"],
        "temporal_energy_p95": output_summary["temporal_energy_p95"],
        "added_rise_nits_p95_max": pair_summary["added_rise_nits_p95_max"],
        "added_fall_nits_p95_max": pair_summary["added_fall_nits_p95_max"],
        "temporal_residual_nits_p95_max": pair_summary["temporal_residual_nits_p95_max"],
        "spatial_residual_log_gradient_p95_max": pair_summary[
            "spatial_residual_log_gradient_p95_max"],
        "spatial_residual_log_gradient_max": pair_summary["spatial_residual_log_gradient_max"],
        "spatial_residual_log_gradient_mean_seconds": sum(
            item.spatial_residual_log_gradient_mean for item in paired) / spec.fps,
        "same_frame_brightening_nits_max": pair_summary["same_frame_brightening_nits_max"],
        "same_frame_dimming_nits_max": pair_summary["same_frame_dimming_nits_max"],
        "mean_dimming_over_1_nit_seconds": pair_summary["mean_dimming_over_1_nit_seconds"],
        "mean_dimming_nit_seconds": sum(item.same_frame_dimming_nits_mean for item in paired) / spec.fps,
        "extended_frames": sum(trace["added_amount"] > 1e-9 for trace in traces),
        "extended_tile_events": sum(
            trace.get("extended_tiles", trace.get("tile_count", 1)
                      if trace["added_amount"] > 1e-9 else 0) for trace in traces),
        "infeasible_frames": sum(trace["infeasible"] for trace in traces),
        "infeasible_tile_events": sum(
            trace.get("infeasible_tiles", trace.get("tile_count", 1)
                      if trace["infeasible"] else 0) for trace in traces),
        "max_envelope_overshoot_log": max(trace["max_envelope_overshoot_log"] for trace in traces),
        "max_realized_added_log_rise_per_second": max(realized_rates) if realized_rates else None,
        "max_feasible_realized_added_log_rise_per_second": (
            max(feasible_rates) if feasible_rates else None),
        "max_neighbor_amount_delta": max(
            trace.get("max_neighbor_amount_delta", 0.0) for trace in traces),
        "max_neighbor_amount_gradient_per_degree": max(
            trace.get("max_neighbor_amount_gradient_per_degree", 0.0) for trace in traces),
        "boundary_control_contrast_p95_max": max(
            trace.get("boundary_control_contrast_p95", 0.0) for trace in traces),
        "boundary_control_contrast_max": max(
            trace.get("boundary_control_contrast_max", 0.0) for trace in traces),
        "boundary_control_contrast_mean_seconds": sum(
            trace.get("boundary_control_contrast_mean", 0.0) for trace in traces) / spec.fps,
        "max_spatial_ceiling_credit_log": max(
            trace.get("max_spatial_ceiling_credit_log", 0.0) for trace in traces),
    }


def _aggregate(rows: list[dict], candidates: list[str], split: str):
    result = []
    for candidate in candidates:
        selected = [row for row in rows if row["candidate"] == candidate and row["split"] == split]
        if not selected:
            continue
        first = selected[0]
        result.append({
            "split": split,
            "candidate": candidate,
            "limiter_mode": first["limiter_mode"],
            "cell_degrees": first["cell_degrees"],
            "log_rate_per_second": first["log_rate_per_second"],
            "output_bits": first["output_bits"],
            "max_amount_gradient_per_degree": first["max_amount_gradient_per_degree"],
            "max_added_rise_nits": max(row["added_rise_nits_p95_max"] for row in selected),
            "total_mean_dimming_nit_seconds": sum(row["mean_dimming_nit_seconds"] for row in selected),
            "max_case_dimming_seconds": max(row["mean_dimming_over_1_nit_seconds"] for row in selected),
            "max_temporal_energy_p95": max(row["temporal_energy_p95"] for row in selected),
            "max_spatial_residual_log_gradient_p95": max(
                row["spatial_residual_log_gradient_p95_max"] for row in selected),
            "max_spatial_residual_log_gradient": max(
                row["spatial_residual_log_gradient_max"] for row in selected),
            "total_spatial_residual_log_gradient_mean_seconds": sum(
                row["spatial_residual_log_gradient_mean_seconds"] for row in selected),
            "max_peak_luminance_nits": max(row["peak_luminance_nits"] for row in selected),
            "extended_frames": sum(row["extended_frames"] for row in selected),
            "extended_tile_events": sum(row["extended_tile_events"] for row in selected),
            "infeasible_frames": sum(row["infeasible_frames"] for row in selected),
            "infeasible_tile_events": sum(row["infeasible_tile_events"] for row in selected),
            "max_realized_added_log_rise_per_second": max(
                (row["max_realized_added_log_rise_per_second"] for row in selected
                 if row["max_realized_added_log_rise_per_second"] is not None), default=None),
            "max_feasible_realized_added_log_rise_per_second": max(
                (row["max_feasible_realized_added_log_rise_per_second"] for row in selected
                 if row["max_feasible_realized_added_log_rise_per_second"] is not None), default=None),
            "max_neighbor_amount_delta": max(row["max_neighbor_amount_delta"] for row in selected),
            "max_neighbor_amount_gradient_per_degree": max(
                row["max_neighbor_amount_gradient_per_degree"] for row in selected),
            "max_boundary_control_contrast_p95": max(
                row["boundary_control_contrast_p95_max"] for row in selected),
            "max_boundary_control_contrast": max(
                row["boundary_control_contrast_max"] for row in selected),
            "total_boundary_control_contrast_mean_seconds": sum(
                row["boundary_control_contrast_mean_seconds"] for row in selected),
            "max_spatial_ceiling_credit_log": max(
                row["max_spatial_ceiling_credit_log"] for row in selected),
            "max_same_frame_brightening_nits": max(
                row["same_frame_brightening_nits_max"] for row in selected),
        })
    objectives = ("max_added_rise_nits", "total_mean_dimming_nit_seconds", "infeasible_frames",
                  "max_boundary_control_contrast")
    for current in result:
        current["pareto"] = not any(
            all(other[key] <= current[key] + 1e-12 for key in objectives) and
            any(other[key] < current[key] - 1e-12 for key in objectives)
            for other in result if other is not current
        )
    return result


def run_recovery_sweep(output: Path, spec: SequenceSpec, profile: DisplayProfile,
                       design: RecoverySweepDesign = RecoverySweepDesign(),
                       perceptual_config: PerceptualConfig = PerceptualConfig(),
                       *, job: JobContext = JobContext()):
    """Run the declared grid and write reproducible JSON/CSV/Markdown evidence."""
    job.check()
    output.mkdir(parents=True, exist_ok=False)
    candidates = [("current_screen_8bit", "current", None)]
    for mode in design.limiter_modes:
        gradients = design.spatial_gradients_per_degree if mode == "local_cone" else (None,)
        for cell, rate, bits, gradient in product(design.cell_degrees,
                                                  design.log_rates_per_second,
                                                  design.output_bits, gradients):
            config = LogRiseLimitConfig(max_added_log_rise_per_second=rate,
                                        cell_degrees=cell, output_bits=bits,
                                        max_amount_gradient_per_degree=gradient)
            candidates.append((_candidate_id(mode, cell, rate, bits, gradient), mode, config))
    case_plan = [("tuning", case) for case in design.tuning_cases]
    case_plan += [("holdout", case) for case in design.holdout_cases]
    total = len(candidates) * len(case_plan) * spec.frame_count
    progress = [0]
    case_rows, audit_rows = [], []
    for split, case_id in case_plan:
        prepared = _prepare(case_id, spec)
        for candidate_id, limiter_mode, limit_config in candidates:
            baseline = limit_config is None
            parameters = {
                "limiter_mode": limiter_mode,
                "cell_degrees": None if baseline else limit_config.cell_degrees,
                "log_rate_per_second": None if baseline else limit_config.max_added_log_rise_per_second,
                "output_bits": 8 if baseline else limit_config.output_bits,
                "max_amount_gradient_per_degree": (
                    None if baseline else limit_config.max_amount_gradient_per_degree),
            }
            if baseline:
                limiter = None
            elif limiter_mode == "scalar_worst":
                limiter = LogRiseLimiter(profile, tiled=True, config=limit_config)
            else:
                limiter = LocalLogRiseLimiter(profile, config=limit_config)
            case_rows.append(_metrics(case_id, split, candidate_id, prepared, profile,
                                      perceptual_config, limiter, spec, audit_rows,
                                      job=job, progress=progress, total=total,
                                      parameters=parameters))
    candidate_ids = [candidate_id for candidate_id, _, _ in candidates]
    aggregates = _aggregate(case_rows, candidate_ids, "tuning")
    aggregates += _aggregate(case_rows, candidate_ids, "holdout")
    tuning_front = {row["candidate"] for row in aggregates if row["split"] == "tuning" and row["pareto"]}
    holdout_front = {row["candidate"] for row in aggregates if row["split"] == "holdout" and row["pareto"]}
    report = {
        "schema_version": 2,
        "kind": "recovery_architecture_parameter_quantization_ablation",
        "scope": "offline causal 8-bit SDR input; output depth above 8-bit is counterfactual",
        "sequence": asdict(spec),
        "display_profile": asdict(profile),
        "perceptual_config": asdict(perceptual_config),
        "design": asdict(design),
        "objectives": ["max added local rise", "integrated mean dimming", "infeasible frames",
                       "max visibility-weighted local-control boundary contrast"],
        "selection_rule": "Pareto membership is computed on tuning cases only; holdout never selects parameters.",
        "candidate_count": len(candidates),
        "tuning_pareto": sorted(tuning_front),
        "holdout_pareto": sorted(holdout_front),
        "front_intersection": sorted(tuning_front & holdout_front),
        "case_metrics": case_rows,
        "candidate_summaries": aggregates,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": source_hashes("recovery.py", "recovery_sweep.py", "perceptual.py",
                                        "perceptual_lab.py", "demo_filter.py", "fixtures.py", "color.py"),
        "limitations": [
            "No result is a medical, comfort or visibility threshold.",
            "Holdout cases are synthetic structural variants, not independent game recordings or participants.",
            "Ten- and twelve-bit outputs are hypothetical; the current capture and swap chain is eight-bit SDR.",
            "Fixed screen cells do not follow gaze, retinal motion or eccentricity.",
            "Local candidates remain piecewise-constant at pixel level; neighboring amount gradient is only an angular seam-risk proxy, not a visibility threshold.",
            "The spatial residual is an uncalibrated adjacent-pixel log-luminance proxy; it is not a spatial CSF or ColorVideoVDP score.",
            "Boundary control contrast weights tile-amount differences by local maximum compressibility; it is an uncalibrated seam-risk proxy.",
            "Pareto membership depends on the declared cases, objectives and assumed display profile.",
            "The experiment observes algorithm pixels, not DWM composition or emitted display light.",
        ],
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    with (output / "case_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    with (output / "candidate_summaries.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with (output / "control_audit.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = (list(audit_rows[0]) if audit_rows else
                  ["split", "case", "candidate", "frame_id", "timestamp", "desired_amount",
                   "selected_amount", "added_amount", "infeasible", "envelope_overshoot_log",
                   "realized_added_log_rate"])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    lines = [
        "# Recovery scale/rate/quantization ablation",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        "Tuning cases determine the first Pareto flag. Holdout cases are reported afterward and never retune it.",
        "",
        "| Split | Candidate | Max added rise (nit) | Integrated mean dimming (nit·s) | Max boundary contrast | Infeasible frames | Neighbor gradient (/deg) | Max realized log rate | Pareto |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in aggregates:
        realized_rate = row["max_feasible_realized_added_log_rise_per_second"]
        realized_text = "n/a" if realized_rate is None else f"{realized_rate:.4f}"
        lines.append(f"| {row['split']} | {row['candidate']} | {row['max_added_rise_nits']:.4f} | "
                     f"{row['total_mean_dimming_nit_seconds']:.4f} | "
                     f"{row['max_boundary_control_contrast']:.4f} | "
                     f"{row['infeasible_frames']} | {row['max_neighbor_amount_gradient_per_degree']:.4f} | "
                     f"{realized_text} | "
                     f"{'yes' if row['pareto'] else 'no'} |")
    lines += ["", f"Tuning Pareto: {', '.join(report['tuning_pareto'])}", "",
              f"Holdout Pareto: {', '.join(report['holdout_pareto'])}", "",
              f"Intersection: {', '.join(report['front_intersection']) or 'none'}", "",
              "No production default is selected by this report."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
