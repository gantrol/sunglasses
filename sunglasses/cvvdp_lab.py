"""Optional ColorVideoVDP validation for recovery candidates.

The official metric is an optional research dependency and is imported only
inside ``run_cvvdp_suite``.  This module feeds in-memory uint8 sRGB arrays to
the metric; it never writes or plays flashing media.  JOD measures fidelity to
the source and is never treated as a flash-safety or comfort score.
"""

from dataclasses import asdict, dataclass
import csv
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from . import demo_filter as screen_reference
from .color import linear_to_srgb
from .detail_protection import StableDetailConfig, StableDetailProtector
from .fixtures import CASE_BY_ID, SequenceSpec, frames
from .i18n import ValidationError
from .jobs import JobContext
from .perceptual import DisplayProfile, PairedPerceptualAnalyzer, PerceptualConfig, summarize_paired
from .provenance import source_hashes
from .recovery import (LocalLogRiseLimiter, LogRiseLimitConfig, LogRiseLimiter,
                       PixelLogRiseLimiter)


DETAIL_CASES = {
    "hud_flash": "Bright scene event behind a fixed protected-luminance HUD.",
    "subtitle_flash": "Bright scene event behind fixed thin white subtitle-like strokes.",
    "dense_detail_flash": "Bright event behind stable marks exceeding the protection area budget.",
    "moving_highlight": "A fixed-area highlight moves through screen-fixed recovery cells.",
}
KNOWN_CASES = set(CASE_BY_ID) | set(DETAIL_CASES)
CANDIDATES = ("identity", "current_screen_8bit", "scalar_rate_0p5_q8",
              "local_cone_4deg_rate_0p5_q8_g0p02",
              "pixel_cone_4deg_rate_0p5_q8_g0p02",
              "pixel_cone_detail_ranked_4deg_rate_0p5_q8_g0p02",
              "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02")
CHANNEL_NAMES = ("achromatic_sustained", "red_green", "yellow_violet",
                 "achromatic_transient")


@dataclass(frozen=True)
class ColorVideoVDPDesign:
    cases: tuple[str, ...] = (
        "hard_cut",
        "localized_burst",
        "white_pulses",
        "hud_flash",
        "subtitle_flash",
        "dense_detail_flash",
        "moving_highlight",
    )
    candidates: tuple[str, ...] = CANDIDATES
    expected_cvvdp_version: str = "0.5.7"
    temporal_padding: str = "replicate"

    def __post_init__(self) -> None:
        if (not isinstance(self.cases, tuple) or not self.cases or
                len(set(self.cases)) != len(self.cases) or
                any(value not in KNOWN_CASES for value in self.cases)):
            raise ValidationError("cvvdp_cases")
        if (not isinstance(self.candidates, tuple) or not self.candidates or
                len(set(self.candidates)) != len(self.candidates) or
                any(value not in CANDIDATES for value in self.candidates)):
            raise ValidationError("cvvdp_candidates")
        if (not isinstance(self.expected_cvvdp_version, str) or
                not self.expected_cvvdp_version.strip()):
            raise ValidationError("cvvdp_version")
        if self.temporal_padding not in ("replicate", "symmetric", "valid"):
            raise ValidationError("cvvdp_padding")


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant")
    result = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def _detail_geometry(spec: SequenceSpec) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:spec.height, :spec.width]
    x, y = xx / spec.width, yy / spec.height
    hud = (((x > .05) & (x < .34) & (y > .86) & (y < .91)) |
           ((np.abs(x - .50) < .004) & (np.abs(y - .50) < .035)) |
           ((np.abs(y - .50) < .006) & (np.abs(x - .50) < .025)))
    subtitle_box = (x > .20) & (x < .80) & (y > .78) & (y < .88)
    unit_x = max(2, spec.width // 80)
    unit_y = max(2, spec.height // 45)
    strokes = subtitle_box & (((xx // unit_x) % 4 == 0) | ((yy // unit_y) % 5 == 0))
    outline = _dilate(strokes) & ~strokes
    surround = subtitle_box & ~(_dilate(outline | strokes))
    dense_step_x = max(4, round(spec.width / 40))
    dense_step_y = max(4, round(spec.height / 23))
    dense_width = max(1, round(dense_step_x * .25))
    dense_height = max(1, round(dense_step_y * .25))
    dense_detail = ((xx % dense_step_x) < dense_width) & ((yy % dense_step_y) < dense_height)
    return {"hud": hud, "subtitle": strokes, "subtitle_outline": outline,
            "subtitle_surround": surround, "dense_detail": dense_detail}


def detail_frames(case_id: str, spec: SequenceSpec):
    """Yield custom linear-light frames and never render them."""
    if case_id not in DETAIL_CASES:
        raise ValidationError("cvvdp_cases")
    yy, xx = np.mgrid[:spec.height, :spec.width]
    x, y = xx / spec.width, yy / spec.height
    geometry = _detail_geometry(spec)
    texture = .012 * (np.sin(x * math.tau * 13) + np.cos(y * math.tau * 7))
    base_y = np.clip(.035 + .075 * x + texture, .015, .16).astype(np.float32)
    base = np.repeat(base_y[..., None], 3, axis=-1)
    for index in range(spec.frame_count):
        timestamp = index / spec.fps
        phase = index / spec.frame_count
        frame = base.copy()
        if case_id in ("hud_flash", "subtitle_flash", "dense_detail_flash") and .28 <= phase < .62:
            progress = (phase - .28) / .34
            center_x = .62 - .10 * progress
            radius = .30 - .08 * progress
            event = (x - center_x) ** 2 + (y - .43) ** 2 < radius**2
            frame[event] = (1.0, .92, .72)
        elif case_id == "moving_highlight" and .20 <= phase < .80:
            progress = (phase - .20) / .60
            center_x = .12 + .76 * progress
            event = (x - center_x) ** 2 + (y - .46) ** 2 < .12**2
            frame[event] = (1.0, .96, .82)
        if case_id == "hud_flash":
            frame[geometry["hud"]] = (.10, .28, .10)
        elif case_id == "subtitle_flash":
            frame[geometry["subtitle_outline"]] = .008
            frame[geometry["subtitle"]] = .92
        elif case_id == "dense_detail_flash":
            frame[geometry["dense_detail"]] = .92
        yield timestamp, frame


def _case_frames(case_id: str, spec: SequenceSpec):
    return detail_frames(case_id, spec) if case_id in DETAIL_CASES else frames(case_id, spec)


def prepare_case(case_id: str, spec: SequenceSpec):
    """Freeze source encoding, detector controls and current screen output."""
    controller = screen_reference.Controller()
    prepared = []
    for index, (timestamp, frame) in enumerate(_case_frames(case_id, spec)):
        encoded = np.rint(linear_to_srgb(frame) * 255).astype(np.uint8)
        linear, y, load, mean = screen_reference.stats(encoded)
        amount = controller.step(load, mean, timestamp)
        current = screen_reference.LINEAR[
            screen_reference.apply(encoded, linear, y, amount, controller.broad, 1.0)]
        prepared.append((index, timestamp, encoded, linear, y, amount, controller.broad, current))
    return prepared


def candidate_sequence(candidate: str, prepared, profile: DisplayProfile):
    if candidate not in CANDIDATES:
        raise ValidationError("cvvdp_candidates")
    if candidate == "scalar_rate_0p5_q8":
        limiter = LogRiseLimiter(profile, tiled=True,
                                 config=LogRiseLimitConfig(max_added_log_rise_per_second=.5,
                                                           cell_degrees=2, output_bits=8))
    elif candidate == "local_cone_4deg_rate_0p5_q8_g0p02":
        limiter = LocalLogRiseLimiter(
            profile,
            config=LogRiseLimitConfig(max_added_log_rise_per_second=.5, cell_degrees=4,
                                      output_bits=8, max_amount_gradient_per_degree=.02),
        )
    elif candidate in ("pixel_cone_4deg_rate_0p5_q8_g0p02",
                       "pixel_cone_detail_ranked_4deg_rate_0p5_q8_g0p02",
                       "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02"):
        limiter = PixelLogRiseLimiter(
            profile,
            config=LogRiseLimitConfig(max_added_log_rise_per_second=.5, cell_degrees=4,
                                      output_bits=8, max_amount_gradient_per_degree=.02),
        )
    else:
        limiter = None
    if candidate == "pixel_cone_detail_ranked_4deg_rate_0p5_q8_g0p02":
        protector = StableDetailProtector(
            profile, StableDetailConfig(area_overflow_policy="ranked_pixels"))
    elif candidate == "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02":
        protector = StableDetailProtector(profile)
    else:
        protector = None
    outputs, traces = [], []
    for index, timestamp, encoded, linear, y, amount, broad, current in prepared:
        if candidate == "identity":
            output = linear
            trace = {"infeasible": False, "max_spatial_ceiling_credit_log": 0.0}
        elif candidate == "current_screen_8bit":
            output = current
            trace = {"infeasible": False, "max_spatial_ceiling_credit_log": 0.0}
        else:
            output, trace = limiter.process(encoded, linear, y, amount, broad, timestamp, index)
            if protector is not None:
                output, detail_trace = protector.process(encoded, linear, output, timestamp, index)
                trace = {**trace,
                         **{f"detail_{key}": value for key, value in detail_trace.items()}}
        outputs.append(output)
        traces.append(trace)
    return outputs, traces


def _encoded_video(outputs) -> np.ndarray:
    return np.stack([np.rint(linear_to_srgb(frame) * 255).astype(np.uint8)
                     for frame in outputs])


def _roi_metrics(case_id: str, prepared, outputs, profile: DisplayProfile, spec: SequenceSpec) -> dict:
    geometry = _detail_geometry(spec)
    result = {
        "hud_dimming_nits_max": None,
        "subtitle_dimming_nits_max": None,
        "subtitle_contrast_retention_min": None,
        "dense_detail_dimming_nits_max": None,
        "dense_detail_dimming_fraction_over_1_nit_max": None,
        "dense_detail_luminance_uniformity_min": None,
    }
    if case_id == "hud_flash":
        dimming = []
        for item, output in zip(prepared, outputs):
            source_y = item[4]
            output_y = np.sum(output * screen_reference.LUMA, axis=-1)
            delta = profile.to_nits(source_y) - profile.to_nits(output_y)
            dimming.append(float(np.maximum(delta[geometry["hud"]], 0).max(initial=0)))
        result["hud_dimming_nits_max"] = max(dimming)
    elif case_id == "subtitle_flash":
        dimming, retention = [], []
        for item, output in zip(prepared, outputs):
            source_y = item[4]
            output_y = np.sum(output * screen_reference.LUMA, axis=-1)
            source_nits, output_nits = profile.to_nits(source_y), profile.to_nits(output_y)
            foreground = geometry["subtitle"]
            surround = geometry["subtitle_surround"]
            dimming.append(float(np.maximum(source_nits[foreground] - output_nits[foreground], 0)
                                 .max(initial=0)))
            source_contrast = ((float(source_nits[foreground].mean()) + .5) /
                               (float(source_nits[surround].mean()) + .5))
            output_contrast = ((float(output_nits[foreground].mean()) + .5) /
                               (float(output_nits[surround].mean()) + .5))
            retention.append(output_contrast / source_contrast)
        result["subtitle_dimming_nits_max"] = max(dimming)
        result["subtitle_contrast_retention_min"] = min(retention)
    elif case_id == "dense_detail_flash":
        dimming, fraction, uniformity = [], [], []
        foreground = geometry["dense_detail"]
        for item, output in zip(prepared, outputs):
            source_nits = profile.to_nits(item[4])
            output_y = np.sum(output * screen_reference.LUMA, axis=-1)
            output_nits = profile.to_nits(output_y)
            delta = np.maximum(source_nits[foreground] - output_nits[foreground], 0)
            dimming.append(float(delta.max(initial=0)))
            fraction.append(float((delta > 1).mean()))
            low, high = np.quantile(output_nits[foreground], (.1, .9))
            uniformity.append(float((low + .5) / (high + .5)))
        result["dense_detail_dimming_nits_max"] = max(dimming)
        result["dense_detail_dimming_fraction_over_1_nit_max"] = max(fraction)
        result["dense_detail_luminance_uniformity_min"] = min(uniformity)
    return result


def summarize_cvvdp_stats(stats: dict) -> dict:
    values = np.asarray(stats["Q_per_ch"], dtype=np.float64)
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValidationError("cvvdp_stats")
    channels = CHANNEL_NAMES[:values.shape[1]]
    return {
        "rho_band_cycles_per_degree": [float(value) for value in np.asarray(stats["rho_band"]).ravel()],
        "difference_max_by_channel": {
            name: float(values[0, index].max(initial=0)) for index, name in enumerate(channels)
        },
        "difference_max_by_spatial_band": [
            float(values[:, :, :, index].max(initial=0)) for index in range(values.shape[3])
        ],
    }


def _load_backend(expected_version: str, device: str):
    try:
        pycvvdp = importlib.import_module("pycvvdp")
        torch = importlib.import_module("torch")
        version = importlib.metadata.version("cvvdp")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ValidationError("cvvdp_dependency") from error
    if version != expected_version:
        raise ValidationError("cvvdp_version")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValidationError("cvvdp_device")
    if device not in ("cpu", "cuda"):
        raise ValidationError("cvvdp_device")
    return pycvvdp, torch, version


def run_cvvdp_suite(output: Path, spec: SequenceSpec, profile: DisplayProfile,
                    design: ColorVideoVDPDesign = ColorVideoVDPDesign(),
                    perceptual_config: PerceptualConfig = PerceptualConfig(), *,
                    device: str = "cuda", job: JobContext = JobContext()):
    """Run official ColorVideoVDP on declared in-memory candidate videos."""
    job.check()
    pycvvdp, torch, version = _load_backend(design.expected_cvvdp_version, device)
    output.mkdir(parents=True, exist_ok=False)
    contrast = profile.peak_luminance_nits / max(profile.black_luminance_nits, .005)
    photometry = pycvvdp.vvdp_display_photo_eotf(
        Y_peak=profile.peak_luminance_nits, contrast=contrast, source_colorspace="sRGB")
    geometry = pycvvdp.vvdp_display_geometry(
        (spec.width, spec.height), diagonal_size_inches=profile.diagonal_inches,
        distance_m=profile.viewing_distance_m)
    metric = pycvvdp.cvvdp(
        display_photometry=photometry, display_geometry=geometry, quiet=True,
        device=torch.device(device), temp_padding=design.temporal_padding)
    rows, total = [], len(design.cases) * len(design.candidates)
    started = time.perf_counter()
    progress = 0
    for case_id in design.cases:
        prepared = prepare_case(case_id, spec)
        source_video = np.stack([item[2] for item in prepared])
        source_linear = [item[3] for item in prepared]
        for candidate in design.candidates:
            job.check()
            outputs, traces = candidate_sequence(candidate, prepared, profile)
            analyzer = PairedPerceptualAnalyzer(profile, perceptual_config)
            paired = [analyzer.process(source, result, item[1], item[0])
                      for item, source, result in zip(prepared, source_linear, outputs)]
            paired_summary = summarize_paired(paired)
            candidate_video = _encoded_video(outputs)
            quality, stats = metric.predict(candidate_video, source_video, dim_order="FHWC",
                                            frames_per_second=spec.fps)
            metric_summary = summarize_cvvdp_stats(stats)
            peak_added = max(paired, key=lambda item: item.added_rise_nits_p95)
            realized_rates = [trace.get("max_realized_added_log_rise_per_second")
                              for trace in traces
                              if trace.get("max_realized_added_log_rise_per_second") is not None]
            feasible_rates = [trace.get("max_realized_added_log_rise_per_second")
                              for trace in traces
                              if (trace.get("max_realized_added_log_rise_per_second") is not None and
                                  not trace["infeasible"])]
            rows.append({
                "case": case_id,
                "candidate": candidate,
                "quality_jod": float(quality),
                "added_rise_nits_p95_max": paired_summary["added_rise_nits_p95_max"],
                "added_rise_nits_max": paired_summary["added_rise_nits_max"],
                "added_rise_area_over_1_nit_max": paired_summary[
                    "added_rise_area_over_1_nit_max"],
                "added_fall_nits_p95_max": paired_summary["added_fall_nits_p95_max"],
                "added_fall_nits_max": paired_summary["added_fall_nits_max"],
                "added_fall_area_over_1_nit_max": paired_summary[
                    "added_fall_area_over_1_nit_max"],
                "temporal_residual_nits_p95_max": paired_summary[
                    "temporal_residual_nits_p95_max"],
                "temporal_residual_nits_max": paired_summary["temporal_residual_nits_max"],
                "added_rise_peak_frame_id": peak_added.frame_id,
                "added_rise_peak_timestamp": peak_added.timestamp,
                "same_frame_brightening_nits_max": paired_summary[
                    "same_frame_brightening_nits_max"],
                "same_frame_dimming_nits_max": paired_summary["same_frame_dimming_nits_max"],
                "mean_dimming_over_1_nit_seconds": paired_summary[
                    "mean_dimming_over_1_nit_seconds"],
                "spatial_residual_log_gradient_max": paired_summary[
                    "spatial_residual_log_gradient_max"],
                "infeasible_frames": sum(bool(trace["infeasible"]) for trace in traces),
                "max_envelope_overshoot_log": max(
                    float(trace.get("max_envelope_overshoot_log", 0.0)) for trace in traces),
                "max_realized_added_log_rise_per_second": (
                    max(realized_rates) if realized_rates else None),
                "max_feasible_realized_added_log_rise_per_second": (
                    max(feasible_rates) if feasible_rates else None),
                "max_boundary_control_contrast": max(
                    float(trace.get("boundary_control_contrast_max", 0.0)) for trace in traces),
                "max_spatial_ceiling_credit_log": max(
                    float(trace.get("max_spatial_ceiling_credit_log", 0.0)) for trace in traces),
                "max_detail_protected_area_fraction": max(
                    float(trace.get("detail_protected_area_fraction", 0.0)) for trace in traces),
                "max_detail_weighted_protection_fraction": max(
                    float(trace.get("detail_weighted_protection_fraction", 0.0))
                    for trace in traces),
                "max_detail_protection_budget_overshoot": max(
                    float(trace.get("detail_protection_budget_overshoot", 0.0))
                    for trace in traces),
                "max_detail_strength": max(
                    float(trace.get("detail_maximum_strength", 0.0)) for trace in traces),
                "max_detail_strength_change": max(
                    float(trace.get("detail_maximum_strength_change", 0.0)) for trace in traces),
                "max_detail_released_dimming_nits": max(
                    float(trace.get("detail_max_released_dimming_nits", 0.0)) for trace in traces),
                "detail_armed_while_dimmed_pixels": sum(
                    int(trace.get("detail_armed_while_dimmed_pixels", 0)) for trace in traces),
                "max_detail_overflow_rejected_pixels": max(
                    int(trace.get("detail_overflow_rejected_pixels", 0)) for trace in traces),
                **_roi_metrics(case_id, prepared, outputs, profile, spec),
                **metric_summary,
            })
            progress += 1
            job.advance(progress, total, f"{case_id}:{candidate}")

    identity_rows = [row for row in rows if row["candidate"] == "identity"]
    if any(abs(row["quality_jod"] - 10) > 1e-5 for row in identity_rows):
        raise ValidationError("cvvdp_identity")
    audit = {
        "same_frame_brightening_nits_max": max(
            row["same_frame_brightening_nits_max"] for row in rows),
        "max_envelope_overshoot_log": max(row["max_envelope_overshoot_log"] for row in rows),
        "max_spatial_ceiling_credit_log": max(
            row["max_spatial_ceiling_credit_log"] for row in rows),
        "infeasible_frames": sum(row["infeasible_frames"] for row in rows),
        "detail_armed_while_dimmed_pixels": sum(
            row["detail_armed_while_dimmed_pixels"] for row in rows),
        "max_detail_protection_budget_overshoot": max(
            row["max_detail_protection_budget_overshoot"] for row in rows),
    }
    if audit["same_frame_brightening_nits_max"] > 1e-5:
        raise ValidationError("cvvdp_brightening")
    if audit["detail_armed_while_dimmed_pixels"]:
        raise ValidationError("cvvdp_detail_arming")
    if audit["max_detail_protection_budget_overshoot"] > 1e-12:
        raise ValidationError("cvvdp_detail_budget")
    report = {
        "schema_version": 2,
        "kind": "official_colorvideovdp_recovery_fidelity",
        "scope": "offline in-memory uint8 sRGB arrays; fidelity only, not flash safety",
        "sequence": asdict(spec),
        "display_profile": asdict(profile),
        "perceptual_config": asdict(perceptual_config),
        "design": asdict(design),
        "audit": audit,
        "metric": {
            "package": "cvvdp",
            "version": version,
            "device": device,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "pixels_per_degree": float(geometry.get_ppd()),
            "input": "in-memory FHWC uint8 display-encoded sRGB",
            "temporal_padding": design.temporal_padding,
        },
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": source_hashes("cvvdp_lab.py", "detail_protection.py", "recovery.py",
                                        "perceptual.py", "demo_filter.py", "fixtures.py",
                                        "color.py"),
        "limitations": [
            "ColorVideoVDP JOD measures fidelity to the unfiltered source, not comfort or flash safety.",
            "Static full-frame dimming may adapt out of JOD; explicit dimming duration remains separate.",
            "Synthetic fixtures are not independent game captures or participant data.",
            "The reduced sample resolution represents the full display field and limits high spatial frequencies.",
            "No heatmap, video, desktop capture, UI, game, compositor or emitted-light measurement was used.",
            "The optional environment lacks the separate ffmpeg distribution; only the working array API was used.",
        ],
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items()
                if key not in ("rho_band_cycles_per_degree", "difference_max_by_channel",
                               "difference_max_by_spatial_band")}
        for name, value in row["difference_max_by_channel"].items():
            flat[f"cvvdp_difference_{name}"] = value
        flat_rows.append(flat)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = [
        "# Official ColorVideoVDP recovery fidelity",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        "## Hard audit",
        "",
        f"- Same-frame brightening: {audit['same_frame_brightening_nits_max']:.6g} nit",
        f"- Envelope overshoot: {audit['max_envelope_overshoot_log']:.6g} log unit",
        f"- Spatial recovery credit: {audit['max_spatial_ceiling_credit_log']:.6g} log unit",
        f"- Infeasible frames: {audit['infeasible_frames']}",
        f"- Details armed while dimmed: {audit['detail_armed_while_dimmed_pixels']} pixels",
        f"- Weighted detail-budget overshoot: {audit['max_detail_protection_budget_overshoot']:.6g}",
        "",
        "| Case | Candidate | JOD | Added rise P95 (nit) | Added rise max (nit) | Added fall max (nit) | Dimming duration (s) | HUD dim (nit) | Subtitle contrast | Dense uniformity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        hud = "n/a" if row["hud_dimming_nits_max"] is None else f"{row['hud_dimming_nits_max']:.3f}"
        subtitle = ("n/a" if row["subtitle_contrast_retention_min"] is None else
                    f"{row['subtitle_contrast_retention_min']:.3f}")
        dense = ("n/a" if row["dense_detail_luminance_uniformity_min"] is None else
                 f"{row['dense_detail_luminance_uniformity_min']:.3f}")
        lines.append(f"| {row['case']} | {row['candidate']} | {row['quality_jod']:.4f} | "
                     f"{row['added_rise_nits_p95_max']:.3f} | "
                     f"{row['added_rise_nits_max']:.3f} | {row['added_fall_nits_max']:.3f} | "
                     f"{row['mean_dimming_over_1_nit_seconds']:.3f} | {hud} | {subtitle} | "
                     f"{dense} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
