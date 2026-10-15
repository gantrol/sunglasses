"""Streaming native-resolution text/detail stress experiments.

The lab renders real font outlines into synthetic linear-light frames, processes
one full-resolution frame at a time, and retains only a fixed text ROI for
perceptual analysis.  It never writes or displays source/output imagery.
"""

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import demo_filter as screen_reference
from .color import linear_to_srgb
from .cvvdp_lab import _load_backend, summarize_cvvdp_stats
from .detail_protection import StableDetailConfig, StableDetailProtector
from .flash_screening import (
    FlashScreeningAnalyzer,
    FlashScreeningConfig,
    compare_flash_screening,
    summarize_flash_screening,
)
from .i18n import ValidationError
from .jobs import JobContext
from .perceptual import DisplayProfile, PairedPerceptualAnalyzer, summarize_paired
from .provenance import source_hashes
from .recovery import LogRiseLimitConfig, PixelLogRiseLimiter


NATIVE_DETAIL_CASES = {
    "prearmed_text": "Static subtitle present before the bright event.",
    "late_text": "Subtitle first appears while the base output is already dimmed.",
    "moving_text": "Pre-armed subtitle starts moving during the bright event.",
    "threshold_noise": "Pre-armed subtitle alternates by one encoded code value.",
    "dense_warning": "Several stable outlined warnings exceed the detail-area budget.",
}
CANDIDATES = ("identity", "current_screen_8bit", "pixel_cone_4deg_rate_0p5_q8_g0p02",
              "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02",
              "pixel_cone_detail_motion_4deg_rate_0p5_q8_g0p02")


@dataclass(frozen=True)
class NativeDetailSpec:
    width: int = 1920
    height: int = 1080
    fps: int = 60
    duration_seconds: float = 1.0
    event_start_seconds: float = .50
    event_end_seconds: float = .75

    def __post_init__(self) -> None:
        if (type(self.width) is not int or type(self.height) is not int or
                not 160 <= self.width <= 3840 or not 90 <= self.height <= 2160):
            raise ValidationError("native_detail_size")
        if type(self.fps) is not int or not 15 <= self.fps <= 240:
            raise ValidationError("native_detail_fps")
        numeric = (self.duration_seconds, self.event_start_seconds, self.event_end_seconds)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise ValidationError("native_detail_time")
        if not (.5 <= self.duration_seconds <= 5 and .4 <= self.event_start_seconds <
                self.event_end_seconds < self.duration_seconds):
            raise ValidationError("native_detail_time")

    @property
    def frame_count(self) -> int:
        return round(self.fps * self.duration_seconds)


@dataclass(frozen=True)
class NativeDetailDesign:
    font_path: str
    cases: tuple[str, ...] = tuple(NATIVE_DETAIL_CASES)
    expected_cvvdp_version: str = "0.5.7"
    temporal_padding: str = "replicate"

    def __post_init__(self) -> None:
        if not isinstance(self.font_path, str) or not self.font_path.strip():
            raise ValidationError("native_detail_font")
        if (not isinstance(self.cases, tuple) or not self.cases or
                len(set(self.cases)) != len(self.cases) or
                any(case not in NATIVE_DETAIL_CASES for case in self.cases)):
            raise ValidationError("native_detail_cases")
        if self.temporal_padding not in ("replicate", "symmetric", "valid"):
            raise ValidationError("cvvdp_padding")


def _font_size(spec: NativeDetailSpec, dense: bool = False) -> int:
    reference = 72 if dense else 34
    return max(8, round(reference * spec.height / 1080))


def _render_line(spec: NativeDetailSpec, font, text: str, center_y: int,
                 stroke_width: int) -> tuple[np.ndarray, np.ndarray]:
    fill = Image.new("L", (spec.width, spec.height))
    outer = Image.new("L", (spec.width, spec.height))
    fill_draw, outer_draw = ImageDraw.Draw(fill), ImageDraw.Draw(outer)
    bounds = fill_draw.textbbox((0, 0), text, font=font, stroke_width=0)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    origin = ((spec.width - width) // 2 - bounds[0], center_y - height // 2 - bounds[1])
    fill_draw.text(origin, text, font=font, fill=255)
    outer_draw.text(origin, text, font=font, fill=255, stroke_width=stroke_width,
                    stroke_fill=255)
    fill_values = np.asarray(fill, dtype=np.uint8)
    outer_values = np.asarray(outer, dtype=np.uint8)
    outline = np.maximum(outer_values.astype(np.int16) - fill_values, 0).astype(np.uint8)
    return fill_values, outline


def _render_layers(spec: NativeDetailSpec, font_path: Path, case_id: str) -> dict:
    dense = case_id == "dense_warning"
    font = ImageFont.truetype(str(font_path), _font_size(spec, dense))
    stroke = max(1, round((3 if dense else 2) * spec.height / 1080))
    if dense:
        texts = ("WARNING — HOLD POSITION", "OBJECTIVE AREA COMPROMISED",
                 "WARNING — HOLD POSITION", "OBJECTIVE AREA COMPROMISED",
                 "WARNING — HOLD POSITION", "OBJECTIVE AREA COMPROMISED")
        centers = np.linspace(.43, .92, len(texts)) * spec.height
    else:
        texts = ("OBJECTIVE UPDATED — HOLD POSITION",)
        centers = (.86 * spec.height,)
    fill = np.zeros((spec.height, spec.width), dtype=np.uint8)
    outline = np.zeros_like(fill)
    for text, center in zip(texts, centers):
        line_fill, line_outline = _render_line(spec, font, text, round(center), stroke)
        fill = np.maximum(fill, line_fill)
        outline = np.maximum(outline, line_outline)
    maximum_shift = round(.06 * spec.width) if case_id == "moving_text" else 0
    union = fill | outline
    ys, xs = np.nonzero(union)
    if not xs.size:
        raise ValidationError("native_detail_font")
    pad_x = max(8, round(.75 * spec.width / 47.0))
    pad_y = max(8, round(.75 * spec.height / 28.0))
    left = max(0, int(xs.min()) - pad_x)
    right = min(spec.width, int(xs.max()) + maximum_shift + pad_x + 1)
    top = max(0, int(ys.min()) - pad_y)
    bottom = min(spec.height, int(ys.max()) + pad_y + 1)
    return {"fill": fill, "outline": outline, "crop": (left, top, right, bottom),
            "maximum_shift": maximum_shift}


def _shift(mask: np.ndarray, amount: int) -> np.ndarray:
    if not amount:
        return mask
    result = np.zeros_like(mask)
    if amount < mask.shape[1]:
        result[:, amount:] = mask[:, :-amount]
    return result


def _active_layers(case_id: str, layers: dict, frame_id: int,
                   spec: NativeDetailSpec) -> tuple[np.ndarray, np.ndarray, int]:
    event_start = round(spec.event_start_seconds * spec.fps)
    if case_id == "late_text" and frame_id < event_start:
        empty = np.zeros_like(layers["fill"])
        return empty, empty, 245
    shift = 0
    if case_id == "moving_text" and frame_id >= event_start:
        progress = (frame_id - event_start) / max(1, spec.frame_count - 1 - event_start)
        shift = round(progress * layers["maximum_shift"])
    code = 244 + frame_id % 2 if case_id == "threshold_noise" else 245
    return _shift(layers["fill"], shift), _shift(layers["outline"], shift), code


def _base_frame(spec: NativeDetailSpec) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:spec.height, :spec.width]
    x = xx.astype(np.float32) / spec.width
    y = yy.astype(np.float32) / spec.height
    texture = .010 * (np.sin(x * math.tau * 13) + np.cos(y * math.tau * 7))
    values = np.clip(.035 + .075 * x + texture, .015, .16).astype(np.float32)
    base = np.repeat(values[..., None], 3, axis=-1)
    event = (x - .55) ** 2 + (y - .43) ** 2 < .31**2
    return base, event


def _compose(base: np.ndarray, event: np.ndarray, event_active: bool,
             fill: np.ndarray, outline: np.ndarray, fill_code: int) -> np.ndarray:
    frame = base.copy()
    if event_active:
        frame[event] = (1.0, .92, .72)
    for mask, color in ((outline, screen_reference.LINEAR[22]),
                        (fill, screen_reference.LINEAR[fill_code])):
        alpha = mask.astype(np.float32) / 255
        frame += alpha[..., None] * (color - frame)
    return frame


def _crop_profile(profile: DisplayProfile, crop: tuple[int, int, int, int]) -> DisplayProfile:
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    diagonal = profile.diagonal_inches * math.hypot(width, height) / math.hypot(*profile.resolution)
    return DisplayProfile(profile.peak_luminance_nits, profile.black_luminance_nits,
                          diagonal, (width, height), profile.viewing_distance_m)


def _encoded(frame: np.ndarray) -> np.ndarray:
    return np.rint(linear_to_srgb(frame) * 255).astype(np.uint8)


def _empty_roi() -> dict:
    return {"core_dimming_nits_max": 0.0, "edge_dimming_nits_max": 0.0,
            "glyph_dimming_area_over_1_nit_max": 0.0,
            "edge_dimming_area_over_1_nit_max": 0.0,
            "contrast_retention_min": 1.0, "attenuation_spread_max": 0.0,
            "attenuation_range_max": 0.0, "core_edge_attenuation_gap_max": 0.0,
            "evaluated_frames": 0}


def _update_roi(summary: dict, profile: DisplayProfile, source: np.ndarray, output: np.ndarray,
                fill: np.ndarray, outline: np.ndarray) -> None:
    core = fill >= 204
    glyph = (fill > 12) | (outline > 12)
    edge = glyph & ~core
    if not np.any(core):
        return
    source_y = np.sum(source * screen_reference.LUMA, axis=-1)
    output_y = np.sum(output * screen_reference.LUMA, axis=-1)
    source_nits, output_nits = profile.to_nits(source_y), profile.to_nits(output_y)
    dimming = np.maximum(source_nits - output_nits, 0)
    summary["core_dimming_nits_max"] = max(
        summary["core_dimming_nits_max"], float(dimming[core].max(initial=0)))
    if np.any(edge):
        summary["edge_dimming_nits_max"] = max(
            summary["edge_dimming_nits_max"], float(dimming[edge].max(initial=0)))
        summary["edge_dimming_area_over_1_nit_max"] = max(
            summary["edge_dimming_area_over_1_nit_max"], float((dimming[edge] > 1).mean()))
    summary["glyph_dimming_area_over_1_nit_max"] = max(
        summary["glyph_dimming_area_over_1_nit_max"], float((dimming[glyph] > 1).mean()))
    surround = ~glyph
    source_contrast = ((float(source_nits[core].mean()) + .5) /
                       (float(source_nits[surround].mean()) + .5))
    output_contrast = ((float(output_nits[core].mean()) + .5) /
                       (float(output_nits[surround].mean()) + .5))
    summary["contrast_retention_min"] = min(
        summary["contrast_retention_min"], output_contrast / source_contrast)
    attenuation = (np.log(source_nits[glyph] + .5) - np.log(output_nits[glyph] + .5))
    if attenuation.size:
        low, high = np.quantile(attenuation, (.1, .9))
        summary["attenuation_spread_max"] = max(
            summary["attenuation_spread_max"], float(high - low))
        summary["attenuation_range_max"] = max(
            summary["attenuation_range_max"], float(np.ptp(attenuation)))
    if np.any(edge):
        core_attenuation = (np.log(source_nits[core] + .5) - np.log(output_nits[core] + .5))
        edge_attenuation = (np.log(source_nits[edge] + .5) - np.log(output_nits[edge] + .5))
        summary["core_edge_attenuation_gap_max"] = max(
            summary["core_edge_attenuation_gap_max"],
            abs(float(core_attenuation.mean()) - float(edge_attenuation.mean())))
    summary["evaluated_frames"] += 1


def _make_metric(pycvvdp, torch, profile: DisplayProfile, design: NativeDetailDesign,
                 device: str):
    contrast = profile.peak_luminance_nits / max(profile.black_luminance_nits, .005)
    photometry = pycvvdp.vvdp_display_photo_eotf(
        Y_peak=profile.peak_luminance_nits, contrast=contrast, source_colorspace="sRGB")
    geometry = pycvvdp.vvdp_display_geometry(
        profile.resolution, diagonal_size_inches=profile.diagonal_inches,
        distance_m=profile.viewing_distance_m)
    metric = pycvvdp.cvvdp(display_photometry=photometry, display_geometry=geometry,
                           quiet=True, device=torch.device(device),
                           temp_padding=design.temporal_padding)
    return metric, geometry


def run_native_detail_suite(output: Path, spec: NativeDetailSpec, profile: DisplayProfile,
                            design: NativeDetailDesign, *, with_cvvdp: bool = True,
                            device: str = "cuda", job: JobContext = JobContext()) -> dict:
    """Run streaming full-resolution processing and optional native-pixel ROI CVVDP."""
    job.check()
    if profile.resolution != (spec.width, spec.height):
        raise ValidationError("native_detail_profile")
    font_path = Path(design.font_path).resolve()
    if not font_path.is_file():
        raise ValidationError("native_detail_font")
    if output.exists():
        raise FileExistsError(output)
    pycvvdp = torch = version = None
    if with_cvvdp:
        pycvvdp, torch, version = _load_backend(design.expected_cvvdp_version, device)
    output.mkdir(parents=True)
    rows, case_records = [], []
    motion_detail_config = StableDetailConfig(motion_propagation=True)
    screening_config = FlashScreeningConfig()
    trace_directory = output / "traces"
    trace_directory.mkdir()
    total = len(design.cases) * spec.frame_count
    completed = 0
    for case_id in design.cases:
        started = time.perf_counter()
        layers = _render_layers(spec, font_path, case_id)
        crop = layers["crop"]
        crop_profile = _crop_profile(profile, crop)
        controller = screen_reference.Controller()
        limiter = PixelLogRiseLimiter(
            profile, config=LogRiseLimitConfig(max_added_log_rise_per_second=.5,
                                               cell_degrees=4, output_bits=8,
                                               max_amount_gradient_per_degree=.02))
        protector = StableDetailProtector(profile)
        motion_protector = StableDetailProtector(profile, motion_detail_config)
        analyzers = {name: PairedPerceptualAnalyzer(crop_profile) for name in CANDIDATES}
        measurements = {name: [] for name in CANDIDATES}
        screeners = {name: FlashScreeningAnalyzer(profile, screening_config)
                     for name in CANDIDATES}
        screenings = {name: [] for name in CANDIDATES}
        roi = {name: _empty_roi() for name in CANDIDATES}
        videos = {name: [] for name in CANDIDATES}
        source_video = []
        traces = []
        frame_rows = []
        full_frame_brightening = 0.0
        base, event = _base_frame(spec)
        for frame_id in range(spec.frame_count):
            job.check()
            timestamp = frame_id / spec.fps
            fill, outline, fill_code = _active_layers(case_id, layers, frame_id, spec)
            active = spec.event_start_seconds <= timestamp < spec.event_end_seconds
            frame = _compose(base, event, active, fill, outline, fill_code)
            encoded = _encoded(frame)
            linear, y, load, mean = screen_reference.stats(encoded)
            amount = controller.step(load, mean, timestamp)
            current = screen_reference.LINEAR[
                screen_reference.apply(encoded, linear, y, amount, controller.broad, 1.0)]
            pixel, pixel_trace = limiter.process(
                encoded, linear, y, amount, controller.broad, timestamp, frame_id)
            coherent, detail_trace = protector.process(
                encoded, linear, pixel, timestamp, frame_id)
            motion, motion_detail_trace = motion_protector.process(
                encoded, linear, pixel, timestamp, frame_id)
            candidates = {"identity": linear, "current_screen_8bit": current,
                          "pixel_cone_4deg_rate_0p5_q8_g0p02": pixel,
                          "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02": coherent,
                          "pixel_cone_detail_motion_4deg_rate_0p5_q8_g0p02": motion}
            left, top, right, bottom = crop
            source_crop = linear[top:bottom, left:right]
            fill_crop, outline_crop = fill[top:bottom, left:right], outline[top:bottom, left:right]
            if with_cvvdp:
                source_video.append(encoded[top:bottom, left:right].copy())
            for name, candidate in candidates.items():
                candidate_crop = candidate[top:bottom, left:right]
                measurement = analyzers[name].process(
                    source_crop, candidate_crop, timestamp, frame_id)
                measurements[name].append(measurement)
                _update_roi(roi[name], crop_profile, source_crop, candidate_crop,
                            fill_crop, outline_crop)
                if with_cvvdp:
                    videos[name].append(_encoded(candidate_crop))
                candidate_y = np.sum(candidate * screen_reference.LUMA, axis=-1)
                full_frame_brightening = max(full_frame_brightening, float(
                    np.maximum(profile.to_nits(candidate_y) - profile.to_nits(y), 0).max()))
                if full_frame_brightening > 1e-5:
                    raise ValidationError("native_detail_brightening")
                screening = screeners[name].process(candidate, timestamp, frame_id)
                screenings[name].append(screening)
                pair_row = asdict(measurement)
                pair_row.pop("source")
                pair_row.pop("output")
                frame_rows.append({
                    "case": case_id, "candidate": name, **pair_row,
                    **{f"source_screening_{key}": value for key, value in
                       asdict(screenings["identity"][-1]).items()},
                    **{f"output_screening_{key}": value for key, value in
                       asdict(screening).items()},
                })
            traces.append({**pixel_trace,
                           **{f"detail_{key}": value for key, value in detail_trace.items()},
                           **{f"motion_detail_{key}": value
                              for key, value in motion_detail_trace.items()}})
            completed += 1
            job.advance(completed, total, f"{case_id}:{frame_id}")

        # Numeric evidence only; synchronous writes run in this offline worker,
        # outside any UI thread. No frame imagery is persisted.
        with (trace_directory / f"{case_id}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
        with (trace_directory / f"{case_id}-controls.csv").open(
                "x", newline="", encoding="utf-8") as stream:
            control_rows = [{"frame_id": index, "timestamp": index / spec.fps, **trace}
                            for index, trace in enumerate(traces)]
            fields = list(dict.fromkeys(key for trace in control_rows for key in trace))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(control_rows)

        metric = None
        if with_cvvdp:
            source_values = np.stack(source_video)
            metric, _ = _make_metric(pycvvdp, torch, crop_profile, design, device)
        for name in CANDIDATES:
            job.check()
            paired = summarize_paired(measurements[name])
            metric_summary = {}
            quality = None
            if with_cvvdp:
                job.advance(completed, total, f"{case_id}:cvvdp:{name}")
                candidate_values = np.stack(videos[name])
                quality, stats = metric.predict(
                    candidate_values, source_values, dim_order="FHWC",
                    frames_per_second=spec.fps)
                metric_summary = summarize_cvvdp_stats(stats)
            rows.append({"case": case_id, "candidate": name,
                         "quality_jod": None if quality is None else float(quality),
                         **paired, **roi[name], **metric_summary,
                         **{f"screening_{key}": value for key, value in
                            summarize_flash_screening(screenings[name]).items()},
                         **{f"source_aware_screening_{key}": value for key, value in
                            compare_flash_screening(screenings["identity"], screenings[name]).items()}})
        case_records.append({
            "case": case_id,
            "crop": crop,
            "crop_profile": asdict(crop_profile),
            "pixels_per_degree": crop_profile.pixels_per_degree,
            "runtime_seconds": time.perf_counter() - started,
            "full_frame_brightening_nits_max": full_frame_brightening,
            "maximum_control_amount": max(
                float(trace.get("selected_amount", 0.0)) for trace in traces),
            "infeasible_frames": sum(bool(trace["infeasible"]) for trace in traces),
            "max_envelope_overshoot_log": max(
                float(trace.get(prefix + "max_envelope_overshoot_log", 0.0))
                for trace in traces for prefix in ("", "detail_", "motion_detail_")),
            "max_spatial_ceiling_credit_log": max(
                float(trace.get("max_spatial_ceiling_credit_log", 0.0)) for trace in traces),
            "detail_armed_while_dimmed_pixels": sum(
                int(trace.get(prefix + "armed_while_dimmed_pixels", 0))
                for trace in traces for prefix in ("detail_", "motion_detail_")),
            "max_detail_protection_budget_overshoot": max(
                float(trace.get(prefix + "protection_budget_overshoot", 0.0))
                for trace in traces for prefix in ("detail_", "motion_detail_")),
            "max_detail_protected_area_fraction": max(
                float(trace.get("detail_protected_area_fraction", 0.0)) for trace in traces),
            "max_motion_detail_protected_area_fraction": max(
                float(trace.get("motion_detail_protected_area_fraction", 0.0))
                for trace in traces),
            "max_detail_eligible_area_fraction": max(
                float(trace.get("detail_eligible_area_fraction", 0.0)) for trace in traces),
            "motion_propagated_frames": sum(
                int(trace.get("motion_detail_motion_propagated_pixels", 0)) > 0
                for trace in traces),
            "max_motion_detail_realized_added_log_rise_per_second": max(
                float(trace.get(
                    "motion_detail_max_realized_added_log_rise_per_second", 0.0))
                for trace in traces),
            "max_detail_realized_added_log_rise_per_second": max(
                float(trace.get("detail_max_realized_added_log_rise_per_second", 0.0))
                for trace in traces),
            "max_motion_detail_envelope_added_log_rise_per_second": max(
                float(trace.get("motion_detail_max_envelope_added_log_rise_per_second", 0.0))
                for trace in traces),
            "max_detail_envelope_added_log_rise_per_second": max(
                float(trace.get("detail_max_envelope_added_log_rise_per_second", 0.0))
                for trace in traces),
        })

    identity = [row for row in rows if row["candidate"] == "identity"]
    if with_cvvdp and any(abs(row["quality_jod"] - 10) > 1e-5 for row in identity):
        raise ValidationError("cvvdp_identity")
    audit = {
        "same_frame_brightening_nits_max": max(
            case["full_frame_brightening_nits_max"] for case in case_records),
        "infeasible_frames": sum(case["infeasible_frames"] for case in case_records),
        "max_envelope_overshoot_log": max(
            case["max_envelope_overshoot_log"] for case in case_records),
        "max_spatial_ceiling_credit_log": max(
            case["max_spatial_ceiling_credit_log"] for case in case_records),
        "detail_armed_while_dimmed_pixels": sum(
            case["detail_armed_while_dimmed_pixels"] for case in case_records),
        "max_detail_protection_budget_overshoot": max(
            case["max_detail_protection_budget_overshoot"] for case in case_records),
        "max_motion_detail_realized_added_log_rise_per_second": max(
            case["max_motion_detail_realized_added_log_rise_per_second"]
            for case in case_records),
        "max_detail_realized_added_log_rise_per_second": max(
            case["max_detail_realized_added_log_rise_per_second"] for case in case_records),
        "max_motion_detail_envelope_added_log_rise_per_second": max(
            case["max_motion_detail_envelope_added_log_rise_per_second"] for case in case_records),
        "max_detail_envelope_added_log_rise_per_second": max(
            case["max_detail_envelope_added_log_rise_per_second"] for case in case_records),
    }
    if (audit["same_frame_brightening_nits_max"] > 1e-5 or
            audit["infeasible_frames"] or
            audit["max_envelope_overshoot_log"] > 1e-7 or
            audit["max_spatial_ceiling_credit_log"] > 1e-7 or
            audit["detail_armed_while_dimmed_pixels"] or
            audit["max_detail_protection_budget_overshoot"] > 1e-12 or
            audit["max_detail_envelope_added_log_rise_per_second"] >
            protector.config.max_added_log_rise_per_second + 1e-7 or
            audit["max_motion_detail_envelope_added_log_rise_per_second"] >
            motion_detail_config.max_added_log_rise_per_second + 1e-7):
        raise ValidationError("native_detail_audit")
    report = {
        "schema_version": 2,
        "kind": "native_resolution_font_detail_stress",
        "scope": "streamed full-resolution synthetic frames; native-pixel fixed ROI analysis",
        "spec": asdict(spec),
        "display_profile": asdict(profile),
        "design": {**asdict(design), "font_path": str(font_path)},
        "font": {"path": str(font_path),
                 "sha256": hashlib.sha256(font_path.read_bytes()).hexdigest()},
        "metric": None if not with_cvvdp else {
            "package": "cvvdp", "version": version, "device": device,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "input": "native-pixel ROI, in-memory FHWC uint8 display-encoded sRGB",
            "temporal_padding": design.temporal_padding,
        },
        "audit": audit,
        "flash_screening": {
            "config": asdict(screening_config),
            "scope": "full native-resolution frame; no ROI cropping or downsampling",
            "comparison": "exact-frame event mismatch, with general pair completion direction",
        },
        "detail_configs": {"static": asdict(protector.config), "motion": asdict(motion_detail_config)},
        "cases": case_records,
        "rows": rows,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": source_hashes(
            "native_detail_lab.py", "cvvdp_lab.py", "detail_protection.py", "recovery.py",
            "perceptual.py", "flash_screening.py", "demo_filter.py", "color.py"),
        "limitations": [
            "Synthetic text and events are not independent game captures or participant data.",
            "ROI ColorVideoVDP preserves native pixel pitch but excludes most full-screen context.",
            "Paired nits/contrast metrics describe the text ROI; flash screening and brightening audit cover the full frame.",
            "Screening is an engineering proxy, not standards conformance or a medical score. Exact-frame mismatches do not establish spatial event identity or causation.",
            "Directional flash counts describe the transition completing a pair, not every added pixel rise/fall; inspect the per-frame numeric traces too.",
            "The strict detail rise envelope covers currently/previously protected screen pixels until they rejoin the base output. Untouched pixels retain the base limiter's one-code quantization credit; full-frame realized rates are reported separately and may exceed its nominal rate.",
            "Font masks are rendered by the recorded local font file and Pillow/FreeType environment.",
            "CPU runtime is an offline research cost, not a real-time implementation measurement.",
            "Motion propagation searches a bounded global integer translation of already-protected detail; it is not optical flow or semantic tracking.",
            "No frame image, video, UI, game, desktop capture, compositor or emitted light was used.",
        ],
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat_rows.append({key: value for key, value in row.items()
                          if key not in ("rho_band_cycles_per_degree",
                                         "difference_max_by_channel",
                                         "difference_max_by_spatial_band")})
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    lines = ["# Native-resolution font/detail stress", "",
             *[f"- {item}" for item in report["limitations"]], "",
             "| Case | Candidate | JOD | Added rise max (nit) | Added fall max (nit) | Core dim (nit) | Edge dim (nit) | Edge area >1 nit | Contrast retained | Attenuation P90-P10 | Attenuation range |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        jod = "n/a" if row["quality_jod"] is None else f"{row['quality_jod']:.4f}"
        lines.append(
            f"| {row['case']} | {row['candidate']} | {jod} | "
            f"{row['added_rise_nits_max']:.3f} | {row['added_fall_nits_max']:.3f} | "
            f"{row['core_dimming_nits_max']:.3f} | {row['edge_dimming_nits_max']:.3f} | "
            f"{row['edge_dimming_area_over_1_nit_max']:.3f} | "
            f"{row['contrast_retention_min']:.3f} | {row['attenuation_spread_max']:.3f} | "
            f"{row['attenuation_range_max']:.3f} |")
    lines += ["", "## Full-frame flash screening", "",
              "| Case | Candidate | Flashes | Max flashes/s | Output-only event frames | Output-only rising pairs | Output-only falling pairs |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['candidate']} | "
                     f"{row['screening_combined_flash_events']} | "
                     f"{row['screening_max_combined_flashes_1s']} | "
                     f"{row['source_aware_screening_added_combined_event_frames']} | "
                     f"{row['source_aware_screening_added_general_rise_event_frames']} | "
                     f"{row['source_aware_screening_added_general_fall_event_frames']} |")
    for case_id in design.cases:
        lines += ["", f"[{case_id} frame trace](traces/{case_id}.csv) · "
                  f"[controls](traces/{case_id}-controls.csv)"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
