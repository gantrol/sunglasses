"""Local SDR video evaluation using the same filter and comparison viewer.

This offline CLI deliberately uses synchronous, bounded pipes and disk writes.
It never starts a browser, records a game, downloads footage or publishes media.
"""

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from .automatic import AutoConfig, AutoFilter, STRONG, HIGHLIGHT, LOAD_START, LOAD_FULL
from .color import linear_to_srgb, luminance
from .comparison import _Stats, _json, png_bytes
from .filter import compress_highlights
from .i18n import LANGUAGES, messages
from .provenance import source_hashes

METHODS = ("original", "automatic", "legacy", "fixed_strong")


@dataclass(frozen=True)
class VideoClip:
    id: str
    input: str
    start: float
    duration: float
    width: int
    height: int
    source_start: float
    source_url: str
    page_url: str
    assume_bt709: bool = False
    warmup: float = 2.0


def _run(command):
    return subprocess.run(command, capture_output=True, check=True, timeout=60,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect(clip, ffprobe):
    path = Path(clip.input)
    if not path.is_file() or "trash" in str(path.resolve()).lower():
        raise ValueError("video_local_input")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", clip.id):
        raise ValueError("video_case_id")
    if not all(math.isfinite(v) for v in (clip.start, clip.duration, clip.source_start, clip.warmup)):
        raise ValueError("video_time")
    if not 0 < clip.duration <= 15 or not 0 <= clip.warmup <= 5 or clip.start < clip.warmup or clip.source_start < 0:
        raise ValueError("video_time")
    if any(type(v) is not int or not 16 <= v <= 1280 or v % 2 for v in (clip.width, clip.height)):
        raise ValueError("video_dimensions")
    if clip.width * clip.height > 921600:
        raise ValueError("video_pixel_budget")
    if type(clip.assume_bt709) is not bool:
        raise ValueError("video_color_assumption")
    for url in (clip.source_url, clip.page_url):
        if not url.startswith("https://"):
            raise ValueError("video_source_url")
    probe = json.loads(_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                            "-show_frames", "-show_entries",
                            "stream:format=duration:frame=best_effort_timestamp_time", "-of", "json", str(path)]).stdout)
    stream = probe["streams"][0]
    if stream.get("pix_fmt") != "yuv420p" or stream.get("field_order") not in (None, "unknown", "progressive"):
        raise ValueError("video_sdr_progressive_yuv420p_only")
    assumed = []
    for key, expected in (("color_space", "bt709"), ("color_transfer", "bt709"),
                          ("color_primaries", "bt709"), ("color_range", "tv")):
        value = stream.get(key)
        if value in (None, "unknown", "unspecified") and clip.assume_bt709:
            assumed.append(key)
        elif value != expected:
            raise ValueError(f"video_color:{key}={value}")
    rate = Fraction(stream["r_frame_rate"])
    if not 1 <= rate <= 120:
        raise ValueError("video_fps")
    fps = float(rate)
    timestamps = np.array([float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]])
    if len(timestamps) < 2 or np.max(np.abs(timestamps - timestamps[0] - np.arange(len(timestamps)) / fps)) > .0011:
        raise ValueError("video_cfr_required")
    if abs(timestamps[0]) > .0011 or clip.start + clip.duration > len(timestamps) / fps + .001:
        raise ValueError("video_interval")
    for time in (clip.start, clip.duration, clip.warmup):
        if abs(time * fps - round(time * fps)) > 1e-5:
            raise ValueError("video_frame_aligned_times_required")
    return dict(stream=stream, fps=fps, rate=str(rate), assumptions=assumed,
                timestamps=timestamps.tolist(), source_sha256=_hash(path))


def decode(clip, info, ffmpeg, log):
    count = round((clip.duration + clip.warmup) * info["fps"])
    # Explicit BT.709 inverse transfer into relative linear RGB. This is not a
    # measurement of display luminance or an assumption that BT.709 == sRGB.
    conversion = (f"zscale=w={clip.width}:h={clip.height}:filter=lanczos:"
                  "pin=709:tin=709:min=709:rin=limited:p=709:t=linear:m=gbr:r=full,format=gbrpf32le")
    command = [ffmpeg, "-v", "error", "-nostdin", "-ss", str(clip.start - clip.warmup), "-i", clip.input,
               "-map", "0:v:0", "-frames:v", str(count), "-an", "-vf", conversion,
               "-fps_mode", "passthrough", "-pix_fmt", "gbrpf32le", "-f", "rawvideo", "pipe:1"]
    with log.open("xb") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            size = clip.width * clip.height * 3 * 4
            for index in range(count):
                raw = process.stdout.read(size)
                if len(raw) != size:
                    raise RuntimeError(f"Incomplete decoded frame {index}; inspect {log}")
                planes = np.frombuffer(raw, "<f4").reshape(3, clip.height, clip.width)
                rgb = np.ascontiguousarray(planes[[2, 0, 1]].transpose(1, 2, 0))
                if not np.isfinite(rgb).all():
                    raise ValueError("video_nonfinite")
                outside = int(np.count_nonzero((rgb < 0) | (rgb > 1)))
                yield index, np.clip(rgb, 0, 1), outside
            process.stdout.close()
            if process.wait(timeout=20):
                raise RuntimeError(f"Decoder failed; inspect {log}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def _encode_command(ffmpeg, video, clip, info):
    return [ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{clip.width * 4}x{clip.height}", "-framerate", info["rate"], "-i", "pipe:0", "-an",
            "-vf", ("scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
                    "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video)]


def _evaluate(root, clip, info, ffmpeg):
    folder = root / clip.id
    folder.mkdir()
    count = round(clip.duration * info["fps"])
    warmup = round(clip.warmup * info["fps"])
    controller = AutoFilter()
    legacy = AutoFilter(AutoConfig(policy="legacy"))
    stats = {method: _Stats() for method in METHODS}
    diagnostics = {method: dict(bright_fraction=[], excess_mean=[], max_positive_delta=0.0,
                               max_negative_delta=0.0, mean_abs_pixel_delta=[]) for method in METHODS}
    video = folder / "comparison.mp4"
    pool, traces, legacy_traces = [], [], []
    off_frames = unchanged_frames = 0
    max_nonhighlight_error = 0.0
    bytes_written = outside_count = 0
    first_means, last_means = [], []
    with (folder / "encoder.log").open("xb") as errors:
        process = subprocess.Popen(_encode_command(ffmpeg, video, clip, info), stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=errors,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for index, rgb, outside in decode(clip, info, ffmpeg, folder / "decode.log"):
                auto, trace = controller.process(rgb, index / info["fps"], index)
                old, old_trace = legacy.process(rgb, index / info["fps"], index)
                if index < warmup:
                    continue
                outside_count += outside
                trace.update(timestamp=(index - warmup) / info["fps"], source_timestamp=clip.source_start + (index - warmup) / info["fps"])
                outputs = (rgb, auto, old, compress_highlights(rgb, STRONG))
                source_y = luminance(rgb)
                off_frames += int(trace["amount"] == 0)
                unchanged_frames += int(np.array_equal(auto, rgb))
                ordinary = source_y <= HIGHLIGHT.knee
                if ordinary.any():
                    max_nonhighlight_error = max(max_nonhighlight_error, float(np.abs(auto[ordinary] - rgb[ordinary]).max()))
                for method, output in zip(METHODS, outputs):
                    y = luminance(output)
                    d = diagnostics[method]
                    d["bright_fraction"].append(float((y > .65).mean()))
                    d["excess_mean"].append(float(np.maximum(y - .65, 0).mean()))
                    if stats[method].previous is not None:
                        delta = y - stats[method].previous
                        d["max_positive_delta"] = max(d["max_positive_delta"], float(delta.max()))
                        d["max_negative_delta"] = max(d["max_negative_delta"], float(-delta.min()))
                        d["mean_abs_pixel_delta"].append(float(np.abs(delta).mean()))
                    stats[method].add(output, source_y)
                packed = np.rint(linear_to_srgb(np.concatenate(outputs, axis=1)) * 255).astype(np.uint8)
                png = png_bytes(packed)
                bytes_written += len(png)
                if bytes_written > 1024**3:
                    raise ValueError("video_case_output_budget")
                relative = Path(clip.id) / f"{index-warmup:04d}.png"
                with (root / relative).open("xb") as file:
                    file.write(png)
                pool.append(dict(file=relative.as_posix(), sha256=hashlib.sha256(png).hexdigest()))
                traces.append(trace)
                legacy_traces.append(old_trace)
                process.stdin.write(packed.tobytes())
                means = packed.reshape(clip.height, 4, clip.width, 3).mean(axis=(0, 2, 3)).tolist()
                if not first_means:
                    first_means = means
                last_means = means
            process.stdin.close()
            if process.wait(timeout=120):
                raise RuntimeError(f"Encoder failed: {folder}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
    if len(pool) != count:
        raise ValueError("video_count")
    metrics = {method: value.result() for method, value in stats.items()}
    for method in METHODS:
        metrics[method].update(diagnostics[method])
        metrics[method]["excess_area_seconds"] = sum(diagnostics[method]["excess_mean"]) / info["fps"]
    sequence = dict(width=clip.width, height=clip.height, fps=info["fps"], duration=count / info["fps"])
    source = dict(page_url=clip.page_url, media_url=clip.source_url, start_seconds=clip.source_start,
                  local_start_seconds=clip.start, input_sha256=info["source_sha256"],
                  original_stream=info["stream"], assumed_bt709_properties=info["assumptions"],
                  warmup_seconds=clip.warmup, out_of_gamut_channel_samples_clipped=outside_count,
                  decoded_channel_samples=count * clip.width * clip.height * 3)
    return dict(id=clip.id, sequence=sequence, frame_count=count, source=source,
                keyframe=int(np.argmax(metrics["original"]["bright_fraction"])),
                gating=dict(off_frames=off_frames, unchanged_frames=unchanged_frames,
                            max_nonhighlight_rgb_error=max_nonhighlight_error),
                traces_by_method={"legacy": legacy_traces},
                metrics=metrics, traces=traces, frame_refs=list(range(count)), png_pool=pool,
                video_file=video.relative_to(root).as_posix(), video_sha256=_hash(video),
                encoded_first_means=first_means, encoded_last_means=last_means)


def generate_video_comparison(clips, output, language="zh-CN"):
    if not 1 <= len(clips) <= 4 or len({clip.id for clip in clips}) != len(clips) or language not in LANGUAGES:
        raise ValueError("video_selection")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg / ffprobe")
    locale = messages(language)
    if any("comparison.case_" + clip.id not in locale for clip in clips):
        raise ValueError("video_case_label")
    inspected = [inspect(clip, ffprobe) for clip in clips]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = [_evaluate(output, clip, info, ffmpeg) for clip, info in zip(clips, inspected)]
    assets = ("comparison_ui/index.html", "comparison_ui/viewer.css", "comparison_ui/viewer.js")
    payload = dict(schema=3, language=language, sequence=cases[0]["sequence"], frame_count=cases[0]["frame_count"],
                   methods=METHODS, cases=cases, scope_key="scope_real", automatic=asdict(AutoConfig()),
                   fixed_strong=asdict(STRONG),
                   highlight_curve=asdict(HIGHLIGHT), highlight_load_start=LOAD_START, highlight_load_full=LOAD_FULL,
                   legacy=asdict(AutoConfig(policy="legacy")),
                   encoder=_run([ffmpeg, "-version"]).stdout.decode().splitlines()[0],
                   source_sha256=source_hashes("video_comparison.py", "comparison.py", "automatic.py", "color.py", "filter.py", *assets,
                                               "locales/en.json", "locales/zh-CN.json"),
                   messages={lang: {k.removeprefix("comparison."): v for k, v in messages(lang).items()
                                    if k.startswith("comparison.")} for lang in LANGUAGES})
    resources = files("sunglasses")
    html, css, js = (resources.joinpath(asset).read_text(encoding="utf-8") for asset in assets)
    page = output / "index.html"
    with page.open("x", encoding="utf-8") as stream:
        stream.write(html.replace("/* STYLE */", css).replace("/* SCRIPT */", js).replace("/* DATA */", _json(payload)))
    manifest = {key: value for key, value in payload.items() if key != "messages"}
    manifest["html_sha256"] = _hash(page)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return page


def main():
    parser = argparse.ArgumentParser(description="Evaluate local, frame-aligned CFR SDR footage; no player is started.")
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", choices=LANGUAGES, default="zh-CN")
    args = parser.parse_args()
    clips = [VideoClip(**item) for item in json.loads(args.clips.read_text(encoding="utf-8"))]
    print(generate_video_comparison(clips, args.output, args.language))


if __name__ == "__main__":
    main()
