"""Bounded, offline synthetic video comparison. Never opens a player or captures a screen.

File I/O and encoding are synchronous in this standalone CLI, outside any UI or
real-time frame loop. Only one case and one encoder are processed at a time.
"""

import argparse
import base64
from dataclasses import asdict
import hashlib
from importlib.resources import files
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, PngImagePlugin

from .automatic import AutoConfig, AutoFilter, STRONG, HIGHLIGHT, LOAD_START, LOAD_FULL
from .color import linear_to_srgb, luminance
from .filter import compress_highlights
from .fixtures import SequenceSpec, frames
from .i18n import LANGUAGES, messages, translator
from .provenance import source_hashes


CASES = ("localized_burst", "single_white_flash", "midtone_recovery", "slow_ramp", "moving_pattern")
METHODS = ("original", "automatic", "fixed_strong", "dim_peak")
EMBED_LIMIT = 64 * 1024 * 1024


def png_bytes(rgb8):
    info = PngImagePlugin.PngInfo()
    info.add(b"sRGB", b"\x00")
    output = io.BytesIO()
    Image.fromarray(rgb8).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def _json(value):
    # Safe in an HTML script element, even for translated strings.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


class _Stats:
    """Streaming diagnostics in relative linear luminance, before media encoding."""

    def __init__(self):
        self.peak = self.delta = self.dark_in = self.dark_out = 0.0
        self.previous = None
        self.means = []

    def add(self, rgb, source_y):
        y = luminance(rgb)
        self.peak = max(self.peak, float(y.max()))
        if self.previous is not None:
            self.delta = max(self.delta, float(np.abs(y - self.previous).max()))
        self.previous = y
        dark = (source_y > 0) & (source_y <= STRONG.knee)
        self.dark_in += float(source_y[dark].sum(dtype=np.float64))
        self.dark_out += float(y[dark].sum(dtype=np.float64))
        self.means.append(float(y.mean(dtype=np.float64)))

    def result(self):
        return dict(peak_y=self.peak, max_pixel_delta=self.delta,
                    dark_luminance_retention=self.dark_out / self.dark_in if self.dark_in else None,
                    frame_mean_y=self.means)


def _case(output, case_id, spec, ffmpeg, config, remaining):
    # Whole-clip peak matching is an explicitly offline control, never prediction.
    control = AutoFilter(config)
    source_peak = auto_peak = 0.0
    for index, (timestamp, frame) in enumerate(frames(case_id, spec)):
        filtered, _ = control.process(frame, timestamp, index)
        source_peak = max(source_peak, float(luminance(frame).max()))
        auto_peak = max(auto_peak, float(luminance(filtered).max()))
    dim_gain = auto_peak / source_peak if source_peak else 1.0
    video = output / f"{case_id}.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{spec.width * 4}x{spec.height}",
               "-framerate", str(spec.fps), "-i", "pipe:0", "-an",
               "-vf", ("scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
                       "setparams=range=limited:color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709"),
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-color_range", "tv", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
               "-colorspace", "bt709", "-movflags", "+faststart", str(video)]
    stats = {method: _Stats() for method in METHODS}
    control = AutoFilter(config)
    pool, lookup, frame_refs, traces = [], {}, [], []
    used = 0
    log = output / f"{case_id}.encoder.log"
    # File stderr avoids pipe deadlock on an encoder error. This log stays local.
    with log.open("xb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            for index, (timestamp, frame) in enumerate(frames(case_id, spec)):
                filtered, trace = control.process(frame, timestamp, index)
                outputs = (frame, filtered, compress_highlights(frame, STRONG), frame * dim_gain)
                source_y = luminance(frame)
                for method, rgb in zip(METHODS, outputs):
                    stats[method].add(rgb, source_y)
                packed = np.concatenate(outputs, axis=1)
                rgb8 = np.rint(linear_to_srgb(packed) * 255).astype(np.uint8)
                png = png_bytes(rgb8)
                digest = hashlib.sha256(png).hexdigest()
                if digest not in lookup:
                    encoded = base64.b64encode(png).decode("ascii")
                    used += len(encoded)
                    if used > remaining:
                        raise ValueError("comparison_budget")
                    lookup[digest] = len(pool)
                    pool.append(dict(sha256=digest, data=encoded))
                frame_refs.append(lookup[digest])
                traces.append(trace)
                process.stdin.write(rgb8.tobytes())
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise RuntimeError(f"Encoder failed; inspect {log}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
    if video.stat().st_size * 4 / 3 + used > remaining:
        raise ValueError("comparison_budget")
    video_bytes = video.read_bytes()
    video_data = base64.b64encode(video_bytes).decode("ascii")
    used += len(video_data)
    metrics = {method: value.result() for method, value in stats.items()}
    if case_id == "slow_ramp":
        keyframe = int(np.argmax(np.abs(np.diff(metrics["automatic"]["frame_mean_y"])))) + 1
    elif case_id == "moving_pattern":
        keyframe = next((i for i, trace in enumerate(traces) if trace["amount"] > .05), 0)
    else:
        keyframe = int(np.argmax(metrics["original"]["frame_mean_y"]))
    return dict(id=case_id, keyframe=keyframe, dim_gain=dim_gain, metrics=metrics, traces=traces,
                frame_refs=frame_refs, png_pool=pool, video_file=video.name, video_data=video_data,
                video_sha256=hashlib.sha256(video_bytes).hexdigest()), used


def generate(output: Path, spec=SequenceSpec(duration=4), cases=CASES, language="zh-CN", ffmpeg=None):
    """Write a self-contained HTML and synchronized four-panel MP4s to a new directory.

    PNGs are exact 8-bit sRGB exports for scrubbing, not lossless float32 samples.
    MP4 is a lossy playback preview. Numeric diagnostics never use decoded MP4s.
    """
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError("comparison_cases")
    if spec.width % 2 or spec.height % 2:
        raise ValueError("comparison_even_size")
    if language not in LANGUAGES:
        raise ValueError("comparison_language")
    ffmpeg = shutil.which(ffmpeg or "ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg")
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.splitlines()[0]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = AutoConfig()
    assets = ("comparison_ui/index.html", "comparison_ui/viewer.css", "comparison_ui/viewer.js")
    payload = dict(schema=1, language=language, sequence=asdict(spec), frame_count=spec.frame_count,
                   methods=METHODS, automatic=asdict(config), fixed_strong=asdict(STRONG), encoder=version,
                   highlight_curve=asdict(HIGHLIGHT), highlight_load_start=LOAD_START, highlight_load_full=LOAD_FULL,
                   source_sha256=source_hashes("automatic.py", "color.py", "filter.py", "fixtures.py", "comparison.py", *assets,
                                               "locales/en.json", "locales/zh-CN.json"),
                   cases=[], messages={lang: {k.removeprefix("comparison."): v for k, v in messages(lang).items()
                                             if k.startswith("comparison.")} for lang in LANGUAGES})
    used = 0
    for case_id in cases:
        case, size = _case(output, case_id, spec, ffmpeg, config, EMBED_LIMIT - used)
        payload["cases"].append(case)
        used += size
    resources = files("sunglasses")
    html, css, js = (resources.joinpath(name).read_text(encoding="utf-8") for name in assets)
    html = html.replace("/* STYLE */", css).replace("/* SCRIPT */", js).replace("/* DATA */", _json(payload))
    if len(html.encode("utf-8")) > EMBED_LIMIT:
        raise ValueError("comparison_budget")
    page = output / "index.html"
    with page.open("x", encoding="utf-8") as stream:
        stream.write(html)
    # The manifest references all embedded media, without duplicating base64 payloads.
    manifest = {k: v for k, v in payload.items() if k not in ("messages", "cases")}
    manifest["html_sha256"] = hashlib.sha256(page.read_bytes()).hexdigest()
    manifest["cases"] = [{k: ([p["sha256"] for p in v] if k == "png_pool" else v)
                           for k, v in case.items() if k != "video_data"} for case in payload["cases"]]
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return page


def main():
    parser = argparse.ArgumentParser(description="Generate a local synthetic video comparison; never opens a player.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", choices=LANGUAGES, default="zh-CN")
    args = parser.parse_args()
    t = translator(args.language)
    try:
        print(generate(args.output, language=args.language))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(t("comparison.generation_error", error=str(error)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
