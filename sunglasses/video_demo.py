"""Archive original excerpts, analyze screen-burst processing, export demos.

All blocking I/O/CPU work is intentional in this standalone offline command.
No application launch, live capture, network access or native build occurs.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import demo_filter as effect
from .demo_media import ffmpeg, frame_hashes, frames, pipe, probe, read, run, save, sha, tool
from .i18n import LANGUAGES, translator


def identity():
    shader = Path(__file__).resolve().parents[1] / "shaders/screen.hlsl"
    if shader.exists() and sha(shader) != effect.SHADER_SHA256:
        raise RuntimeError("shader_changed_revalidate_reference")
    return {"shader_sha256": effect.SHADER_SHA256, "reference_sha256": sha(effect.__file__)}


def checked_manifest(path):
    manifest = read(path)
    if manifest.get("schema") != 1 or manifest["algorithm"] != identity():
        raise RuntimeError("unsupported_archive_or_reference_changed")
    for clip in manifest["clips"]:
        candidate = (path.parent / clip["file"]).resolve()
        if not candidate.is_relative_to(path.parent.resolve()) or sha(candidate) != clip["sha256"]:
            raise RuntimeError("archive_file_mismatch")
    return manifest


def archive(source, ranges, output, say):
    algorithm = identity()
    importer_sha256 = sha(__file__)
    source = source.resolve(strict=True)
    info = probe(source)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    fps = Fraction(video["avg_frame_rate"])
    # A deliberately explicit SDR scope: never silently reinterpret HDR/VFR.
    if (video.get("pix_fmt") != "yuv420p" or video.get("color_space") != "bt709" or
            video.get("color_transfer") != "bt709" or video.get("color_primaries") != "bt709" or
            video.get("color_range") != "tv" or video.get("sample_aspect_ratio") != "1:1" or
            video.get("field_order") != "progressive" or not 24 <= fps <= 120 or
            float(video.get("start_time", 0)) != 0 or video["width"] < video["height"] or
            any(s.get("rotation", 0) for s in video.get("side_data_list", []))):
        raise ValueError("requires_progressive_landscape_8bit_limited_bt709_zero_start")
    clips = []
    for part, item in enumerate(ranges, 1):
        start, end = map(Fraction, item.split(":"))
        if start < 0 or not 0 < end - start <= 30 or (start * fps).denominator != 1 or (end * fps).denominator != 1:
            raise ValueError("invalid_frame_aligned_range_max_30_seconds")
        clips.append({"file": f"original-{part:02d}.mkv", "start_frame": int(start * fps),
                      "frames": int((end - start) * fps), "source_start_seconds": float(start),
                      "duration_seconds": float(end - start)})
    if not 1 <= len(clips) <= 12 or sum(c["duration_seconds"] for c in clips) > 120:
        raise ValueError("archive_limit_12_clips_120_seconds")
    output.mkdir(parents=True, exist_ok=False)
    say("archive")
    save(output / "source-probe.json", info)
    last_end = max(c["source_start_seconds"] + c["duration_seconds"] for c in clips)
    pts_result = run([tool("ffprobe"), "-v", "error", "-threads", "3", "-read_intervals", f"%{last_end + 1}",
                      "-select_streams", "v:0", "-show_frames",
                      "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)],
                     capture_output=True)
    pts = [float(f["best_effort_timestamp_time"]) for f in json.loads(pts_result.stdout)["frames"]]
    end_frame = max(c["start_frame"] + c["frames"] for c in clips)
    if len(pts) < end_frame or any(abs(t - n / float(fps)) > 1e-6 for n, t in enumerate(pts[:end_frame])):
        raise ValueError("requires_constant_cadence_through_selected_ranges")
    manifest = {"schema": 1, "algorithm": algorithm, "importer_sha256": importer_sha256,
                "source_name": source.name,
                "source_sha256": sha(source), "source_bytes": source.stat().st_size,
                "source_size": [video["width"], video["height"]], "fps": str(fps),
                "analysis_size": [860, round(video["height"] * 860 / video["width"])],
                "color_interpretation": "BT.709 limited YUV to RGB; sRGB relative Y, not cd/m2",
                "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
                "ffmpeg_version": run([tool("ffmpeg"), "-version"], capture_output=True).stdout.decode().splitlines()[0],
                "clips": clips}
    for clip in clips:
        target = output / clip["file"]
        args = ffmpeg("-threads", 3, "-ss", clip["source_start_seconds"], "-i", source,
                      "-t", clip["duration_seconds"], "-map", "0:v:0", "-map", "0:a:0?",
                      "-c:v", "ffv1", "-level", 3, "-threads", 4, "-pix_fmt", "yuv420p",
                      "-slicecrc", 1, "-fps_mode", "passthrough", "-c:a", "flac",
                      "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
                      "-color_primaries", "bt709", "-n", target)
        with (output / f"{target.stem}-encode.log").open("wb") as log:
            run(args, stderr=log)
        original_hashes = frame_hashes(source, clip["frames"], clip["source_start_seconds"])
        stored_hashes = frame_hashes(target, clip["frames"])
        actual = next(s for s in probe(target, count=True)["streams"] if s["codec_type"] == "video")
        if (original_hashes != stored_hashes or int(actual["nb_read_frames"]) != clip["frames"] or
                [actual["width"], actual["height"]] != manifest["source_size"]):
            raise RuntimeError("lossless_archive_validation_failed")
        clip.update(sha256=sha(target), bytes=target.stat().st_size,
                    source_pts_seconds=pts[clip["start_frame"]:clip["start_frame"] + clip["frames"]],
                    decoded_yuv_frame_md5=stored_hashes)
        say("clip_done", file=clip["file"])
    # Scan the full prefix once, not only the demonstration ranges. A clip's
    # initial state includes all preceding frames even when output order differs.
    say("history")
    control, starts = effect.Controller(), {c["start_frame"] for c in clips}
    states = {}
    width, height = manifest["analysis_size"]
    for n, raw in enumerate(frames(source, (width, height), max(starts) + 1, output / "history-decode.log")):
        if n in starts:
            states[n] = control.snapshot()
        frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        _, _, load, mean = effect.stats(frame)
        control.step(load, mean, pts[n])
        if (n + 1) % 1200 == 0:
            say("progress", frame=n + 1, total=max(starts) + 1)
    for clip in clips:
        clip["initial_controller"] = states[clip["start_frame"]]
    save(output / "manifest.json", manifest)
    say("done", file=output / "manifest.json")


def analyze(manifest_path, output, say):
    manifest = checked_manifest(manifest_path)
    analyzer_sha256 = sha(__file__)
    output.mkdir(parents=True, exist_ok=False)
    width, height = manifest["analysis_size"]
    master = output / "panels-lossless.mkv"
    args = ffmpeg("-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height * 4}",
                  "-framerate", manifest["fps"], "-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", 3,
                  "-threads", 4, "-pix_fmt", "bgr0", "-n", master)
    columns = ["output_frame", "source_frame", "source_seconds", "segment",
               "original_peak", "a_peak", "b_peak", "c_peak", "original_mean", "a_mean", "b_mean", "c_mean",
               "gain", "broad"]
    digest, n, total = hashlib.sha256(), 0, sum(c["frames"] for c in manifest["clips"])
    with pipe(args, output / "master-encode.log", writing=True) as encoder, \
            (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for part, clip in enumerate(manifest["clips"], 1):
            control = effect.Controller(**clip["initial_controller"])
            for offset, raw in enumerate(frames(manifest_path.parent / clip["file"], (width, height),
                                                clip["frames"], output / f"decode-{part}.log")):
                frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                linear, y, load, mean = effect.stats(frame)
                t = clip["source_pts_seconds"][offset]
                gain = control.step(load, mean, t)
                pictures, peaks, means = [frame], [float(y.max())], [mean]
                for strength in effect.STRENGTHS[1:]:
                    processed = effect.apply(frame, linear, y, gain, control.broad, strength)
                    if np.any(processed > pictures[-1]) or np.any((processed != frame) & (y <= .45)[..., None]):
                        raise RuntimeError("filter_pixel_invariant_failed")
                    _, output_y, _, output_mean = effect.stats(processed)
                    pictures.append(processed)
                    peaks.append(float(output_y.max()))
                    means.append(output_mean)
                pixels = np.concatenate(pictures, axis=0).tobytes()
                digest.update(pixels)
                encoder.write(pixels)
                writer.writerow([n, clip["start_frame"] + offset, t, part, *peaks, *means, gain, control.broad])
                n += 1
                if n % 180 == 0:
                    say("progress", frame=n, total=total)
    # The copied manifest and local audio make analysis sufficient for re-layout.
    save(output / "archive-manifest.json", manifest)
    audio = None
    if manifest["audio"]:
        inputs, filters, labels = [], [], []
        for i, clip in enumerate(manifest["clips"]):
            inputs += ["-i", str(manifest_path.parent / clip["file"])]
            filters.append(f"[{i}:a]atrim=duration={clip['duration_seconds']},asetpts=PTS-STARTPTS,"
                           f"apad=whole_dur={clip['duration_seconds']},atrim=duration={clip['duration_seconds']}[a{i}]")
            labels.append(f"[a{i}]")
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
        audio = output / "excerpt-audio.flac"
        run(ffmpeg(*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-c:a", "flac", "-n", audio),
            capture_output=True)
    save(output / "analysis.json", {"schema": 1, "algorithm": identity(), "frames": n,
         "analyzer_sha256": analyzer_sha256, "numpy": np.__version__, "python": sys.version,
         "strengths": effect.STRENGTHS, "raw_rgb_sha256": digest.hexdigest(),
         "master_sha256": sha(master), "metrics_sha256": sha(output / "metrics.csv"),
         "manifest_sha256": sha(output / "archive-manifest.json"), "audio_sha256": sha(audio) if audio else None,
         "metric": "sRGB relative Y at analysis resolution after 8-bit quantization, before layout/encoding",
         "checks": "All pixels original>=A>=B>=C; original Y<=0.45 unchanged. Same-frame offline output."})
    say("done", file=output / "analysis.json")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--language", choices=LANGUAGES, default="zh-CN")
    language, _ = pre.parse_known_args(argv)
    tr = translator(language.language)
    parser = argparse.ArgumentParser(parents=[pre], description=tr("demo.help"))
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("archive", help=tr("demo.help_archive"))
    importer.add_argument("--source", type=Path, required=True, help=tr("demo.source"))
    importer.add_argument("--ranges", nargs="+", required=True, help=tr("demo.ranges"))
    importer.add_argument("--output", type=Path, required=True, help=tr("demo.output"))
    analyzer = commands.add_parser("analyze", help=tr("demo.help_analyze"))
    analyzer.add_argument("--manifest", type=Path, required=True, help=tr("demo.manifest"))
    analyzer.add_argument("--output", type=Path, required=True, help=tr("demo.output"))
    renderer = commands.add_parser("render", help=tr("demo.help_render"))
    renderer.add_argument("--analysis", type=Path, required=True, help=tr("demo.analysis"))
    renderer.add_argument("--output", type=Path, required=True, help=tr("demo.output"))
    renderer.add_argument("--font", type=Path, required=True, help=tr("demo.font"))
    renderer.add_argument("--layout", choices=("both", "landscape", "portrait"), default="both", help=tr("demo.layout"))
    args = parser.parse_args(argv)

    def say(key, **kwargs):
        print(tr(f"demo.{key}", **kwargs), flush=True)

    try:
        if args.command == "archive":
            archive(args.source, args.ranges, args.output, say)
        elif args.command == "analyze":
            analyze(args.manifest.resolve(), args.output, say)
        else:
            from .demo_render import render
            render(args.analysis.resolve(), args.output, args.font.resolve(strict=True), args.layout, tr, say)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(1, tr("demo.error", detail=str(error)) + "\n")


if __name__ == "__main__":
    main()
