"""Reusable offline video layouts and frame-synchronized scientific plots."""
from contextlib import ExitStack
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager

from .demo_media import ffmpeg, pipe, probe, read, run, save, sha

BG = "#17191c"
COLORS = ("#e4e1da", "#eac664", "#ef9560", "#65c6a4")


def chart(values, part, portrait, kind, fps, output, labels, tr):
    width, height = (860, 206) if portrait else (920, 216)
    indices = [0, 2, 3] if portrait else [0, 1, 2, 3]
    column = 8 if kind == "mean" else 4
    chunk = values[values[:, 3] == part]
    duration = len(chunk) / fps
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)
    left, bottom, aw, ah = .066, .235, .91, .57
    ax = fig.add_axes([left, bottom, aw, ah], facecolor=BG)
    for index in reversed(indices):
        ax.plot(np.arange(len(chunk)) / fps, chunk[:, column + index], color=COLORS[index],
                linestyle=["--", ":", "-", "-"][index], linewidth=1.7, label=labels[index], zorder=5-index)
    ax.set(xlim=(0, duration), ylim=(0, 1.04), yticks=[0, .5, 1])
    ax.set_xlabel(tr("demo.chart_time"), fontsize=11, color="#b9b9b5", labelpad=3)
    ax.tick_params(colors="#b9b9b5", labelsize=12, length=0, pad=5)
    ax.grid(axis="y", color="#35383a", linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], names[::-1], loc="upper right", bbox_to_anchor=(.99, 1.01),
               ncol=len(indices), frameon=False, labelcolor="#e4e1da", fontsize=12,
               handlelength=1.5, columnspacing=.9, handletextpad=.4)
    fig.text(left, .98, tr(f"demo.chart_{kind}"), color="#e4e1da", fontsize=15, va="top")
    stem = f"{kind}-{'9x16' if portrait else '16x9'}-segment-{part}"
    fig.savefig(output / f"{stem}.png", dpi=100, facecolor=BG)
    fig.savefig(output / f"{stem}.svg", facecolor=BG)
    plt.close(fig)
    with Image.open(output / f"{stem}.png") as image:
        raster = image.convert("RGB")
    return raster, (round(left * width), round((1 - bottom - ah) * height),
                    round(aw * width), round(ah * height)), column


def layout(portrait, font, fps, labels, tr):
    canvas = Image.new("RGB", (1080, 1920) if portrait else (1920, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(font), 32)
    label = ImageFont.truetype(str(font), 25)
    mean_font = ImageFont.truetype(str(font), 28)
    small = ImageFont.truetype(str(font), 21)
    if portrait:
        indices = [0, 2, 3]
        positions = [(40, 140), (40, 532), (40, 924)]
        picture_size, clock_pos = (860, 360), (560, 29)
        plots = [(40, 1306), (40, 1516)]
        draw.text((40, 24), tr("demo.title"), font=title, fill="#f1f1ef")
        draw.text((40, 73), tr("demo.units"), font=small, fill="#b9b9b5")
        draw.text((40, 1750), tr("demo.method", fps=f"{fps:g}"), font=small, fill="#b9b9b5")
    else:
        indices = [0, 1, 2, 3]
        positions = [(70, 86), (1035, 86), (70, 477), (1035, 477)]
        picture_size, clock_pos = (817, 342), (1515, 19)
        plots = [(20, 844), (980, 844)]
        draw.text((28, 15), tr("demo.title"), font=title, fill="#f1f1ef")
        draw.text((435, 24), tr("demo.units"), font=small, fill="#b9b9b5")
        draw.text((1025, 24), tr("demo.method", fps=f"{fps:g}"), font=small, fill="#b9b9b5")
    for index, (x, y) in zip(indices, positions):
        draw.text((x, y - 31), labels[index], font=label, anchor="lt", fill=COLORS[index])
    return {"base": canvas, "positions": positions, "picture_size": picture_size, "indices": indices,
            "plots": plots, "clock_pos": clock_pos, "mean_font": mean_font, "small": small,
            "portrait": portrait, "name": f"comparison-{'9x16' if portrait else '16x9'}"}


def metric_check(frame, row, height):
    """Independent float64 check at segment boundaries and a flash sample."""
    worst = 0.
    for i in range(4):
        rgb = frame[i * height:(i + 1) * height].astype(np.float64) / 255
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        y = linear[..., 0] * .2126 + linear[..., 1] * .7152 + linear[..., 2] * .0722
        worst = max(worst, abs(float(y.mean()) - row[8 + i]), abs(float(y.max()) - row[4 + i]))
    if worst > 5e-7:
        raise RuntimeError("independent_luminance_check_failed")
    return worst


def render(analysis_dir, output, font, selected, tr, say):
    analysis = read(analysis_dir / "analysis.json")
    if analysis.get("schema") != 1:
        raise ValueError("unsupported_analysis_schema")
    for name, field in [("panels-lossless.mkv", "master_sha256"), ("metrics.csv", "metrics_sha256"),
                        ("archive-manifest.json", "manifest_sha256")]:
        if sha(analysis_dir / name) != analysis[field]:
            raise RuntimeError(f"analysis_file_mismatch: {name}")
    manifest = read(analysis_dir / "archive-manifest.json")
    audio = analysis_dir / "excerpt-audio.flac" if analysis["audio_sha256"] else None
    if audio and sha(audio) != analysis["audio_sha256"]:
        raise RuntimeError("analysis_audio_mismatch")
    values = np.genfromtxt(analysis_dir / "metrics.csv", delimiter=",", skip_header=1, ndmin=2)
    count = analysis["frames"]
    if (values.shape != (count, 14) or not np.isfinite(values).all() or
            not np.array_equal(values[:, 0], np.arange(count)) or
            np.any(values[:, 4:12] < 0) or np.any(values[:, 4:12] > 1.000001)):
        raise RuntimeError("invalid_metrics")
    output.mkdir(parents=True, exist_ok=False)
    fps = float(Fraction(manifest["fps"]))
    width, height = manifest["analysis_size"]
    labels = [tr(f"demo.{name}") for name in ["original", "a", "b", "c"]]
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "path.simplify": False})
    modes = [False, True] if selected == "both" else [selected == "portrait"]
    layouts = [layout(mode, font, fps, labels, tr) for mode in modes]
    charts = [[[chart(values, p + 1, cfg["portrait"], kind, fps, output, labels, tr)
                for kind in ["mean", "peak"]] for p in range(len(manifest["clips"]))] for cfg in layouts]
    boundaries = np.cumsum([0, *[c["frames"] for c in manifest["clips"]]])
    samples = set(boundaries[:-1]) | set(boundaries[1:] - 1) | {min(75, count - 1)}
    digest, checked, commands = hashlib.sha256(), [], []
    with ExitStack() as stack:
        encoders = []
        for cfg in layouts:
            cw, ch = cfg["base"].size
            args = ffmpeg("-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{cw}x{ch}",
                          "-framerate", manifest["fps"], "-i", "pipe:0")
            if audio:
                args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
            else:
                args += ["-an"]
            args += ["-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
                     "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "3",
                     "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
                     "-color_primaries", "bt709", "-t", str(count / fps), "-movflags", "+faststart",
                     "-n", str(output / f"{cfg['name']}.mp4")]
            commands.append(args)
            encoders.append(stack.enter_context(pipe(args, output / f"{cfg['name']}-encode.log", writing=True)))
        decoder = stack.enter_context(pipe(ffmpeg("-threads", 3, "-i", analysis_dir / "panels-lossless.mkv",
                                                  "-an", "-pix_fmt", "rgb24", "-fps_mode", "passthrough",
                                                  "-f", "rawvideo", "pipe:1"), output / "master-decode.log"))
        for n, row in enumerate(values):
            raw = decoder.read(width * height * 4 * 3)
            if len(raw) != width * height * 4 * 3:
                raise RuntimeError("truncated_master")
            digest.update(raw)
            master = np.frombuffer(raw, np.uint8).reshape(height * 4, width, 3)
            pictures = [Image.fromarray(master[i * height:(i + 1) * height]) for i in range(4)]
            if n in samples:
                checked.append({"frame": n, "max_metric_error": metric_check(master, row, height)})
            part = int(row[3]) - 1
            offset = n - int(boundaries[part])
            for cfg, plots, encoder in zip(layouts, charts, encoders):
                canvas = cfg["base"].copy()
                draw = ImageDraw.Draw(canvas)
                for index, (x, y) in zip(cfg["indices"], cfg["positions"]):
                    picture = ImageOps.contain(pictures[index], cfg["picture_size"], Image.Resampling.LANCZOS)
                    px = x + (cfg["picture_size"][0] - picture.width) // 2
                    py = y + (cfg["picture_size"][1] - picture.height) // 2
                    canvas.paste(picture, (px, py))
                    draw.text((x + 285, y - 31), tr("demo.mean", value=row[8 + index]), font=cfg["mean_font"],
                              fill=COLORS[index], anchor="lt")
                    draw.text((x + cfg["picture_size"][0] - 160, y - 27), tr("demo.peak", value=row[4 + index]),
                              font=cfg["small"], fill=COLORS[index], anchor="lt")
                for (image, (px, py, pw, ph), column), (gx, gy) in zip(plots[part], cfg["plots"]):
                    canvas.paste(image, (gx, gy))
                    cx = gx + px + round(pw * offset / manifest["clips"][part]["frames"])
                    draw.line((cx, gy + py, cx, gy + py + ph), fill="#8b8e8d", width=2)
                    for index in cfg["indices"]:
                        cy = gy + py + round(ph * (1 - row[column + index] / 1.04))
                        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=COLORS[index])
                source_time = row[2]
                draw.text(cfg["clock_pos"], tr("demo.clock", time=f"{int(source_time)//60:02d}:{source_time%60:06.3f}",
                          part=part + 1, total=len(manifest["clips"])), font=cfg["small"], fill="#b9b9b5")
                encoder.write(canvas.tobytes())
            if (n + 1) % 180 == 0:
                say("progress", frame=n + 1, total=count)
        if decoder.read(1) or digest.hexdigest() != analysis["raw_rgb_sha256"]:
            raise RuntimeError("master_rgb_hash_mismatch")
    results = []
    for cfg in layouts:
        target = output / f"{cfg['name']}.mp4"
        info = probe(target, count=True)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        if (int(video["nb_read_frames"]) != count or Fraction(video["avg_frame_rate"]) != Fraction(manifest["fps"]) or
                abs(float(video["duration"]) - count / fps) > .001 or
                [video.get(key) for key in ["color_range", "color_space", "color_transfer", "color_primaries"]]
                != ["tv", "bt709", "bt709", "bt709"]):
            raise RuntimeError("encoded_video_count_or_timing_mismatch")
        save(output / f"{cfg['name']}-probe.json", info)
        # A decoded final-video preview includes the actual encoding, not just a canvas.
        run(ffmpeg("-i", target, "-vf", f"select=eq(n\\,{min(75, count - 1)})", "-frames:v", 1,
                   "-fps_mode", "passthrough", "-n", output / f"{cfg['name']}-preview.png"), capture_output=True)
        results.append({"file": target.name, "bytes": target.stat().st_size, "sha256": sha(target),
                        "dimensions": [video["width"], video["height"]], "frames": count, "seconds": count / fps})
    save(output / "delivery.json", {"schema": 1, "outputs": results, "analysis_sha256": sha(analysis_dir / "analysis.json"),
         "renderer_sha256": sha(__file__), "font_sha256": sha(font), "matplotlib": matplotlib.__version__,
         "labels": labels, "raw_rgb_sha256": digest.hexdigest(), "independent_metric_checks": checked,
         "commands": commands, "smoothing": False, "chart_range": [0, 1.04], "audio": bool(audio),
         "scope": "Offline same-frame rendering; no screen luminance or real-time latency measurement."})
    say("done", file=output / "delivery.json")
