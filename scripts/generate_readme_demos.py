"""Generate masked README comparisons and separate original GIF attachments.

Synchronous rendering and file I/O belong to this offline CLI. One six-second
clip (at most three 288x162 views) is held at a time; no private media, network
requests or external fonts. Charts use all 60 Hz samples, without smoothing.
The original picture is fully covered in every comparison frame; its measured
brightness remains visible. Uncovered originals are exported as separate files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunglasses import demo_filter
from sunglasses.apple_vfm_screen import AppleVFMScreen
from sunglasses.i18n import translator

WIDTH, HEIGHT = 288, 162
SAMPLE_HZ, EXPORT_HZ, SECONDS = 60, 20, 6
STRENGTH = .5
INPUTS = (
    "scripts/generate_readme_demos.py", "shaders/screen.hlsl",
    "sunglasses/demo_filter.py", "sunglasses/apple_vfm_screen.py",
    "sunglasses/apple_vfm.py", "sunglasses/color.py", "sunglasses/i18n.py",
    "sunglasses/locales/en.json", "sunglasses/locales/zh-CN.json",
    "sunglasses/data/apple_ufm_kernels_6e9604d.txt",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def scene(bright):
    """An abstract room, with a bright opening and unchanged dark foreground."""
    image = Image.new("L", (WIDTH, HEIGHT), 36)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, WIDTH - 13, HEIGHT - 30), fill=48)
    draw.rectangle((28, 22, WIDTH - 29, HEIGHT - 43), fill=255 if bright else 76)
    draw.rectangle((40, 54, 75, HEIGHT - 43), fill=58)
    draw.rectangle((WIDTH - 79, 72, WIDTH - 42, HEIGHT - 43), fill=62)
    draw.rectangle((125, 85, 161, HEIGHT - 31), fill=24)
    draw.rectangle((119, 79, 167, 88), fill=24)
    draw.line((0, HEIGHT - 29, WIDTH, HEIGHT - 29), fill=96)
    for index in range(5):
        draw.rectangle((16 + index * 12, HEIGHT - 19, 23 + index * 12, HEIGHT - 13), fill=155)
    draw.line((WIDTH - 50, HEIGHT - 16, WIDTH - 16, HEIGHT - 16), fill=155, width=2)
    return np.asarray(image.convert("RGB"))


def luminance_metrics(frame):
    """Common display-sRGB metric, independent of either detector's statistics."""
    rgb = frame.astype(np.float64) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    y = linear @ np.array([.2126, .7152, .0722])
    return float(y.mean()), float(y.max())


def panel_base(metrics, labels):
    image = Image.new("RGB", ((WIDTH + 12) * len(labels), 494), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=11)
    tr = translator("en")
    for column, label in enumerate(labels):
        x = 6 + column * (WIDTH + 12)
        if draw.textlength(label, font=font) > WIDTH:
            raise RuntimeError("Comparison label exceeds its column")
        draw.text((x, 9), label, font=font, fill=(220, 220, 220))
        draw.text((x, 304), tr("demo.chart_peak"), font=font, fill=(220, 220, 220))
        left, right, top, bottom = x + 24, x + WIDTH - 4, 332, 412
        for value in (0., .5, 1.):
            y = bottom - round((bottom - top) * value)
            draw.line((left, y, right, y), fill=(65, 65, 65))
            draw.text((x, y - 6), f"{value:g}", font=small, fill=(160, 160, 160))
        points = [(left + (right - left) * i / SAMPLE_HZ / SECONDS,
                   bottom - (bottom - top) * values[column, 1])
                  for i, values in enumerate(metrics)]
        draw.line(points, fill=(220, 220, 220), width=2)
        for seconds in (0, SECONDS / 2, SECONDS):
            px = left + (right - left) * seconds / SECONDS
            draw.text((px, bottom + 6), f"{seconds:g}", font=small,
                      anchor="mt", fill=(160, 160, 160))
        draw.text((x, 440), tr("demo.chart_time"), font=small, fill=(180, 180, 180))
    draw.text((6, 473), tr("demo.units"), font=small, fill=(180, 180, 180))
    return image


def panel(base, pictures, metrics, index):
    image = base.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    tr = translator("en")
    for column, picture in enumerate(pictures):
        x = 6 + column * (WIDTH + 12)
        if column == 0:
            # Never put source pixels into the comparison's original-video area.
            draw.rectangle((x, 38, x + WIDTH - 1, 38 + HEIGHT - 1), fill=(32, 32, 32))
            for key, y in (("demo.original_hidden", 106), ("demo.original_attachment", 134)):
                label = tr(key)
                if draw.textlength(label, font=font) > WIDTH - 12:
                    raise RuntimeError("Original-video notice exceeds its column")
                draw.text((x + WIDTH / 2, y), label, anchor="mm", font=font, fill=(180, 180, 180))
        else:
            image.paste(Image.fromarray(picture), (x, 38))
        for metric, y, value in zip(("mean", "peak"), (212, 254), metrics[column]):
            draw.text((x, y), tr(f"demo.{metric}", value=value), font=font, fill=(220, 220, 220))
            draw.rectangle((x, y + 22, x + WIDTH - 1, y + 31), fill=(65, 65, 65))
            length = round(WIDTH * value)
            if length:
                draw.rectangle((x, y + 22, x + length - 1, y + 31), fill=(205, 205, 205))
        left, right = x + 24, x + WIDTH - 4
        cx = left + round((right - left) * index / SAMPLE_HZ / SECONDS)
        cy = 412 - round(80 * metrics[column, 1])
        draw.line((cx, 332, cx, 412), fill=(115, 115, 115))
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(245, 245, 245))
    # All pixels are neutral gray: a fixed palette preserves every 8-bit level.
    palette = image.convert("L").convert("P")
    palette.putpalette([value for level in range(256) for value in (level, level, level)])
    return palette


def save_gif(target, frames, *, masked_source=False):
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=1000 // EXPORT_HZ, loop=0, disposal=2, optimize=False)
    duration = 0
    mask = np.asarray(frames[0].convert("RGB"))[38:38 + HEIGHT, 6:6 + WIDTH] if masked_source else None
    with Image.open(target) as decoded:
        for index in range(decoded.n_frames):
            decoded.seek(index)
            decoded.load()
            pixels = np.asarray(decoded.convert("RGB"))
            if mask is not None and not np.array_equal(pixels[38:38 + HEIGHT, 6:6 + WIDTH], mask):
                raise RuntimeError("Original-video cover changed during playback")
            # GIF may coalesce identical frames. Verify every covered export tick.
            frame_duration = decoded.info["duration"]
            if frame_duration % (1000 // EXPORT_HZ):
                raise RuntimeError("Encoded GIF changed the export clock")
            for tick in range(duration // (1000 // EXPORT_HZ),
                              (duration + frame_duration) // (1000 // EXPORT_HZ)):
                if tick >= len(frames) or not np.array_equal(pixels, np.asarray(frames[tick].convert("RGB"))):
                    raise RuntimeError("Encoded GIF pixels differ from the rendered frames")
            duration += frame_duration
        if duration != SECONDS * 1000 or decoded.size != frames[0].size:
            raise RuntimeError("Encoded GIF dimensions or duration changed")
    return {"file": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
            "width": frames[0].width, "height": frames[0].height}


def render(output, repeated):
    burst = demo_filter.Controller()
    apple = AppleVFMScreen() if repeated else None
    algorithms = ["original", "screen_burst_reference"]
    tr = translator("en")
    labels = [tr("demo.original"),
              tr("demo.filter_level", name=tr("desktop.screen_highlights"), strength=STRENGTH)]
    if apple is not None:
        algorithms.append("apple_matlab_vfm_60hz_v1")
        labels.append(tr("demo.filter_level", name=tr("desktop.screen_apple_vfm"), strength=STRENGTH))
    dark, bright = scene(False), scene(True)
    sampled, measurements, active_samples = [], [], []
    hashes = [hashlib.sha256() for _ in algorithms]
    changed_frames, first_changed = [0] * len(algorithms), [None] * len(algorithms)
    max_mitigation = [0.] * len(algorithms)
    for index in range(SAMPLE_HZ * SECONDS):
        active = (60 <= index < 240 and index % 12 < 6) if repeated else 90 <= index < 96
        source = bright if active else dark
        linear, luminance, load, mean = demo_filter.stats(source)
        gain = burst.step(load, mean, index / SAMPLE_HZ)
        filtered = demo_filter.apply(source, linear, luminance, gain, burst.broad, STRENGTH)
        pictures = [source, filtered]
        max_mitigation[1] = max(max_mitigation[1], gain)
        if apple is not None:
            result = apple.process(source, index / SAMPLE_HZ, index, strength=STRENGTH)
            filtered = np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8)
            pictures.append(filtered)
            if apple.measurement is None:
                raise RuntimeError("Apple reference did not produce a measurement")
            max_mitigation[2] = max(max_mitigation[2], apple.measurement.applied_mitigation)
        for column, picture in enumerate(pictures):
            hashes[column].update(picture.tobytes())
            changed = not np.array_equal(source, picture)
            changed_frames[column] += int(changed)
            if changed and first_changed[column] is None:
                first_changed[column] = index / SAMPLE_HZ
        measurements.append([luminance_metrics(picture) for picture in pictures])
        active_samples.append(active)
        if index % (SAMPLE_HZ // EXPORT_HZ) == 0:
            sampled.append((index, pictures))
    if any(count == 0 for count in changed_frames[1:]) or min(max_mitigation[1:]) <= 0:
        raise RuntimeError("The comparison did not exercise the selected filter")
    metrics = np.asarray(measurements, dtype=np.float64)
    active = np.asarray(active_samples)
    base = panel_base(metrics, labels)
    exported = [panel(base, pictures, metrics[index], index) for index, pictures in sampled]
    name = "repeated-flashes.gif" if repeated else "single-burst.gif"
    video = save_gif(output / name, exported, masked_source=True)
    originals = []
    for _, pictures in sampled:
        original = Image.fromarray(pictures[0]).convert("L").convert("P")
        original.putpalette([value for level in range(256) for value in (level, level, level)])
        originals.append(original)
    original_video = save_gif(output / ("original-" + name), originals)
    comparisons = []
    for column, algorithm in enumerate(algorithms):
        comparisons.append({"algorithm": algorithm, "label": labels[column],
                            "rgb_sha256": hashes[column].hexdigest(),
                            "filtered_frames": changed_frames[column],
                            "first_changed_seconds": first_changed[column],
                            "max_reference_control": max_mitigation[column],
                            "bright_phase_mean_y_range": [float(metrics[active, column, 0].min()),
                                                          float(metrics[active, column, 0].max())],
                            "bright_phase_peak_y_range": [float(metrics[active, column, 1].min()),
                                                          float(metrics[active, column, 1].max())],
                            "dark_phase_mean_y_range": [float(metrics[~active, column, 0].min()),
                                                        float(metrics[~active, column, 0].max())]})
    return {**video,
            "algorithm": "apple_matlab_vfm_60hz_v1" if repeated else "screen_burst_reference",
            "strength": STRENGTH, "sample_hz": SAMPLE_HZ, "export_hz": EXPORT_HZ,
            "duration_seconds": SECONDS, "flash_hz": 5 if repeated else None,
            "flash_interval_seconds": [1, 4] if repeated else [1.5, 1.6],
            "filtered_frames": changed_frames[-1], "max_reference_control": max_mitigation[-1],
            "input_rgb_sha256": hashes[0].hexdigest(), "output_rgb_sha256": hashes[-1].hexdigest(),
            "comparisons": comparisons,
            "original_video": {"visibility": "fully_covered_in_comparison",
                               "cover_xywh": [6, 38, WIDTH, HEIGHT],
                               "attachment": original_video},
            "luminance_metrics": {"space": "display-sRGB relative linear Y", "range": [0, 1],
                                  "bars": ["frame_mean", "frame_peak"], "chart": "frame_peak",
                                  "sample_hz": SAMPLE_HZ, "smoothing": False,
                                  "includes_layout": False,
                                  "samples_f64le_sha256": hashlib.sha256(metrics.astype('<f8').tobytes()).hexdigest()},
            "verification": "Every decoded comparison and original export tick matches its rendered frame; original-video cover is constant"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/demos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if "trash" in str(args.output).lower() or args.output.is_symlink():
        parser.error("Excluded output path")
    output = args.output.resolve()
    if "trash" in str(output).lower() or not output.is_relative_to(ROOT):
        parser.error("Output must be inside the project")
    names = ("single-burst.gif", "repeated-flashes.gif", "original-single-burst.gif",
             "original-repeated-flashes.gif", "manifest.json")
    if any((output / name).is_symlink() for name in names):
        parser.error("Output files cannot be symbolic links")
    if not args.overwrite and any((output / name).exists() for name in names):
        parser.error("Output exists; pass --overwrite to regenerate")
    if digest(ROOT / "shaders/screen.hlsl") != demo_filter.SHADER_SHA256:
        raise RuntimeError("Frozen screen reference changed")
    inputs = {name: source_digest(ROOT / name) for name in INPUTS}
    output.mkdir(parents=True, exist_ok=True)
    clips = [render(output, False), render(output, True)]
    if inputs != {name: source_digest(ROOT / name) for name in INPUTS}:
        raise RuntimeError("Demonstration inputs changed during rendering")
    report = {"schema": 1, "kind": "synthetic-offline-readme-demos",
              "scope": "CPU references only; no live capture, presentation timing or game compatibility evidence",
              "numpy": np.__version__, "pillow": pillow_version,
              "input_hash_format": "SHA-256 after CRLF-to-LF normalization of text sources",
              "inputs": inputs, "clips": clips}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for clip in clips:
        print(f"{clip['file']}: {clip['bytes']} bytes, {clip['duration_seconds']} seconds")


if __name__ == "__main__":
    main()
