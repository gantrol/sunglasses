"""Offline media/data verification only. Does not start a browser or desktop UI."""

import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

import numpy as np
from PIL import Image

from sunglasses.automatic import AutoFilter, STRONG
from sunglasses.color import linear_to_srgb
from sunglasses.comparison import generate
from sunglasses.filter import compress_highlights
from sunglasses.fixtures import SequenceSpec, frames
from sunglasses.i18n import messages
from tests.support import workspace_temp


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
class ComparisonTests(unittest.TestCase):
    def test_synchronized_media_and_exact_scrubbing_frames(self):
        spec = SequenceSpec(32, 18, 30, .4)
        with workspace_temp() as temporary:
            output = Path(temporary) / "comparison"
            page = generate(output, spec, ("single_white_flash",))
            html = page.read_text(encoding="utf-8")
            data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)[1])
            case = data["cases"][0]
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["html_sha256"], hashlib.sha256(page.read_bytes()).hexdigest())
            self.assertEqual(len(case["frame_refs"]), spec.frame_count)
            self.assertNotIn("autoplay", html)
            self.assertNotRegex(html, r'(?:src|href)=["\']https?://')
            control = AutoFilter()
            for index, (timestamp, original) in enumerate(frames(case["id"], spec)):
                automatic, _ = control.process(original, timestamp, index)
                expected = np.concatenate((original, automatic, compress_highlights(original, STRONG),
                                           original * case["dim_gain"]), axis=1)
                expected = np.rint(linear_to_srgb(expected) * 255).astype(np.uint8)
                png_record = case["png_pool"][case["frame_refs"][index]]
                png = base64.b64decode(png_record["data"])
                self.assertEqual(hashlib.sha256(png).hexdigest(), png_record["sha256"])
                with Image.open(io.BytesIO(png)) as image:
                    self.assertIn("srgb", image.info)
                    np.testing.assert_array_equal(np.asarray(image), expected)
            video = output / case["video_file"]
            self.assertEqual(base64.b64decode(case["video_data"]), video.read_bytes())
            probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_frames",
                                               "-of", "json", str(video)], capture_output=True, check=True).stdout)
            stream = probe["streams"][0]
            self.assertEqual((stream["width"], stream["height"]), (4 * spec.width, spec.height))
            self.assertEqual(stream["color_space"], "bt709")
            self.assertEqual(stream["color_range"], "tv")
            self.assertEqual(stream["color_transfer"], "iec61966-2-1")
            self.assertEqual(stream["r_frame_rate"], f"{spec.fps}/1")
            self.assertEqual(len(probe["frames"]), spec.frame_count)
            np.testing.assert_allclose([float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]],
                                       np.arange(spec.frame_count) / spec.fps, atol=1e-6)
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                     capture_output=True, check=True).stdout
            decoded = np.frombuffer(decoded, np.uint8).reshape(spec.frame_count, spec.height, 4 * spec.width, 3)
            # Each of the four tiles must put the ONLY white pulse at the same timestamp.
            peaks = decoded.reshape(spec.frame_count, spec.height, 4, spec.width, 3).mean(axis=(1, 3, 4))
            np.testing.assert_array_equal(peaks.argmax(axis=0), [spec.frame_count // 2] * 4)
            flash_png = np.asarray(Image.open(io.BytesIO(base64.b64decode(case["png_pool"][case["frame_refs"][spec.frame_count // 2]]["data"]))))
            self.assertLessEqual(float(np.abs(decoded[spec.frame_count // 2].astype(float) - flash_png).mean()), 3.0)
            self.assertAlmostEqual(case["metrics"]["automatic"]["peak_y"], .7854542, places=5)
            self.assertEqual(case["metrics"]["automatic"]["dark_luminance_retention"], 1.0)
            self.assertLess(case["metrics"]["dim_peak"]["dark_luminance_retention"], .79)
            original_html = page.read_bytes()
            with self.assertRaises(FileExistsError):
                generate(output, spec, ("single_white_flash",))
            self.assertEqual(page.read_bytes(), original_html)

    def test_highlight_ramp_has_no_added_step_and_invalid_input_writes_nothing(self):
        with workspace_temp() as temporary:
            output = Path(temporary) / "failure"
            generate(output, SequenceSpec(32, 18, 60, 4), ("slow_ramp",))
            case = json.loads((output / "manifest.json").read_text())["cases"][0]
            self.assertLessEqual(case["metrics"]["automatic"]["max_pixel_delta"],
                                 case["metrics"]["original"]["max_pixel_delta"] + 1e-6)
            self.assertGreater(case["keyframe"], 0)
            invalid = Path(temporary) / "invalid"
            with self.assertRaises(ValueError):
                generate(invalid, SequenceSpec(17, 18, 30, .4), ("slow_ramp",))
            self.assertFalse(invalid.exists())
            with self.assertRaises(ValueError):
                generate(invalid, cases=("../arbitrary",))
            self.assertFalse(invalid.exists())


class ComparisonResourceTests(unittest.TestCase):
    def test_locales_cover_viewer_and_cases(self):
        resource = Path(__file__).resolve().parents[1] / "sunglasses/comparison_ui"
        en = {k.removeprefix("comparison.") for k in messages("en") if k.startswith("comparison.")}
        zh = {k.removeprefix("comparison.") for k in messages("zh-CN") if k.startswith("comparison.")}
        self.assertEqual(en, zh)
        keys = set(re.findall(r'data-text="([^"]+)"', (resource / "index.html").read_text()))
        keys.update(re.findall(r'\bt\("([^"]+)"\)', (resource / "viewer.js").read_text()))
        keys.discard("case_")
        self.assertTrue(keys <= en, keys - en)

    @unittest.skipUnless(shutil.which("node"), "Node required")
    def test_javascript_syntax_and_frame_time_boundaries_without_browser(self):
        script = Path(__file__).resolve().parents[1] / "sunglasses/comparison_ui/viewer.js"
        subprocess.run(["node", "--check", str(script)], check=True, capture_output=True)
        checks = """
          const assert = require('node:assert/strict');
          const m = require(process.argv[1]);
          assert.equal(m.frameAt(0, 60, 240), 0);
          assert.equal(m.frameAt(1.999999999, 60, 240), 120);
          assert.equal(m.frameAt(4, 60, 240), 239);
          assert.equal(m.frameAt(-1, 60, 240), 0);
          assert.equal(m.clamp(1100, 0, 1000), 1000);
        """
        subprocess.run(["node", "-e", checks, str(script)], check=True, capture_output=True)
