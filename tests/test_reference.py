import math
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
import subprocess
import sys

import numpy as np
from PIL import Image

from sunglasses.color import linear_to_srgb, luminance, srgb_to_linear
from sunglasses.filter import FilterConfig, compress_highlights
from sunglasses.fixtures import CASES, SequenceSpec, frames
from sunglasses.image_io import read_sdr_png, save_sdr_png
from sunglasses.lab import run_suite
from sunglasses.metrics import matched_gains, measure
from sunglasses.cli import main
from sunglasses.i18n import messages
from tests.support import workspace_temp


class ColorTests(unittest.TestCase):
    def test_srgb_reference_values_and_round_trip(self):
        encoded = np.array([[0, 128 / 255, 1]], dtype=np.float32)
        linear = srgb_to_linear(encoded)
        np.testing.assert_allclose(linear, [[0, .2158605, 1]], atol=1e-7)
        np.testing.assert_allclose(linear_to_srgb(linear), encoded, atol=1e-7)

    def test_gamma_is_not_applied_to_linear_data_twice(self):
        np.testing.assert_allclose(luminance(np.eye(3, dtype=np.float32)), [.2126, .7152, .0722])
        self.assertGreater(float(linear_to_srgb(np.full((1, 3), .5))[0, 0]), .73)

    def test_reject_invalid_color_data(self):
        for data in (np.zeros((2, 4)), np.empty((0, 3)), [[math.nan, 0, 0]], [[1.1, 0, 0]], [[-1, 0, 0]]):
            with self.subTest(data=str(data)), self.assertRaises(ValueError):
                srgb_to_linear(data)


class FilterTests(unittest.TestCase):
    def test_black_shadows_and_disabled_mode_are_exact(self):
        image = np.random.default_rng(2).uniform(0, .19, (13, 17, 3)).astype(np.float32)
        image[0, 0] = 0
        np.testing.assert_array_equal(compress_highlights(image), image)
        bright = image * 5
        np.testing.assert_array_equal(compress_highlights(bright, FilterConfig(strength=0)), bright)

    def test_monotonic_grayscale_and_peak_limit(self):
        ramp = np.repeat(np.linspace(0, 1, 4096, dtype=np.float32)[:, None], 3, axis=1)
        result = compress_highlights(ramp)
        self.assertTrue(np.all(np.diff(result[:, 0]) >= 0))
        self.assertTrue(np.all(result <= ramp))
        self.assertLess(float(result.max()), .55)
        self.assertAlmostEqual(float(result[-1, 0]), .51440455, places=6)

    def test_no_mutation_and_colored_channels_keep_proportions(self):
        image = np.array([[[1, .8, .4], [.1, .2, .3]]], dtype=np.float32)
        before = image.copy()
        result = compress_highlights(image)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_allclose(result[0, 0] / image[0, 0], np.repeat(result[0, 0, 0], 3), atol=1e-7)

    def test_strength_interpolates_without_increasing_light(self):
        image = np.ones((2, 2, 3), dtype=np.float32)
        full = compress_highlights(image)
        half = compress_highlights(image, FilterConfig(strength=.5))
        np.testing.assert_allclose(half, (image + full) / 2, atol=1e-7)

    def test_invalid_parameters_fail_before_processing(self):
        for kwargs in ({"knee": 0}, {"ceiling": .1}, {"ceiling": 2}, {"strength": -1}, {"strength": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FilterConfig(**kwargs)


class FixtureAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.spec = SequenceSpec(32, 24, 30, 1, 7)

    def test_cases_are_reproducible_and_timestamped(self):
        for case in CASES:
            first = list(frames(case.id, self.spec))
            second = list(frames(case.id, self.spec))
            self.assertEqual(len(first), 30)
            self.assertTrue(all(a[0] < b[0] for a, b in zip(first, first[1:])))
            for (timestamp, a), (_, b) in zip(first, second):
                np.testing.assert_array_equal(a, b)
                self.assertTrue(np.all((a >= 0) & (a <= 1)))

    def test_single_flash_is_reduced_on_the_first_bright_frame(self):
        sequence = list(frames("single_white_flash", self.spec))
        self.assertEqual(sum(np.all(frame == 1) for _, frame in sequence), 1)
        result = compress_highlights(sequence[15][1])
        self.assertLess(float(luminance(result).max()), .55)

    def test_antiphase_is_invisible_to_average_but_not_local_metric(self):
        y = np.stack([luminance(frame) for _, frame in frames("antiphase_blocks", self.spec)])
        metrics = measure(y, y, .2, .5, 1 / 30)
        self.assertLess(metrics["mean_frame_delta"], 1e-6)
        self.assertGreater(metrics["mean_pixel_delta"], .1)

    def test_red_case_documents_static_limiter_blind_spot(self):
        red = np.array([[[1, 0, 0]]], dtype=np.float32)
        self.assertGreater(float(compress_highlights(red)[0, 0, 0]), .99)

    def test_all_three_matching_criteria_hold(self):
        source = np.random.default_rng(10).uniform(0, 1, (12, 8, 10)).astype(np.float32)
        rgb = np.repeat(source[..., None], 3, axis=-1)
        target = luminance(compress_highlights(rgb))
        gains = matched_gains(source, target, .5)
        self.assertAlmostEqual(float((source * gains["dim_peak"]).max()), float(target.max()), places=6)
        self.assertAlmostEqual(float((source * gains["dim_mean"]).mean()), float(target.mean()), places=6)
        self.assertAlmostEqual(float(np.maximum(source * gains["dim_exposure"] - .5, 0).mean()),
                               float(np.maximum(target - .5, 0).mean()), places=7)

    def test_no_dark_pixels_are_reported_as_not_applicable(self):
        data = np.ones((2, 4, 4), dtype=np.float32)
        self.assertIsNone(measure(data, data, .2, .5, .1)["dark_luminance_retention"])

    def test_case_memory_budget_is_enforced(self):
        with self.assertRaises(ValueError):
            SequenceSpec(4096, 4096, 60, 2)

    def test_refresh_rates_and_odd_width_antiphase(self):
        for fps in (30, 60, 120, 144):
            spec = SequenceSpec(17, 19, fps, 1)
            sequence = list(frames("single_white_flash", spec))
            self.assertEqual(sum(np.all(frame == 1) for _, frame in sequence), 1)
            self.assertAlmostEqual(sequence[-1][0] + 1 / fps, 1)
            y = np.stack([luminance(frame) for _, frame in frames("antiphase_blocks", spec)])
            self.assertLess(measure(y, y, .2, .5, 1 / fps)["mean_frame_delta"], 1e-6)


class FileWorkflowTests(unittest.TestCase):
    def test_tagged_png_processing_and_non_overwrite(self):
        with workspace_temp() as directory:
            path = Path(directory) / "sample.png"
            image = np.ones((16, 16, 3), dtype=np.float32) * .4
            save_sdr_png(path, image)
            np.testing.assert_allclose(read_sdr_png(path), image, atol=.004)
            with self.assertRaises(FileExistsError):
                save_sdr_png(path, image)

    def test_untagged_and_transparent_png_require_explicit_handling(self):
        with workspace_temp() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (16, 16), (128, 128, 128)).save(path)
            with self.assertRaisesRegex(ValueError, "png_untagged"):
                read_sdr_png(path)
            self.assertEqual(read_sdr_png(path, assume_srgb=True).shape, (16, 16, 3))
            Image.new("RGBA", (16, 16), (255, 255, 255, 10)).save(path)
            with self.assertRaisesRegex(ValueError, "png_transparent"):
                read_sdr_png(path, assume_srgb=True)

    def test_suite_outputs_reproducible_report_without_animation(self):
        with workspace_temp() as directory:
            path = Path(directory) / "run"
            report = run_suite(path, ["single_white_flash", "saturated_red"], SequenceSpec(16, 16, 30, .2), FilterConfig())
            self.assertEqual(len(report["cases"]), 2)
            self.assertTrue((path / "summary.json").is_file())
            self.assertEqual(len((path / "metrics.csv").read_text().splitlines()), 11)
            self.assertEqual(len(list(path.glob("previews/*.png"))), 2)
            self.assertFalse(any(p.suffix in (".gif", ".mp4", ".html") for p in path.rglob("*")))
            with self.assertRaises(FileExistsError):
                run_suite(path, ["dark_control"], SequenceSpec(16, 16, 30, .2), FilterConfig())

    def test_png_profile_and_bit_depth_are_not_silently_discarded(self):
        with workspace_temp() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (16, 16)).save(path, icc_profile=b"unknown-profile")
            with self.assertRaisesRegex(ValueError, "png_profile"):
                read_sdr_png(path, assume_srgb=True)
            Image.fromarray(np.zeros((16, 16), dtype=np.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "png_format"):
                read_sdr_png(path, assume_srgb=True)


class CliTests(unittest.TestCase):
    def test_supported_languages_share_keys(self):
        self.assertEqual(messages("en").keys(), messages("zh-CN").keys())

    def test_redirected_windows_style_encoding_handles_chinese(self):
        with workspace_temp() as directory:
            source, target = Path(directory) / "in.png", Path(directory) / "out.png"
            save_sdr_png(source, np.ones((16, 16, 3), dtype=np.float32))
            environment = dict(os.environ, PYTHONIOENCODING="ascii")
            process = subprocess.run([sys.executable, "-m", "sunglasses", "--language", "zh-CN", "process", str(source), str(target)],
                                     cwd=Path(__file__).resolve().parents[1], env=environment,
                                     capture_output=True, encoding="utf-8", timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("已保存", process.stdout)

    def test_cli_process_preserves_source_and_reports_errors(self):
        with workspace_temp() as directory:
            source, target = Path(directory) / "in.png", Path(directory) / "out.png"
            save_sdr_png(source, np.ones((16, 16, 3), dtype=np.float32))
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = main(["--language", "zh-CN", "process", str(source), str(target)])
            self.assertEqual(status, 0)
            self.assertIn("已保存", output.getvalue())
            self.assertEqual(source.read_bytes(), original)
            self.assertLess(float(read_sdr_png(target).max()), .52)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["process", str(source), str(source)]), 2)
            self.assertEqual(source.read_bytes(), original)

    def test_cli_suite_and_invalid_parameters(self):
        with workspace_temp() as directory:
            output = Path(directory) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["suite", "--output", str(output), "--case", "dark_control", "--width", "16", "--height", "16", "--duration", ".1"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads((output / "summary.json").read_text())["cases"][0]["id"], "dark_control")
            invalid = Path(directory) / "invalid"
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                status = main(["--language", "zh-CN", "suite", "--output", str(invalid), "--strength", "nan"])
            self.assertEqual(status, 2)
            self.assertIn("有限数值", errors.getvalue())
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
