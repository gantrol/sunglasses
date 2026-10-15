"""Headless checks for streamed native-resolution font/detail experiments."""

from pathlib import Path
import unittest
import csv
import json

import numpy as np

from sunglasses.i18n import ValidationError
from sunglasses.native_detail_lab import (
    NATIVE_DETAIL_CASES,
    NativeDetailDesign,
    NativeDetailSpec,
    _render_layers,
    run_native_detail_suite,
)
from sunglasses.perceptual import DisplayProfile
from tests.support import workspace_temp


FONT = Path("C:/Windows/Fonts/seguisb.ttf")


class NativeDetailLabTests(unittest.TestCase):
    def test_real_font_masks_have_antialiasing_and_bounded_crops(self):
        spec = NativeDetailSpec(320, 180, 30, 1)
        areas = {}
        for case_id in NATIVE_DETAIL_CASES:
            layers = _render_layers(spec, FONT, case_id)
            fill = layers["fill"]
            self.assertGreater(int((fill > 0).sum()), 0)
            self.assertTrue(np.any((fill > 0) & (fill < 255)))
            left, top, right, bottom = layers["crop"]
            self.assertTrue(0 <= left < right <= spec.width)
            self.assertTrue(0 <= top < bottom <= spec.height)
            areas[case_id] = float((fill >= 204).mean())
        self.assertGreater(areas["dense_warning"], areas["prearmed_text"])
        native_scale = NativeDetailSpec(960, 540, 30, 1)
        dense = _render_layers(native_scale, FONT, "dense_warning")["fill"]
        self.assertGreater(float((dense >= 204).mean()), .05)

    def test_streamed_suite_distinguishes_prearmed_and_late_text(self):
        spec = NativeDetailSpec(320, 180, 30, 1)
        profile = DisplayProfile(200, .2, 27, (320, 180), .65)
        design = NativeDetailDesign(str(FONT), cases=("prearmed_text", "late_text"))
        with workspace_temp() as temporary:
            report = run_native_detail_suite(
                Path(temporary) / "native-detail", spec, profile, design, with_cvvdp=False)
            output = Path(temporary) / "native-detail"
            self.assertEqual(json.loads((output / "summary.json").read_text()), report)
            with (output / "traces/prearmed_text.csv").open(newline="") as stream:
                trace = list(csv.DictReader(stream))
            self.assertEqual(len(trace), spec.frame_count * 5)
            self.assertEqual({row["frame_id"] for row in trace},
                             {str(index) for index in range(spec.frame_count)})
            self.assertIn("added_fall_nits_max", trace[0])
            self.assertIn("source_screening_general_fall_flash_event", trace[0])
            self.assertIn("output_screening_general_fall_flash_event", trace[0])
            self.assertTrue((output / "traces/prearmed_text-controls.csv").is_file())
            self.assertFalse(any(path.suffix.lower() in (".png", ".mp4", ".mkv")
                                 for path in output.rglob("*")))
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(len(report["rows"]), 10)
        self.assertEqual(report["audit"]["same_frame_brightening_nits_max"], 0)
        self.assertEqual(report["audit"]["detail_armed_while_dimmed_pixels"], 0)
        self.assertLessEqual(
            report["audit"]["max_motion_detail_envelope_added_log_rise_per_second"],
            .5 + 1e-7)
        self.assertGreaterEqual(
            report["audit"]["max_motion_detail_realized_added_log_rise_per_second"],
            report["audit"]["max_motion_detail_envelope_added_log_rise_per_second"])
        cases = {item["case"]: item for item in report["cases"]}
        self.assertGreater(cases["prearmed_text"]["max_detail_protected_area_fraction"], 0)
        self.assertEqual(cases["late_text"]["max_detail_protected_area_fraction"], 0)
        self.assertEqual(cases["late_text"]["max_motion_detail_protected_area_fraction"], 0)
        self.assertEqual(cases["late_text"]["motion_propagated_frames"], 0)
        identity = [row for row in report["rows"] if row["candidate"] == "identity"]
        self.assertTrue(all(row["source_aware_screening_added_combined_event_frames"] == 0
                            for row in identity))
        self.assertTrue(all(row["screening_combined_flash_events"] > 0 for row in identity))
        rows = {(row["case"], row["candidate"]): row for row in report["rows"]}
        pixel = rows[("prearmed_text", "pixel_cone_4deg_rate_0p5_q8_g0p02")]
        coherent = rows[
            ("prearmed_text", "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02")]
        motion = rows[
            ("prearmed_text", "pixel_cone_detail_motion_4deg_rate_0p5_q8_g0p02")]
        self.assertGreaterEqual(coherent["contrast_retention_min"],
                                pixel["contrast_retention_min"])
        self.assertEqual(motion["contrast_retention_min"],
                         coherent["contrast_retention_min"])
        self.assertGreaterEqual(coherent["attenuation_range_max"],
                                coherent["attenuation_spread_max"])

    def test_invalid_specs_and_designs_are_rejected(self):
        with self.assertRaises(ValidationError):
            NativeDetailSpec(width=100)
        with self.assertRaises(ValidationError):
            NativeDetailSpec(event_start_seconds=.2)
        with self.assertRaises(ValidationError):
            NativeDetailDesign(str(FONT), cases=("unknown",))


if __name__ == "__main__":
    unittest.main()
