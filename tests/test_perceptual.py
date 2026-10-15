"""Synthetic, headless checks for perception proxies and flash attribution."""

import math
import json
from pathlib import Path
import unittest

import numpy as np

from sunglasses.fixtures import SequenceSpec, frames
from sunglasses.i18n import ValidationError
from sunglasses.perceptual import (
    DisplayProfile,
    LayerSample,
    PairedPerceptualAnalyzer,
    PerceptualAnalyzer,
    PerceptualConfig,
    attribute_layer_changes,
    summarize_measurements,
    summarize_paired,
)
from sunglasses.perceptual_lab import run_perceptual_suite
from tests.support import workspace_temp


def solid(value, shape=(36, 64)):
    return np.full((*shape, 3), value, dtype=np.float32)


class PerceptualProxyTests(unittest.TestCase):
    def setUp(self):
        self.profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)

    def analyze(self, case, *, fps=60, duration=2):
        analyzer = PerceptualAnalyzer(self.profile)
        spec = SequenceSpec(64, 36, fps, duration)
        return [analyzer.process(frame, timestamp, index)
                for index, (timestamp, frame) in enumerate(frames(case, spec))]

    def test_profile_converts_linear_luminance_and_records_viewing_geometry(self):
        converted = self.profile.to_nits(np.array([0, .5, 1]))
        np.testing.assert_allclose(converted, [.2, 100.1, 200])
        horizontal, vertical = self.profile.visual_angle_degrees
        self.assertGreater(horizontal, vertical)
        self.assertGreater(horizontal, 40)
        self.assertLess(horizontal, 60)
        self.assertAlmostEqual(self.profile.pixels_per_degree[0] * horizontal, 1920)

    def test_static_sequence_has_no_temporal_activity(self):
        measurements = self.analyze("dark_control", duration=1)
        summary = summarize_measurements(measurements)
        self.assertEqual(summary["max_opposing_transitions_1s"], 0)
        self.assertEqual(summary["temporal_energy_max"], 0)
        self.assertEqual(summary["local_log_change_p95"], 0)
        self.assertEqual(summary["discontinuities"], 0)

    def test_hard_cut_is_separate_from_an_opposing_flash_transition(self):
        hard_cut = summarize_measurements(self.analyze("hard_cut"))
        single_flash = summarize_measurements(self.analyze("single_white_flash"))
        self.assertGreater(hard_cut["coherent_step_frames"], 0)
        self.assertEqual(hard_cut["max_opposing_transitions_1s"], 0)
        self.assertGreaterEqual(single_flash["max_opposing_transitions_1s"], 1)
        self.assertGreater(single_flash["max_opposed_area_fraction"], .9)

    def test_equal_mean_antiphase_and_fine_swaps_are_not_hidden_by_average(self):
        for case in ("antiphase_blocks", "fine_antiphase"):
            measurements = self.analyze(case)
            means = np.array([item.mean_nits for item in measurements])
            self.assertLess(float(np.ptp(means)), 1e-4, case)
            summary = summarize_measurements(measurements)
            self.assertGreater(summary["max_opposed_area_fraction"], .9, case)
            self.assertGreaterEqual(summary["max_opposing_transitions_1s"], 3, case)

    def test_saturated_red_has_an_independent_color_change_measure(self):
        summary = summarize_measurements(self.analyze("saturated_red"))
        self.assertGreater(summary["red_changed_area_fraction_max"], .99)
        self.assertGreater(summary["red_change_mean_p95"], .5)

    def test_temporal_filters_use_real_timestamps_and_reset_at_discontinuity(self):
        analyzer = PerceptualAnalyzer(self.profile)
        analyzer.process(solid(.2), 0, 0)
        changed = analyzer.process(solid(.8), 1 / 60, 1)
        self.assertAlmostEqual(changed.nyquist_hz, 30)
        reset = analyzer.process(solid(.2), 10, 8, discontinuity=True)
        self.assertEqual(reset.continuity, "discontinuity")
        self.assertEqual(reset.temporal_energy_mean, 0)
        self.assertEqual(reset.opposed_area_fraction, 0)

    def test_pair_measurement_separates_filter_recovery_from_content_change(self):
        analyzer = PairedPerceptualAnalyzer(self.profile)
        first = analyzer.process(solid(.5), solid(.3), 0, 0)
        recovery = analyzer.process(solid(.5), solid(.35), 1 / 60, 1)
        self.assertEqual(first.added_rise_nits_mean, 0)
        self.assertGreater(recovery.added_rise_nits_mean, 9)
        self.assertEqual(recovery.added_fall_nits_mean, 0)
        self.assertEqual(recovery.same_frame_brightening_nits_max, 0)
        self.assertGreater(recovery.same_frame_dimming_nits_mean, 29)
        self.assertEqual(recovery.spatial_residual_log_gradient_max, 0)
        summary = summarize_paired((first, recovery))
        self.assertGreater(summary["added_rise_nits_p95_max"], 9)
        self.assertGreater(summary["same_frame_dimming_nits_max"], 29)

        control = PairedPerceptualAnalyzer(self.profile)
        control.process(solid(.3), solid(.3), 0, 0)
        content = control.process(solid(.5), solid(.5), 1 / 60, 1)
        self.assertEqual(content.added_rise_nits_mean, 0)
        self.assertEqual(content.temporal_residual_nits_mean, 0)

        spatial = PairedPerceptualAnalyzer(self.profile)
        source = solid(.8)
        output = solid(.8)
        output[:, :output.shape[1] // 2] = .4
        edge = spatial.process(source, output, 0, 0)
        self.assertGreater(edge.spatial_residual_log_gradient_max, .1)

        local = PairedPerceptualAnalyzer(self.profile)
        source = solid(.5)
        local.process(source, source, 0, 0)
        one_pixel_darker = source.copy()
        one_pixel_darker[0, 0] = .3
        changed = local.process(source, one_pixel_darker, 1 / 60, 1)
        self.assertEqual(changed.added_fall_nits_p95, 0)
        self.assertGreater(changed.added_fall_nits_max, 39)
        self.assertAlmostEqual(changed.added_fall_area_over_1_nit, 1 / (64 * 36))

    def test_invalid_profile_and_config_are_rejected(self):
        with self.assertRaises(ValidationError):
            DisplayProfile(100, 100, 27, (1920, 1080), .6)
        with self.assertRaises(ValidationError):
            PerceptualConfig(temporal_bands_hz=(8, 2))
        with self.assertRaises(ValidationError):
            PerceptualConfig(changed_log_contrast=.2, transition_log_contrast=.1)


class LayerAttributionTests(unittest.TestCase):
    def classify(self, before, after):
        result = attribute_layer_changes((before, after), change_threshold=1)
        self.assertEqual(len(result), 1)
        return result[0]

    def test_content_algorithm_compositor_and_display_are_distinguished(self):
        content = self.classify(LayerSample(0, 10, 8, 8, 8), LayerSample(.1, 20, 18, 18, 18))
        self.assertEqual(content.origin, "source_content")
        self.assertEqual(content.observed_change, 10)

        algorithm = self.classify(LayerSample(0, 10, 6, 6, 6), LayerSample(.1, 10, 9, 9, 9))
        self.assertEqual(algorithm.origin, "algorithm_state")

        composition = self.classify(
            LayerSample(0, 10, 6, 6, 6, overlay_cloaked=False),
            LayerSample(.1, 10, 6, 10, 10, overlay_cloaked=True))
        self.assertEqual(composition.origin, "composition_or_overlay")
        self.assertIn("overlay_cloak_changed", composition.evidence)

        display = self.classify(LayerSample(0, 10, 6, 6, 6), LayerSample(.1, 10, 6, 6, 10))
        self.assertEqual(display.origin, "display_path")

    def test_cancellation_missing_output_and_discontinuity_remain_explicit(self):
        cancelled = self.classify(LayerSample(0, 0, 0), LayerSample(.1, 10, 0))
        self.assertIn("downstream_cancelled", cancelled.evidence)
        self.assertIn("unobserved_after_algorithm", cancelled.evidence)

        reset = attribute_layer_changes(
            (LayerSample(0, 0), LayerSample(.1, 10, capture_discontinuity=True)), change_threshold=1)
        self.assertEqual(reset[0].origin, "capture_discontinuity")

    def test_layer_contract_rejects_invalid_order_and_time(self):
        with self.assertRaises(ValidationError):
            attribute_layer_changes((LayerSample(1, 0), LayerSample(0, 1)), change_threshold=1)
        with self.assertRaises(ValidationError):
            attribute_layer_changes((LayerSample(0, 0, composed=0),), change_threshold=1)
        with self.assertRaises(ValidationError):
            attribute_layer_changes((LayerSample(0, 0),), change_threshold=math.nan)


class PerceptualLabTests(unittest.TestCase):
    def test_numeric_scorecard_is_reproducible_and_refuses_overwrite(self):
        profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)
        spec = SequenceSpec(32, 18, 30, 1)
        cases = ["dark_control", "antiphase_blocks", "saturated_red", "hard_cut"]
        with workspace_temp() as temporary:
            output = Path(temporary) / "perceptual"
            report = run_perceptual_suite(output, cases, spec, profile)
            stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, report)
            self.assertEqual(stored["schema_version"], 2)
            self.assertEqual([case["id"] for case in stored["cases"]], cases)
            antiphase = stored["cases"][1]["measurements"]["original"]
            self.assertGreaterEqual(antiphase["max_opposing_transitions_1s"], 3)
            self.assertGreater(antiphase["max_opposed_area_fraction"], .9)
            self.assertGreaterEqual(
                stored["cases"][1]["flash_screening"]["original"][
                    "max_combined_flashes_1s"], 1)
            self.assertEqual(stored["cases"][1]["algorithm_diagnostics"]["apple_matlab_vfm"]["max_risk"], 0)
            screen_cut = stored["cases"][3]["source_aware_comparison"]["screen_burst_100"]
            self.assertGreater(screen_cut["added_rise_nits_p95_max"], 1)
            self.assertGreater(screen_cut["mean_dimming_over_1_nit_seconds"], .25)
            held_cut = stored["cases"][3]["source_aware_comparison"]["screen_source_hold"]
            limited_cut = stored["cases"][3]["source_aware_comparison"]["screen_tile_log_limit"]
            self.assertEqual(held_cut["added_rise_nits_p95_max"], 0)
            self.assertGreater(held_cut["mean_dimming_over_1_nit_seconds"],
                               screen_cut["mean_dimming_over_1_nit_seconds"])
            self.assertLess(limited_cut["added_rise_nits_p95_max"],
                            screen_cut["added_rise_nits_p95_max"])
            tile_diagnostics = stored["cases"][3]["algorithm_diagnostics"]["screen_tile_log_limit"]
            self.assertGreater(tile_diagnostics["extended_frames"], 0)
            self.assertEqual(tile_diagnostics["infeasible_frames"], 0)
            self.assertGreater(tile_diagnostics["tile_count"], 1)
            apple_cut = stored["cases"][3]["algorithm_diagnostics"]["apple_matlab_vfm"]
            self.assertGreater(apple_cut["max_unmasked_risk"], 0)
            self.assertEqual(apple_cut["max_risk"], 0)
            self.assertEqual(len(stored["screen_reference_shader_sha256"]), 64)
            self.assertEqual(len(stored["apple_reference"]["kernel_sha256"]), 64)
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "frequency_sweep.csv").is_file())
            self.assertTrue((output / "traces/antiphase_blocks.csv").is_file())
            self.assertTrue((output / "controls/hard_cut.csv").is_file())
            self.assertFalse(any(path.suffix.lower() in (".png", ".mp4", ".mkv")
                                 for path in output.rglob("*")))
            frequencies = [row["frequency_hz"] for row in stored["frequency_sweep"]["rows"]]
            self.assertEqual(frequencies[0], .5)
            self.assertLessEqual(frequencies[-1], spec.fps / 2)
            sweep = {row["frequency_hz"]: row for row in stored["frequency_sweep"]["rows"]}
            self.assertFalse(sweep[2.0]["screening_threshold_exceeded"])
            self.assertTrue(sweep[5.0]["screening_threshold_exceeded"])
            with self.assertRaises(FileExistsError):
                run_perceptual_suite(output, cases, spec, profile)


if __name__ == "__main__":
    unittest.main()
