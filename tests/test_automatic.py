"""Synthetic, headless automatic-controller contracts; no UI or real application use."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import math
from pathlib import Path
import unittest

import numpy as np

from sunglasses.automatic import AutoConfig, AutoFilter, FrameClock, TemporalControl, Timing, apply_auto, features, grid_shape
from sunglasses.cli import main
from sunglasses.color import luminance
from sunglasses.i18n import ValidationError
from sunglasses.fixtures import SequenceSpec, frames
from tests.support import workspace_temp


def solid(value, shape=(9, 17)):
    return np.full((*shape, 3), value, np.float32)


class AutomaticTests(unittest.TestCase):
    def test_first_frame_white_uses_current_frame_and_retains_shadows(self):
        frame = solid(1)
        frame[:, :3] = .1
        saved = frame.copy()
        output, trace = AutoFilter(AutoConfig(policy="legacy")).process(frame, 0, 0)
        self.assertEqual(trace["amount"], 1)
        self.assertEqual(trace["repeat"], 0)
        self.assertAlmostEqual(float(luminance(output).max()), .5144045, places=5)
        np.testing.assert_array_equal(frame, saved)
        np.testing.assert_array_equal(output[:, :3], frame[:, :3])

    def test_midtone_identity_zero_strength_and_no_rgb_history(self):
        np.testing.assert_array_equal(AutoFilter(AutoConfig(policy="legacy")).process(solid(.35), 0, 0)[0], solid(.35))
        frame = np.random.default_rng(5).random((9, 17, 3), dtype=np.float32)
        np.testing.assert_array_equal(AutoFilter(AutoConfig(policy="legacy", strength=0)).process(frame, 0, 0)[0], frame)
        processor = AutoFilter(AutoConfig(policy="legacy"))
        processor.process(solid(1), 0, 0)
        output, trace = processor.process(solid(0), 1 / 60, 1)
        self.assertEqual(trace["amount"], 1)
        np.testing.assert_array_equal(output, solid(0))
        for amount in (0, .25, .5, 1):
            filtered = apply_auto(frame, amount, policy="legacy")
            self.assertTrue(np.all(filtered <= frame))
            self.assertTrue(np.all(filtered >= 0))

    def test_recovery_is_time_based_and_monotonic(self):
        for fps in (30, 60, 120, 144):
            processor = AutoFilter(AutoConfig(policy="legacy"))
            processor.process(solid(1), 0, 0)
            amounts = []
            for index in range(1, fps * 4 + 1):
                _, trace = processor.process(solid(.35), index / fps, index)
                amounts.append(trace["amount"])
                expected = math.exp(-max(0, index / fps - .2) / .5)
                if expected < .001:
                    expected = 0
                self.assertAlmostEqual(trace["amount"], expected, places=10)
            self.assertTrue(np.all(np.diff(amounts) <= 0))
            self.assertEqual(amounts[-1], 0)

    def test_duplicate_does_not_advance_or_change_output(self):
        processor = AutoFilter(AutoConfig(policy="legacy"))
        first, trace = processor.process(solid(1), 0, 0)
        output, duplicate = processor.process(solid(0), 0, 0)
        np.testing.assert_array_equal(output, first)
        self.assertEqual(duplicate["continuity"], "duplicate")
        self.assertEqual(duplicate["amount"], trace["amount"])
        output[:] = 0
        self.assertTrue(np.any(processor.process(solid(0), 0, 0)[0]))

    def test_clock_marks_gaps_rollbacks_and_shape_change(self):
        for changed in ((6, 4 / 60, (9, 17, 3)), (4, .2, (9, 17, 3)),
                        (4, 0, (9, 17, 3)), (4, 4 / 60, (8, 17, 3))):
            clock = FrameClock()
            for i in range(4):
                clock.accept(i, i / 60, (9, 17, 3))
            self.assertEqual(clock.accept(*changed).continuity, "discontinuity")
            self.assertFalse(clock.intervals)
        for frame_id, timestamp in ((-1, 0), (2**64, 0), (0, float("nan")), (0, -1)):
            with self.assertRaises(ValidationError):
                FrameClock().accept(frame_id, timestamp, (1, 1, 3))

    def test_discontinuity_preserves_amount_but_clears_temporal_evidence(self):
        processor = AutoFilter(AutoConfig(policy="legacy"))
        processor.process(solid(1), 10, 0)
        _, trace = processor.process(solid(.35, (10, 18)), 0, 10)
        self.assertEqual(trace["continuity"], "discontinuity")
        self.assertEqual(trace["amount"], 1)
        self.assertEqual(trace["rise"], 0)
        self.assertEqual(trace["repeat"], 0)
        self.assertFalse(trace["latch"])
        _, trace = processor.process(solid(.35, (10, 18)), .1, 11)
        self.assertEqual(trace["amount"], 1)

    def test_blocks_cover_odd_dimensions_and_difference_before_reduction(self):
        y = np.random.default_rng(1).random((101, 131), dtype=np.float32)
        old = 1 - y
        actual = features(y, old)
        rows, cols = grid_shape(*y.shape)
        expected = []
        for row in range(rows):
            for col in range(cols):
                block = y[row * 101 // rows:(row + 1) * 101 // rows, col * 131 // cols:(col + 1) * 131 // cols]
                delta = 2 * block - 1
                expected.append([block.mean(dtype=np.float64), (block > .65).mean(),
                                 np.maximum(block - .65, 0).mean(dtype=np.float64),
                                 np.maximum(delta, 0).mean(dtype=np.float64), np.maximum(-delta, 0).mean(dtype=np.float64)])
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertTrue(np.any((actual[:, 3] > 0) & (actual[:, 4] > 0)))

    def test_peak_valley_uses_cumulative_excursions_and_expires(self):
        controller = TemporalControl(AutoConfig(policy="legacy"))
        # Every step is < .08, but full reversals exceed it. First excursion is not an event.
        values = [.25, .30, .35, .40, .35, .30, .25, .30, .35, .40, .35, .30, .25,
                  .30, .35, .40]
        for i, value in enumerate(values):
            trace = controller.update(np.array([[value, 0, 0, 0, 0]]), Timing(.02 if i else 0, i == 0, i == 0, False, ""))
            if i <= 3:
                self.assertEqual(trace["repeat"], 0)
        self.assertEqual(trace["repeat"], 1)
        trace = controller.update(np.array([[.40, 0, 0, 0, 0]]), Timing(.8, False, False, False, ""))
        self.assertEqual(trace["repeat"], 0)

    def test_hysteresis_and_refreshed_hold(self):
        controller = TemporalControl(AutoConfig(policy="legacy"))
        def update(excess, dt, first=False):
            return controller.update(np.array([[.35, 0, excess, 0, 0]]), Timing(dt, first, first, False, ""))
        self.assertFalse(update(.03, 0, True)["latch"])  # score .5 cannot enter
        self.assertTrue(update(.05, .1)["latch"])
        self.assertTrue(update(.03, .1)["latch"])        # score .5 stays latched
        self.assertEqual(update(.05, .1)["hold_remaining"], .2)
        trace = update(0, .1)
        self.assertFalse(trace["latch"])
        self.assertEqual(trace["amount"], 1)
        self.assertAlmostEqual(update(0, .15)["amount"], math.exp(-.05 / .5))

    def test_red_failure_and_repeat_ablation_are_explicit(self):
        controllers = [AutoFilter(AutoConfig(policy="legacy")), AutoFilter(AutoConfig(policy="legacy", repeat_enabled=False))]
        max_repeat = [0, 0]
        for index in range(16):
            frame = solid(0)
            frame[..., 0] = 1 if index % 2 else .04
            for j, controller in enumerate(controllers):
                output, trace = controller.process(frame, index / 30, index)
                max_repeat[j] = max(max_repeat[j], trace["repeat"])
                if index % 2:
                    self.assertLess(float(luminance(frame).max() - luminance(output).max()), .0003)
        self.assertEqual(max_repeat, [1, 0])

    def test_repeat_branch_gain_and_known_motion_and_fine_pattern_failures(self):
        # 64x32 gives even 2x2 blocks, so fine swaps preserve each block mean.
        spec = SequenceSpec(64, 32, 30, 2)
        for case, should_trigger in (("midtone_pulses", True), ("moving_pattern", True), ("fine_antiphase", False)):
            controller = AutoFilter(AutoConfig(policy="legacy"))
            ablated = AutoFilter(AutoConfig(policy="legacy", repeat_enabled=False))
            amounts, repeats, without = [], [], []
            for i, (t, frame) in enumerate(frames(case, spec)):
                _, trace = controller.process(frame, t, i)
                _, other = ablated.process(frame, t, i)
                amounts.append(trace["amount"])
                repeats.append(trace["repeat"])
                without.append(other["amount"])
            self.assertEqual(max(repeats), 1 if should_trigger else 0, case)
            self.assertEqual(max(amounts), 1 if should_trigger else 0, case)
            self.assertEqual(max(without), 0, case)

    def test_cli_report_trace_annotations_and_overwrite_protection(self):
        with workspace_temp() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output = Path(temporary) / "auto"
            args = ["--language", "zh-CN", "auto-suite", "--policy", "legacy", "--case", "midtone_recovery", "--width", "16", "--height", "16",
                    "--fps", "30", "--duration", "4", "--output", str(output)]
            self.assertEqual(main(args), 0)
            report = json.loads((output / "summary.json").read_text())
            case = report["cases"][0]
            self.assertEqual(case["event_frame"], 12)
            self.assertAlmostEqual(case["recovery_to_5_percent_seconds"], 1.7)
            self.assertLess(case["metrics"]["automatic"]["outside_event_mean_abs_y_error"],
                            case["metrics"]["fixed_strong"]["outside_event_mean_abs_y_error"])
            self.assertTrue((output / "traces/midtone_recovery.csv").exists())
            self.assertEqual(main(args), 2)
            self.assertEqual(main([*args[:-1], str(Path(temporary) / "bad"), "--release", "0"]), 2)

    def test_invalid_automatic_parameters(self):
        for parameters in ({"strength": 2}, {"hold_seconds": -1}, {"release_seconds": 0},
                           {"release_seconds": float("nan")}, {"repeat_enabled": 1}):
            with self.assertRaises(ValidationError):
                AutoConfig(policy="legacy", **parameters)
        with self.assertRaises(ValidationError):
            AutoFilter(AutoConfig(policy="legacy")).process(np.ones((4, 3), np.float32), 0, 0)


class HighlightOnlyTests(unittest.TestCase):
    def test_small_bright_regions_are_identity_and_area_gate_is_weighted(self):
        processor = AutoFilter()
        # Odd, tall partitions have unequal sizes; a block maximum is not frame area.
        shape = (131, 35)
        for index, fraction in enumerate((0, .001, .009, .011, .02, .035, .041, .10, .001)):
            frame = solid(.35, shape)
            count = int(np.prod(shape) * fraction)
            frame.reshape(-1, 3)[:count] = 1
            output, trace = processor.process(frame, index / 60, index)
            area = count / np.prod(shape)
            self.assertAlmostEqual(trace["bright_area"], area, places=12)
            if area <= .01:
                self.assertEqual(trace["amount"], 0)
                np.testing.assert_array_equal(output, frame)
            else:
                self.assertGreater(trace["amount"], 0)
                self.assertLess(output.reshape(-1, 3)[0, 0], 1)
            np.testing.assert_array_equal(output.reshape(-1, 3)[count:], frame.reshape(-1, 3)[count:])

    def test_bright_frame_then_normal_frame_has_no_hold_tail(self):
        processor = AutoFilter()
        output, trace = processor.process(solid(1), 0, 0)
        self.assertEqual(trace["amount"], 1)
        self.assertAlmostEqual(float(output[0, 0, 0]), .7854542, places=6)
        for i, value in enumerate((.1, .35, .65), 1):
            frame = solid(value)
            output, trace = processor.process(frame, i / 60, i)
            self.assertEqual(trace["amount"], 0)
            np.testing.assert_array_equal(output, frame)
        np.testing.assert_array_equal(apply_auto(solid(1), 0), solid(1))

    def test_moving_midtones_remain_exact_and_uniform_ramp_stays_monotone(self):
        processor = AutoFilter()
        for i, (t, frame) in enumerate(frames("moving_pattern", SequenceSpec(64, 32, 60, 4))):
            output, trace = processor.process(frame, t, i)
            self.assertEqual(trace["amount"], 0)
            np.testing.assert_array_equal(output, frame)
        processor = AutoFilter()
        values = np.linspace(.01, 1, 1001, dtype=np.float32)
        outputs = [float(processor.process(solid(v), i / 120, i)[0][0, 0, 0]) for i, v in enumerate(values)]
        self.assertGreaterEqual(min(np.diff(outputs)), 0)
        self.assertLessEqual(max(np.diff(outputs)), max(np.diff(values)) + 1e-6)

    def test_nonhighlight_pixels_remain_exact_even_when_triggered(self):
        frame = solid(1, (25, 40))
        frame[:15] = np.random.default_rng(6).uniform(0, .65, (15, 40, 3)).astype(np.float32)
        output, trace = AutoFilter().process(frame, 0, 0)
        self.assertEqual(trace["amount"], 1)
        np.testing.assert_array_equal(output[:15], frame[:15])
        self.assertTrue(np.all(output <= frame))
        for value in ("", "unknown", None):
            with self.assertRaises(ValidationError):
                AutoConfig(policy=value)

    def test_large_region_crossing_knee_has_no_hard_area_jump(self):
        frame = solid(.64999, (100, 100))
        frame[:2] = 1
        next_frame = frame.copy()
        next_frame[2:] = .65001
        processor = AutoFilter()
        first, _ = processor.process(frame, 0, 0)
        second, _ = processor.process(next_frame, 1/60, 1)
        # Raw area counting caused a 0.21455 step here; continuous load removes
        # that discontinuity. This is not a universal temporal Lipschitz bound.
        self.assertLess(float(np.abs(second - first).max()), .001)


if __name__ == "__main__":
    unittest.main()
