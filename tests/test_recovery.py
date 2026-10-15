"""Authorized synthetic/headless checks for experimental recovery policies."""

import unittest

import numpy as np

from sunglasses import demo_filter as screen_reference
from sunglasses.i18n import ValidationError
from sunglasses.perceptual import DisplayProfile
from sunglasses.recovery import (
    LocalLogRiseLimiter,
    LogRiseLimitConfig,
    LogRiseLimiter,
    PixelLogRiseLimiter,
    SourceHoldConfig,
    SourceHoldController,
    _boundary_control_contrast,
    apply_quantized_screen_curve,
    apply_quantized_screen_curve_map,
)


def encoded(value, shape=(18, 32)):
    return np.full((*shape, 3), value, dtype=np.uint8)


class SourceHoldTests(unittest.TestCase):
    def test_hold_removes_timer_recovery_until_source_returns(self):
        current = screen_reference.Controller()
        held = SourceHoldController()
        current_gains, held_gains = [], []
        values = [60] * 6 + [255] * 42 + [60] * 3
        for index, value in enumerate(values):
            _, _, load, mean = screen_reference.stats(encoded(value))
            timestamp = index / 60
            current_gains.append(current.step(load, mean, timestamp))
            gain, _ = held.step(load, mean, timestamp, coherent_step=index == 6)
            held_gains.append(gain)
        self.assertGreater(current_gains[6], .9)
        self.assertAlmostEqual(current_gains[6], held_gains[6], places=6)
        self.assertEqual(current_gains[30], 0)
        self.assertGreater(held_gains[30], .9)
        self.assertEqual(held_gains[-1], 0)

    def test_tentative_step_escalates_only_after_opposing_transition(self):
        control = SourceHoldController(SourceHoldConfig(tentative_step_scale=.35))
        dark = screen_reference.stats(encoded(60))[2:]
        bright = screen_reference.stats(encoded(255))[2:]
        control.step(*dark, 0)
        first, trace = control.step(*bright, 1 / 60, coherent_step=True)
        self.assertTrue(trace["tentative_step"])
        self.assertAlmostEqual(first, .35, places=6)
        returned, _ = control.step(*dark, 2 / 60, opposed_area=1)
        self.assertEqual(returned, 0)
        repeated, trace = control.step(*bright, 3 / 60, opposed_area=1)
        self.assertFalse(trace["tentative_step"])
        self.assertAlmostEqual(repeated, 1, places=6)

    def test_invalid_hold_inputs_are_rejected(self):
        with self.assertRaises(ValidationError):
            SourceHoldConfig(tentative_step_scale=2)
        control = SourceHoldController()
        control.step(0, .1, 1)
        with self.assertRaises(ValidationError):
            control.step(0, .1, 1)


class LogRiseLimiterTests(unittest.TestCase):
    def setUp(self):
        self.profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)

    def run_step(self, *, tiled):
        control = screen_reference.Controller()
        limiter = LogRiseLimiter(self.profile, tiled=tiled)
        rows = []
        for index, value in enumerate([60] * 6 + [255] * 90):
            frame = encoded(value)
            linear, y, load, mean = screen_reference.stats(frame)
            desired = control.step(load, mean, index / 60)
            output, trace = limiter.process(frame, linear, y, desired, control.broad,
                                            index / 60, index)
            rows.append((desired, float(np.sum(output * screen_reference.LUMA, axis=-1).mean()), trace))
            self.assertTrue(np.all(output <= linear + 1e-7))
        return rows

    def test_explicit_eight_bit_curve_is_identical_to_frozen_reference(self):
        rng = np.random.default_rng(284)
        frame = rng.integers(0, 256, (19, 37, 3), dtype=np.uint8)
        linear, y, *_ = screen_reference.stats(frame)
        for amount in (0, .13, .5, 1):
            for broad in (0, .4, 1):
                with self.subTest(amount=amount, broad=broad):
                    expected = screen_reference.LINEAR[
                        screen_reference.apply(frame, linear, y, amount, broad, 1.0)]
                    actual = apply_quantized_screen_curve(frame, linear, y, amount, broad, 8)
                    np.testing.assert_array_equal(actual, expected)

    def test_varying_eight_bit_amounts_share_exact_decoded_code_values(self):
        frame = encoded(255)
        frame[:3] = 80
        linear, y, *_ = screen_reference.stats(frame)
        amounts = np.broadcast_to(np.linspace(0, 1, frame.shape[1]), y.shape)
        actual = apply_quantized_screen_curve_map(frame, linear, y, amounts, 1, 8)
        expected = np.empty_like(actual)
        for column, amount in enumerate(amounts[0]):
            expected[:, column:column + 1] = apply_quantized_screen_curve(
                frame[:, column:column + 1], linear[:, column:column + 1],
                y[:, column:column + 1], float(amount), 1, 8)
        # Identical output codes must be identical linear values, otherwise the
        # detail envelope never recognizes that it has rejoined the base output.
        np.testing.assert_array_equal(actual, expected)

    def test_higher_depth_counterfactual_keeps_identity_and_never_brightens(self):
        rng = np.random.default_rng(913)
        frame = rng.integers(0, 256, (17, 31, 3), dtype=np.uint8)
        linear, y, *_ = screen_reference.stats(frame)
        for bits in (10, 12):
            identity = apply_quantized_screen_curve(frame, linear, y, 0, 1, bits)
            np.testing.assert_array_equal(identity, linear)
            output = apply_quantized_screen_curve(frame, linear, y, .73, .8, bits)
            self.assertTrue(np.all(output <= linear + 1e-7))
            np.testing.assert_array_equal(output[y <= .49], linear[y <= .49])

    def test_constant_amount_map_is_identical_to_scalar_curve(self):
        rng = np.random.default_rng(1207)
        frame = rng.integers(0, 256, (17, 31, 3), dtype=np.uint8)
        linear, y, *_ = screen_reference.stats(frame)
        for bits in (8, 10, 12):
            for amount in (0, .37, 1):
                with self.subTest(bits=bits, amount=amount):
                    expected = apply_quantized_screen_curve(frame, linear, y, amount, .8, bits)
                    actual = apply_quantized_screen_curve_map(
                        frame, linear, y, np.full(y.shape, amount), .8, bits)
                    np.testing.assert_array_equal(actual, expected)

    def test_local_amount_map_reduces_unnecessary_global_dimming(self):
        frame = encoded(220)
        frame[:, :frame.shape[1] // 2] = 255
        linear, y, *_ = screen_reference.stats(frame)
        config = LogRiseLimitConfig(cell_degrees=4, output_bits=8)
        scalar = LogRiseLimiter(self.profile, tiled=True, config=config)
        local = LocalLogRiseLimiter(self.profile, config=config)
        for limiter in (scalar, local):
            limiter.process(frame, linear, y, 1, 1, 0, 0)
        scalar_output, scalar_trace = scalar.process(frame, linear, y, 0, 0, 1 / 60, 1)
        local_output, local_trace = local.process(frame, linear, y, 0, 0, 1 / 60, 1)
        self.assertTrue(np.all(local_output <= linear + 1e-7))
        self.assertGreaterEqual(float(local_output.mean()), float(scalar_output.mean()))
        self.assertGreater(local_trace["max_neighbor_amount_delta"], 0)
        self.assertLess(local_trace["selected_amount_mean"], scalar_trace["selected_amount"])

    def test_spatial_majorant_bounds_angular_gradient_without_relaxing_tiles(self):
        frame = encoded(220)
        frame[:, :frame.shape[1] // 2] = 255
        linear, y, *_ = screen_reference.stats(frame)
        raw = LocalLogRiseLimiter(
            self.profile, config=LogRiseLimitConfig(cell_degrees=2, output_bits=8))
        smooth = LocalLogRiseLimiter(
            self.profile,
            config=LogRiseLimitConfig(cell_degrees=2, output_bits=8,
                                      max_amount_gradient_per_degree=.005),
        )
        for limiter in (raw, smooth):
            limiter.process(frame, linear, y, 1, 1, 0, 0)
        raw_output, raw_trace = raw.process(frame, linear, y, 0, 0, 1 / 60, 1)
        smooth_output, smooth_trace = smooth.process(frame, linear, y, 0, 0, 1 / 60, 1)
        self.assertLessEqual(smooth_trace["max_neighbor_amount_gradient_per_degree"], .005 + 1e-12)
        self.assertLess(smooth_trace["max_neighbor_amount_delta"],
                        raw_trace["max_neighbor_amount_delta"])
        self.assertGreater(smooth_trace["spatially_added_amount_mean"], 0)
        self.assertGreater(smooth_trace["spatially_regularized_tiles"], 0)
        self.assertGreaterEqual(smooth_trace["selected_amount_mean"],
                                smooth_trace["required_amount_mean"])
        self.assertTrue(np.all(smooth_output <= raw_output + 1e-7))
        self.assertLess(smooth_trace["max_envelope_overshoot_log"], 1e-8)
        self.assertEqual(smooth_trace["max_spatial_ceiling_credit_log"], 0)

    def test_local_and_scalar_limiters_match_on_uniform_frames(self):
        config = LogRiseLimitConfig(cell_degrees=2, output_bits=8)
        scalar = LogRiseLimiter(self.profile, tiled=True, config=config)
        local = LocalLogRiseLimiter(self.profile, config=config)
        for index, (value, desired, broad) in enumerate(((60, 0, 0), (255, 1, 1),
                                                         (255, 0, 0))):
            frame = encoded(value)
            linear, y, *_ = screen_reference.stats(frame)
            scalar_output, scalar_trace = scalar.process(
                frame, linear, y, desired, broad, index / 60, index)
            local_output, local_trace = local.process(
                frame, linear, y, desired, broad, index / 60, index)
            np.testing.assert_array_equal(local_output, scalar_output)
            self.assertAlmostEqual(local_trace["selected_amount_mean"],
                                   scalar_trace["selected_amount"], places=6)
            self.assertEqual(local_trace["max_neighbor_amount_delta"], 0)

    def test_pixel_envelope_limits_high_resolution_local_recovery(self):
        shape = (90, 160)
        frame = encoded(220, shape)
        frame[:, :shape[1] // 2] = 255
        local = LocalLogRiseLimiter(
            self.profile,
            config=LogRiseLimitConfig(cell_degrees=4, max_amount_gradient_per_degree=.02),
        )
        pixel = PixelLogRiseLimiter(
            self.profile,
            config=LogRiseLimitConfig(cell_degrees=4, max_amount_gradient_per_degree=.02),
        )
        outputs = {"local": [], "pixel": []}
        for index in range(12):
            moving = frame.copy()
            if index >= 2:
                offset = min((index - 2) * 5, shape[1] // 2)
                moving[:, :shape[1] // 2] = 220
                moving[:, offset:offset + shape[1] // 2] = 255
            linear, y, *_ = screen_reference.stats(moving)
            desired = 1.0 if index == 0 else 0.0
            for name, limiter in (("local", local), ("pixel", pixel)):
                output, trace = limiter.process(moving, linear, y, desired, 1,
                                                index / 60, index)
                outputs[name].append((linear, output, trace))
                self.assertTrue(np.all(output <= linear + 1e-7))
        def max_added(rows):
            result = 0.0
            for before, after in zip(rows, rows[1:]):
                source_delta = after[0] - before[0]
                output_delta = after[1] - before[1]
                result = max(result, float(np.maximum(output_delta - np.maximum(source_delta, 0), 0)
                                           .max(initial=0)))
            return result
        self.assertLessEqual(max_added(outputs["pixel"]), max_added(outputs["local"]) + 1e-7)
        self.assertTrue(all(row[2]["constraint_domain"] == "pixel_log_luminance"
                            for row in outputs["pixel"]))

    def test_invalid_amount_map_is_rejected(self):
        frame = encoded(200)
        linear, y, *_ = screen_reference.stats(frame)
        with self.assertRaises(ValidationError):
            apply_quantized_screen_curve_map(frame, linear, y, np.ones((2, 2)), 1, 8)
        invalid = np.zeros_like(y)
        invalid[0, 0] = 2
        with self.assertRaises(ValidationError):
            apply_quantized_screen_curve_map(frame, linear, y, invalid, 1, 8)

    def test_boundary_proxy_ignores_uncompressible_pixels(self):
        config = LogRiseLimitConfig(cell_degrees=4)
        amounts = np.zeros((1, 2), dtype=np.float64)
        amounts[:, 1] = 1
        for value, visible in ((128, False), (240, True)):
            frame = encoded(value)
            linear, y, *_ = screen_reference.stats(frame)
            partitions = (np.array((0, y.shape[0])), np.array((0, y.shape[1] // 2, y.shape[1])),
                          np.array(((y.shape[0] * y.shape[1] // 2,) * 2,)))
            _, _, maximum = _boundary_control_contrast(
                frame, linear, y, amounts, partitions, 1, self.profile, config)
            self.assertEqual(maximum > 0, visible)

    def test_global_and_angular_limits_extend_recovery_without_brightening(self):
        for tiled in (False, True):
            with self.subTest(tiled=tiled):
                rows = self.run_step(tiled=tiled)
                self.assertEqual(rows[30][0], 0)
                self.assertGreater(rows[30][2]["selected_amount"], 0)
                self.assertGreater(rows[30][2]["added_amount"], 0)
                self.assertFalse(any(row[2]["infeasible"] for row in rows))
                self.assertEqual(rows[30][2]["tile_count"] > 1, tiled)
                self.assertEqual(rows[-1][2]["selected_amount"], 0)
                self.assertLess(max(row[2]["max_envelope_overshoot_log"] for row in rows), 1e-8)

    def test_local_limit_is_stricter_for_a_localized_recovery(self):
        global_limit = LogRiseLimiter(self.profile, tiled=False)
        local_limit = LogRiseLimiter(self.profile, tiled=True)
        control = screen_reference.Controller()
        selected = {False: [], True: []}
        for index in range(70):
            frame = encoded(60)
            if index >= 6:
                frame[3:9, 8:18] = 255
            linear, y, load, mean = screen_reference.stats(frame)
            desired = control.step(load, mean, index / 60)
            for tiled, limiter in ((False, global_limit), (True, local_limit)):
                _, trace = limiter.process(frame, linear, y, desired, control.broad,
                                           index / 60, index)
                selected[tiled].append(trace["selected_amount"])
        self.assertGreater(max(np.subtract(selected[True], selected[False])), .01)

    def test_uncompressible_change_is_reported_and_rebased(self):
        limiter = LogRiseLimiter(
            self.profile, tiled=False,
            config=LogRiseLimitConfig(max_added_log_rise_per_second=.01),
        )
        first = encoded(0)
        first[:, : first.shape[1] // 4] = 255
        linear, y, *_ = screen_reference.stats(first)
        limiter.process(first, linear, y, 1, 1, 0, 0)
        # The source mean stays near 0.25 while spatial energy moves entirely
        # below the curve knee.  No legal amount can retain the old dim output.
        protected_value = int(np.argmin(np.abs(screen_reference.LINEAR - .25)))
        protected = encoded(protected_value)
        linear, y, *_ = screen_reference.stats(protected)
        _, trace = limiter.process(protected, linear, y, 0, 0, 1 / 60, 1)
        self.assertTrue(trace["infeasible"])
        self.assertGreater(trace["max_envelope_overshoot_log"], 0)
        _, rebased = limiter.process(protected, linear, y, 0, 0, 2 / 60, 2)
        self.assertFalse(rebased["infeasible"])

    def test_invalid_limit_configuration_and_shape_are_rejected(self):
        with self.assertRaises(ValidationError):
            LogRiseLimitConfig(max_added_log_rise_per_second=0)
        with self.assertRaises(ValidationError):
            LogRiseLimitConfig(output_bits=7)
        with self.assertRaises(ValidationError):
            LogRiseLimitConfig(max_amount_gradient_per_degree=0)
        limiter = LogRiseLimiter(self.profile, tiled=True)
        frame = encoded(60)
        linear, y, *_ = screen_reference.stats(frame)
        with self.assertRaises(ValidationError):
            limiter.process(frame.astype(np.float32), linear, y, 0, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
