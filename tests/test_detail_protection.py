"""Authorized offline checks for stable high-frequency detail protection."""

import unittest
import math

import numpy as np

from sunglasses import demo_filter as screen_reference
from sunglasses.detail_protection import (
    StableDetailConfig,
    StableDetailProtector,
    _advance_strength,
    _limited_mask,
    _motion_match,
    _translate,
)
from sunglasses.i18n import ValidationError
from sunglasses.perceptual import DisplayProfile


class StableDetailProtectionTests(unittest.TestCase):
    def setUp(self):
        self.profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)

    def frame(self, with_line=True):
        encoded = np.full((90, 160, 3), 50, dtype=np.uint8)
        if with_line:
            encoded[70:73, 35:125] = 245
        linear, *_ = screen_reference.stats(encoded)
        return encoded, linear

    def test_preexisting_detail_is_armed_before_filter_dimming(self):
        protector = StableDetailProtector(self.profile)
        encoded, source = self.frame()
        filtered = source.copy()
        trace = None
        for index in range(30):
            if index >= 20:
                filtered = source.copy()
                filtered[70:73, 35:125] = screen_reference.LINEAR[150]
            output, trace = protector.process(encoded, source, filtered, index / 60, index)
            self.assertTrue(np.all(output <= source + 1e-7))
        self.assertGreater(trace["protected_area_fraction"], 0)
        self.assertGreater(float(output[71, 80, 0]), float(filtered[71, 80, 0]))

    def test_new_detail_does_not_arm_while_already_dimmed(self):
        protector = StableDetailProtector(self.profile)
        dark_encoded, dark = self.frame(False)
        for index in range(15):
            protector.process(dark_encoded, dark, dark, index / 60, index)
        encoded, source = self.frame(True)
        filtered = source.copy()
        filtered[70:73, 35:125] = screen_reference.LINEAR[150]
        outputs = []
        for index in range(15, 45):
            output, trace = protector.process(encoded, source, filtered, index / 60, index)
            outputs.append(output)
        self.assertEqual(trace["maximum_strength"], 0)
        self.assertEqual(trace["armed_while_dimmed_pixels"], 0)
        np.testing.assert_array_equal(outputs[-1], filtered)

    def test_invalid_configuration_and_values_are_rejected(self):
        with self.assertRaises(ValidationError):
            StableDetailConfig(maximum_area_fraction=.5)
        with self.assertRaises(ValidationError):
            StableDetailConfig(motion_propagation=1)
        encoded, source = self.frame()
        protector = StableDetailProtector(self.profile)
        with self.assertRaises(ValidationError):
            protector.process(encoded, source, source + 1, 0, 0)

    def test_area_limit_retains_previously_protected_pixels(self):
        eligible = np.ones((2, 3), dtype=bool)
        score = np.arange(6, dtype=np.float64).reshape(2, 3)
        preferred = np.zeros_like(eligible)
        preferred[0, :2] = True
        selected = _limited_mask(eligible, score, 2, preferred)
        np.testing.assert_array_equal(selected, preferred)

    def test_area_overflow_can_reject_or_rank_a_detail_batch(self):
        encoded = np.full((90, 160, 3), 50, dtype=np.uint8)
        encoded[::4, ::4] = 245
        source, *_ = screen_reference.stats(encoded)
        for policy, expected_protection in (("reject_batch", False), ("ranked_pixels", True)):
            with self.subTest(policy=policy):
                protector = StableDetailProtector(
                    self.profile, StableDetailConfig(area_overflow_policy=policy))
                for index in range(30):
                    _, trace = protector.process(encoded, source, source, index / 60, index)
                self.assertGreater(trace["overflow_rejected_pixels"], 0)
                self.assertEqual(trace["maximum_strength"] > 0, expected_protection)
                self.assertLessEqual(trace["protected_area_fraction"], .05)

    def test_strength_handoff_never_exceeds_the_shared_budget(self):
        previous = np.array((1., 1., 1., 0., 0., 0.))
        target = np.array((0., 0., 0., 1., 1., 1.))
        advanced = _advance_strength(previous, target, .5, .25, 3)
        self.assertLessEqual(float(advanced.sum()), 3)
        np.testing.assert_allclose(advanced[:3], .75)
        np.testing.assert_allclose(advanced[3:], .25)

    def test_supported_small_translation_inherits_existing_protection(self):
        base = np.full((90, 160, 3), 50, dtype=np.uint8)
        base[60:76, 40:42] = 245
        base[60:76, 70:72] = 245
        base[60:76, 100:102] = 245
        shifted = np.full_like(base, 50)
        shifted[:, 1:] = base[:, :-1]
        base_source, *_ = screen_reference.stats(base)
        shifted_source, *_ = screen_reference.stats(shifted)
        filtered = shifted_source.copy()
        detail = shifted[..., 0] == 245
        filtered[detail] = screen_reference.LINEAR[150]
        static = StableDetailProtector(self.profile)
        moving = StableDetailProtector(
            self.profile, StableDetailConfig(motion_propagation=True))
        for index in range(30):
            static.process(base, base_source, base_source, index / 60, index)
            moving.process(base, base_source, base_source, index / 60, index)
        static_output, _ = static.process(shifted, shifted_source, filtered, .5, 30)
        moving_output, trace = moving.process(shifted, shifted_source, filtered, .5, 30)
        self.assertEqual(trace["motion_shift_pixels"], (1, 0))
        self.assertGreater(trace["motion_propagated_pixels"], 0)
        self.assertGreater(trace["motion_best_support"], trace["motion_zero_support"])
        self.assertGreater(float(moving_output[detail].mean()),
                           float(static_output[detail].mean()))
        self.assertTrue(np.all(moving_output <= shifted_source + 1e-7))
        self.assertLessEqual(trace["max_envelope_overshoot_log"], 1e-12)
        self.assertLessEqual(trace["max_realized_added_log_rise_per_second"], .5 + 1e-7)

    def test_sparse_motion_scoring_matches_full_array_reference(self):
        rng = np.random.default_rng(130)
        for shift_y, shift_x in ((0, 0), (-2, 3), (2, -3), (1, 1)):
            protected = rng.random((15, 23)) < .2
            previous = rng.random(protected.shape)
            current = _translate(previous, shift_y, shift_x, 0.0)
            candidate = _translate(protected, shift_y, shift_x, False)
            # Occlusion and cropped edges remain part of the support denominator.
            candidate[4:7, 9:14] = False
            scores = []
            for dy in range(-2, 3):
                for dx in range(-3, 4):
                    moved = _translate(protected, dy, dx, False)
                    moved_log = _translate(previous, dy, dx, np.nan)
                    matched = (moved & candidate & np.isfinite(moved_log) &
                               (np.abs(moved_log - current) <= .03))
                    fraction = float(matched.sum() / protected.sum())
                    error = (float(np.abs(moved_log[matched] - current[matched]).mean())
                             if np.any(matched) else math.inf)
                    scores.append(((fraction, -error, -abs(dx) - abs(dy), dy, dx), dy, dx))
            zero = next(score[0][0] for score in scores if score[1:] == (0, 0))
            score, dy, dx = max(scores)
            expected = (dy, dx, zero, score[0])
            if score[0] < .6 or score[0] < zero + .1:
                expected = (0, 0, zero, score[0])
            self.assertEqual(_motion_match(protected, previous, current, candidate,
                                           2, 3, .03, .6, .1), expected)

    def test_detail_envelope_does_not_modify_untouched_pixels_or_hide_their_rate(self):
        protector = StableDetailProtector(self.profile)
        encoded, source = self.frame()
        for index in range(30):
            protector.process(encoded, source, source, index / 60, index)
        filtered = source.copy()
        filtered[70:73, 35:125] = screen_reference.LINEAR[150]
        filtered[:20] = 0
        protector.process(encoded, source, filtered, .5, 30)
        # A base-only step must bypass detail processing, including quantization.
        filtered[:20] = source[:20]
        output, trace = protector.process(encoded, source, filtered, 31 / 60, 31)
        np.testing.assert_array_equal(output[:20], filtered[:20])
        self.assertGreater(trace["max_realized_added_log_rise_per_second"], .5)
        self.assertLessEqual(trace["max_envelope_added_log_rise_per_second"], .5 + 1e-7)
        self.assertGreater(trace["envelope_active_area_fraction"], 0)
        self.assertLess(trace["envelope_active_area_fraction"], .05)


if __name__ == "__main__":
    unittest.main()
