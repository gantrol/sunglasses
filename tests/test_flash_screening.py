"""Synthetic, headless checks for conservative flash screening."""

import unittest
from dataclasses import replace

import numpy as np

from sunglasses.flash_screening import (
    FlashScreeningAnalyzer,
    FlashScreeningConfig,
    compare_flash_screening,
    summarize_flash_screening,
)
from sunglasses.i18n import ValidationError
from sunglasses.perceptual import DisplayProfile


def solid(rgb, shape=(60, 100)):
    values = np.asarray(rgb, dtype=np.float32)
    return np.broadcast_to(values, (*shape, 3)).copy()


class FlashScreeningTests(unittest.TestCase):
    def setUp(self):
        self.profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)

    def test_cut_arms_once_and_return_completes_one_general_flash(self):
        analyzer = FlashScreeningAnalyzer(self.profile)
        first = analyzer.process(solid((0, 0, 0)), 0, 0)
        cut = analyzer.process(solid((1, 1, 1)), 1 / 60, 1)
        returned = analyzer.process(solid((0, 0, 0)), 2 / 60, 2)
        self.assertFalse(first.general_flash_event)
        self.assertFalse(cut.general_flash_event)
        self.assertTrue(returned.general_flash_event)
        self.assertEqual(returned.general_flashes_1s, 1)
        self.assertEqual(returned.general_pair_area_fraction, 1)

    def test_four_non_overlapping_cycles_exceed_rolling_threshold(self):
        analyzer = FlashScreeningAnalyzer(self.profile)
        measurements = [analyzer.process(solid((0, 0, 0)), 0, 0)]
        for index in range(1, 9):
            value = 1 if index % 2 else 0
            measurements.append(analyzer.process(
                solid((value, value, value)), index / 10, index))
        summary = summarize_flash_screening(measurements)
        self.assertEqual(summary["general_flash_events"], 4)
        self.assertEqual(summary["max_general_flashes_1s"], 4)
        self.assertTrue(summary["screening_threshold_exceeded"])

    def test_small_local_pair_stays_below_angular_area_threshold(self):
        analyzer = FlashScreeningAnalyzer(self.profile)
        dark = solid((0, 0, 0))
        local = dark.copy()
        local[28:32, 48:52] = 1
        analyzer.process(dark, 0, 0)
        analyzer.process(local, 1 / 60, 1)
        returned = analyzer.process(dark, 2 / 60, 2)
        self.assertGreater(returned.general_pair_area_fraction, 0)
        self.assertLess(returned.general_pair_field_fraction_max, .25)
        self.assertFalse(returned.general_flash_event)

    def test_equal_luminance_red_change_has_an_independent_pair(self):
        analyzer = FlashScreeningAnalyzer(self.profile)
        gray = solid((.2126, .2126, .2126))
        red = solid((1, 0, 0))
        analyzer.process(gray, 0, 0)
        analyzer.process(red, 1 / 60, 1)
        returned = analyzer.process(gray, 2 / 60, 2)
        self.assertFalse(returned.general_flash_event)
        self.assertTrue(returned.red_flash_event)
        self.assertEqual(returned.red_pair_area_fraction, 1)

    def test_monotonic_ramp_and_discontinuity_do_not_form_a_pair(self):
        analyzer = FlashScreeningAnalyzer(self.profile)
        measurements = []
        for index, value in enumerate(np.linspace(0, 1, 11)):
            measurements.append(analyzer.process(
                solid((value, value, value)), index / 60, index))
        self.assertEqual(summarize_flash_screening(measurements)["general_flash_events"], 0)
        reset = analyzer.process(solid((0, 0, 0)), 1, 20, discontinuity=True)
        self.assertEqual(reset.continuity, "discontinuity")
        self.assertFalse(reset.combined_flash_event)

    def test_duplicate_and_invalid_configuration_are_explicit(self):
        analyzer = FlashScreeningAnalyzer(self.profile)
        first = analyzer.process(solid((0, 0, 0)), 0, 0)
        duplicate = analyzer.process(solid((0, 0, 0)), 0, 0)
        self.assertEqual(first.continuity, "first")
        self.assertEqual(duplicate.continuity, "duplicate")
        with self.assertRaises(ValidationError):
            FlashScreeningConfig(area_fraction=0)
        with self.assertRaises(ValidationError):
            FlashScreeningConfig(maximum_flashes_per_second=3.0)

    def test_pair_completion_direction_exposes_coincident_opposite_events(self):
        source = FlashScreeningAnalyzer(self.profile)
        output = FlashScreeningAnalyzer(self.profile)
        source_frames, output_frames = [], []
        # Both complete a flash on frame 2; only the output completes a fall.
        for index, (left, right) in enumerate(((.6, 0), (.4, .2), (.6, 0))):
            source_frames.append(source.process(solid((left,) * 3), index / 60, index))
            output_frames.append(output.process(solid((right,) * 3), index / 60, index))
        comparison = compare_flash_screening(source_frames, output_frames)
        self.assertEqual(comparison["added_combined_event_frames"], 0)
        self.assertEqual(comparison["added_general_fall_event_frames"], 1)
        self.assertEqual(comparison["added_general_rise_event_frames"], 0)
        self.assertEqual(source_frames[-1].general_rise_pair_field_fraction_max, 1)
        self.assertEqual(output_frames[-1].general_fall_pair_field_fraction_max, 1)
        self.assertEqual(output_frames[-1].general_fall_transition_area_fraction, 1)
        # Repeated delivery of an event does not create a second event.
        source_frames.append(source.process(solid((.6,) * 3), 2 / 60, 2))
        output_frames.append(output.process(solid((0,) * 3), 2 / 60, 2))
        self.assertEqual(compare_flash_screening(source_frames, output_frames), comparison)
        with self.assertRaises(ValidationError):
            compare_flash_screening(source_frames, output_frames[:-1])
        with self.assertRaises(ValidationError):
            compare_flash_screening(source_frames, [
                *output_frames[:-1], replace(output_frames[-1], timestamp=.2)])
        with self.assertRaises(ValidationError):
            compare_flash_screening(source_frames, [
                *output_frames[:-1], replace(output_frames[-1], continuity="contiguous")])

    def test_shifted_pair_is_reported_as_a_timing_mismatch(self):
        source = FlashScreeningAnalyzer(self.profile)
        output = FlashScreeningAnalyzer(self.profile)
        source_frames, output_frames = [], []
        for index, (left, right) in enumerate(((0, 0), (1, .8), (1, .8), (0, 0))):
            source_frames.append(source.process(solid((left,) * 3), index / 60, index))
            output_frames.append(output.process(solid((right,) * 3), index / 60, index))
        comparison = compare_flash_screening(source_frames, output_frames)
        self.assertEqual(comparison["added_combined_event_frames"], 0)
        # Independent shifted sequence: compare by time, never zip just event lists.
        delayed = FlashScreeningAnalyzer(self.profile)
        delayed_frames = [delayed.process(solid((value,) * 3), index / 60, index)
                          for index, value in enumerate((0, .8, 0, 0))]
        comparison = compare_flash_screening(source_frames, delayed_frames)
        self.assertEqual(comparison["added_combined_event_frames"], 1)
        self.assertEqual(comparison["removed_combined_event_frames"], 1)


if __name__ == "__main__":
    unittest.main()
