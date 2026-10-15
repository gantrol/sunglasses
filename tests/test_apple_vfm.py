"""Synthetic checks for the pinned Apple MATLAB temporal-reference port."""

import math
import unittest

import numpy as np

from sunglasses.apple_vfm import (
    AppleVFMConfig,
    AppleVFMReference,
    KERNEL_SHA256,
    STANDARD_FRAME_RATES,
    apply_matlab_mitigation,
    load_ufm_kernels,
)
from sunglasses.i18n import ValidationError


class AppleVFMReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = AppleVFMConfig(60, 200, 1265.625)

    def analyze(self, values):
        reference = AppleVFMReference(self.config)
        return [reference.process_apl(value) for value in values]

    def test_pinned_official_kernel_data_and_matlab_selection(self):
        kernels = load_ufm_kernels()
        self.assertEqual(len(kernels), 3)
        self.assertEqual([len(levels) for levels in kernels], [5, 5, 5])
        self.assertTrue(all(len(rates) == len(STANDARD_FRAME_RATES)
                            for size in kernels for rates in size))
        self.assertEqual(len(KERNEL_SHA256), 64)
        reference = AppleVFMReference(self.config)
        self.assertEqual(reference.kernel_frame_rate, 60)
        self.assertEqual(reference.kernel_size_degrees, 45)
        self.assertEqual([len(kernel) for kernel in reference.kernels], [15, 14, 9, 5, 4])
        self.assertEqual(len(reference.gamma_kernel), 61)
        self.assertAlmostEqual(reference.response_adjust, 1 / math.sqrt(60))

    def test_static_signal_is_zero_and_black_uses_explicit_numeric_floor(self):
        stable = self.analyze([.2] * 180)
        self.assertEqual(max(item.risk for item in stable), 0)
        self.assertEqual(max(item.next_mitigation for item in stable), 0)
        self.assertEqual(max(abs(item.contrast) for item in stable), 0)
        black = self.analyze([0] * 4)
        self.assertTrue(all(item.adaptation_floor_applied for item in black))
        self.assertTrue(all(math.isfinite(item.contrast) and item.risk == 0 for item in black))

    def test_step_mask_separates_hard_cut_and_single_pulse_from_repetition(self):
        hard_cut = self.analyze([.04] * 60 + [.85] * 180)
        single = self.analyze([.04] * 60 + [1] + [.04] * 179)
        repeated = self.analyze([1.0 if int(index * 10 / 60) % 2 else .04 for index in range(240)])
        for isolated in (hard_cut, single):
            self.assertGreater(max(item.unmasked_risk for item in isolated), 99)
            self.assertEqual(max(item.risk for item in isolated), 0)
            self.assertLess(max(item.normalized_step_energy for item in isolated), 1.8)
        self.assertGreater(max(item.risk for item in repeated), 99)
        self.assertGreater(max(item.normalized_step_energy for item in repeated), 1.8)

    def test_mitigation_is_causal_and_applied_one_frame_after_detection(self):
        repeated = self.analyze([1.0 if int(index * 10 / 60) % 2 else .04 for index in range(240)])
        attack = next(index for index, item in enumerate(repeated) if item.next_mitigation > 0)
        self.assertEqual(repeated[attack].applied_mitigation, 0)
        self.assertAlmostEqual(repeated[attack + 1].applied_mitigation,
                               repeated[attack].next_mitigation)
        self.assertGreater(repeated[attack + 1].applied_mitigation, 0)

    def test_reference_mapping_can_brighten_pixels_below_adaptation(self):
        frame = np.array([[[0, 0, 0], [.5, .5, .5], [1, 1, 1]]], dtype=np.float32)
        output = apply_matlab_mitigation(frame, .5, .5)
        np.testing.assert_allclose(output[0, :, 0], [.125, .25, .375])
        self.assertTrue(np.all(output[0, 0] > frame[0, 0]))
        self.assertTrue(np.all(output[0, 2] < frame[0, 2]))

    def test_process_linear_rgb_returns_identity_before_any_detection(self):
        frame = np.full((8, 12, 3), .4, dtype=np.float32)
        output, measurement = AppleVFMReference(self.config).process_linear_rgb(frame)
        np.testing.assert_array_equal(output, frame)
        self.assertEqual(measurement.applied_mitigation, 0)

    def test_invalid_parameters_and_values_are_rejected(self):
        for config in ((0, 200, 100), (60, 0, 100), (60, 200, 0)):
            with self.assertRaises(ValidationError):
                AppleVFMConfig(*config)
        reference = AppleVFMReference(self.config)
        for value in (-1, 2, math.nan):
            with self.assertRaises(ValidationError):
                reference.process_apl(value)
        with self.assertRaises(ValidationError):
            apply_matlab_mitigation(np.zeros((2, 2, 3), np.float32), 2, .5)


if __name__ == "__main__":
    unittest.main()
