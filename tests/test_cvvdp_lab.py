"""Headless checks for ColorVideoVDP inputs without requiring the optional metric."""

import unittest

import numpy as np

from sunglasses.cvvdp_lab import (
    ColorVideoVDPDesign,
    _detail_geometry,
    candidate_sequence,
    detail_frames,
    prepare_case,
    summarize_cvvdp_stats,
)
from sunglasses.fixtures import SequenceSpec
from sunglasses.i18n import ValidationError
from sunglasses.perceptual import DisplayProfile


class ColorVideoVDPLabTests(unittest.TestCase):
    def setUp(self):
        self.spec = SequenceSpec(160, 90, 30, 1)
        self.profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)

    def test_detail_cases_are_finite_and_keep_declared_regions(self):
        geometry = _detail_geometry(self.spec)
        self.assertGreater(geometry["hud"].sum(), 0)
        self.assertGreater(geometry["subtitle"].sum(), 0)
        self.assertGreater(geometry["subtitle_surround"].sum(), 0)
        self.assertGreater(float(geometry["dense_detail"].mean()), .05)
        for case in ("hud_flash", "subtitle_flash", "dense_detail_flash", "moving_highlight"):
            sequence = list(detail_frames(case, self.spec))
            self.assertEqual(len(sequence), self.spec.frame_count)
            values = np.stack([frame for _, frame in sequence])
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(float(values.min()), 0)
            self.assertLessEqual(float(values.max()), 1)

    def test_candidate_sequences_never_brighten_and_keep_spatial_credit_zero(self):
        prepared = prepare_case("moving_highlight", self.spec)
        for candidate in ("identity", "current_screen_8bit", "scalar_rate_0p5_q8",
                          "local_cone_4deg_rate_0p5_q8_g0p02",
                          "pixel_cone_4deg_rate_0p5_q8_g0p02",
                          "pixel_cone_detail_ranked_4deg_rate_0p5_q8_g0p02",
                          "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02"):
            with self.subTest(candidate=candidate):
                outputs, traces = candidate_sequence(candidate, prepared, self.profile)
                for item, output in zip(prepared, outputs):
                    self.assertTrue(np.all(output <= item[3] + 1e-7))
                self.assertEqual(max(trace.get("max_spatial_ceiling_credit_log", 0)
                                     for trace in traces), 0)
                self.assertEqual(sum(trace.get("detail_armed_while_dimmed_pixels", 0)
                                     for trace in traces), 0)

    def test_backend_statistics_are_flattened_with_named_channels(self):
        stats = {
            "Q_per_ch": np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2),
            "rho_band": np.array((2.0, .5)),
        }
        summary = summarize_cvvdp_stats(stats)
        self.assertEqual(summary["rho_band_cycles_per_degree"], [2.0, .5])
        self.assertEqual(set(summary["difference_max_by_channel"]),
                         {"achromatic_sustained", "red_green", "yellow_violet",
                          "achromatic_transient"})
        self.assertEqual(len(summary["difference_max_by_spatial_band"]), 2)

    def test_coherent_detail_policy_preserves_sparse_details_and_rejects_dense_batch(self):
        coherent = "pixel_cone_detail_coherent_4deg_rate_0p5_q8_g0p02"
        base = "pixel_cone_4deg_rate_0p5_q8_g0p02"

        dense = prepare_case("dense_detail_flash", self.spec)
        base_outputs, _ = candidate_sequence(base, dense, self.profile)
        coherent_outputs, coherent_traces = candidate_sequence(coherent, dense, self.profile)
        np.testing.assert_array_equal(np.stack(coherent_outputs), np.stack(base_outputs))
        self.assertEqual(max(trace.get("detail_protected_area_fraction", 0)
                             for trace in coherent_traces), 0)
        self.assertGreater(max(trace.get("detail_overflow_rejected_pixels", 0)
                               for trace in coherent_traces), 0)
        self.assertEqual(max(trace.get("detail_protection_budget_overshoot", 0)
                             for trace in coherent_traces), 0)
        _, ranked_traces = candidate_sequence(
            "pixel_cone_detail_ranked_4deg_rate_0p5_q8_g0p02", dense, self.profile)
        self.assertEqual(max(trace.get("detail_protection_budget_overshoot", 0)
                             for trace in ranked_traces), 0)

        subtitle = prepare_case("subtitle_flash", self.spec)
        base_outputs, _ = candidate_sequence(base, subtitle, self.profile)
        coherent_outputs, coherent_traces = candidate_sequence(coherent, subtitle, self.profile)
        self.assertGreater(max(trace.get("detail_protected_area_fraction", 0)
                               for trace in coherent_traces), 0)
        self.assertTrue(any(np.any(coherent_output > base_output)
                            for coherent_output, base_output in zip(coherent_outputs, base_outputs)))

    def test_invalid_design_and_case_are_rejected(self):
        with self.assertRaises(ValidationError):
            ColorVideoVDPDesign(cases=("unknown",))
        with self.assertRaises(ValidationError):
            ColorVideoVDPDesign(candidates=("unknown",))
        with self.assertRaises(ValidationError):
            next(detail_frames("hard_cut", self.spec))


if __name__ == "__main__":
    unittest.main()
