"""Authorized offline/headless checks for recovery parameter ablation."""

import json
from pathlib import Path
import unittest

from sunglasses.fixtures import SequenceSpec
from sunglasses.i18n import ValidationError
from sunglasses.perceptual import DisplayProfile
from sunglasses.recovery_sweep import RecoverySweepDesign, run_recovery_sweep
from tests.support import workspace_temp


class RecoverySweepTests(unittest.TestCase):
    def test_small_declared_grid_writes_reproducible_numeric_evidence(self):
        design = RecoverySweepDesign(
            cell_degrees=(2.0,),
            log_rates_per_second=(.5,),
            output_bits=(8, 10),
            tuning_cases=("hard_cut",),
            holdout_cases=("redistributed_midtones",),
        )
        spec = SequenceSpec(32, 18, 30, 1)
        profile = DisplayProfile(200, .2, 27, (1920, 1080), .65)
        with workspace_temp() as temporary:
            output = Path(temporary) / "sweep"
            report = run_recovery_sweep(output, spec, profile, design)
            stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report, stored)
            self.assertEqual(stored["candidate_count"], 7)
            self.assertEqual(len(stored["case_metrics"]), 14)
            self.assertIn("current_screen_8bit", {row["candidate"] for row in stored["case_metrics"]})
            self.assertEqual({"current", "scalar_worst", "local_map", "local_cone"},
                             {row["limiter_mode"] for row in stored["case_metrics"]})
            self.assertTrue(stored["tuning_pareto"])
            self.assertTrue(stored["holdout_pareto"])
            self.assertEqual(max(row["same_frame_brightening_nits_max"]
                                 for row in stored["case_metrics"]), 0)
            self.assertGreater(sum(row["extended_frames"] for row in stored["case_metrics"]), 0)
            self.assertTrue(all(row["infeasible_frames"] >= 0 for row in stored["case_metrics"]))
            baseline = next(row for row in stored["candidate_summaries"]
                            if row["candidate"] == "current_screen_8bit")
            self.assertIsNone(baseline["max_realized_added_log_rise_per_second"])
            self.assertIsNone(baseline["max_feasible_realized_added_log_rise_per_second"])
            self.assertEqual(baseline["max_neighbor_amount_delta"], 0)
            self.assertEqual(baseline["max_neighbor_amount_gradient_per_degree"], 0)
            self.assertEqual(baseline["max_boundary_control_contrast"], 0)
            self.assertEqual(baseline["max_spatial_ceiling_credit_log"], 0)
            self.assertIn("recovery_sweep.py", stored["source_sha256"])
            for name in ("case_metrics.csv", "candidate_summaries.csv", "control_audit.csv", "report.md"):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse(any(path.suffix.lower() in (".png", ".mp4", ".mkv", ".gif")
                                 for path in output.rglob("*")))
            with self.assertRaises(FileExistsError):
                run_recovery_sweep(output, spec, profile, design)

    def test_invalid_or_leaking_designs_are_rejected(self):
        with self.assertRaises(ValidationError):
            RecoverySweepDesign(cell_degrees=(2.0, 2.0))
        with self.assertRaises(ValidationError):
            RecoverySweepDesign(output_bits=(7,))
        with self.assertRaises(ValidationError):
            RecoverySweepDesign(limiter_modes=("unknown",))
        with self.assertRaises(ValidationError):
            RecoverySweepDesign(spatial_gradients_per_degree=(0,))
        with self.assertRaises(ValidationError):
            RecoverySweepDesign(tuning_cases=("hard_cut",), holdout_cases=("hard_cut",))
        with self.assertRaises(ValidationError):
            RecoverySweepDesign(tuning_cases=("not_a_case",))


if __name__ == "__main__":
    unittest.main()
