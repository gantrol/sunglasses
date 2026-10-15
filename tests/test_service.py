"""Offline public API and worker boundary tests. No desktop UI is started."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from sunglasses import (AutoConfig, ExperimentRequest, FilterConfig, ImageRequest, JobCancelled, JobContext,
                        experiment_preset, process_image, run_experiment)
from sunglasses.fixtures import SequenceSpec
from sunglasses.i18n import ValidationError
from sunglasses.image_io import read_sdr_png, save_sdr_png
from tests.support import workspace_temp


class ServiceTests(unittest.TestCase):
    def test_public_image_result_progress_and_source_preservation(self):
        with workspace_temp() as folder:
            source, output = Path(folder) / "source.png", Path(folder) / "output.png"
            save_sdr_png(source, np.ones((16, 16, 3), np.float32))
            original = source.read_bytes()
            updates = []
            result = process_image(ImageRequest(source, output, FilterConfig(strength=.5)), JobContext(progress=updates.append))
            self.assertEqual(result.output, output)
            self.assertIsNone(result.report)
            self.assertEqual([p.completed for p in updates], [0, 1, 2])
            self.assertLess(float(read_sdr_png(output).max()), 1)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                process_image(ImageRequest(source, output))

    def test_cancel_before_start_and_during_sequence(self):
        with workspace_temp() as folder:
            output = Path(folder) / "cancelled"
            request = ExperimentRequest(output, sequence=SequenceSpec(16, 16, 30, .5), cases=("dark_control",))
            with self.assertRaises(JobCancelled):
                run_experiment(request, JobContext(cancelled=lambda: True))
            self.assertFalse(output.exists())
            updates = []
            with self.assertRaises(JobCancelled):
                run_experiment(request, JobContext(cancelled=lambda: len(updates) >= 3, progress=updates.append))
            self.assertFalse((output / "report.md").exists())

    def test_modes_presets_and_source_provenance(self):
        with workspace_temp() as folder:
            for mode in ("static", "automatic"):
                result = run_experiment(ExperimentRequest(Path(folder) / mode, mode=mode,
                    sequence=SequenceSpec(16, 16, 30, .2), cases=("single_white_flash",), automatic=AutoConfig(strength=.5)))
                self.assertEqual(result.case_count, 1)
                self.assertTrue(result.report.exists())
                report = json.loads((result.output / "summary.json").read_text())
                self.assertTrue(all(len(value) == 64 for value in report["source_sha256"].values()))
        self.assertIn("moving_pattern", experiment_preset(Path("unused")).cases)
        with self.assertRaises(ValidationError):
            experiment_preset(Path("unused"), "unknown")
        with self.assertRaises(ValidationError):
            ExperimentRequest(Path("unused"), mode="unknown")


class WorkerTests(unittest.TestCase):
    def commands(self):
        commands = [[sys.executable, "-m", "sunglasses.worker"]]
        if os.environ.get("SUNGLASSES_WORKER"):
            commands.append([os.environ["SUNGLASSES_WORKER"]])
            if os.environ.get("SUNGLASSES_WORKER_HOST"):
                commands.append([os.environ["SUNGLASSES_WORKER_HOST"], os.environ["SUNGLASSES_WORKER"]])
        return commands

    def execute(self, command, *, cancel=False, cwd=None):
        environment = dict(os.environ)
        if os.environ.get("SUNGLASSES_WORKER") in command:
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding="utf-8", cwd=cwd, env=environment) as process:
            try:
                if cancel:
                    process.stdin.write("cancel\n")
                    process.stdin.flush()
                # Inputs keep stdin open; tiny jobs emit < pipe capacity. Timeout is enforced.
                code = process.wait(timeout=60)
                stdout, stderr = process.stdout.read(), process.stderr.read()
            except BaseException:
                process.kill()
                process.wait()
                raise
        return code, stdout, stderr

    def test_worker_png_unicode_paths_and_error_protocol(self):
        for executable in self.commands():
            with self.subTest(executable=executable), workspace_temp() as folder:
                source, output = Path(folder) / "输入 空格.png", Path(folder) / "新文件夹" / "输出.png"
                save_sdr_png(source, np.ones((16, 16, 3), np.float32))
                command = [*executable, "--language", "zh-CN", "image", str(source), str(output)]
                code, stdout, stderr = self.execute(command, cwd=folder if len(executable) == 1 else None)
                self.assertEqual(code, 0, (stdout, stderr))
                self.assertIn("ready\t1", stdout)
                self.assertIn("done\t" + str(output), stdout)
                self.assertTrue(output.exists())
                code, stdout, stderr = self.execute(command, cwd=folder if len(executable) == 1 else None)
                self.assertEqual(code, 2)
                self.assertIn("error\t", stdout)
                self.assertIn("输出已存在", stdout)

    def test_worker_quick_report_and_cooperative_cancel(self):
        for executable in self.commands():
            with self.subTest(executable=executable), workspace_temp() as folder:
                output = Path(folder) / "quick"
                code, stdout, stderr = self.execute([*executable, "experiment", str(output), "--preset", "quick"],
                                                   cwd=folder if len(executable) == 1 else None)
                self.assertEqual(code, 0, (stdout, stderr))
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(len(summary["cases"]), 4)
                self.assertTrue(summary["source_sha256"])
                cancelled = Path(folder) / "cancelled"
                code, stdout, stderr = self.execute([*executable, "experiment", str(cancelled), "--preset", "full"], cancel=True)
                self.assertEqual(code, 3, (stdout, stderr))
                self.assertIn("cancelled\t", stdout)
                self.assertFalse((cancelled / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
