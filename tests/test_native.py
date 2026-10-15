"""Headless shader parity, enabled with SUNGLASSES_COMPUTE pointing to the built runner."""

import os
from pathlib import Path
import struct
import subprocess
import unittest

import numpy as np

from sunglasses.filter import FilterConfig, compress_highlights
from tests.support import workspace_temp


@unittest.skipUnless(os.environ.get("SUNGLASSES_COMPUTE"), "Native runner not configured")
class NativeParityTests(unittest.TestCase):
    def test_warp_and_requested_hardware_against_reference(self):
        executable = Path(os.environ["SUNGLASSES_COMPUTE"]).resolve()
        backends = ["warp"]
        if os.environ.get("SUNGLASSES_TEST_HARDWARE") == "1":
            backends.append("hardware")
        # Odd dimensions exercise dispatch bounds and padded GPU row pitch.
        rgba = np.random.default_rng(42).uniform(0, 1, (19, 37, 4)).astype("<f4")
        rgba[0, 0, :3] = 0
        rgba[0, 1, :3] = 1
        rgba[0, 2, :3] = (1, 0, 0)
        configurations = [FilterConfig(strength=s) for s in (0, .5, 1)]
        configurations += [FilterConfig(.08, .35, 1), FilterConfig(.4, .9, .7)]
        for backend in backends:
            for config in configurations:
                with self.subTest(backend=backend, config=config), workspace_temp() as directory:
                    source, target = Path(directory) / "in.sgf", Path(directory) / "out.sgf"
                    source.write_bytes(struct.pack("<4sII", b"SGF1", 37, 19) + rgba.tobytes())
                    process = subprocess.run([str(executable), str(source), str(target), str(config.knee), str(config.ceiling), str(config.strength), backend],
                                             capture_output=True, text=True, timeout=30)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertIn(f"backend={backend}", process.stdout)
                    payload = target.read_bytes()
                    self.assertEqual(struct.unpack("<4sII", payload[:12]), (b"SGF1", 37, 19))
                    actual = np.frombuffer(payload[12:], dtype="<f4").reshape(rgba.shape)
                    expected = compress_highlights(rgba[..., :3], config)
                    np.testing.assert_allclose(actual[..., :3], expected, rtol=2e-5, atol=2e-6)
                    np.testing.assert_array_equal(actual[..., 3], rgba[..., 3])
                    print(f"{process.stdout.strip()} config={config}: max_abs_error={np.max(np.abs(actual[..., :3] - expected)):.9g}")

    def test_invalid_payload_and_existing_output_fail_without_overwrite(self):
        executable = os.environ["SUNGLASSES_COMPUTE"]
        with workspace_temp() as directory:
            source, target = Path(directory) / "in.sgf", Path(directory) / "out.sgf"
            source.write_bytes(struct.pack("<4sII", b"SGF1", 4, 4))
            command = [executable, str(source), str(target), ".2", ".55", "1", "warp"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse(target.exists())
            target.write_bytes(b"keep")
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(target.read_bytes(), b"keep")
