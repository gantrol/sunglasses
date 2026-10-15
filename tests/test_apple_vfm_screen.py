"""Authorized numeric/headless adapter checks; never launches or captures UI."""
import os
from pathlib import Path
import struct
import subprocess
import unittest

import numpy as np

from sunglasses.apple_vfm import AppleVFMConfig, AppleVFMReference
from sunglasses.apple_vfm_screen import AppleVFMScreen
from tests.support import workspace_temp


def frame(code):
    return np.full((16, 20, 3), code, dtype=np.uint8)


class AppleVFMScreenTests(unittest.TestCase):
    def test_cfr_reproduces_existing_matlab_core_and_one_sample_delay(self):
        adapter = AppleVFMScreen()
        reference = AppleVFMReference(AppleVFMConfig(60, 100, 1265.625))
        attacks = []
        for index in range(240):
            source = frame(255 if index % 12 < 6 else 40)
            output = adapter.process(source, index / 60, index)
            expected, measurement = reference.process_linear_rgb((source.astype(np.float32) / 255) ** 2.2)
            np.testing.assert_allclose(output, expected ** (1 / 2.2), atol=2e-7)
            self.assertAlmostEqual(adapter.measurement.risk, measurement.risk)
            if measurement.next_mitigation > 0:
                attacks.append(index)
        self.assertTrue(attacks)

    def test_constant_white_black_and_dark_are_identity_including_idle(self):
        for code in (0, 40, 255):
            adapter = AppleVFMScreen()
            original = frame(code)
            adapter.process(original, 0, 0)
            for tick in range(1, 601):
                output = adapter.refresh(tick / 60)
                np.testing.assert_array_equal(output, original.astype(np.float32) / 255)
                self.assertEqual(adapter.measurement.risk, 0)

    def test_duplicate_metadata_does_not_advance_or_replace_pixels(self):
        adapter = AppleVFMScreen()
        expected = adapter.process(frame(40), 0, 0)
        for _ in range(8):
            np.testing.assert_array_equal(adapter.process(frame(255), 0, 0), expected)
        self.assertEqual(adapter.reference.frame_index, 1)

    def test_sparse_and_dense_observations_agree_on_held_apl(self):
        dense, sparse = AppleVFMScreen(), AppleVFMScreen()
        # 30-Hz content carried by 60-Hz capture, or 30-Hz capture plus idle ticks.
        for tick in range(180):
            source = frame(255 if (tick // 2) % 6 < 3 else 40)
            expected = dense.process(source, tick / 60, tick)
            actual = (sparse.process(source, tick / 60, tick // 2) if tick % 2 == 0 else
                      sparse.refresh(1 / 60))
            np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_between_tick_capture_cannot_rewrite_an_earlier_sample(self):
        adapter = AppleVFMScreen()
        adapter.process(frame(40), 0, 0)
        adapter.process(frame(255), .010, 1)
        self.assertEqual(adapter.reference.frame_index, 1)
        self.assertLess(adapter.measurement.apl_relative, .1)
        adapter.refresh(1 / 60 - .010)
        self.assertEqual(adapter.reference.frame_index, 2)
        self.assertAlmostEqual(adapter.measurement.apl_relative, 1)

    def test_gaps_rebaseline_and_strength_does_not_feed_back_into_detector(self):
        full, disabled = AppleVFMScreen(), AppleVFMScreen()
        for index in range(150):
            source = frame(255 if index % 12 < 6 else 40)
            full.process(source, index / 60, index)
            output = disabled.process(source, index / 60, index, strength=0)
            np.testing.assert_array_equal(output, source.astype(np.float32) / 255)
            self.assertEqual(full.measurement.risk, disabled.measurement.risk)
        self.assertGreater(full.measurement.applied_mitigation, 0)
        output = full.process(frame(255), 10, 150)
        np.testing.assert_array_equal(output, np.ones_like(output))
        self.assertEqual(full.measurement.applied_mitigation, 0)

    def test_idle_gap_preserves_held_apl_and_late_capture_is_not_backdated(self):
        adapter = AppleVFMScreen()
        adapter.process(frame(255), 0, 0)
        adapter.refresh(2)
        self.assertAlmostEqual(adapter.measurement.apl_relative, 1)
        self.assertAlmostEqual(adapter.measurement.adaptation_luminance_nits, 100)
        samples = adapter.reference.frame_index
        adapter.process(frame(40), 1.99, 1)
        self.assertEqual(adapter.reference.frame_index, samples)
        self.assertAlmostEqual(adapter.measurement.apl_relative, 1)
        adapter.refresh(.01 + 1 / 60)
        self.assertLess(adapter.measurement.apl_relative, .1)


@unittest.skipUnless(os.environ.get('SUNGLASSES_SCREEN_COMPUTE'), 'Fresh Apple-capable native probe required')
class AppleVFMNativeTests(unittest.TestCase):
    def test_cfr_rgb_mapping_matches_cpu_replica(self):
        # This optional suite must be invoked only after a corresponding build.
        frames = [frame(255 if i % 12 < 6 else 40) for i in range(180)]
        for source in frames:
            source[4:7, 3:10] = [0, 32, 80]
        chunks = [struct.pack('<4sIII', b'SGB2', 20, 16, len(frames))]
        cpu = AppleVFMScreen()
        expected = []
        for i, source in enumerate(frames):
            chunks.append(struct.pack('<IdQf', 0, i / 60, i, .5))
            chunks.append(np.concatenate((source[..., ::-1], np.full((16, 20, 1), 255, np.uint8)), axis=2).tobytes())
            expected.append(cpu.process(source, i / 60, i, .5))
        for backend in (['warp', 'hardware'] if os.environ.get('SUNGLASSES_TEST_HARDWARE') == '1' else ['warp']):
            with workspace_temp() as temporary:
                source_path, output_path = Path(temporary) / 'input.sgb', Path(temporary) / 'mask.bin'
                source_path.write_bytes(b''.join(chunks))
                result = subprocess.run([os.environ['SUNGLASSES_SCREEN_COMPUTE'], str(source_path),
                                         str(output_path), backend, 'apple-vfm'], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'synthetic_mask_only apple_matlab_vfm_60hz_v1')
                mask = np.frombuffer(output_path.read_bytes(), '<f4').reshape(-1, 16, 20, 4)
                actual = np.asarray(frames, np.float32) / 255 * (1 - mask[..., 3:]) + mask[..., :3]
                np.testing.assert_allclose(actual, expected, atol=2e-4)
