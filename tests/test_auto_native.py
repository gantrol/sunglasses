"""SGA1/SGO1 sequence parity; runs only explicitly configured headless compute programs."""

import os
from pathlib import Path
import struct
import subprocess
import unittest

import numpy as np

from sunglasses.automatic import AutoConfig, AutoFilter, FrameClock, features
from sunglasses.color import luminance
from tests.support import workspace_temp

FRAME = struct.Struct("<IIQdII")
STATE = struct.Struct("<5f3I")
CONTINUITY = ("first", "warmup", "continuous", "discontinuity", "duplicate")


def encode(sequence):
    result = bytearray(struct.pack("<4sI", b"SGA1", len(sequence)))
    for frame_id, timestamp, flags, rgba in sequence:
        height, width = rgba.shape[:2]
        result += FRAME.pack(width, height, frame_id, timestamp, flags, 0)
        result += rgba.astype("<f4").tobytes()
    return result


def decode(payload):
    magic, count = struct.unpack_from("<4sI", payload)
    if magic != b"SGO1":
        raise ValueError("Invalid output")
    offset = 8
    result = []
    for _ in range(count):
        width, height, frame_id, timestamp, flags, reserved = FRAME.unpack_from(payload, offset)
        offset += FRAME.size
        continuity, blocks = struct.unpack_from("<II", payload, offset)
        offset += 8
        state = STATE.unpack_from(payload, offset)
        offset += STATE.size
        stats = np.frombuffer(payload, "<f4", blocks * 20, offset).reshape(blocks, 20)
        offset += blocks * 80
        image = np.frombuffer(payload, "<f4", width * height * 4, offset).reshape(height, width, 4)
        offset += width * height * 16
        result.append((frame_id, timestamp, flags, reserved, CONTINUITY[continuity], state, stats, image))
    if offset != len(payload):
        raise ValueError("Output size mismatch")
    return result


def sequence_fixture():
    sequence = []
    def add(value, *, shape=(19, 37), timestamp=None, frame_id=None, flags=0):
        index = len(sequence)
        rgba = np.full((*shape, 4), value, np.float32)
        rgba[..., 3] = np.random.default_rng(9).random(shape, dtype=np.float32)
        rgba[:2, :2, :3] = .1
        sequence.append((index if frame_id is None else frame_id,
                         index / 30 if timestamp is None else timestamp, flags, rgba))
    add(1)
    for _ in range(120):
        add(.35)
    for index in range(40):
        add(.03 if (index // 3) % 2 else .40)
    # Odd and tall grids exercise non-uniform partition and the 64-row cap.
    add(.35, shape=(131, 35))
    add(1, shape=(131, 35))
    last = sequence[-1]
    add(.1, shape=(131, 35), timestamp=last[1], frame_id=last[0])
    add(.35, shape=(131, 35), timestamp=0, flags=1)
    for index in range(6):
        add(.35, shape=(131, 35), timestamp=(index + 1) / 30)
    add(.35, shape=(131, 35), timestamp=1)
    add(.35, shape=(131, 35), timestamp=1 + 1 / 30, frame_id=500)
    return sequence


@unittest.skipUnless(os.environ.get("SUNGLASSES_AUTO_COMPUTE"), "Automatic native runner not configured")
class AutomaticNativeTests(unittest.TestCase):
    def backends(self):
        return ["warp", "hardware"] if os.environ.get("SUNGLASSES_TEST_HARDWARE") == "1" else ["warp"]

    def invoke(self, source, target, config=AutoConfig(), backend="warp"):
        return subprocess.run([os.environ["SUNGLASSES_AUTO_COMPUTE"], str(source), str(target), str(config.strength),
                               str(config.hold_seconds), str(config.release_seconds), str(int(config.repeat_enabled)), backend, config.policy],
                              capture_output=True, text=True, timeout=60)

    def check_sequence(self, sequence, config, backend):
        with workspace_temp() as directory:
            source, target = Path(directory) / "in.sga", Path(directory) / "out.sgo"
            source.write_bytes(encode(sequence))
            process = self.invoke(source, target, config, backend)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn(f"backend={backend}", process.stdout)
            actual = decode(target.read_bytes())
        reference, clock = AutoFilter(config), FrameClock()
        previous_y = last_stats = last_alpha = None
        max_rgb, max_state, max_stats = 0, 0, 0
        self.assertEqual(len(actual), len(sequence))
        for original, computed in zip(sequence, actual):
            frame_id, timestamp, flags, rgba = original
            with self.subTest(frame=frame_id, time=timestamp, shape=rgba.shape):
                expected, trace = reference.process(rgba[..., :3], timestamp, frame_id, discontinuity=bool(flags))
                timing = clock.accept(frame_id, timestamp, rgba.shape, bool(flags))
                y = luminance(rgba[..., :3])
                if not timing.duplicate:
                    last_stats = features(y, None if timing.reset else previous_y)
                    previous_y, last_alpha = y.copy(), rgba[..., 3].copy()
                ident, time, actual_flags, reserved, continuity, state, stats, image = computed
                self.assertEqual((ident, time, actual_flags, reserved), (frame_id, timestamp, flags, 0))
                self.assertEqual(continuity, trace["continuity"])
                expected_state = [trace[key] for key in ("amount", "bright", "rise", "repeat", "hold_remaining")]
                np.testing.assert_allclose(state[:5], expected_state, rtol=2e-5, atol=2e-6)
                self.assertEqual(state[5:8], (int(trace["latch"]), trace["reason_bits"], 0))
                np.testing.assert_allclose(stats[:, :5], last_stats, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(image[..., :3], expected, rtol=2e-5, atol=2e-6)
                np.testing.assert_array_equal(image[..., 3], last_alpha)
                max_rgb = max(max_rgb, float(np.abs(image[..., :3] - expected).max()))
                max_state = max(max_state, float(np.abs(np.array(state[:5]) - expected_state).max()))
                max_stats = max(max_stats, float(np.abs(stats[:, :5] - last_stats).max()))
        print(f"{process.stdout.strip()} config={config} max_rgb={max_rgb:.9g} max_state={max_state:.9g} max_stats={max_stats:.9g}")

    def test_persistent_state_recovery_duplicates_resize_and_gaps(self):
        configs = [AutoConfig(strength=s) for s in (0, .5, 1)]
        configs += [AutoConfig(policy="legacy"), AutoConfig(hold_seconds=0, release_seconds=.1, policy="legacy"), AutoConfig(repeat_enabled=False, policy="legacy")]
        for backend in self.backends():
            for config in configs:
                with self.subTest(backend=backend, config=config):
                    self.check_sequence(sequence_fixture(), config, backend)

    def test_random_block_statistics_and_fractional_demand(self):
        rng = np.random.default_rng(23)
        sequence = []
        for index in range(12):
            rgba = rng.uniform(.1, .9, (53, 101, 4)).astype(np.float32)
            # Colored frames and a local patch exercise fractional continuous highlight load.
            rgba[10:20, 30:40, :3] = .68 if index % 2 else .4
            sequence.append((index, index / 60, 0, rgba))
        for backend in self.backends():
            self.check_sequence(sequence, AutoConfig(), backend)

    def test_invalid_files_are_rejected_without_overwrite(self):
        rgba = np.ones((1, 1, 4), np.float32)
        valid = encode([(0, 0, 0, rgba)])
        malformed = [valid[:-1], valid + b"extra", encode([(0, float("nan"), 0, rgba)]),
                     encode([(0, 0, 2, rgba)]), encode([(0, 0, 0, rgba * float("nan"))]),
                     struct.pack("<4sI", b"SGA1", 1) + FRAME.pack(8193, 1, 0, 0, 0, 0)]
        with workspace_temp() as directory:
            source, target = Path(directory) / "bad.sga", Path(directory) / "out.sgo"
            for payload in malformed:
                source.write_bytes(payload)
                result = self.invoke(source, target)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(target.exists())
            source.write_bytes(valid)
            target.write_bytes(b"preserve")
            self.assertNotEqual(self.invoke(source, target).returncode, 0)
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_highlight_gate_odd_area_and_exact_normal_frame_passthrough(self):
        sequence = []
        for index, fraction in enumerate((0, .001, .009, .011, .02, .035, .041, .10, 1, .001)):
            rgba = np.full((131, 35, 4), .35, np.float32)
            rgba.reshape(-1, 4)[:int(131 * 35 * fraction), :3] = 1
            sequence.append((index, index / 60, 0, rgba))
        for backend in self.backends():
            self.check_sequence(sequence, AutoConfig(), backend)

    def test_latch_boundary_records_rounding_instead_of_hiding_state_changes(self):
        # Invert smoothstep(.01,.05,E)=.6, then inspect adjacent float32 luminances.
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if mid * mid * (3 - 2 * mid) < .6:
                lo = mid
            else:
                hi = mid
        middle = np.float32(.65 + .01 + .04 * (lo + hi) / 2)
        levels = [np.nextafter(middle, np.float32(0)), middle, np.nextafter(middle, np.float32(1))]
        for backend in self.backends():
            differences = 0
            for level in levels:
                rgba = np.full((19, 37, 4), level, np.float32)
                sequence = [(0, 0, 0, rgba)]
                with workspace_temp() as directory:
                    source, target = Path(directory) / "in.sga", Path(directory) / "out.sgo"
                    source.write_bytes(encode(sequence))
                    process = self.invoke(source, target, AutoConfig(policy="legacy"), backend)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    computed = decode(target.read_bytes())[0]
                _, trace = AutoFilter(AutoConfig(policy="legacy")).process(rgba[..., :3], 0, 0)
                state = computed[5]
                self.assertLess(abs(state[1] - trace["bright"]), 5e-6)
                if bool(state[5]) != trace["latch"]:
                    differences += 1
                    # These are declared boundary differences, not a relaxed parity pass.
                    self.assertLess(abs(state[1] - .6), 5e-6)
                    self.assertLess(abs(trace["bright"] - .6), 5e-6)
            print(f"boundary_diagnostic backend={backend} samples={len(levels)} latch_disagreements={differences}")


if __name__ == "__main__":
    unittest.main()
