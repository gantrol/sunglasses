"""Authorized synthetic/headless checks. Never opens a window or captures a screen."""
import os
from pathlib import Path
import struct
import subprocess
import unittest

import numpy as np

from sunglasses.automatic import AutoConfig, AutoFilter
from sunglasses.color import srgb_to_linear, linear_to_srgb
from tests.support import workspace_temp


@unittest.skipUnless(os.environ.get('SUNGLASSES_SCREEN_COMPUTE'), 'Synthetic screen pipeline runner not configured')
class ScreenPipelineTests(unittest.TestCase):
    def render(self, frames, backend, strength=1, refresh=(), metadata=None):
        h, w = frames[0].shape[:2]
        chunks = [struct.pack('<4sIII', b'SGB2', w, h, len(frames)+len(refresh))]
        for index, rgb8 in enumerate(frames):
            time, frame_id = metadata[index] if metadata else (index/60, index)
            amount = strength[index] if isinstance(strength, list) else strength
            chunks.append(struct.pack('<IdQf', 0, time, frame_id, amount))
            bgra = np.concatenate((rgb8[..., ::-1], np.full((h, w, 1), 255, np.uint8)), axis=2)
            chunks.append(bgra.tobytes())
        for age, amount in refresh:
            chunks.append(struct.pack('<IdQf', 1, age, 0, amount))
        with workspace_temp() as temporary:
            source, output = Path(temporary)/'input.sgb', Path(temporary)/'mask.bin'
            source.write_bytes(b''.join(chunks))
            result = subprocess.run([os.environ['SUNGLASSES_SCREEN_COMPUTE'], str(source), str(output), backend],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'synthetic_mask_only')
            return np.frombuffer(output.read_bytes(), '<f4').reshape(-1, h, w, 4)

    def backends(self):
        return ['warp', 'hardware'] if os.environ.get('SUNGLASSES_TEST_HARDWARE') == '1' else ['warp']

    def test_startup_and_static_bright_scenes_are_exactly_transparent(self):
        white = np.full((35, 131, 3), 255, np.uint8)
        mixed = white.copy()
        mixed[8:24, 15:85] = 60
        for backend in self.backends():
            for frame in (white, mixed):
                with self.subTest(backend=backend):
                    masks = self.render([frame]*32, backend, refresh=[(.5, .5), (2, 1)])
                    np.testing.assert_array_equal(masks, np.zeros_like(masks))

    def test_small_changes_and_equal_brightness_motion_do_not_trigger(self):
        dark = np.full((32, 100, 3), 160, np.uint8)
        tiny = dark.copy()
        tiny[:2, :4] = 255
        # Quantization-scale changes around a bright stable background.
        noise = [np.full_like(dark, 229+i%2) for i in range(32)]
        moving = []
        for x in range(0, 81, 4):
            frame = dark.copy()
            frame[:, x:x+20] = 255
            moving.append(frame)
        for backend in self.backends():
            for frames in ([dark, tiny, dark], noise, moving):
                masks = self.render(frames, backend)
                np.testing.assert_array_equal(masks, np.zeros_like(masks))

    def test_sudden_highlights_compose_to_cpu_curve_and_preserve_other_pixels(self):
        rng = np.random.default_rng(77)
        ordinary = np.full((53, 101, 3), 80, np.uint8)
        colored = rng.integers(0, 256, ordinary.shape, dtype=np.uint8)
        colored[:20, :60] = 255
        colored[-5:] = 60
        for backend in self.backends():
            # The old full-strength curve is the new balanced 50% anchor.
            for strength in (.125, .25, .5):
                with self.subTest(backend=backend, strength=strength):
                    mask = self.render([ordinary, colored], backend, strength)[1]
                    self.assertTrue(np.isfinite(mask).all())
                    self.assertTrue(np.all(mask >= 0))
                    self.assertTrue(np.all(mask <= 1))
                    self.assertTrue(np.all(mask[..., :3] <= mask[..., 3:4]+1e-7))
                    original = colored.astype(np.float32)/255
                    filtered, _ = AutoFilter(AutoConfig(strength=2*strength)).process(srgb_to_linear(original), 0, 0)
                    composite = mask[..., :3] + (1-mask[..., 3:4])*original
                    np.testing.assert_allclose(composite, linear_to_srgb(filtered), rtol=0, atol=3e-6)
                    np.testing.assert_array_equal(mask[-5:], np.zeros_like(mask[-5:]))
                    self.assertGreater(float(mask[:20, :60, 3].max()), .01)

    def test_balanced_to_strong_is_continuous_monotone_and_retains_shadows(self):
        dark = np.full((32, 256, 3), 40, np.uint8)
        flash = np.full_like(dark, 255)
        flash[-8:] = np.arange(256, dtype=np.uint8)[None, :, None]
        original = flash.astype(np.float32)/255
        y = srgb_to_linear(original)[..., 0]
        for backend in self.backends():
            outputs = []
            for strength in (.5, .5001, .75, 1):
                mask = self.render([dark, flash], backend, strength)[1]
                self.assertTrue(np.isfinite(mask).all())
                self.assertTrue(np.all((mask >= 0) & (mask <= 1)))
                self.assertTrue(np.all(mask[..., :3] <= mask[..., 3:4]+1e-7))
                composite = mask[..., :3]+(1-mask[..., 3:4])*original
                self.assertTrue(np.all(composite <= original+1e-7))
                np.testing.assert_array_equal(mask[y <= .45], np.zeros_like(mask[y <= .45]))
                if strength == .5:
                    np.testing.assert_array_equal(mask[y <= .65], np.zeros_like(mask[y <= .65]))
                # The luminance ramp retains its ordering, even at maximum strength.
                self.assertTrue(np.all(np.diff(composite[-1, :, 0]) >= -1e-7))
                outputs.append(composite)
            self.assertLess(float(np.abs(outputs[1]-outputs[0]).max()), 2e-6)
            for previous, current in zip(outputs, outputs[1:]):
                self.assertTrue(np.all(current <= previous+1e-7))
            white = float(srgb_to_linear(outputs[-1])[0, 0, 0])
            self.assertGreater(white, .54)
            self.assertLess(white, .56)

    def test_extra_strength_is_reserved_for_large_highlight_events(self):
        dark = np.full((40, 100, 3), 60, np.uint8)
        small = dark.copy()
        small[:2] = 255  # 5% white load: triggers, but below the extra-strength area gate.
        for backend in self.backends():
            balanced = self.render([dark, small], backend, .5)[1]
            strong = self.render([dark, small], backend, 1)[1]
            self.assertGreater(float(balanced[..., 3].max()), 0)
            np.testing.assert_array_equal(strong, balanced)

    def test_bright_freeze_expires_and_does_not_retrigger(self):
        dark = np.full((19, 37, 3), 100, np.uint8)
        white = np.full_like(dark, 255)
        for backend in self.backends():
            masks = self.render([dark]+[white]*45, backend)
            opacity = masks[..., 3].max(axis=(1, 2))
            self.assertGreater(opacity[1], .05)
            self.assertTrue(np.all(np.diff(opacity[1:]) <= 1e-6))
            np.testing.assert_array_equal(masks[21:], np.zeros_like(masks[21:]))

    def test_strong_curve_does_not_release_when_highlight_area_shrinks(self):
        dark = np.full((32, 100, 3), 60, np.uint8)
        flash = np.full_like(dark, 170)
        flash[:, :22] = 255
        fading = flash.copy()
        fading[:, 17:22] = 170
        for backend in self.backends():
            masks = self.render([dark, flash, fading], backend, 1)
            self.assertGreater(float(masks[1, 0, 0, 3]), .1)
            # Both event ages are in the hold interval. Contracting bright area
            # must not brighten unchanged pixels by moving the curve's knee.
            np.testing.assert_allclose(masks[1, :, :17], masks[2, :, :17], rtol=0, atol=3e-7)
            outputs = [mask[..., :3]+(1-mask[..., 3:4])*frame/255
                       for mask, frame in zip(masks, [dark, flash, fading])]
            self.assertLess(float(srgb_to_linear(outputs[2]).mean()),
                            float(srgb_to_linear(outputs[1]).mean()))

    def test_no_new_capture_and_strength_changes_cannot_keep_mask_alive(self):
        dark = np.full((19, 37, 3), 100, np.uint8)
        white = np.full_like(dark, 255)
        for backend in self.backends():
            masks = self.render([dark, white], backend, refresh=[(.02, 0), (.03, 1), (.2, 1), (.32, 1), (2, .5), (3, 1)])
            np.testing.assert_array_equal(masks[2], np.zeros_like(masks[2]))
            np.testing.assert_allclose(masks[3], masks[1], atol=1e-7)
            self.assertLess(masks[4, ..., 3].max(), masks[3, ..., 3].max())
            np.testing.assert_array_equal(masks[5:], np.zeros_like(masks[5:]))
            # Moving the slider on a never-triggered white screen also stays transparent.
            stable = self.render([white], backend, 0, refresh=[(0, 1), (.5, .5), (2, 1)])
            np.testing.assert_array_equal(stable, np.zeros_like(stable))

    def test_return_to_normal_is_immediate_and_repeated_flashes_still_trigger(self):
        dark = np.full((19, 37, 3), 100, np.uint8)
        white = np.full_like(dark, 255)
        for backend in self.backends():
            masks = self.render([dark, white, dark, white, dark], backend)
            np.testing.assert_array_equal(masks[[0, 2, 4]], np.zeros_like(masks[[0, 2, 4]]))
            self.assertGreater(masks[1, ..., 3].max(), .05)
            np.testing.assert_allclose(masks[3], masks[1], atol=1e-7)

    def test_capture_metadata_does_not_create_false_events(self):
        dark = np.full((19, 37, 3), 100, np.uint8)
        white = np.full_like(dark, 255)
        for backend in self.backends():
            # Duplicate metadata must not re-analyze; gaps/backward time rebaseline.
            for metadata in ([(0, 0), (0, 0)], [(0, 0), (1/60, 2)], [(1, 0), (.5, 1)]):
                masks = self.render([dark, white], backend, metadata=metadata)
                np.testing.assert_array_equal(masks, np.zeros_like(masks))
            # Idle capture is sparse: a later genuine bright frame must still be recognized.
            masks = self.render([dark, white], backend, metadata=[(0, 0), (3, 1)])
            self.assertGreater(masks[1, ..., 3].max(), .05)
