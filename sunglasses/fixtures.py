"""Deterministic synthetic signals, not recreations or recordings of any game."""

from dataclasses import dataclass
import math

import numpy as np

from .i18n import ValidationError


@dataclass(frozen=True)
class Case:
    id: str
    description: str
    limitation: str = ""


CASES = (
    Case("dark_control", "Dark scene and low-luminance HUD; identity control."),
    Case("single_white_flash", "Exactly one full-screen white frame; first-frame amplitude."),
    Case("localized_burst", "Localized bright skill-like burst over a dark scene; abstract Wuthering Waves lead."),
    Case("neon_pulses", "Alternating neon panels in a dark scene; abstract NTE event lead.", "Not a reconstruction of a specific event."),
    Case("antiphase_blocks", "Two opposing bright/dark panels with constant frame-average luminance.", "Global averages can hide large local temporal changes."),
    Case("saturated_red", "Saturated red on/off pulses.", "A luminance shoulder alone barely reduces these red pulses."),
    Case("hard_cut", "One dark-to-bright scene cut; not a repeated event."),
    Case("slow_ramp", "Smooth full-screen increase; monotonicity control."),
)
CASE_BY_ID = {case.id: case for case in CASES}
AUTO_CASES = CASES + (
    Case("white_pulses", "Repeated full-screen white/dark pulses after a dark lead-in.",
         "A causal controller cannot know at the first rise whether a reversal will follow."),
    Case("midtone_recovery", "One white frame at 10% of the clip, then a constant 0.35 midtone.",
         "The recovery tail dims ordinary midtones; no comfort inference."),
    Case("midtone_pulses", "Full-screen 0.25/0.36 pulses, below the strong-brightness trigger.",
         "Repeated-change evidence starts after several sampled reversals, not the first pulse."),
    Case("moving_pattern", "A 0.25/0.36 striped pattern translates horizontally at 80 pixels/second.",
         "Normal motion can trigger repeated-change evidence; the controller has no motion compensation."),
    Case("fine_antiphase", "A one-pixel 0.25/0.36 checkerboard swaps its two levels.",
         "With even block sizes the block means are constant: repeated-change evidence misses the swaps."),
)
CASE_BY_ID.update({case.id: case for case in AUTO_CASES})


@dataclass(frozen=True)
class SequenceSpec:
    width: int = 320
    height: int = 180
    fps: int = 60
    duration: float = 2.0
    seed: int = 7

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.width, self.height, self.fps, self.seed)):
            raise ValidationError("fixture_integers")
        if not (16 <= self.width <= 4096 and 16 <= self.height <= 4096):
            raise ValidationError("fixture_dimensions")
        if not (1 <= self.fps <= 240 and math.isfinite(self.duration) and 0 < self.duration <= 30):
            raise ValidationError("fixture_time")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValidationError("fixture_seed")
        if self.frame_count < 4 or self.frame_count * self.width * self.height > 20_000_000:
            raise ValidationError("fixture_budget")

    @property
    def frame_count(self) -> int:
        return round(self.duration * self.fps)


def frames(case_id: str, spec: SequenceSpec):
    """Yield (timestamp_seconds, linear_rgb); never display or encode a flashing video."""
    if case_id not in CASE_BY_ID:
        raise ValidationError("fixture_unknown")
    yy, xx = np.mgrid[:spec.height, :spec.width]
    x, y = xx / spec.width, yy / spec.height
    base = np.repeat((0.025 + 0.10 * x)[..., None], 3, axis=-1).astype(np.float32)
    base[(y > .78) & (y < .84) & (x > .10) & (x < .48)] = (.10, .16, .10)
    base[(y > .42) & (y < .60) & (x > .12) & (x < .20)] = (.15, .11, .07)
    rng = np.random.default_rng(spec.seed)
    cx, cy = .60 + rng.uniform(-.04, .04), .44 + rng.uniform(-.04, .04)
    burst = ((x - cx) ** 2 + (y - cy) ** 2) < .16**2
    left = xx < spec.width // 2
    right = xx >= spec.width - spec.width // 2
    panels = (yy // max(1, spec.height // 8) + xx // max(1, spec.width // 12)) % 2 == 0
    for index in range(spec.frame_count):
        t = index / spec.fps
        phase = index / spec.frame_count
        frame = base.copy()
        pulse = int(t * 8) % 2 == 0
        if case_id == "single_white_flash" and index == spec.frame_count // 2:
            frame[:] = 1
        elif case_id == "localized_burst" and (.30 <= phase < .40 or .55 <= phase < .63):
            frame[burst] = (1, .95, .85)
        elif case_id == "neon_pulses":
            frame[panels if pulse else ~panels] = (.95, .85, 1)
        elif case_id == "antiphase_blocks":
            frame[:] = .03
            frame[~(left | right)] = .515
            frame[left if pulse else right] = 1
        elif case_id == "saturated_red":
            frame[:] = (1 if pulse else .04, 0, 0)
        elif case_id == "hard_cut":
            frame[:] = .04 if phase < .5 else .85
        elif case_id == "slow_ramp":
            frame[:] = .03 + .97 * index / (spec.frame_count - 1)
        elif case_id == "white_pulses":
            event_time = t - .25
            frame[:] = 1 if event_time >= 0 and int(event_time * 10) % 2 == 0 else .04
        elif case_id == "midtone_recovery":
            frame[:] = 1 if index == spec.frame_count // 10 else .35
        elif case_id == "midtone_pulses":
            frame[:] = .36 if pulse else .25
        elif case_id == "moving_pattern":
            pattern = ((xx + int(t * 80)) // max(1, spec.width // 16)) % 2 == 0
            frame[:] = np.where(pattern[..., None], .36, .25)
        elif case_id == "fine_antiphase":
            pattern = (xx + yy) % 2 == int(pulse)
            frame[:] = np.where(pattern[..., None], .36, .25)
        yield t, frame
