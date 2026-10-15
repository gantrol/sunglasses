"""Procedural screen scenes for behaviour-contract evaluation.

Every scene is a deterministic function of time rendered on a reduced grid that
keeps the monitor aspect ratio.  Layouts are generic (no captured user content)
and use relative coordinates, so the same scene renders on landscape and
portrait profiles.  Text is approximated by thin dark rows; glyph detail needs
native-resolution checks elsewhere.  Values are 8-bit sRGB-encoded RGB.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Profile:
    """Detected display, simulated on a grid that preserves its aspect ratio."""

    name: str
    native: tuple[int, int]  # width, height in physical pixels
    refresh_hz: float
    grid: tuple[int, int]  # width, height of the simulation grid


# Detected on 2026-10-13 via Win32_VideoController / WmiMonitorID / EDID size.
PROFILES = {
    "main": Profile("PHL 40B1U5600 3440x1440@120", (3440, 1440), 120., (172, 72)),
    "portrait": Profile("DELL U2520DR 1440x2560@60", (1440, 2560), 60., (72, 128)),
}


@dataclass(frozen=True)
class Scene:
    id: str
    contracts: tuple[str, ...]
    duration: float
    render: Callable[[float], np.ndarray]
    rois: dict[str, np.ndarray]
    # Intervals (start, end) in which a capture source delivers no frames.
    capture_gaps: tuple[tuple[float, float], ...] = ()
    # ROIs whose source content never changes; any filter-added change there is a violation.
    unchanged: tuple[str, ...] = ()
    notes: str = ""
    meta: dict = field(default_factory=dict)


class Canvas:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.v, self.u = np.mgrid[0:height, 0:width].astype(np.float32)
        self.u = (self.u + .5) / width
        self.v = (self.v + .5) / height

    def box(self, u0, v0, u1, v1):
        return (self.u >= u0) & (self.u < u1) & (self.v >= v0) & (self.v < v1)

    def fill(self, value):
        return np.broadcast_to(np.asarray(value, np.float32), (self.h, self.w, 3)).copy()


def _text_rows(c: Canvas, u0, v0, u1, v1, pitch, progress=1.):
    """Thin dark rows standing in for text lines; progress reveals rows left to right."""
    rows = np.floor((c.v - v0) / pitch)
    line = (((c.v - v0) / pitch) % 1) < .35
    total = max(1, int((v1 - v0) / pitch))
    shown = progress * total
    reveal = (rows < np.floor(shown)) | ((rows == np.floor(shown)) & (c.u < u0 + (u1 - u0) * (shown % 1)))
    return c.box(u0, v0, u1, v1) & line & reveal


def light_app(c: Canvas, text_progress=1., cursor=True, t=0.):
    img = c.fill(255.)
    img[c.box(0, 0, 1, .06)] = 240.          # title bar
    img[c.box(0, .06, .18, 1)] = 243.        # sidebar
    img[_text_rows(c, .22, .1, .70, .9, .05, text_progress)] = 45.
    img[_text_rows(c, .02, .1, .16, .6, .07)] = 90.
    if cursor and (t % 1.06) < .53:
        img[c.box(.70, .5, .705, .56)] = 20.
    return img


def dark_app(c: Canvas):
    img = c.fill(30.)
    img[c.box(0, 0, 1, .06)] = 24.
    img[_text_rows(c, .22, .1, .70, .9, .05)] = 200.
    return img


def dark_desktop(c: Canvas):
    img = c.fill(0.)
    img[...] = (40 + 25 * c.u)[..., None] * np.array([.8, .9, 1.2], np.float32)
    img[c.box(0, .95, 1, 1)] = 32.
    return img


def light_desktop(c: Canvas):
    img = c.fill(230.)
    img[c.box(0, .95, 1, 1)] = 238.
    return img


def dark_game(c: Canvas, t: float):
    """Night scene with constant per-frame motion, as games present every frame."""
    img = c.fill(0.)
    img[...] = (18 + 30 * (1 - c.v) + 6 * np.sin(6.3 * c.u + t))[..., None] * np.array([.7, .8, 1.], np.float32)
    x = .3 + .2 * np.sin(.7 * t)
    img[c.box(x, .55, x + .06, .7)] = (90, 80, 70)        # moving character
    img[c.box(.03, .9, .25, .94)] = (150, 40, 40)          # HUD health bar
    img[_text_rows(c, .35, .86, .65, .92, .03)] = 200.     # subtitle
    return img


def snow_scene(c: Canvas, t: float):
    img = c.fill(0.)
    img[...] = (205 + 30 * (1 - c.v) + 8 * np.sin(23 * c.u + 17 * c.v + 3 * t))[..., None]
    img[c.box(.03, .9, .25, .94)] = (150, 40, 40)
    return np.clip(img, 0, 255)


def _blend(a, b, w):
    return a * (1 - w) + b * w


def _ease(x):
    x = min(max(x, 0.), 1.)
    return x * x * (3 - 2 * x)


def build(profile: Profile) -> list[Scene]:
    w, h = profile.grid
    c = Canvas(w, h)
    L = light_app(c)
    D = dark_app(c)
    dd, ld = dark_desktop(c), light_desktop(c)
    blank = c.box(.75, .15, .97, .85)            # white page area without text rows
    sidebar = c.box(.0, .62, .18, .94)
    full = np.ones((h, w), bool)
    scenes: list[Scene] = []

    def q(img):
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)

    # D1/D2 static
    scenes.append(Scene("D1_static_light", ("S1",), 30., lambda t: q(light_app(c, t=t)),
                        {"blank": blank, "full": full}, unchanged=("blank",)))
    scenes.append(Scene("D2_static_dark", ("S1",), 15., lambda t: q(D), {"full": full}, unchanged=("full",)))

    # D3 dark game -> light app and stay (field symptom)
    scenes.append(Scene("D3_game_to_light", ("S1", "S2"), 12.,
                        lambda t: q(dark_game(c, t) if t < 2 else light_app(c, t=t)),
                        {"blank": blank, "full": full}, meta={"switch": 2.}))
    # D4 light -> dark game
    scenes.append(Scene("D4_light_to_game", ("S3",), 8.,
                        lambda t: q(light_app(c, t=t) if t < 2 else dark_game(c, t)),
                        {"blank": blank, "full": full}, meta={"switch": 2.}))
    # D5 ping-pong every 2 s
    scenes.append(Scene("D5_pingpong", ("S2", "S3"), 14.,
                        lambda t: q(light_app(c, t=t) if int(t // 2) % 2 else dark_game(c, t)),
                        {"blank": blank, "full": full}))

    # D6 virtual desktop slide (250 ms) dark <-> light
    def slide(t):
        def shifted(a, b, x):
            k = int(round(x * w))
            out = np.empty_like(a)
            out[:, :w - k] = a[:, k:]
            out[:, w - k:] = b[:, :k]
            return out
        if t < 2:
            return q(dd)
        if t < 2.25:
            return q(shifted(dd, ld, _ease((t - 2) / .25)))
        if t < 6:
            return q(ld)
        if t < 6.25:
            return q(shifted(ld, dd, _ease((t - 6) / .25)))
        return q(dd)
    scenes.append(Scene("D6_desktop_slide", ("S2", "S3"), 10., slide, {"full": full}))

    # D7 light window opens (200 ms zoom+fade) on dark desktop and closes (150 ms fade)
    def window(t):
        if 2 <= t < 6.15:
            grow = _ease((t - 2) / .2)
            fade = 1 - _ease((t - 6) / .15) if t >= 6 else grow
            s = .6 + .4 * grow
            box = c.box(.5 - .3 * s, .5 - .35 * s, .5 + .3 * s, .5 + .35 * s)
            img = dd.copy()
            img[box] = _blend(dd[box], L[box], fade)
            return q(img)
        return q(dd)
    win_roi = c.box(.25, .2, .75, .8)
    scenes.append(Scene("D7_window_open_close", ("S2", "S3"), 10., window, {"window": win_roi, "full": full}))

    # D8 scrolling light page with mixed images; sidebar stays put
    page_h = 6.
    rng = np.random.default_rng(8)
    tiles = rng.uniform(20, 250, size=40)

    def scroll(t):
        img = L.copy()
        offset = .35 * max(0., min(t, 15.) - 1.)
        vv = (c.v + offset) % page_h
        area = c.box(.22, .1, .95, 1.)
        idx = (vv / .3).astype(int) % len(tiles)
        tile_on = ((vv / .3) % 1 < .6) & ((idx % 3) == 0)
        img[area & tile_on] = tiles[idx][area & tile_on][:, None]
        text = ((vv / .05) % 1 < .35) & ~tile_on
        img[area & text] = 45.
        img[area & ~tile_on & ~text] = 255.
        return q(img)
    scenes.append(Scene("D8_scroll", ("S4",), 18., scroll, {"sidebar": sidebar, "full": full},
                        unchanged=("sidebar",)))

    # D9 streaming text (Codex-like)
    scenes.append(Scene("D9_streaming_text", ("S1", "S4"), 25.,
                        lambda t: q(light_app(c, text_progress=min(1., t / 20.), t=t)),
                        {"blank": blank, "full": full}, unchanged=("blank",)))

    # D10 menu, tooltip, toast
    def popups(t):
        img = L.copy()
        if 2 <= t < 3:
            img[c.box(.40, .30, .52, .62)] = 252.
            img[c.box(.40, .30, .52, .62) & ((c.v * 40) % 1 < .3)] = 60.
        if 5 <= t < 6.5:
            img[c.box(.30, .70, .40, .74)] = 50.
        if 8 <= t < 12.3:
            enter = _ease((t - 8) / .3)
            leave = _ease((t - 12) / .3)
            x0 = 1. - .2 * enter + .2 * leave
            img[c.box(x0, .8, x0 + .19, .92)] = 60.
        return q(img)
    scenes.append(Scene("D10_popups", ("S4",), 14., popups, {"blank": blank, "full": full},
                        unchanged=("blank",)))

    # D11 carousel (25% area), 0.4 s cross-fade every 3 s
    car = c.box(.22, .12, .62, .5)
    a_img, b_img = L.copy(), L.copy()
    a_img[car] = 235.
    b_img[car] = 35.

    def carousel(t):
        phase = t % 6
        wgt = _ease((phase - 3) / .4) if phase < 5.6 else 1 - _ease((phase - 5.6) / .4)
        if phase < 3:
            wgt = 0.
        img = a_img.copy()
        img[car] = _blend(a_img[car], b_img[car], wgt)
        return q(img)
    scenes.append(Scene("D11_carousel", ("S4",), 24., carousel, {"blank": blank, "carousel": car, "full": full},
                        unchanged=("blank",)))

    # D12 windowed video (30% area) with 3 Hz white flashes, on a light page
    vid = c.box(.25, .15, .70, .70)

    def video(t):
        img = L.copy()
        on = 2 <= t < 6 and (t * 3) % 1 < .5
        img[vid] = 245. if on else 25.
        return q(img)
    scenes.append(Scene("D12_window_video_flash", ("S4", "S5", "S7"), 9., video,
                        {"video": vid, "blank": blank, "full": full}, unchanged=("blank",)))

    # D13 capture gap (secure desktop) with content change during the gap
    scenes.append(Scene("D13_capture_gap", ("S9",), 9.,
                        lambda t: q(light_app(c, t=t) if t < 4 else dark_game(c, t)),
                        {"blank": blank, "full": full}, capture_gaps=((3., 5.),)))

    # D14 cursor movement only
    def cursor(t):
        img = L.copy()
        x = .3 + .25 * np.sin(1.3 * t)
        y = .4 + .2 * np.sin(.9 * t)
        img[c.box(x, y, x + .008, y + .03)] = 10.
        return q(img)
    scenes.append(Scene("D14_cursor", ("S1", "S4"), 10., cursor, {"blank": blank, "full": full},
                        unchanged=("blank",)))

    # ---- Games
    center = c.box(.3, .25, .7, .75)
    scenes.append(Scene("G1_single_flash", ("S6",), 8.,
                        lambda t: q(c.fill(255.) if (2 <= t < 2 + 1 / 60 or 5 <= t < 5 + 3 / 60)
                                    else dark_game(c, t)),
                        {"center": center, "full": full}, notes="1 and 3 game frames at 60 fps"))

    def burst(t):
        img = dark_game(c, t)
        if 2 <= t < 3.05:
            k = _ease((t - 2) / .05) if t < 2.45 else 1 - _ease((t - 2.45) / .6)
            r = np.hypot((c.u - .5) * 2.4, c.v - .5)
            glow = np.clip(1.3 - r * 1.5, 0, 1)[..., None] * 255.
            img = img + (np.maximum(glow, 0) - img) * k
        return q(img)
    scenes.append(Scene("G2_burst", ("S6",), 6., burst, {"center": center, "full": full}))

    def strobe(t):
        img = dark_game(c, t)
        for start, end, hz, local in ((2, 4, 2, False), (5, 7, 5, False), (8, 10, 10, False), (11, 13, 5, True)):
            if start <= t < end and (t * hz) % 1 < .5:
                if local:
                    img[c.box(.6, .1, .95, .6)] = 250.
                else:
                    img[...] = 250.
        return q(img)
    scenes.append(Scene("G3_strobe", ("S7",), 14., strobe,
                        {"center": center, "local": c.box(.6, .1, .95, .6), "full": full}))

    scenes.append(Scene("G4_enter_snow", ("S2", "S8"), 12.,
                        lambda t: q(dark_game(c, t) if t < 2 else snow_scene(c, t)),
                        {"center": center, "full": full}, meta={"switch": 2.}))

    def menu(t):
        img = snow_scene(c, t)
        if 2 <= t < 5:
            img = img * .4
            img[_text_rows(c, .4, .3, .6, .7, .08)] = 230.
        return q(img)
    scenes.append(Scene("G5_menu_overlay", ("S2", "S3"), 8., menu, {"center": center, "full": full}))

    def red(t):
        img = dark_game(c, t)
        if 2 <= t < 4 and (t * 3) % 1 < .5:
            img[...] = (255, 0, 0)
        return q(img)
    scenes.append(Scene("G6_red_flash", ("S7",), 6., red, {"center": center, "full": full}))

    left = c.u < .5

    def antiphase(t):
        img = dark_game(c, t)
        if 2 <= t < 4:
            on = (t * 5) % 1 < .5
            img[left == on] = 240.
            img[left != on] = 20.
        return q(img)
    scenes.append(Scene("G7_antiphase", ("S7",), 6., antiphase,
                        {"left": left, "right": ~left, "full": full}, notes="known global-mean blind spot"))
    return scenes
