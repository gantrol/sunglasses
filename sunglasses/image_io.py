"""Small synchronous offline file operations; never called by a UI/render thread."""

from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from .color import linear_to_srgb, srgb_to_linear
from .i18n import ValidationError


def read_sdr_png(path: Path, *, assume_srgb: bool = False) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(29)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) < 29 or header[24] != 8:
        raise ValidationError("png_format")
    with Image.open(path) as image:
        if image.width * image.height > 16_000_000:
            raise ValidationError("png_size")
        if image.mode not in ("RGB", "RGBA"):
            raise ValidationError("png_mode")
        if image.info.get("icc_profile") or "cicp" in image.info or "cICP" in image.info:
            raise ValidationError("png_profile")
        gamma = image.info.get("gamma")
        if gamma is not None and abs(gamma - 0.45455) > .0001:
            raise ValidationError("png_gamma")
        if "srgb" not in image.info and not assume_srgb:
            raise ValidationError("png_untagged")
        pixels = np.array(image)
        if image.mode == "RGBA" and np.any(pixels[..., 3] != 255):
            raise ValidationError("png_transparent")
        return srgb_to_linear(pixels[..., :3].astype(np.float32) / 255)


def save_sdr_png(path: Path, linear_rgb: np.ndarray) -> None:
    pixels = np.rint(np.clip(linear_to_srgb(linear_rgb), 0, 1) * 255).astype(np.uint8)
    metadata = PngImagePlugin.PngInfo()
    metadata.add(b"sRGB", b"\x00")
    # Exclusive creation prevents accidental replacement of user material.
    with path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG", pnginfo=metadata)
