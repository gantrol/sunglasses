"""Generate native and package icons from the existing vector geometry, without a window.

Pillow is a build dependency only. File I/O is synchronous in this offline command.
"""

import argparse
import io
import re
import struct

from PIL import Image, ImageDraw, ImageFilter

from release_support import ROOT

LOGOS = {"StoreLogo": 50, "Square44x44Logo": 44, "Square150x150Logo": 150}
SCALES = (100, 125, 150, 200, 250, 300, 400)
TARGET_SIZES = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 256)
ICO_SIZES = tuple(sorted((*TARGET_SIZES, 128)))
ICON_SOURCE = ROOT / "native/screen/assets/Sunglasses.ico"


def asset_sizes():
    result = {f"{name}.scale-{scale}.png": (base*scale+99)//100 for name, base in LOGOS.items() for scale in SCALES}
    for size in TARGET_SIZES:
        for suffix in ("", "_altform-unplated", "_altform-lightunplated"):
            result[f"Square44x44Logo.targetsize-{size}{suffix}.png"] = size
    return result


def render_icon(size):
    """Keep the pixel silhouette, with transparent corners and a neutral contrast rim."""
    geometry = (ROOT / "native/screen/pixel_glasses.h").read_text(encoding="utf-8")
    width = int(re.search(r"GlassesWidth=(\d+)", geometry)[1])
    height = int(re.search(r"GlassesHeight=(\d+)", geometry)[1])
    def points(name):
        match = re.search(name + r"\[\]=\{(.*?)\};", geometry, re.S)
        if not match:
            raise ValueError(f"Missing native icon geometry: {name}")
        return [tuple(map(int, pair)) for pair in re.findall(r"\{(\d+),(\d+)\}", match[1])]
    outline, lens = points("GlassesOutline"), points("GlassesLens")
    palette = (ROOT / "native/screen/pixel_theme.h").read_text(encoding="utf-8")
    colors = re.search(r"return \{false,RGB\((\d+,\d+,\d+)\),on\?RGB\((\d+,\d+,\d+)\)", palette)
    if not colors:
        raise ValueError("Native icon palette changed; update the package asset generator")
    frame_color, lens_color = [tuple(map(int, value.split(","))) for value in colors.groups()]
    image = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(image)
    scale_x = size * .82 / width
    # At 16-24 px retain seven rows of silhouette instead of reducing it to a thin line.
    scale_y = max(height*scale_x, 7) / height
    left, top = size*.09, (size-height*scale_y)/2
    def polygon(vertices, color, mirror=False):
        draw.polygon([(round(left+(width-x if mirror else x)*scale_x), round(top+y*scale_y))
                      for x, y in vertices], fill=color)
    polygon(outline, frame_color)
    polygon(lens, lens_color)
    polygon(lens, lens_color, mirror=True)
    rim = Image.new("RGBA", image.size, (199, 199, 199, 0))
    rim.putalpha(image.getchannel("A").filter(ImageFilter.MaxFilter(2*max(1, round(size/150))+1)))
    return Image.alpha_composite(rim, image)


def native_icon_bytes():
    # Encode each size independently, with DIB+AND masks below 256 and PNG at 256.
    # This avoids deriving small pixel artwork by resampling the largest image.
    entries, payloads = [], []
    offset = 6 + 16*len(ICO_SIZES)
    for size in ICO_SIZES:
        icon = render_icon(size)
        if size == 256:
            stream = io.BytesIO()
            icon.save(stream, format="PNG")
            payload = stream.getvalue()
        else:
            pixels = icon.tobytes("raw", "BGRA", 0, -1)
            stride = ((size+31)//32)*4
            mask = bytearray(stride*size)
            alpha = icon.getchannel("A")
            for y in range(size):
                for x in range(size):
                    if alpha.getpixel((x, size-1-y)) == 0:
                        mask[y*stride+x//8] |= 0x80 >> (x % 8)
            payload = struct.pack("<IiiHHIIiiII", 40, size, size*2, 1, 32, 0, len(pixels), 0, 0, 0, 0) + pixels + mask
        entries.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(payload), offset))
        payloads.append(payload)
        offset += len(payload)
    return struct.pack("<HHH", 0, 1, len(entries)) + b"".join(entries) + b"".join(payloads)


def check_native_icon():
    with Image.open(ICON_SOURCE) as icon:
        if icon.ico.sizes() != {(size, size) for size in ICO_SIZES}:
            raise ValueError("Native icon sizes are stale; run scripts/msix_assets.py --write-native-icon")
        for size in ICO_SIZES:
            if icon.ico.getimage((size, size)).convert("RGBA").tobytes() != render_icon(size).tobytes():
                raise ValueError("Native icon artwork is stale; run scripts/msix_assets.py --write-native-icon")


def generate_assets(destination):
    check_native_icon()
    destination.mkdir()
    rendered = {}
    for name, size in asset_sizes().items():
        if size not in rendered:
            rendered[size] = render_icon(size)
        rendered[size].save(destination / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-native-icon", action="store_true", help="Regenerate the checked-in native icon after changing its vector artwork")
    args = parser.parse_args()
    if args.write_native_icon:
        ICON_SOURCE.parent.mkdir(parents=True, exist_ok=True)
        ICON_SOURCE.write_bytes(native_icon_bytes())
    check_native_icon()
    print("Native icon geometry, transparency and all sizes verified.")
