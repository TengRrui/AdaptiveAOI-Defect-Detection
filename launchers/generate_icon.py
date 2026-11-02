from __future__ import annotations

import argparse
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SIZES = (16, 24, 32, 48, 64, 128, 256)


def ico_sizes(path: Path) -> list[int]:
    """Read the ICO directory because Pillow exposes only the selected frame."""
    data = path.read_bytes()
    if len(data) < 6:
        raise RuntimeError("ICO file is too small.")
    reserved, icon_type, count = struct.unpack_from("<HHH", data, 0)
    if reserved != 0 or icon_type != 1 or count < 1:
        raise RuntimeError("Generated file does not have a valid ICO header.")
    sizes: list[int] = []
    for index in range(count):
        offset = 6 + index * 16
        if len(data) < offset + 16:
            raise RuntimeError("ICO directory is truncated.")
        width = data[offset] or 256
        height = data[offset + 1] or 256
        if width != height:
            raise RuntimeError("ICO contains a non-square icon entry.")
        sizes.append(width)
    return sorted(set(sizes))


def build_icon(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = max(1, round(size * 0.06))
    radius = max(2, round(size * 0.22))
    draw.rounded_rectangle((margin, margin, size - margin, size - margin), radius=radius, fill=(10, 28, 51, 255))

    center = size / 2
    outer = max(3, round(size * 0.28))
    inner = max(2, round(size * 0.16))
    draw.ellipse((center - outer, center - outer, center + outer, center + outer), outline=(69, 189, 255, 255), width=max(1, round(size * 0.045)))
    draw.ellipse((center - inner, center - inner, center + inner, center + inner), fill=(20, 99, 154, 255), outline=(170, 235, 255, 255), width=max(1, round(size * 0.03)))
    highlight = max(1, round(size * 0.04))
    draw.ellipse((center - inner * 0.35, center - inner * 0.55, center - inner * 0.35 + highlight * 2, center - inner * 0.55 + highlight * 2), fill=(235, 251, 255, 220))

    if size >= 48:
        box = round(size * 0.22)
        stroke = max(1, round(size * 0.035))
        left = round(size * 0.58)
        top = round(size * 0.56)
        right = left + box
        bottom = top + box
        color = (87, 226, 171, 255)
        draw.line((left, top + box * 0.28, left, top, left + box * 0.28, top), fill=color, width=stroke)
        draw.line((right - box * 0.28, top, right, top, right, top + box * 0.28), fill=color, width=stroke)
        draw.line((left, bottom - box * 0.28, left, bottom, left + box * 0.28, bottom), fill=color, width=stroke)
        draw.line((right - box * 0.28, bottom, right, bottom, right, bottom - box * 0.28), fill=color, width=stroke)
        draw.ellipse((left + box * 0.44, top + box * 0.44, left + box * 0.60, top + box * 0.60), fill=(255, 192, 68, 255))

    if size >= 128:
        try:
            font = ImageFont.truetype("arialbd.ttf", round(size * 0.16))
        except OSError:
            font = ImageFont.load_default()
        label = "AOI"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(((size - text_width) / 2, size * 0.73), label, font=font, fill=(231, 246, 255, 255))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Create AdaptiveAOI launcher icon assets.")
    parser.add_argument("--ico", required=True)
    parser.add_argument("--preview", required=True)
    args = parser.parse_args()
    ico_path = Path(args.ico)
    preview_path = Path(args.preview)
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    master = build_icon(256)
    master.save(ico_path, format="ICO", sizes=[(size, size) for size in SIZES])
    master.save(preview_path, format="PNG")
    with Image.open(ico_path) as check:
        if check.format != "ICO" or check.size != (256, 256):
            raise RuntimeError("Generated icon is not a valid 256x256 ICO.")
    available = ico_sizes(ico_path)
    missing = sorted(set(SIZES) - set(available))
    if missing:
        raise RuntimeError(f"ICO is missing requested sizes: {missing}")
    print(f"ICO={ico_path}")
    print(f"PREVIEW={preview_path}")
    print(f"SIZES={','.join(map(str, available))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
