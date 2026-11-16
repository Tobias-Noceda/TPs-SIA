import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# Builds the dataset of ex. 2: face emojis rendered from a color emoji font installed in the
# machine, one small PNG per emoji (white background). A single family of images (faces) so a
# generated sample can be judged as belonging to the dataset or not.

# Face emojis (smileys, no cats, gestures or people)
# @link https://unicode.org/emoji/charts/full-emoji-list.html#face-smiling
FACE_RANGES = (
    (0x1F600, 0x1F637), (0x1F641, 0x1F644), (0x1F910, 0x1F917), (0x1F920, 0x1F925), (0x1F927, 0x1F92F),
    (0x1F970, 0x1F976), (0x1F978, 0x1F97A), (0x1F9D0, 0x1F9D0), (0x1FAE0, 0x1FAE5), (0x1FAE8, 0x1FAE8),
)

DEFAULT_FONTS = (
    "/mnt/c/Windows/Fonts/seguiemj.ttf",                      # Windows font seen from WSL
    "C:/Windows/Fonts/seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
)

def face_codepoints() -> List[int]:
    return [code for first, last in FACE_RANGES for code in range(first, last + 1)]

def render(font: ImageFont.FreeTypeFont, char: str, size: int, canvas: int) -> Image.Image:
    """The emoji centered on a white square of `canvas` pixels, reduced to `size` (antialiased)."""
    image = Image.new("RGB", (canvas, canvas), "white")
    ImageDraw.Draw(image).text((canvas // 2, canvas // 2), char, font=font, embedded_color=True, anchor="mm")
    return image.resize((size, size), Image.Resampling.LANCZOS)

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 2: dataset of face emojis rendered from a color emoji font.")
    parser.add_argument("-o", "--output", default="res/datasets/emojis24", help="Output folder (must not exist)")
    parser.add_argument("-s", "--size", type=int, default=24, help="Side of every image in pixels (default 24)")
    parser.add_argument("-f", "--font", default=None, help="Color emoji font file (default: the first one found in the system)")
    args = parser.parse_args()

    font_path = args.font or next((path for path in DEFAULT_FONTS if Path(path).is_file()), None)
    if font_path is None:
        parser.error("no color emoji font found; pass one with --font")
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists; choose another --output or remove it")

    # Only the emojis the font really has (a missing one would be drawn as an empty box)
    available = TTFont(font_path, fontNumber=0).getBestCmap()
    codes = [code for code in face_codepoints() if code in available]
    if not codes:
        parser.error(f"{font_path} has none of the face emojis")

    canvas = 8 * args.size
    font = ImageFont.truetype(font_path, int(canvas * 0.72))
    output.mkdir(parents=True)
    for code in codes:
        render(font, chr(code), args.size, canvas).save(output / f"{code:05X}.png")

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "font": Path(font_path).name,
        "size": args.size,
        "count": len(codes),
        "codepoints": [f"U+{code:05X}" for code in codes],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(codes)} emojis of {args.size}x{args.size} written to {output} (font {Path(font_path).name})")

if __name__ == "__main__":
    main()
