"""Draw the app icon: 字 over A on a diagonally split blue badge.

    python scripts/make_icon.py

Writes icon.ico (every size Windows asks for) to packaging/, for the .exe
files, and to src/whisper_subs/assets/, for the window; and
src/whisper_subs/assets/icon.png, shown in the window's header. The lettering is Noto Sans
JP Bold, SIL Open Font License 1.1, fetched from google/fonts into build/fonts
on first run so the 9.6 MB font never goes into git.
"""

import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
FONT = ROOT / "build" / "fonts" / "NotoSansJP[wght].ttf"

S = 1024  # drawn large, scaled down: smooth edges at every size
BLUE, BLUE_DARK = (21, 101, 192), (13, 71, 161)  # the window's accent colour
WHITE, YELLOW = (255, 255, 255), (255, 213, 79)
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]


def noto_bold(size: int) -> ImageFont.FreeTypeFont:
    if not FONT.is_file():
        FONT.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Noto Sans JP -> {FONT}")
        urllib.request.urlretrieve(FONT_URL, FONT)
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_name("Bold")
    return f


def centred(d: ImageDraw.ImageDraw, x: float, y: float, text: str, f, fill) -> None:
    left, top, right, bottom = d.textbbox((0, 0), text, font=f)
    d.text((x - (left + right) / 2, y - (top + bottom) / 2), text, font=f, fill=fill)


def draw() -> Image.Image:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    box = (60, 60, 964, 964)
    badge = Image.new("L", (S, S), 0)
    ImageDraw.Draw(badge).rounded_rectangle(box, radius=200, fill=255)
    lower = Image.new("L", (S, S), 0)
    ImageDraw.Draw(lower).polygon([(964, 60), (964, 964), (60, 964)], fill=255)

    im.paste(Image.new("RGBA", (S, S), BLUE), (0, 0), badge)
    im.paste(Image.new("RGBA", (S, S), BLUE_DARK), (0, 0),
             Image.composite(lower, Image.new("L", (S, S), 0), badge))
    d = ImageDraw.Draw(im)
    centred(d, 345, 355, "字", noto_bold(400), WHITE)
    centred(d, 690, 655, "A", noto_bold(440), YELLOW)
    return im


def main() -> None:
    im = draw()
    ico = ROOT / "packaging" / "icon.ico"
    im.resize((256, 256), Image.LANCZOS).save(ico, sizes=[(s, s) for s in ICO_SIZES])
    assets = ROOT / "src" / "whisper_subs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    im.resize((256, 256), Image.LANCZOS).save(assets / "icon.ico",
                                              sizes=[(s, s) for s in ICO_SIZES])
    png = assets / "icon.png"
    im.resize((256, 256), Image.LANCZOS).save(png)
    preview = Image.new("RGB", (560, 300), (250, 250, 250))
    preview.paste(im.resize((256, 256), Image.LANCZOS), (20, 20), im.resize((256, 256)))
    x = 300
    for s in (64, 32, 16):
        small = im.resize((s, s), Image.LANCZOS)
        preview.paste(small, (x, 150 - s // 2), small)
        x += s + 30
    preview.save(ROOT / "build" / "icon_preview.png")
    print(f"{ico}\n{assets / 'icon.ico'}\n{png}")


if __name__ == "__main__":
    main()
