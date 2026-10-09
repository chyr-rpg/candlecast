"""Draw the CandleCast logo: three candlesticks and a dotted forecast line on a rounded tile.

Writes assets/candlecast.ico (16-256 px, for the exe/installer/shortcuts) and assets/logo.png (512 px,
for the app's sidebar). Run: .venv\\Scripts\\python tools\\make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parents[1] / "src" / "candlecast" / "assets"
BG_TOP, BG_BOTTOM = (31, 41, 64), (14, 18, 30)       # deep navy gradient
BLUE, ORANGE, WICK = (57, 135, 229), (235, 104, 52), (200, 210, 230)


def draw(size: int = 512) -> Image.Image:
    s = size / 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / max(size - 1, 1)
        gd.line([(0, y), (size, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(110 * s), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)

    def candle(cx, wick_top, body_top, body_bot, wick_bot, color):
        d.line([(cx * s, wick_top * s), (cx * s, wick_bot * s)], fill=WICK, width=max(1, int(10 * s)))
        d.rounded_rectangle([(cx - 34) * s, body_top * s, (cx + 34) * s, body_bot * s],
                            radius=max(1, int(10 * s)), fill=color)

    candle(130, 250, 285, 395, 430, ORANGE)
    candle(230, 175, 205, 330, 365, BLUE)
    candle(330, 110, 140, 255, 300, BLUE)
    # dotted forecast continuation
    for i, (x, y) in enumerate([(380, 150), (410, 128), (440, 110), (468, 96)]):
        r = (13 - i * 1.5) * s
        d.ellipse([x * s - r, y * s - r, x * s + r, y * s + r], fill=(109, 167, 236, 255 - i * 40))
    return img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    big = draw(512)
    big.save(ASSETS / "logo.png")
    big.save(ASSETS / "candlecast.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", ASSETS / "logo.png", ASSETS / "candlecast.ico")


if __name__ == "__main__":
    main()
