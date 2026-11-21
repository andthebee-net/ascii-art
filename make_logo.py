#!/usr/bin/env python3
"""make_logo.py - an andthebee logo sting to feed glyph_engine.py.

Scene 1: the "A" rises, the bee buzzes in from the left and lands in its place,
the ANDTHEBEE wordmark and the brand colour blocks pop in.
Scene 2: SMART / CONSTRUCTION / SOLUTIONS slides in big, with the 2x2 colour grid.
Mark: assets/andthebee-mark.svg. Colours: the andthebee.net brand tokens.

    python3 make_logo.py              # -> demo/andthebee_logo.mp4 (1080x1080, 30 fps, 10.5 s)
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glyph_engine import codec
from make_demo import SQUARE, TRIANGLE, hz, voice

W = H = 1080
FPS = 30
TOTAL = 10.5
SR = 44100
MARK = Path(__file__).with_name("assets") / "andthebee-mark.svg"
SANS = ["/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/HelveticaNeue.ttc"]

# brand tokens from andthebee.net
BLUE, TEAL, LIME, GRAY = (0x55, 0x68, 0xB6), (0x64, 0xC8, 0xC6), (0xE0, 0xFF, 0x40), (0xEC, 0xEC, 0xEC)
DARK, DARK2 = (0x0B, 0x0F, 0x1E), (0x13, 0x1A, 0x33)

MARK_W, MARK_TOP = 520, 100     # where the mark sits in scene 1
LAND = 3.1                      # the bee lands
SCENE2 = 6.0                    # scene 1 fades out, scene 2 slides in


def sans(size: int, weight: str):
    """Avenir Next Heavy / Demi Bold, falling back to Helvetica Neue Bold."""
    index = {"heavy": 8, "demi": 2}[weight]
    for path in SANS:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index if "Avenir" in path else 1)
    return ImageFont.load_default(size)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def pop(u, s=1.7):
    """Ease-out-back: 0 -> overshoot -> 1."""
    u = min(max(u, 0.0), 1.0) - 1
    return u * u * ((s + 1) * u + s) + 1


def scaled(sprite: Image.Image, s: float, sy: float = 1.0) -> Image.Image:
    return sprite.resize((max(1, round(sprite.width * s)), max(1, round(sprite.height * s * sy))),
                         Image.Resampling.BILINEAR)


# ---------------------------------------------------------------- the mark

def mark_paths(svg: Path):
    """viewBox and subpaths of the mark's single <path> (absolute M/L/C/Z) as point lists."""
    text = svg.read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', text).group(1).split()]
    toks = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?", re.search(r'\sd="([^"]+)"', text).group(1))
    polys, pts, cur, cmd, i = [], [], (0.0, 0.0), "", 0
    while i < len(toks):
        if toks[i].isalpha():
            cmd, i = toks[i], i + 1
            if cmd not in "MLCZ":
                sys.exit(f"make_logo: path command {cmd!r} not supported (absolute M/L/C/Z only)")
            if cmd == "Z":
                polys, pts = polys + [pts], []
                continue
        k = 6 if cmd == "C" else 2
        n, i = [float(v) for v in toks[i:i + k]], i + k
        if cmd == "M":
            if pts:
                polys.append(pts)
            pts, cur, cmd = [tuple(n)], tuple(n), "L"
        elif cmd == "L":
            pts.append(tuple(n))
            cur = tuple(n)
        else:  # cubic Bezier, flattened
            (x0, y0), (x1, y1, x2, y2, x3, y3) = cur, n
            for s in np.linspace(0, 1, 9)[1:]:
                u = 1 - s
                pts.append((u ** 3 * x0 + 3 * u * u * s * x1 + 3 * u * s * s * x2 + s ** 3 * x3,
                            u ** 3 * y0 + 3 * u * u * s * y1 + 3 * u * s * s * y2 + s ** 3 * y3))
            cur = (x3, y3)
    return vb, polys + ([pts] if pts else [])


def mark_masks(width: float):
    """The mark at a given width, split in two 'L' masks: the "A" (largest subpath) and the bee."""
    vb, polys = mark_paths(MARK)
    scale = width / vb[2]
    area = [abs(np.sum(np.array(p)[:, 0] * np.roll(np.array(p)[:, 1], -1)
                       - np.roll(np.array(p)[:, 0], -1) * np.array(p)[:, 1])) for p in polys]
    a = int(np.argmax(area))
    ss, w, h = 4, round(vb[2] * scale), round(vb[3] * scale)
    out = []
    for group in ([polys[a]], [p for j, p in enumerate(polys) if j != a]):
        acc = np.zeros((h * ss, w * ss), bool)
        for p in group:  # even-odd fill, 4x supersampled
            m = Image.new("1", (w * ss, h * ss))
            ImageDraw.Draw(m).polygon([((x - vb[0]) * scale * ss, (y - vb[1]) * scale * ss) for x, y in p], fill=1)
            acc ^= np.asarray(m)
        out.append(Image.fromarray(acc.astype(np.uint8) * 255).resize((w, h), Image.Resampling.LANCZOS))
    return out


def text_sprite(text: str, font, colour, track: float) -> Image.Image:
    """A line of text with extra letter spacing, as an RGBA sprite."""
    ascent, descent = font.getmetrics()
    width = sum(font.getlength(c) for c in text) + track * font.size * (len(text) - 1)
    im = Image.new("RGBA", (round(width) + 4, ascent + descent))
    d, x = ImageDraw.Draw(im), 2.0
    for c in text:
        d.text((x, ascent), c, fill=colour + (255,), font=font, anchor="ls")
        x += font.getlength(c) + track * font.size
    return im


# ---------------------------------------------------------------- the sting

class Sting:
    def __init__(self):
        a_mask, bee_mask = mark_masks(MARK_W)
        ox = (W - a_mask.width) // 2
        box = a_mask.getbbox()
        self.a = a_mask.crop(box)
        self.a_base = (ox + (box[0] + box[2]) / 2, MARK_TOP + box[3])  # bottom centre of the "A"
        box = bee_mask.getbbox()
        self.bee = bee_mask.crop(box)
        self.home = (ox + (box[0] + box[2]) / 2, MARK_TOP + (box[1] + box[3]) / 2)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        glow = np.exp(-(((xx - W / 2) / 520) ** 2 + ((yy - 380) / 420) ** 2))[..., None]
        self.bg = Image.fromarray(np.uint8(np.array(DARK) + (np.array(DARK2) - np.array(DARK)) * glow))

        # scene 1: the wordmark, one sprite per letter so each can pop on its own
        word, track = "ANDTHEBEE", 0.08
        probe = sans(100, "heavy")
        font = sans(round(100 * 820 / (sum(probe.getlength(c) for c in word) + track * 100 * (len(word) - 1))),
                    "heavy")
        x = (W - sum(font.getlength(c) for c in word) - track * font.size * (len(word) - 1)) / 2
        self.letters = []
        for c in word:
            w = font.getlength(c)
            self.letters.append((text_sprite(c, font, GRAY, 0), x + w / 2, 545))
            x += w + track * font.size
        side, gap = 92, 18
        x0 = (W - 4 * side - 3 * gap) / 2
        self.blocks = [(x0 + k * (side + gap) + side / 2, 820 + side / 2, side, c)
                       for k, c in enumerate((GRAY, BLUE, TEAL, LIME))]

        # scene 2: the tagline big enough to read as glyphs, the colour grid, a small mark
        lines, probe = ("SMART", "CONSTRUCTION", "SOLUTIONS"), sans(100, "heavy")
        font = sans(round(100 * 940 / (probe.getlength("CONSTRUCTION") + 0.04 * 100 * 11)), "heavy")
        self.lines = [(text_sprite(s, font, c, 0.04), 200 + 150 * k)
                      for k, (s, c) in enumerate(zip(lines, (LIME, GRAY, TEAL)))]
        side, gap = 104, 14
        self.grid = [(W - 70 - side / 2 - (1 - k % 2) * (side + gap), H - 70 - side / 2 - (1 - k // 2) * (side + gap),
                      side, c) for k, c in enumerate((GRAY, BLUE, TEAL, LIME))]
        a_small, bee_small = mark_masks(300)
        small = Image.new("RGBA", a_small.size)
        small.paste(Image.new("RGBA", a_small.size, LIME + (255,)), (0, 0), a_small)
        small.paste(Image.new("RGBA", a_small.size, GRAY + (255,)), (0, 0), bee_small)
        self.small = small

    # -- motion ------------------------------------------------------
    def bee_at(self, t):
        """Bee centre: an S-shaped flight in from the left, then a gentle hover at home."""
        if t >= LAND:
            k = min(1.0, (t - LAND) / 0.5)
            return self.home[0], self.home[1] + 4 * math.sin(2 * math.pi * 1.2 * (t - LAND)) * k
        u = ease((t - 0.7) / (LAND - 0.7))
        p = [(-160, 330), (420, 20), (40, 660), self.home]
        v = 1 - u
        x = v ** 3 * p[0][0] + 3 * v * v * u * p[1][0] + 3 * v * u * u * p[2][0] + u ** 3 * p[3][0]
        y = v ** 3 * p[0][1] + 3 * v * v * u * p[1][1] + 3 * v * u * u * p[2][1] + u ** 3 * p[3][1]
        return x, y + 16 * math.sin(2 * math.pi * 1.7 * (t - 0.7)) * (1 - u)

    def scene1(self, t) -> Image.Image:
        im = self.bg.copy()
        d = ImageDraw.Draw(im)
        if t >= 0.1:  # the "A" rises from its base, and dips when the bee lands
            s = pop((t - 0.1) / 0.8)
            dip = 1 - 0.06 * math.sin(math.pi * min(1.0, (t - LAND) / 0.35)) if LAND <= t <= LAND + 0.35 else 1
            mask = scaled(self.a, s, dip)
            paint = Image.new("RGB", mask.size, LIME)
            if 4.8 <= t <= 5.7:  # a shine sweeps across
                w, h = mask.size
                x = (t - 4.8) / 0.9 * (w + h) - h
                ImageDraw.Draw(paint).polygon([(x, h), (x + 70, h), (x + 70 + h * 0.6, 0), (x + h * 0.6, 0)],
                                              fill=(250, 255, 215))
            im.paste(paint, (round(self.a_base[0] - mask.width / 2), round(self.a_base[1] - mask.height)), mask)

        if 0.7 <= t < LAND:  # buzz trail
            for k in range(1, 11):
                x, y = self.bee_at(max(0.7, t - k * 0.055))
                r = 7 * (1 - k / 11)
                d.ellipse((x - r, y - r, x + r, y + r), fill=TEAL)
        if t >= 0.7:  # the bee: tilted along its flight, wings fluttering
            x, y = self.bee_at(t)
            if t < LAND:
                x1, y1 = self.bee_at(t + 0.02)
                x0, y0 = self.bee_at(max(0.7, t - 0.02))
                tilt = max(-25.0, min(25.0, -math.degrees(math.atan2(y1 - y0, x1 - x0))))
                tilt *= 1 - ease((t - (LAND - 0.5)) / 0.5)
                flutter = 1 - 0.1 * abs(math.sin(2 * math.pi * 13 * t))
            else:
                tilt, flutter = 0.0, 1 - 0.03 * abs(math.sin(2 * math.pi * 11 * t))
            bee = scaled(self.bee, 1.0, flutter)
            if tilt:
                bee = bee.rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True)
            im.paste(Image.new("RGB", bee.size, GRAY), (round(x - bee.width / 2), round(y - bee.height / 2)), bee)

        if LAND <= t <= LAND + 0.7:  # landing sparkle
            u = (t - LAND) / 0.7
            for j in range(12):
                a = 2 * math.pi * j / 12
                r, s = 40 + 260 * ease(u), 9 * (1 - u)
                x, y = self.home[0] + r * math.cos(a), self.home[1] + r * math.sin(a)
                d.ellipse((x - s, y - s, x + s, y + s), fill=LIME)

        for k, (sprite, cx, top) in enumerate(self.letters):  # the wordmark, letter by letter
            s = pop((t - 3.3 - 0.1 * k) / 0.3)
            if s > 0.02:
                sp = scaled(sprite, s)
                im.paste(sp, (round(cx - sp.width / 2), round(top + sprite.height / 2 - sp.height / 2)), sp)

        for k, (cx, cy, side, colour) in enumerate(self.blocks):  # brand colour blocks
            s = pop((t - 4.4 - 0.12 * k) / 0.3)
            if s > 0.02:
                r = side * s / 2
                d.rounded_rectangle((cx - r, cy - r, cx + r, cy + r), radius=round(10 * s), fill=colour)
        return im

    def frame(self, t) -> Image.Image:
        if t < SCENE2:
            return self.scene1(t)
        fade = 1 - ease((t - SCENE2) / 0.45)  # scene 1 dissolves - as glyphs, it thins out
        im = Image.blend(self.bg, self.scene1(t), fade) if fade > 0 else self.bg.copy()
        d = ImageDraw.Draw(im)
        for k, (sprite, top) in enumerate(self.lines):  # the tagline slides in, line by line
            u = (t - SCENE2 - 0.25 - 0.25 * k) / 0.55
            if u > 0:
                im.paste(sprite, (round(70 - W * (1 - pop(u, 1.2))), top), sprite)
        for k, (cx, cy, side, colour) in enumerate(self.grid):  # the colour grid of the share card
            s = pop((t - SCENE2 - 1.2 - 0.1 * k) / 0.3)
            if s > 0.02:
                r = side * s / 2
                d.rounded_rectangle((cx - r, cy - r, cx + r, cy + r), radius=round(10 * s), fill=colour)
        s = pop((t - SCENE2 - 1.7) / 0.35)
        if s > 0.02:  # and the mark again, small
            sp = scaled(self.small, s)
            im.paste(sp, (round(70 + self.small.width / 2 - sp.width / 2),
                          round(H - 70 - self.small.height / 2 - sp.height / 2)), sp)
        return im


# ---------------------------------------------------------------- sound

def bell(freq, dur):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 4) + 0.35 * np.sin(2 * np.pi * 2.76 * freq * t) * np.exp(-t * 7)


def sting_audio(path: Path, sting: Sting):
    n = int((TOTAL + 0.3) * SR)
    mix = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(5)

    def put(sig, at, gain=1.0, pan=0.0):
        i = int(at * SR)
        j = min(n, i + len(sig))
        s = sig[: j - i] * gain
        if np.ndim(pan):
            pan = pan[: j - i]
        mix[i:j, 0] += s * (1 - np.maximum(pan, 0))
        mix[i:j, 1] += s * (1 + np.minimum(pan, 0))

    def whoosh(at, dur, gain):
        t = np.arange(int(dur * SR)) / SR
        put(np.convolve(rng.standard_normal(len(t)), np.ones(24) / 24, "same") * np.sin(np.pi * t / dur) ** 2,
            at, gain)

    whoosh(0.05, 0.9, 1.6)  # the "A" rises

    t = np.arange(int((LAND - 0.7 + 0.25) * SR)) / SR  # the bee's buzz, panned with its flight
    f = 190 + 25 * np.sin(2 * np.pi * 5.5 * t) + 10 * np.sin(2 * np.pi * 31 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    buzz = sum(np.sin(k * ph) / k for k in range(1, 9))
    buzz *= np.clip(np.minimum(t / 0.3, (t[-1] - t) / 0.35), 0, 1) * (0.65 + 0.35 * np.sin(2 * np.pi * 3 * t) ** 2)
    grid = np.linspace(0.7, LAND, 120)
    pan = np.clip((np.interp(0.7 + t, grid, [sting.bee_at(g)[0] for g in grid]) - W / 2) / (W / 2), -1, 1)
    put(buzz, 0.7, 0.22, pan)

    for k, m in enumerate((84, 88, 91)):  # landing chime
        put(bell(hz(m), 1.2), LAND + 0.07 * k, 0.32)
    for k, m in enumerate((72, 74, 76, 79, 81, 84, 86, 88, 91)):  # one pluck per letter
        put(voice(hz(m), 0.18, SQUARE, 14, 0.0), 3.3 + 0.1 * k, 0.12)
    for k, m in enumerate((79, 84, 88, 91)):  # a blip per colour block
        put(voice(hz(m), 0.12, TRIANGLE, 10, 0.0), 4.4 + 0.12 * k, 0.3)
    pad = TOTAL - 4.2
    for m in (48, 60, 64, 67, 71):  # a warm Cmaj7 underneath
        put(voice(hz(m), pad, TRIANGLE, 0.4, 0.6) * np.clip(np.linspace(0, 3, int(pad * SR)), 0, 1), 4.2, 0.1)

    whoosh(SCENE2 - 0.1, 0.7, 1.2)  # into scene 2
    for k, m in enumerate((79, 84, 88)):  # a bell per tagline line
        put(bell(hz(m), 1.0), SCENE2 + 0.25 + 0.25 * k, 0.25)
    for k, m in enumerate((84, 88, 91, 96)):  # grid blips
        put(voice(hz(m), 0.12, TRIANGLE, 10, 0.0), SCENE2 + 1.2 + 0.1 * k, 0.25)
    for m in (72, 79, 84):  # the small mark: a last chime
        put(bell(hz(m), 2.0), SCENE2 + 1.7, 0.2)

    mix *= 0.89 / np.abs(mix).max()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((mix * 32767).astype("<i2").tobytes())


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/andthebee_logo.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    sting = Sting()
    with tempfile.TemporaryDirectory() as tmp:
        sound = Path(tmp) / "sting.wav"
        sting_audio(sound, sting)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(sound)]
            + codec() + ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
        n, clock = int(TOTAL * FPS), time.time()
        for i in range(n):
            enc.stdin.write(np.asarray(sting.frame(i / FPS)).tobytes())
            if i % 15 == 0 or i + 1 == n:
                print(f"\r  drawing {i + 1}/{n}  {(i + 1) / (time.time() - clock):5.1f} fps",
                      end="", file=sys.stderr)
        enc.stdin.close()
        enc.wait()
    print(f"\n  -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
