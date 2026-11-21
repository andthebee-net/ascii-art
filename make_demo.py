#!/usr/bin/env python3
"""make_demo.py - an original dancing duo, Kongi (pea sprout) & Gyuli (tangerine).

A test clip for glyph_engine.py, drawn from scratch with Pillow, with a little
chiptune synthesised in numpy.

    python3 make_demo.py              # -> demo/kongi_gyuli.mp4 (1080x1080, 30 fps, ~17 s)
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glyph_engine import codec

W = H = 1080
FPS = 30
BPM = 120
BEAT = 60 / BPM
SONG = 32 * BEAT                # 8 bars of dancing
TOTAL = SONG + 1.2              # + the final pose
SR = 44100

FLOOR = int(H * 0.76)           # where the dance floor starts
FEET = int(H * 0.93)            # where the duo stands
BIG = 1.6                       # duo size: a close camera keeps their faces readable as glyphs
VANISH = FLOOR - 520            # vanishing point height of the floor grid
GLOW = [(255, 84, 170), (72, 190, 255), (255, 196, 64), (124, 236, 120)]
INK = (36, 26, 32)
CHEEK = (255, 120, 150)
DUO = {
    "kongi": dict(body=(150, 214, 96), edge=(58, 120, 46), limb=(122, 188, 78),
                  shine=(214, 246, 176), x=0.27),
    "gyuli": dict(body=(255, 162, 54), edge=(190, 94, 24), limb=(240, 132, 42),
                  shine=(255, 218, 156), x=0.73),
}
_rng = random.Random(7)
STARS = [(_rng.uniform(0, W), _rng.uniform(0, FLOOR - 40), _rng.uniform(0, 6.3),
          _rng.uniform(1.5, 4.0), _rng.choice((2, 2, 3, 4))) for _ in range(80)]
CONFETTI = [(_rng.uniform(0, W), _rng.uniform(0, H), _rng.uniform(150, 260),
             _rng.uniform(0, 6.3), _rng.choice(GLOW)) for _ in range(90)]


# ---------------------------------------------------------------- stage

def backdrop():
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    sky = np.array([10, 8, 26], np.float32) * (1 - y) + np.array([34, 16, 58], np.float32) * y
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.exp(-1.8 * (((xx - W / 2) / (W * 0.46)) ** 2 + ((yy - H * 0.55) / (H * 0.42)) ** 2))
    return sky, glow


def stars(d, t):
    for x, y, phase, speed, r in STARS:
        b = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * speed + phase)) ** 2
        c = tuple(int(v * b) for v in (255, 244, 214))
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
        if r >= 3:
            d.line((x - 3 * r, y, x + 3 * r, y), fill=c)
            d.line((x, y - 3 * r, x, y + 3 * r), fill=c)


def floor(d, beat):
    b, p = int(beat), beat % 1
    ys = [FLOOR + (H - FLOOR) * (k / 6) ** 1.35 for k in range(7)]

    def x(j, y):
        return W / 2 + j * 150 * (y - VANISH) / (H - VANISH)

    lit = random.Random(b * 7919 + 11)
    flash = 0.25 + 0.75 * math.exp(-p * 3)
    for k in range(6):
        for j in range(-6, 6):
            tile = (30, 20, 62) if (j + k) % 2 else (40, 28, 80)
            if beat < 32 and lit.random() < 0.2:
                g = GLOW[(j + k + b) % 4]
                tile = tuple(int(a + (c - a) * flash * 0.55) for a, c in zip(tile, g))
            d.polygon([(x(j, ys[k]), ys[k]), (x(j + 1, ys[k]), ys[k]),
                       (x(j + 1, ys[k + 1]), ys[k + 1]), (x(j, ys[k + 1]), ys[k + 1])],
                      fill=tile, outline=(60, 42, 110))


def confetti(d, t, amount):
    for x0, y0, v, ph, c in CONFETTI[: int(len(CONFETTI) * amount)]:
        y = (y0 + v * t) % (H + 40) - 20
        x = x0 + 18 * math.sin(t * 2.2 + ph)
        a = t * 4 + ph
        dx, dy, ex, ey = 9 * math.cos(a), 9 * math.sin(a), -4 * math.sin(a), 4 * math.cos(a)
        d.polygon([(x - dx - ex, y - dy - ey), (x + dx - ex, y + dy - ey),
                   (x + dx + ex, y + dy + ey), (x - dx + ex, y - dy + ey)], fill=c)


def confetti_amount(beat):
    if 8 <= beat < 16:
        return min(1.0, (beat - 8) / 2)
    if 16 <= beat < 18:
        return 1 - (beat - 16) / 2
    return min(1.0, max(0.0, beat - 30))


# ---------------------------------------------------------------- the duo

def ellipse_pts(cx, cy, rx, ry, ang=0.0, n=40):
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(n):
        u = 2 * math.pi * i / n
        x, y = rx * math.cos(u), ry * math.sin(u)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def leaf(d, x, y, length, ang, fill, edge, ow):
    """A leaf growing from (x, y) towards angle ang (0 = right, -pi/2 = up)."""
    pts = ellipse_pts(x + math.cos(ang) * length / 2, y + math.sin(ang) * length / 2,
                      length / 2, length / 4.4, ang)
    d.polygon(pts, fill=fill)
    d.line(pts + pts[:1], fill=edge, width=ow, joint="curve")
    d.line([(x, y), (x + math.cos(ang) * length * 0.78, y + math.sin(ang) * length * 0.78)],
           fill=edge, width=max(1, ow // 2))


def limb(d, p0, p1, w, fill, edge, ow):
    for colour, ww in ((edge, w + 2 * ow), (fill, w)):
        d.line([p0, p1], fill=colour, width=round(ww))
        for x, y in (p0, p1):
            d.ellipse((x - ww / 2, y - ww / 2, x + ww / 2, y + ww / 2), fill=colour)


def buddy(kind, P, L=round(880 * BIG)):
    """Draw one of the duo on a transparent 2x layer. Returns the layer and its feet point."""
    c = DUO[kind]
    im = Image.new("RGBA", (L, L), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ow = round(L * 0.013)
    feet = L * 0.92
    rx, ry = L * 0.29 * P["sx"], L * 0.26 * P["sy"]
    cx, cy = L / 2, feet - L * 0.035 - ry
    front, turn = math.cos(P["spin"]), math.sin(P["spin"])

    for s, a in ((-1, P["armL"]), (1, P["armR"])):  # arms sit behind the body
        a = math.radians(a)
        x0, y0 = cx + s * rx * 0.84 * max(abs(front), 0.25), cy + ry * 0.15
        limb(d, (x0, y0), (x0 + s * math.sin(a) * L * 0.12, y0 + math.cos(a) * L * 0.12),
             L * 0.065, c["limb"], c["edge"], ow)
    for s, lift in ((-1, P["footL"]), (1, P["footR"])):
        fx, fy = cx + s * L * 0.115, feet - L * 0.03 - lift
        d.ellipse((fx - L * 0.07, fy - L * 0.038, fx + L * 0.07, fy + L * 0.038),
                  fill=c["limb"], outline=c["edge"], width=ow)

    top, sway = cy - ry, math.radians(P["wiggle"])
    if kind == "kongi":  # a sprout with two leaves
        tip = (cx + math.sin(sway) * L * 0.05, top - L * 0.085)
        d.line([(cx, top + L * 0.03), tip], fill=c["edge"], width=round(L * 0.02))
        for ang in (-150, -30):
            leaf(d, *tip, L * 0.13 * max(abs(front), 0.5), math.radians(ang) + sway,
                 (108, 198, 78), c["edge"], ow)
    else:  # a stem nub with one leaf
        leaf(d, cx + L * 0.01, top - L * 0.012, L * 0.14, math.radians(-28) + sway,
             (96, 182, 74), (46, 104, 36), ow)
        d.ellipse((cx - L * 0.026, top - L * 0.03, cx + L * 0.026, top + L * 0.014),
                  fill=(120, 94, 44), outline=(72, 54, 24), width=max(1, ow // 2))

    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=c["body"], outline=c["edge"], width=ow)
    d.ellipse((cx - rx * 0.66, cy - ry * 0.78, cx - rx * 0.22, cy - ry * 0.42), fill=c["shine"])
    if kind == "gyuli":  # peel pores that turn with the body
        for lon, v in ((0.9, -0.2), (1.6, 0.3), (2.4, -0.45), (3.3, 0.15), (4.1, 0.5), (5.2, -0.3)):
            if math.cos(lon + P["spin"]) > 0.15:
                px = cx + rx * 0.85 * math.sin(lon + P["spin"]) * math.sqrt(1 - v * v)
                py = cy + ry * v
                d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(232, 136, 40))

    if front > 0.15:  # the face slides round the body as it spins
        k = front
        fx, ey = cx + turn * rx * 0.62, cy - ry * 0.02
        gap = L * (0.105 if kind == "kongi" else 0.112) * k  # big eyes survive small grids
        ew, eh = L * 0.04 * max(k, 0.4), L * 0.056
        if kind == "gyuli":
            ew, eh = ew * 1.12, eh * 1.06
        cheek = tuple(int(a * 0.45 + b * 0.55) for a, b in zip(c["body"], CHEEK))
        for s in (-1, 1):
            x = fx + s * gap
            if P["happy"]:
                d.line([(x - ew * 1.2, ey + eh * 0.25), (x, ey - eh * 0.55), (x + ew * 1.2, ey + eh * 0.25)],
                       fill=INK, width=ow + 2, joint="curve")
            elif P["blink"]:
                d.line([(x - ew * 1.1, ey), (x + ew * 1.1, ey)], fill=INK, width=ow + 2)
            else:
                d.ellipse((x - ew, ey - eh, x + ew, ey + eh), fill=INK)
                r = ew * 0.42
                d.ellipse((x - ew * 0.15 - r, ey - eh * 0.42 - r, x - ew * 0.15 + r, ey - eh * 0.42 + r),
                          fill=(255, 255, 255))
            ch = fx + s * gap * 1.75
            d.ellipse((ch - L * 0.04 * k, ey + L * 0.06, ch + L * 0.04 * k, ey + L * 0.09), fill=cheek)
        my, mw = ey + L * 0.08, L * 0.04 * max(k, 0.4)
        if P["mouth"]:
            d.ellipse((fx - mw, my - L * 0.012, fx + mw, my + L * 0.05), fill=(140, 36, 52),
                      outline=INK, width=max(1, ow // 2))
            d.ellipse((fx - mw * 0.6, my + L * 0.022, fx + mw * 0.6, my + L * 0.048), fill=(255, 128, 150))
        else:
            d.arc((fx - mw, my - L * 0.03, fx + mw, my + L * 0.012), 20, 160, fill=INK, width=ow)
    return im, (L / 2, feet)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def pose(t, who):
    """Choreography: A sway (bars 1-2), B jumps (3-4), C spin canon (5-6), D waves + finale."""
    beat = t / BEAT
    b, p = int(beat), beat % 1
    mirror = 1 if who == 0 else -1
    P = dict(sx=1.0, sy=1.0, lift=0.0, tilt=0.0, x=0.0, armL=25.0, armR=25.0, spin=0.0,
             blink=(t + who * 1.37) % 3.3 < 0.13, mouth=False, happy=False,
             wiggle=10 * math.sin(2 * math.pi * beat + who), footL=0.0, footR=0.0)
    land = math.exp(-(p / 0.11) ** 2) + math.exp(-((1 - p) / 0.11) ** 2)  # peaks on each beat
    hop = math.sin(math.pi * p)  # 0 -> 1 -> 0 between beats

    if beat >= 32:  # finale: hold the pose, breathing
        br = math.sin(t * 6)
        P.update(armL=155, armR=155, happy=True, mouth=True, sx=1 + 0.02 * br, sy=1 - 0.02 * br)
    elif beat < 8:  # A: side-step sway, in sync
        s = math.sin(math.pi * beat)
        P.update(x=34 * s, tilt=-7 * s, lift=14 * hop,
                 sx=1 + 0.07 * land, sy=1 - 0.07 * land,
                 armL=30 + 70 * max(0, s), armR=30 + 70 * max(0, -s),
                 footL=18 * max(0, s), footR=18 * max(0, -s))
    elif beat < 16:  # B: hooray jumps
        P.update(lift=95 * hop, sx=1 + 0.14 * land - 0.05 * hop, sy=1 - 0.14 * land + 0.07 * hop,
                 armL=40 + 115 * hop, armR=40 + 115 * hop, happy=hop > 0.35, mouth=hop > 0.2,
                 tilt=4 * math.sin(math.pi * beat) * mirror)
    elif beat < 24:  # C: a spin each, one beat apart, then a wiggle
        q = beat - 16 - who
        if q < 0:
            P.update(lift=10 * hop, sx=1 + 0.05 * land, sy=1 - 0.05 * land)
        elif q % 4 < 2:
            u = (q % 4) / 2
            P.update(spin=2 * math.pi * ease(u) * mirror, lift=40 * math.sin(math.pi * u),
                     armL=70, armR=70, mouth=True)
        else:
            w = math.sin(4 * math.pi * (q % 4 - 2))
            P.update(tilt=12 * w, armL=50 + 40 * w, armR=50 - 40 * w,
                     sx=1 + 0.05 * land, sy=1 - 0.05 * land, happy=True)
    elif beat < 30:  # D: arm waves
        w = math.sin(math.pi * beat + who * math.pi / 2)
        P.update(lift=20 * hop, sx=1 + 0.08 * land, sy=1 - 0.08 * land,
                 armL=100 + 55 * w, armR=100 - 55 * w, tilt=5 * w * mirror, mouth=b % 2 == 0)
    else:  # one big jump into the finale
        u = (beat - 30) / 2
        up = math.sin(math.pi * u)
        P.update(lift=150 * up, armL=60 + 95 * ease(u * 1.5), armR=60 + 95 * ease(u * 1.5),
                 happy=True, mouth=True, sy=1 + 0.08 * up, sx=1 - 0.05 * up)
    return P


def frame(t, sky, glow):
    beat = t / BEAT
    b, p = int(beat), beat % 1
    level = 0.06 + 0.16 * math.exp(-p * 3.2) if beat < 32 else 0.16  # faint glow: glyphs go to the duo
    tint = np.array(GLOW[(b // 4) % 4] if beat < 32 else (255, 220, 120), np.float32)
    im = Image.fromarray(np.clip(sky + glow[..., None] * tint * level, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    stars(d, t)
    floor(d, beat)
    poses = [pose(t, who) for who in range(len(DUO))]
    for P, c in zip(poses, DUO.values()):  # shadows shrink as they jump
        x, k = W * c["x"] + P["x"] * BIG, 1 - min(P["lift"], 200) / 330
        d.ellipse((x - 185 * k, FEET - 22 * k, x + 185 * k, FEET + 22 * k), fill=(14, 9, 30))
    for P, (kind, c) in zip(poses, DUO.items()):
        layer, (ax, ay) = buddy(kind, P)
        if P["tilt"]:
            layer = layer.rotate(P["tilt"], resample=Image.Resampling.BICUBIC, center=(ax, ay))
        layer = layer.resize((layer.width // 2, layer.height // 2), Image.Resampling.LANCZOS)
        x = W * c["x"] + P["x"] * BIG
        im.paste(layer, (round(x - ax / 2), round(FEET - ay / 2 - P["lift"] * BIG)), layer)
    confetti(d, t, confetti_amount(beat))
    return np.asarray(im)


# ---------------------------------------------------------------- the tune

SQUARE = [(k, 1 / k) for k in (1, 3, 5, 7, 9, 11, 13)]
TRIANGLE = [(k, (-1) ** ((k - 1) // 2) / k ** 2) for k in (1, 3, 5, 7, 9)]
ROOTS = [48, 45, 41, 43, 48, 45, 41, 43]          # C Am F G, twice
MELODY = [                                         # eighth notes; 0 holds, None rests
    [76, 79, 81, 79, 76, 0, 72, 74], [76, 0, 69, 72, 76, 74, 72, 0],
    [81, 0, 79, 77, 76, 77, 81, 0], [79, 0, 74, 76, 79, 81, 83, 0],
    [84, 83, 81, 79, 76, 0, 79, 81], [76, 0, 72, 74, 76, 79, 76, 74],
    [72, 74, 77, 81, 79, 77, 76, 74], [79, 0, 0, None, 83, 0, 84, 0],
]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def voice(freq, dur, harmonics, decay, sustain=0.5):
    t = np.arange(int(dur * SR)) / SR
    sig = sum(a * np.sin(2 * np.pi * k * freq * t) for k, a in harmonics if k * freq < SR * 0.45)
    env = np.minimum(1, t / 0.006) * (sustain + (1 - sustain) * np.exp(-t * decay))
    return sig * env * np.clip((dur - t) / 0.02, 0, 1)


def soundtrack(path: Path):
    n = int((TOTAL + 0.5) * SR)
    mix = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(3)

    def put(sig, at, gain=1.0, pan=0.0):
        i = int(at * SR)
        j = min(n, i + len(sig))
        s = sig[: j - i] * gain
        mix[i:j, 0] += s * (1 - max(pan, 0))
        mix[i:j, 1] += s * (1 + min(pan, 0))

    t = np.arange(int(0.4 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-t * 30)) / SR) * np.exp(-t * 8)
    ts = t[: int(0.22 * SR)]
    snare = (np.diff(rng.standard_normal(len(ts) + 1)) * 0.3 * np.exp(-ts * 20)
             + np.sin(2 * np.pi * 190 * ts) * 0.6 * np.exp(-ts * 28))
    th = t[: int(0.06 * SR)]
    hat = np.diff(rng.standard_normal(len(th) + 2), 2) * 0.12 * np.exp(-th * 75)
    tc = np.arange(int(1.4 * SR)) / SR
    crash = np.diff(rng.standard_normal(len(tc) + 1)) * 0.3 * np.exp(-tc * 2.6)

    for b in range(32):
        put(kick, b * BEAT, 0.9)
        if b % 4 in (1, 3):
            put(snare, b * BEAT, 0.5)
        put(hat, b * BEAT + BEAT / 2, 0.5, pan=0.3)
    for k in range(8):  # a little snare fill into the finale
        put(snare, 30 * BEAT + k * BEAT / 4, 0.15 + 0.05 * k)
    for bar, root in enumerate(ROOTS):
        for e in range(8):
            put(voice(hz(root + 12 * (e % 2)), 0.2, TRIANGLE, 9, 0.3), bar * 4 * BEAT + e * BEAT / 2, 0.42)
    slots = [m for bar in MELODY for m in bar]
    i = 0
    while i < len(slots):
        j = i + 1
        while j < len(slots) and slots[j] == 0:
            j += 1
        if slots[i]:
            put(voice(hz(slots[i]), (j - i) * BEAT / 2 * 0.9, SQUARE, 6), i * BEAT / 2, 0.22, pan=-0.1)
        i = j
    put(crash, SONG, 0.35)
    put(kick, SONG, 1.0)
    for m in (60, 64, 67, 72, 76):
        put(voice(hz(m), 1.15, SQUARE, 3, 0.25), SONG, 0.09)
    put(voice(hz(36), 1.15, TRIANGLE, 2, 0.4), SONG, 0.5)

    mix *= 0.89 / np.abs(mix).max()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((mix * 32767).astype("<i2").tobytes())


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/kongi_gyuli.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        song = Path(tmp) / "song.wav"
        soundtrack(song)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(song)]
            + codec() + ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
        sky, glow = backdrop()
        n, clock = int(TOTAL * FPS), time.time()
        for i in range(n):
            enc.stdin.write(frame(i / FPS, sky, glow).tobytes())
            if i % 15 == 0 or i + 1 == n:
                print(f"\r  drawing {i + 1}/{n}  {(i + 1) / (time.time() - clock):5.1f} fps",
                      end="", file=sys.stderr)
        enc.stdin.close()
        enc.wait()
    print(f"\n  -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
