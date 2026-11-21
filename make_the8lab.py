#!/usr/bin/env python3
"""make_the8lab.py - a The8.Lab logo sting to feed glyph_engine.py.

Scene 1: a pen of light draws the symbol's two rings as one figure-8 stroke, the two
spokes click in, a pulse keeps running round the 8 and the wordmark pops in.
Scene 2: AX & SMART / CONSTRUCTION is swept in by a scan line, then the horizontal logo.
Logo: assets/the8lab-logo.svg (the8lab.co.kr). Colours: the site's hero and accents.

    python3 make_the8lab.py           # -> demo/the8lab_logo.mp4 (1080x1080, 30 fps, 10.5 s)
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
from PIL import Image, ImageDraw

from glyph_engine import codec
from make_demo import SQUARE, TRIANGLE, hz, voice
from make_logo import bell, ease, pop, sans, text_sprite

W = H = 1080
FPS = 30
TOTAL = 10.5
SR = 44100
LOGO = Path(__file__).with_name("assets") / "the8lab-logo.svg"

# the8lab.co.kr: the accent teal, its light purple (the accent #5B1FDB is too dark to read as
# glyphs), and the hero - a dark teal backdrop with colour fields drifting across it
TEAL, LILAC, WHITE = np.float32([0, 190, 190]), np.float32([172, 136, 255]), np.float32([242, 244, 245])
AQUA = (90, 224, 220)           # the teal, lightened so big type reads as dense glyphs
HERO = [(0.0, (0x10, 0x21, 0x27)), (0.52, (0x07, 0x13, 0x13)), (1.0, (0x02, 0x05, 0x04))]
FIELDS = [((255, 235, 32), 0.08, 0.78), ((255, 104, 22), 0.20, 0.18), ((107, 18, 255), 0.42, 0.82),
          ((0, 215, 143), 0.82, 0.72), ((16, 126, 92), 0.92, 0.18)]
FIELD = 0.18                    # how strongly the colour fields light the backdrop

# sizes keep everything inside ~930 px across, so --fill in a portrait terminal crops nothing
SYMBOL_W, SYMBOL_TOP = 500, 145  # scene 1: the symbol big, the wordmark under it
WORD_W, WORD_TOP = 880, 740
LINE_TOP, LINE_GAP = 220, 150    # scene 2: the tagline, then the horizontal logo
LOCKUP_W, LOCKUP_TOP = 920, 615
DRAW = (0.4, 2.6)               # the pen draws the figure-8
CLICK = 3.0                     # the spokes lock in
LETTERS = 3.35                  # the wordmark starts popping in
SHINE = 4.7
LOOP = 1.9                      # seconds per lap of the pulse
SCENE2 = 6.0
LOCKUP = SCENE2 + 1.6


# ---------------------------------------------------------------- the logo

def flatten(d: str):
    """An SVG path's subpaths as point arrays (absolute M/L/H/V/C/Z; curves flattened)."""
    toks = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?", d)
    polys, pts, cur, cmd, i = [], [], (0.0, 0.0), "", 0
    while i < len(toks):
        if toks[i].isalpha():
            cmd, i = toks[i], i + 1
            if cmd not in "MLHVCZ":
                sys.exit(f"make_the8lab: path command {cmd!r} not supported (absolute M/L/H/V/C/Z only)")
            if cmd == "Z":
                if pts:
                    polys.append(np.array(pts))
                pts = []
                continue
        k = {"C": 6, "H": 1, "V": 1}.get(cmd, 2)
        n, i = [float(v) for v in toks[i:i + k]], i + k
        if cmd == "M":
            if pts:
                polys.append(np.array(pts))
            pts, cur, cmd = [tuple(n)], tuple(n), "L"
        elif cmd == "C":  # cubic Bezier, flattened
            (x0, y0), (x1, y1, x2, y2, x3, y3) = cur, n
            for s in np.linspace(0, 1, 9)[1:]:
                u = 1 - s
                pts.append((u ** 3 * x0 + 3 * u * u * s * x1 + 3 * u * s * s * x2 + s ** 3 * x3,
                            u ** 3 * y0 + 3 * u * u * s * y1 + 3 * u * s * s * y2 + s ** 3 * y3))
            cur = (x3, y3)
        else:
            cur = (n[0], cur[1]) if cmd == "H" else (cur[0], n[0]) if cmd == "V" else tuple(n)
            pts.append(cur)
    return polys + ([np.array(pts)] if pts else [])


def logo_paths(svg: Path = LOGO):
    """Every <path> of the logo as a list of point arrays in viewBox units, transforms applied."""
    out = []
    for attrs in re.findall(r"<path\b([^>]*)>", svg.read_text()):
        m, tr = np.eye(3), re.search(r'\stransform="([^"]*)"', attrs)
        if tr and re.sub(r"matrix\([^)]*\)", "", tr.group(1)).strip():
            sys.exit(f"make_the8lab: transform {tr.group(1)!r} not supported (matrix() only)")
        for args in re.findall(r"matrix\(([^)]*)\)", tr.group(1) if tr else ""):
            a, b, c, d, e, f = (float(v) for v in args.replace(",", " ").split())
            m = m @ np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        out.append([p @ m[:2, :2].T + m[:2, 2] for p in flatten(re.search(r'\sd="([^"]+)"', attrs).group(1))])
    return out


def bounds(paths):
    pts = np.concatenate([p for polys in paths for p in polys])
    return pts.min(0), pts.max(0)


def flat(parts):
    """Parts (lists of paths) -> one list of paths."""
    return [path for part in parts for path in part]


def logo_parts():
    """The logo as (rings, spokes, letters), each part a list of paths.

    Every shape in the file has a twin path, its outline stroke, sharing its centre - the two
    are kept together. Rings: the two biggest parts, top-left first. Spokes: the parts inside
    the rings' box, left first. Letters: the rest, left to right."""
    parts = []
    for polys in logo_paths():
        lo, hi = bounds([polys])
        c = (lo + hi) / 2
        twin = next((part for part in parts if np.hypot(*(part[0] - c)) < 3), None)
        if twin:
            twin[1].append(polys)
        else:
            parts.append([c, [polys]])
    groups = sorted((g for _, g in parts), key=lambda g: -np.prod(np.subtract(*bounds(g)[::-1])))
    rings = sorted(groups[:2], key=lambda g: bounds(g)[0][1])
    lo, hi = bounds(rings[0] + rings[1])
    inside = [np.all(bounds(g)[0] >= lo) and np.all(bounds(g)[1] <= hi) for g in groups[2:]]
    spokes = sorted((g for g, i in zip(groups[2:], inside) if i), key=lambda g: bounds(g)[0][0])
    letters = sorted((g for g, i in zip(groups[2:], inside) if not i), key=lambda g: bounds(g)[0][0])
    return rings, spokes, letters


def raster(paths, scale, lo, size, ss=4):
    """Paths -> anti-aliased 0..1 mask of `size` px whose top-left is `lo` (viewBox units).
    Even-odd inside each path, union across paths, 4x supersampled."""
    acc = np.zeros((size[1] * ss, size[0] * ss), bool)
    for polys in paths:
        one = np.zeros_like(acc)
        for p in polys:
            im = Image.new("1", (size[0] * ss, size[1] * ss))
            ImageDraw.Draw(im).polygon([tuple(v) for v in (p - lo) * scale * ss], fill=1)
            one ^= np.asarray(im)
        acc |= one
    im = Image.fromarray(acc.astype(np.uint8) * 255).resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im, np.float32) / 255


def placed(paths, scale):
    """Top-left (viewBox units, one pixel of margin) and pixel size that fit these paths."""
    lo, hi = bounds(paths)
    lo = lo - 1 / scale
    return lo, tuple(int(v) for v in np.ceil((hi - lo) * scale) + 2)


class Mark:
    """The symbol at one size. Its rings are one figure-8 stroke: ring 1 runs 270 degrees
    clockwise from 6 o'clock, ring 2 from 12 o'clock, and they meet at the spokes' joints.
    Each spoke grows out of its joint: the left one up, the right one down."""
    START = (90.0, 270.0)

    def __init__(self, rings, spokes, scale):
        self.lo, self.size = placed(flat(rings + spokes), scale)
        self.rings = [raster(g, scale, self.lo, self.size) for g in rings]
        # logo.svg fills the spokes even-odd, which leaves them hollow; the site's symbol file
        # (logo-symbol.svg) has them solid - so each spoke is its outline's box, filled
        self.spokes = [raster([[np.array([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])])]], scale, self.lo, self.size)
                       for lo, hi in map(bounds, spokes)]
        yy, xx = np.mgrid[0:self.size[1], 0:self.size[0]].astype(np.float32)
        self.rows = yy[:, :1]
        self.centre, self.radius, self.angle, self.loop = [], [], [], []
        for k, (g, m) in enumerate(zip(rings, self.rings)):
            lo, hi = bounds(g)
            c = ((lo + hi) / 2 - self.lo) * scale
            a = (np.degrees(np.arctan2(yy - c[1], xx - c[0])) - self.START[k] + 45) % 360 - 45
            self.centre.append(c)
            self.radius.append(float((np.hypot(xx - c[0], yy - c[1]) * m).sum() / m.sum()))
            self.angle.append(a)                         # degrees along this ring, from its start
            self.loop.append((a + 270 * k) / 540)        # position along the whole 8, 0..1
        self.spans = [np.nonzero(m.max(1) > 0.5)[0][[0, -1]] + (0, 1) for m in self.spokes]
        self.spoke_x = [float(np.nonzero(m.max(0) > 0.5)[0].mean()) for m in self.spokes]

    def at(self, s):
        """The point at loop position s (0..1, wrapping) on the figure-8's centre line."""
        s %= 1.0
        k = int(s >= 0.5)
        a = math.radians(self.START[k] + (s - 0.5 * k) * 540)
        return self.centre[k][0] + self.radius[k] * math.cos(a), self.centre[k][1] + self.radius[k] * math.sin(a)

    def drawn(self, p):
        """The rings as far as a pen that has gone a fraction p round the 8 has drawn them."""
        out = []
        for k in range(2):
            q = min(max(p * 2 - k, 0.0), 1.0)
            sweep = -90.0 if q <= 0 else 400.0 if q >= 1 else q * 270
            out.append(self.rings[k] * np.clip((sweep - self.angle[k]) / 4 + 0.5, 0, 1))
        return out

    def grown(self, g):
        """The spokes grown a fraction g out of their joints."""
        (t0, b0), (t1, b1) = self.spans
        return (self.spokes[0] * np.clip(self.rows - (b0 - (b0 - t0) * g) + 0.5, 0, 1),
                self.spokes[1] * np.clip(t1 + (b1 - t1) * g - self.rows + 0.5, 0, 1))

    def tips(self, g):
        (t0, b0), (t1, b1) = self.spans
        return (self.spoke_x[0], b0 - (b0 - t0) * g), (self.spoke_x[1], t1 + (b1 - t1) * g)

    def comet(self, s, tail=0.14):
        """Ring weights of a pulse whose head is at loop position s, fading along its tail."""
        return [np.clip(1 - ((s - lp) % 1.0) / tail, 0, 1) ** 1.5 * m for lp, m in zip(self.loop, self.rings)]


class Word:
    """The wordmark at one size: a mask per letter and its pixel offset from `origin` (viewBox
    units, less the letters' pixel of margin - so offsets start at 0)."""

    def __init__(self, letters, scale, origin):
        origin, self.letters = origin - 1 / scale, []
        for g in letters:
            lo, size = placed(g, scale)
            self.letters.append((raster(g, scale, lo, size), (lo - origin) * scale))
        self.size = tuple(int(max(o[k] + m.shape[1 - k] for m, o in self.letters)) + 1 for k in range(2))

    def union(self):
        out = np.zeros(self.size[::-1], np.float32)
        for m, (ox, oy) in self.letters:
            x, y = int(round(ox)), int(round(oy))
            np.maximum(out[y:y + m.shape[0], x:x + m.shape[1]], m, out=out[y:y + m.shape[0], x:x + m.shape[1]])
        return out


# ---------------------------------------------------------------- drawing helpers

def paint(arr, m, x, y, colour, alpha=1.0):
    """Blend a flat colour into the frame through mask m (0..1) with its top-left at (x, y)."""
    x, y = int(round(x)), int(round(y))
    h, w = m.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 < x1 and y0 < y1:
        a = m[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
        arr[y0:y1, x0:x1] += (colour - arr[y0:y1, x0:x1]) * a


def glow(arr, x, y, r, colour, gain=1.0):
    """Add a soft round light of radius ~r at (x, y)."""
    x0, y0 = max(int(x - 3 * r), 0), max(int(y - 3 * r), 0)
    x1, y1 = min(int(x + 3 * r) + 1, W), min(int(y + 3 * r) + 1, H)
    if x0 < x1 and y0 < y1:
        yy, xx = np.ogrid[y0:y1, x0:x1]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (r * r)).astype(np.float32)
        arr[y0:y1, x0:x1] += g[..., None] * (colour * gain)


def resized(m, s):
    h, w = m.shape
    return np.asarray(Image.fromarray(m).resize((max(1, round(w * s)), max(1, round(h * s))),
                                                Image.Resampling.BILINEAR))


def mix(a, b, u):
    return a + (b - a) * u


# ---------------------------------------------------------------- the sting

class Sting:
    def __init__(self):
        rings, spokes, letters = logo_parts()
        (s_lo, s_hi), (w_lo, w_hi) = bounds(flat(rings + spokes)), bounds(flat(letters))
        l_lo, l_hi = bounds(flat(rings + spokes + letters))

        self.mark = Mark(rings, spokes, SYMBOL_W / (s_hi[0] - s_lo[0]))
        self.mark_xy = ((W - self.mark.size[0]) / 2, SYMBOL_TOP)
        self.word = Word(letters, WORD_W / (w_hi[0] - w_lo[0]), w_lo)
        self.word_xy = ((W - self.word.size[0]) / 2, WORD_TOP)
        self.word_mask = self.word.union()
        self.accent = {3: TEAL, 4: LILAC}           # the "8" and the "." light up as they land

        scale = LOCKUP_W / (l_hi[0] - l_lo[0])      # scene 2: the horizontal logo
        self.lock_mark, self.lock_word = Mark(rings, spokes, scale), Word(letters, scale, l_lo)
        self.lock_xy = ((W - LOCKUP_W) / 2, LOCKUP_TOP)
        self.lock_mark_xy = tuple(self.lock_xy[k] + (self.lock_mark.lo[k] - l_lo[k]) * scale + 1 for k in range(2))

        probe = sans(100, "heavy")
        font = sans(round(100 * 940 / (probe.getlength("CONSTRUCTION") + 0.04 * 100 * 11)), "heavy")
        self.lines = []
        for k, (text, colour) in enumerate(zip(("AX & SMART", "CONSTRUCTION"), (np.float32(AQUA), WHITE))):
            sprite = text_sprite(text, font, (255, 255, 255), 0.04)
            m = np.asarray(sprite.crop(sprite.getbbox()).getchannel("A"), np.float32) / 255
            self.lines.append((m, (W - m.shape[1]) / 2, LINE_TOP + LINE_GAP * k, colour))

        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot(xs - 0.5 * W, ys - 0.45 * H) / math.hypot(0.5 * W, 0.55 * H)
        self.base = np.stack([np.interp(d, [s for s, _ in HERO], [c[k] for _, c in HERO]) for k in range(3)],
                             -1).astype(np.float32)
        mx, my = self.mark_xy
        self.ring_dist = np.hypot(xs - mx - self.mark.size[0] / 2, ys - my - self.mark.size[1] / 2)
        self.xs, self.ys = xs[0], ys[:, 0]

    def backdrop(self, t, amount):
        """The hero: a dark teal glow with soft colour fields drifting across it."""
        arr = self.base.copy()
        for k, (colour, fx, fy) in enumerate(FIELDS):
            cx = (fx + 0.06 * math.sin(2 * math.pi * t / 7 + 1.3 * k)) * W
            cy = (fy + 0.05 * math.cos(2 * math.pi * t / 9 + 2.1 * k)) * H
            g = np.outer(np.exp(-((self.ys - cy) / (0.24 * H)) ** 2), np.exp(-((self.xs - cx) / (0.24 * W)) ** 2))
            arr += g[..., None] * (np.float32(colour) * amount)
        return arr

    def scene1(self, t):
        arr = self.backdrop(t, FIELD * ease(t / 1.2))
        m, (mx, my) = self.mark, self.mark_xy

        p = ease((t - DRAW[0]) / (DRAW[1] - DRAW[0]))  # the pen draws ring 1, then ring 2
        cool = 1 - ease((t - DRAW[1]) / 0.3)
        locked = ease((t - CLICK) / 0.35)            # brand colours while drawing, white once locked
        for k, (ring, colour) in enumerate(zip(m.drawn(p), (TEAL, LILAC))):
            paint(arr, ring, mx, my, mix(colour, WHITE, locked))
            if cool > 0:                             # white-hot just behind the pen
                paint(arr, ring * np.clip(1 - (p - m.loop[k]) / 0.08, 0, 1), mx, my, WHITE, cool)
        if DRAW[0] <= t and cool > 0:
            x, y = m.at(p)
            fade = min(1.0, (t - DRAW[0]) / 0.15) * cool
            glow(arr, mx + x, my + y, 34, TEAL if p < 0.5 else LILAC, 0.9 * fade)
            glow(arr, mx + x, my + y, 10, WHITE, 1.2 * fade)

        g = ease((t - DRAW[1]) / (CLICK - DRAW[1]))  # the spokes shoot out of the joints
        if g > 0:
            for spoke in m.grown(g):
                paint(arr, spoke, mx, my, WHITE)
            if g < 1:
                for x, y in m.tips(g):
                    glow(arr, mx + x, my + y, 16, WHITE, 0.9)

        if CLICK <= t < CLICK + 0.7:                 # click: a shock ring
            u = (t - CLICK) / 0.7
            r = 150 + 560 * (1 - (1 - u) ** 3)
            band = np.clip(1 - np.abs(self.ring_dist - r) / (4 + 16 * (1 - u)), 0, 1) * (1 - u)
            arr += band[..., None] * (mix(TEAL, WHITE, 0.4) * 0.9)

        if t >= CLICK + 0.2:                         # a pulse keeps running round the 8
            s, fade = (t - CLICK - 0.2) / LOOP, ease((t - CLICK - 0.2) / 0.4)
            for w in m.comet(s):
                paint(arr, w, mx, my, TEAL, fade)
            x, y = m.at(s)
            glow(arr, mx + x, my + y, 22, TEAL, 0.7 * fade)

        wx, wy = self.word_xy                        # the wordmark, letter by letter
        for k, (mask, (ox, oy)) in enumerate(self.word.letters):
            t0 = LETTERS + 0.08 * k
            s = pop((t - t0) / 0.32)
            if s <= 0.02:
                continue
            sm = mask if abs(s - 1) < 1e-3 else resized(mask, s)
            colour = mix(self.accent[k], WHITE, ease((t - t0 - 0.3) / 0.5)) if k in self.accent else WHITE
            paint(arr, sm, wx + ox + (mask.shape[1] - sm.shape[1]) / 2, wy + oy + mask.shape[0] - sm.shape[0],
                  colour)

        if SHINE <= t <= SHINE + 0.8:                # a teal shine sweeps across the wordmark
            h, w = self.word_mask.shape
            x = -140 + (w + 280) * (t - SHINE) / 0.8
            yy, xx = np.ogrid[0:h, 0:w]
            band = np.clip(1 - np.abs(xx + 0.45 * (yy - h / 2) - x) / 50, 0, 1)
            paint(arr, self.word_mask * band, wx, wy, TEAL)
        return arr

    def frame(self, t) -> np.ndarray:
        if t < SCENE2:
            arr = self.scene1(t)
        else:
            fade = 1 - ease((t - SCENE2) / 0.45)     # scene 1 dissolves - as glyphs, it thins out
            arr = self.backdrop(t, FIELD)
            if fade > 0:
                arr = mix(arr, self.scene1(t), fade)

            for k, (m, x, y, colour) in enumerate(self.lines):  # the tagline, swept in by a scan line
                u = (t - SCENE2 - 0.35 - 0.45 * k) / 0.6
                if u <= 0:
                    continue
                h, w = m.shape
                front = -40 + (w + 80) * min(u, 1.0)
                behind = front - np.arange(w, dtype=np.float32)
                paint(arr, m * np.clip(behind, 0, 1), x, y, colour)
                paint(arr, m * (np.clip(1 - behind / 80, 0, 1) * (behind > 0)), x, y, WHITE)
                if u < 1:
                    bx, y0, y1 = x + front, max(int(y - 30), 0), min(int(y + h + 30), H)
                    x0, x1 = max(int(bx - 40), 0), min(int(bx + 41), W)
                    if x0 < x1:
                        col = np.exp(-((np.arange(x0, x1) - bx) / 9) ** 2)[None, :, None]
                        arr[y0:y1, x0:x1] += col * (mix(TEAL, WHITE, 0.5) * 1.3)
                    rng = np.random.default_rng(int(t * FPS) * 7 + k)   # scanned points left behind
                    for px, py in zip(bx - rng.exponential(60, 24), rng.uniform(y - 20, y + h + 20, 24)):
                        if 0 <= px < W - 4 and 0 <= py < H - 4:
                            arr[int(py):int(py) + 4, int(px):int(px) + 4] += TEAL * 0.8

            a = ease((t - LOCKUP) / 0.5)             # the horizontal logo rises in
            if a > 0:
                lift = 24 * (1 - a)
                m, (mx, my) = self.lock_mark, self.lock_mark_xy
                for part in m.rings + m.spokes:
                    paint(arr, part, mx, my + lift, WHITE, a)
                lx, ly = self.lock_xy
                for mask, (ox, oy) in self.lock_word.letters:
                    paint(arr, mask, lx + ox, ly + oy + lift, WHITE, a)
                if t >= LOCKUP + 0.4:
                    s, fade = (t - LOCKUP - 0.4) / LOOP, ease((t - LOCKUP - 0.4) / 0.4)
                    for w in m.comet(s):
                        paint(arr, w, mx, my, TEAL, fade)
                    x, y = m.at(s)
                    glow(arr, mx + x, my + y, 14, TEAL, 0.6 * fade)
        return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- sound

def sting_audio(path: Path, sting: Sting):
    n = int((TOTAL + 0.3) * SR)
    out = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(8)

    def put(sig, at, gain=1.0, pan=0.0):
        i = int(at * SR)
        j = min(n, i + len(sig))
        s = sig[: j - i] * gain
        if np.ndim(pan):
            pan = pan[: j - i]
        out[i:j, 0] += s * (1 - np.maximum(pan, 0))
        out[i:j, 1] += s * (1 + np.minimum(pan, 0))

    def whoosh(at, dur, gain):
        t = np.arange(int(dur * SR)) / SR
        put(np.convolve(rng.standard_normal(len(t)), np.ones(24) / 24, "same") * np.sin(np.pi * t / dur) ** 2,
            at, gain)

    def chirp(f0, f1, dur):
        t = np.arange(int(dur * SR)) / SR
        f = f0 * (f1 / f0) ** (t / dur)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** 2

    m, (mx, _) = sting.mark, sting.mark_xy

    def pan_at(p):
        return float(np.clip((mx + m.at(p)[0] - W / 2) / (W / 2), -1, 1)) * 0.8

    whoosh(0.0, 1.2, 0.8)  # the colour fields come up

    grid = np.linspace(DRAW[0], DRAW[1], 2000)  # the pen: a note every 45 degrees round the 8
    prog = np.array([ease((g - DRAW[0]) / (DRAW[1] - DRAW[0])) for g in grid])
    for k, note in enumerate((57, 60, 62, 64, 65, 69, 72, 74, 76, 77, 81, 84)):
        put(voice(hz(note), 0.5, TRIANGLE, 9, 0.0), float(np.interp(k / 12, prog, grid)), 0.3, pan_at(k / 12))
    t = np.arange(int((DRAW[1] - DRAW[0]) * SR)) / SR  # and a soft hum that rises with it
    f = hz(45) * 2 ** np.array([ease(u) for u in np.linspace(0, 1, len(t))])
    hum = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.clip(np.minimum(t / 0.3, (t[-1] - t) / 0.3), 0, 1)
    put(hum, DRAW[0], 0.12)

    put(chirp(380, 1400, CLICK - DRAW[1]), DRAW[1], 0.12, -0.5)  # the spokes zip out
    put(chirp(420, 1500, CLICK - DRAW[1]), DRAW[1] + 0.03, 0.12, 0.5)

    t = np.arange(int(0.6 * SR)) / SR  # click: a low thump, a snap and a bell chord
    put(np.sin(2 * np.pi * np.cumsum(42 + 140 * np.exp(-t * 28)) / SR) * np.exp(-t * 7), CLICK, 0.9)
    put(rng.standard_normal(int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 120), CLICK, 0.25)
    for k, note in enumerate((74, 81, 86)):
        put(bell(hz(note), 1.6), CLICK + 0.04 * k, 0.26)

    pad = TOTAL - CLICK  # a Dm9 pad underneath from the click on
    for note in (50, 57, 60, 64, 65):
        put(voice(hz(note), pad, TRIANGLE, 0.4, 0.6) * np.clip(np.linspace(0, 3, int(pad * SR)), 0, 1),
            CLICK, 0.09)

    for k, note in enumerate((69, 72, 74, 76, 77, 81, 84, 86)):  # a pluck per letter
        put(voice(hz(note), 0.18, SQUARE, 14, 0.0), LETTERS + 0.08 * k, 0.11)
    for k in sting.accent:  # the 8 and the dot ring
        put(bell(hz(88 + 5 * (k - 3)), 0.8), LETTERS + 0.08 * k + 0.05, 0.12)
    for k, note in enumerate((93, 96, 98)):  # the shine
        put(bell(hz(note), 0.7), SHINE + 0.15 + 0.08 * k, 0.07)

    whoosh(SCENE2 - 0.1, 0.7, 1.0)  # into scene 2
    for k, (mask, x, _, _) in enumerate(sting.lines):  # each scan: a rising sweep riding the beam
        at, dur = SCENE2 + 0.35 + 0.45 * k, 0.6
        t = np.arange(int(dur * SR)) / SR
        beam = np.clip(((x - 40 + (mask.shape[1] + 80) * t / dur) - W / 2) / (W / 2), -1, 1) * 0.8
        put(chirp(700 + 200 * k, 2100 + 300 * k, dur), at, 0.07, beam)
        for j in range(12):  # data ticks
            put(rng.standard_normal(200) * np.exp(-np.arange(200) / 40), at + j * dur / 12, 0.05, beam[0] + (beam[-1] - beam[0]) * j / 12)
        put(bell(hz((81, 84)[k]), 1.0), at + dur, 0.2)
    for k, note in enumerate((62, 69, 74, 77, 81)):  # the logo: a last chord
        put(bell(hz(note), 2.4), LOCKUP + 0.05 * k, 0.17)

    out *= 0.89 / np.abs(out).max()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((out * 32767).astype("<i2").tobytes())


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/the8lab_logo.mp4")
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
            enc.stdin.write(sting.frame(i / FPS).tobytes())
            if i % 15 == 0 or i + 1 == n:
                print(f"\r  drawing {i + 1}/{n}  {(i + 1) / (time.time() - clock):5.1f} fps",
                      end="", file=sys.stderr)
        enc.stdin.close()
        enc.wait()
    print(f"\n  -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
