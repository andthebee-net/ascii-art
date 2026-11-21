#!/usr/bin/env python3
"""glyph_engine.py - turn any video into coloured glyph (ASCII) art.

    python3 glyph_engine.py clip.mp4                 # -> out/clip.glyph.mp4 (studio layout)
    python3 glyph_engine.py clip.mp4 --layout full   # just the glyph picture, source aspect
    python3 glyph_engine.py clip.mp4 --ramp hangul   # draw with Hangul instead of ASCII
    python3 glyph_engine.py clip.mp4 --still 4.2     # one frame at 4.2s -> PNG, to tune
    python3 glyph_engine.py clip.mp4 --play          # play live in this terminal

Needs numpy + pillow, and ffmpeg/ffprobe on PATH (ffplay for sound in --play).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import keyword
import shutil
import subprocess
import sys
import time
import tokenize
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
# fill glyphs carry no direction, so / | - \ only ever mean "an outline runs here"
RAMP = " .'`,:;^\"~+=*icvxzsunoaewmXZOQ#%&8B@$"
HANGUL = " ·ㆍㅇㅅㄱㄴㅎ가나다마바사아자하한글곰돌뭉뚫흙밟뷁쀍"
GLYPHS = {"ascii": RAMP, "hangul": HANGUL}
EDGES = {"ascii": "|/-\\", "hangul": "ㅣ/ㅡ\\"}  # vertical, rising, flat, falling
FONTS = {
    "mono": ["/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "C:/Windows/Fonts/consola.ttf"],
    "kr": ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
           "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "C:/Windows/Fonts/malgun.ttf"],
}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16 - 0.5
BACKDROP = {"dark": (9, 10, 12), "paper": (243, 238, 226)}
UI = dict(bg=(11, 12, 15), panel=(18, 19, 24), rule=(44, 47, 56), text=(234, 236, 240),
          dim=(118, 124, 138), green=(61, 220, 132), amber=(255, 178, 80), hl=(31, 35, 48))
SYNTAX = dict(kw=(199, 146, 234), str=(158, 206, 106), com=(96, 103, 118), num=(247, 140, 108),
              fn=(97, 175, 239), builtin=(86, 182, 194), name=(214, 218, 226), op=(160, 168, 184))


def font(kind: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    for path in FONTS[kind]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                return ImageFont.truetype(path, size)
    sys.exit(f"glyph_engine: no {kind} font found - add one to FONTS")


# ---------------------------------------------------------------- reading the clip

def probe(src: Path) -> dict:
    """Display size, duration and whether there is sound, via ffprobe."""
    run = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                          "-show_streams", "-show_format", str(src)],
                         capture_output=True, text=True)
    if run.returncode:
        sys.exit(f"glyph_engine: ffprobe could not read {src}\n{run.stderr.strip()}")
    meta = json.loads(run.stdout)
    video = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        sys.exit(f"glyph_engine: no video stream in {src}")
    w, h = int(video["width"]), int(video["height"])
    num, _, den = video.get("sample_aspect_ratio", "1:1").partition(":")
    if num.isdigit() and den.isdigit() and int(num) and int(den) and num != den:
        w = round(w * int(num) / int(den))
    rotation = int(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation))
    if abs(rotation) % 180 == 90:
        w, h = h, w
    duration = float(meta["format"].get("duration") or video.get("duration") or 0)
    audio = any(s["codec_type"] == "audio" for s in meta["streams"])
    return dict(w=w, h=h, duration=duration, audio=audio)


def frames(src: Path, size, start=0.0, count=None, aspect=None):
    """Decode the clip with ffmpeg; yield (h, w, 3) uint8 frames at FPS, resized to size.

    With aspect, the picture is first centre-cropped to that display aspect (cover fit)."""
    w, h = size
    crop = f"crop=w='min(iw,ih*{aspect:.6f})':h='min(ih,iw/{aspect:.6f})'," if aspect else ""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.4f}", "-i", str(src), "-an",
           "-vf", f"fps={FPS},{crop}scale={w}:{h}:flags=area"]
    cmd += ["-frames:v", str(count)] if count else []
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while (buf := p.stdout.read(n)) and len(buf) == n:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        if p.poll() is None:
            p.kill()
        p.wait()


# ---------------------------------------------------------------- glyphs

class Glyphs:
    """One glyph set rasterised at one cell size: masks, measured ink, brightness -> glyph."""

    def __init__(self, ramp: str = "ascii", cell_w: int = 6, edges: bool = True):
        self.wide = ramp == "hangul"
        kind = "kr" if self.wide else "mono"
        em = font(kind, 100).getlength("한" if self.wide else "M") / 100
        self.font = font(kind, max(4, round(cell_w / em)))
        self.cw, self.ch = cell_w, round(self.font.size * 1.1)
        chars = GLYPHS[ramp]
        self.chars = chars + "".join(c for c in EDGES[ramp] if c not in chars)
        self.masks = np.stack([self.raster(c) for c in self.chars])
        # sort by how much ink each glyph really puts down, not by a guess
        ink = self.masks[: len(chars)].mean(axis=(1, 2))
        ink = (ink - ink.min()) / (ink.max() - ink.min())
        self.lut = np.abs(ink[None, :] - np.linspace(0, 1, 256)[:, None]).argmin(axis=1)
        self.edge = np.array([self.chars.index(c) for c in EDGES[ramp]]) if edges else None
        self.space = self.chars.index(" ")

    def raster(self, c: str) -> np.ndarray:
        im = Image.new("L", (self.cw, self.ch))
        ascent, descent = self.font.getmetrics()
        baseline = (self.ch - ascent - descent) / 2 + ascent
        ImageDraw.Draw(im).text((self.cw / 2, baseline), c, fill=255, font=self.font, anchor="ms")
        return np.asarray(im, np.float32) / 255


# ---------------------------------------------------------------- one frame -> glyph grid

def blur(y):
    """[1 2 1] separable blur - takes the grain off before edge detection."""
    p = np.pad(y, 1, mode="edge")
    y = (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]) / 4
    return (y[:-2] + 2 * y[1:-1] + y[2:]) / 4


def sobel(y):
    """Brightness gradient (gx, gy); y grows downwards like the image rows."""
    p = np.pad(y, 1, mode="edge")
    gx = p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]
    gy = p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
    return gx, gy


def grow(mask):
    """Binary dilation on the cell grid: a cell or any of its 8 neighbours."""
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[y:y + h, x:x + w] for y in range(3) for x in range(3)])


def analyze(rgb, grid, glyphs: Glyphs, look):
    """Fold a frame of samples (sx * sy per cell) into a glyph index and an ink colour per cell."""
    cols, rows, sx, sy = grid
    f = rgb.astype(np.float32) / 255
    cell = f.reshape(rows, sy, cols, sx, 3).mean(axis=(1, 3))
    ink = np.clip((cell @ LUMA - look.black) / (1 - look.black), 0, 1) ** look.gamma
    if look.theme == "paper":
        ink = 1 - ink  # on paper, dark areas take the dense glyphs
    lum = ink + BAYER[np.arange(rows)[:, None] % 4, np.arange(cols) % 4] * look.dither
    idx = glyphs.lut[np.clip(lum * 255, 0, 255).astype(np.intp)]
    if glyphs.edge is not None:
        # where a cell sits on a clear edge, draw the stroke that runs along it: | / - \
        gx, gy = sobel(blur(f @ LUMA))
        hit = np.hypot(gx, gy) > look.edge
        side = ((np.arctan2(gy, gx) % np.pi + np.pi / 8) // (np.pi / 4)).astype(np.intp) % 4
        votes = np.stack([(hit & (side == k)).reshape(rows, sy, cols, sx).sum(axis=(1, 3))
                          for k in range(4)])
        stroke = (votes.max(axis=0) >= look.edge_share * sx * sy) & (ink > 0.1)  # no strokes for stray specks in the dark
        # dark specks a cell or two wide (eyes, a mouth) stay holes in the fill:
        # ringing them with strokes smudges a face at small sizes
        solid = ink > 0.3
        specks = ~grow(~grow(solid)) & ~solid
        stroke &= ~grow(specks)
        idx = np.where(stroke, glyphs.edge[votes.argmax(axis=0)], idx)
    return idx, grade(cell, look)


def grade(cell, look):
    """Ink colour per cell. A glyph inks only part of its cell, so lift saturation and level."""
    y = (cell @ LUMA)[..., None]
    ink = np.clip(y + (cell - y) * look.sat, 0, 1)
    ink = ink * 0.7 if look.theme == "paper" else np.clip(ink ** 0.85 * look.gain, 0, 1)
    return (ink * 255).astype(np.uint8)


def paint(idx, ink, glyphs: Glyphs, backdrop):
    """Rasterise the grid: each cell is its glyph mask tinted with its ink, over the backdrop."""
    rows, cols = idx.shape
    bg = np.asarray(backdrop, np.float32)
    img = bg + glyphs.masks[idx][..., None] * (ink.astype(np.float32)[:, :, None, None] - bg)
    return img.transpose(0, 2, 1, 3, 4).reshape(rows * glyphs.ch, cols * glyphs.cw, 3).astype(np.uint8)


# ---------------------------------------------------------------- studio layout

class CodeView:
    """This very file, syntax-coloured once into a tall strip; each frame shows a window of it."""

    def __init__(self, path: Path, width: int, size: int = 19, line_h: int = 28):
        self.src = path.read_text(encoding="utf-8")
        self.lines = self.src.splitlines()
        self.lh = line_h
        mono, kr = font("mono", size), font("kr", size)
        adv = mono.getlength("M")
        x0 = 4 * adv + 30
        ink = Image.new("RGBA", (width, line_h * (len(self.lines) + 40)), (0, 0, 0, 0))
        d = ImageDraw.Draw(ink)
        for r in range(len(self.lines)):
            d.text((x0 - 16, (r + 0.72) * line_h), str(r + 1), fill=UI["dim"], font=mono, anchor="rs")
        for r, a, b, colour in self.spans():
            line, base = self.lines[r], (r + 0.72) * line_h
            col = [0]
            for ch in line:
                col.append(col[-1] + (2 if unicodedata.east_asian_width(ch) in "WF" else 1))
            if line[a:b].isascii():
                d.text((x0 + col[a] * adv, base), line[a:b], fill=colour, font=mono, anchor="ls")
                continue
            for i in range(a, b):  # Menlo has no Hangul: fall back per character
                wide = unicodedata.east_asian_width(line[i]) in "WF"
                d.text((x0 + (col[i] + (1 if wide else 0.5)) * adv, base), line[i], fill=colour,
                       font=kr if wide or ord(line[i]) > 0x2E7F else mono, anchor="ms")
        self.strip = self.onto(ink, UI["panel"], None)
        self.lit = self.onto(ink, UI["hl"], UI["amber"])

    @staticmethod
    def onto(ink: Image.Image, colour, bar) -> np.ndarray:
        bg = Image.new("RGBA", ink.size, colour + (255,))
        if bar:
            ImageDraw.Draw(bg).rectangle((0, 0, 3, ink.height), fill=bar + (255,))
        return np.asarray(Image.alpha_composite(bg, ink).convert("RGB"))

    def spans(self):
        """(row, col0, col1, colour) for every token, straight from the tokenize module."""
        prev, names = "", set(dir(builtins))
        for tok in tokenize.generate_tokens(io.StringIO(self.src).readline):
            kind, text = tokenize.tok_name[tok.type], tok.string
            if kind == "COMMENT":
                colour = SYNTAX["com"]
            elif kind == "STRING" or kind.startswith(("FSTRING", "TSTRING")):
                colour = SYNTAX["str"]
            elif kind == "NUMBER":
                colour = SYNTAX["num"]
            elif kind == "NAME":
                colour = (SYNTAX["kw"] if keyword.iskeyword(text) else
                          SYNTAX["fn"] if prev in ("def", "class") else
                          SYNTAX["builtin"] if text in names else SYNTAX["name"])
            elif kind == "OP":
                colour = SYNTAX["op"]
            else:
                continue
            prev = text
            (r0, c0), (r1, c1) = tok.start, tok.end
            for r in range(r0, r1 + 1):
                a, b = (c0 if r == r0 else 0), (c1 if r == r1 else len(self.lines[r - 1]))
                if b > a:
                    yield r - 1, a, b, colour

    def tour(self, names):
        """Camera stops: (first, last) line of each named top-level def/class, in that order."""
        defs = {n.name: (n.lineno - 1, n.end_lineno - 1) for n in ast.parse(self.src).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.stops = [defs[n] for n in names if n in defs] or list(defs.values())

    def window(self, t: float, total: float, height: int) -> np.ndarray:
        """The strip at time t: dwell on each stop while a highlighted line walks its body."""
        n = len(self.stops)
        seg = max(2.5, total / n)
        k, u = int(t // seg), (t % seg) / seg
        rows = height / self.lh

        def place(i, u):
            a, b = self.stops[i % n]
            pc = a + u * (b - a + 1)
            return max(a - 2.0, pc - rows + 5), min(int(pc), b)

        top, pc = place(k, u)
        if k and u < 0.15:  # glide over from the previous stop
            e = u / 0.15
            e = e * e * (3 - 2 * e)
            top = place(k - 1, 1.0)[0] * (1 - e) + top * e
        y0 = max(0, round(top * self.lh))
        win = self.strip[y0:y0 + height].copy()
        ly = pc * self.lh - y0
        if -self.lh < ly < len(win):
            a, b = max(ly, 0), min(ly + self.lh, len(win))
            win[a:b] = self.lit[y0 + a:y0 + b]
        return win


class Studio:
    """The 1080x1920 frame: a header, the glyph picture, and this file running underneath."""
    W, H, M, PAD, HEAD, PIC_MAX = 1080, 1920, 24, 10, 128, 1090
    TOUR = ("frames", "Glyphs", "analyze", "sobel", "grade", "paint", "Studio", "render")

    def __init__(self, src: Path, glyphs: Glyphs, grid, ramp: str, total: float, backdrop):
        cols, rows, _, _ = grid
        W, H, M, P = self.W, self.H, self.M, self.PAD
        gw, gh = cols * glyphs.cw, rows * glyphs.ch
        pic = (M, M + self.HEAD, W - M, M + self.HEAD + gh + 2 * P)
        code = (M, pic[3] + M, W - M, H - M)
        self.pic_xy = ((W - gw) // 2, pic[1] + P)
        self.code_xy = (code[0] + 2, code[1] + 58)
        self.code_h = code[3] - 16 - self.code_xy[1]
        self.total = total
        self.view = CodeView(Path(__file__), code[2] - code[0] - 4)
        self.view.tour(self.TOUR)
        self.f_bold, self.f_reg, self.f_time = font("mono", 30, 1), font("mono", 21), font("mono", 26)

        img = Image.new("RGB", (W, H), UI["bg"])
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(pic, 16, fill=backdrop, outline=UI["rule"])
        d.rounded_rectangle(code, 16, fill=UI["panel"], outline=UI["rule"])
        d.text((M + 4, M + 2), f"glyph_engine.py · {ramp}", fill=UI["text"], font=self.f_bold)
        d.text((M + 4, M + 50), f"{src.name} · {cols}x{rows} cells · {FPS} fps",
               fill=UI["dim"], font=self.f_reg)
        d.text((code[0] + 20, code[1] + 17), "glyph_engine.py", fill=UI["text"], font=self.f_reg)
        label = f"running · {FPS} fps"
        d.text((code[2] - 20, code[1] + 17), label, fill=UI["dim"], font=self.f_reg, anchor="ra")
        dot = code[2] - 20 - self.f_reg.getlength(label) - 16
        d.ellipse((dot - 6, code[1] + 22, dot + 6, code[1] + 34), fill=UI["amber"])
        d.line((code[0] + 1, code[1] + 52, code[2] - 1, code[1] + 52), fill=UI["rule"])
        self.base = np.asarray(img).copy()

    def frame(self, pic: np.ndarray, t: float, letters: int) -> np.ndarray:
        out = self.base.copy()
        x, y = self.pic_xy
        out[y:y + pic.shape[0], x:x + pic.shape[1]] = pic
        x, y = self.code_xy
        win = self.view.window(t, self.total, self.code_h)
        out[y:y + win.shape[0], x:x + win.shape[1]] = win
        head = Image.fromarray(out[: self.M + self.HEAD])
        d = ImageDraw.Draw(head)
        stamp, right = f"t {t:05.2f}s", self.W - self.M - 4
        d.text((right, self.M + 4), stamp, fill=UI["text"], font=self.f_time, anchor="ra")
        dot = right - self.f_time.getlength(stamp) - 16
        d.ellipse((dot - 7, self.M + 12, dot + 7, self.M + 26), fill=UI["green"])
        d.text((self.M + 4, self.M + 82), f"letters {letters:,}", fill=UI["dim"], font=self.f_reg)
        out[: head.height] = np.asarray(head)
        return out


# ---------------------------------------------------------------- render / play

def plan(info: dict, args):
    """Pick the cell size and grid. Each cell is sampled sx * sy times (square samples)."""
    aspect = info["w"] / info["h"]
    if args.layout == "studio":
        inner = Studio.W - 2 * (Studio.M + Studio.PAD)
        glyphs = Glyphs(args.ramp, max(3, inner // (args.cols or (72 if args.ramp == "hangul" else 144))),
                        args.edges)
        cols = inner // glyphs.cw
        rows = min((Studio.PIC_MAX - 2 * Studio.PAD) // glyphs.ch,
                   round(cols * glyphs.cw / aspect / glyphs.ch))
    else:
        glyphs = Glyphs(args.ramp, max(3, args.width // (args.cols or (80 if args.ramp == "hangul" else 160))),
                        args.edges)
        cols = args.width // glyphs.cw
        rows = max(1, round(cols * glyphs.cw / aspect / glyphs.ch))
    return glyphs, (cols, rows, 4, 4 * max(1, round(glyphs.ch / glyphs.cw)))


def codec() -> list[str]:
    have = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if " libx264 " in have:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if " h264_videotoolbox " in have:
        return ["-c:v", "h264_videotoolbox", "-b:v", "20M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p"]


def render(args):
    src = Path(args.src)
    info = probe(src)
    glyphs, grid = plan(info, args)
    cols, rows, sx, sy = grid
    backdrop = BACKDROP[args.theme]
    total = max(0.0, info["duration"] - args.start)
    total = min(total, args.duration) if args.duration else total
    count = int(round(total * FPS)) or None
    aspect = cols * glyphs.cw / (rows * glyphs.ch)
    studio = Studio(src, glyphs, grid, args.ramp, total or 10.0, backdrop) if args.layout == "studio" else None

    def compose(rgb, t):
        idx, ink = analyze(rgb, grid, glyphs, args)
        pic = paint(idx, ink, glyphs, backdrop)
        if studio:
            return studio.frame(pic, t, int((idx != glyphs.space).sum()))
        return np.pad(pic, ((0, pic.shape[0] % 2), (0, pic.shape[1] % 2), (0, 0)), mode="edge")

    print(f"  {src.name}: {info['w']}x{info['h']}, {total:.2f}s -> {cols}x{rows} glyphs "
          f"({glyphs.cw}x{glyphs.ch}px cells, {args.ramp}, {args.layout})", file=sys.stderr)

    if args.still is not None:
        rgb = next(frames(src, (cols * sx, rows * sy), args.start + args.still, 1, aspect), None)
        if rgb is None:
            sys.exit(f"glyph_engine: no frame at {args.still}s")
        out = Path(args.out or f"out/{src.stem}.t{args.still:05.2f}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(compose(rgb, args.still)).save(out)
        print(f"  -> {out}", file=sys.stderr)
        return

    out = Path(args.out or f"out/{src.stem}.glyph.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    enc, clock = None, time.time()
    for i, rgb in enumerate(frames(src, (cols * sx, rows * sy), args.start, count, aspect)):
        frame = compose(rgb, i / FPS)
        if enc is None:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-r", str(FPS), "-i", "-"]
            if info["audio"] and not args.mute:  # carry the original sound across
                cmd += ["-ss", f"{args.start:.4f}", "-t", f"{total:.4f}", "-i", str(src),
                        "-map", "0:v", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
            enc = subprocess.Popen(cmd + codec() + ["-movflags", "+faststart", str(out)],
                                   stdin=subprocess.PIPE)
        enc.stdin.write(frame.tobytes())
        if i % 10 == 0 or i + 1 == count:
            rate = (i + 1) / max(time.time() - clock, 1e-6)
            print(f"\r  frame {i + 1}/{count or '?'}  {rate:5.1f} fps", end="", file=sys.stderr)
    if enc is None:
        sys.exit("glyph_engine: the clip gave no frames")
    enc.stdin.close()
    enc.wait()
    print(f"\n  -> {out}", file=sys.stderr)


def ansi(idx, ink, shown, space, bg) -> str:
    """One frame as 24-bit colour escape codes; colour only changes when the ink does."""
    q = ink.astype(np.int32) & 0xF8
    keys = (q[..., 0] << 16 | q[..., 1] << 8 | q[..., 2]).tolist()
    lines = []
    for krow, irow in zip(keys, idx.tolist()):
        parts, last = [bg], -1
        for k, i in zip(krow, irow):
            if i == space:
                parts.append(shown[space])
                continue
            if k != last:
                parts.append(f"\x1b[38;2;{k >> 16};{k >> 8 & 255};{k & 255}m")
                last = k
            parts.append(shown[i])
        lines.append("".join(parts))
    return "\x1b[H" + "\x1b[0m\n".join(lines) + "\x1b[0m"


def play(args):
    if not sys.stdout.isatty():
        sys.exit("glyph_engine: --play draws into a live terminal window - run it in a terminal,\n"
                 "or leave out --play to render a video file instead")
    src = Path(args.src)
    info = probe(src)
    tw, th = shutil.get_terminal_size((120, 40))
    glyphs = Glyphs(args.ramp, 12, args.edges)  # its tables only; the terminal draws
    span = 2 if glyphs.wide else 1  # terminal columns per glyph
    cell = 0.5 * span  # a terminal cell is about twice as tall as it is wide
    aspect, fit = info["w"] / info["h"], None
    cols = min(args.cols or tw // span, tw // span)
    rows = max(1, round(cols * cell / aspect))
    if args.fill:  # use the whole terminal, cropping the picture's edges to its shape
        rows = th - 2
        fit = cols * cell / rows
    elif rows > th - 2:
        rows = th - 2
        cols = max(1, round(rows * aspect / cell))
    grid = (cols, rows, 4, 4 * max(1, round(1 / cell)))
    shown = [c if unicodedata.east_asian_width(c) in "WF" else c + " " * (span - 1) for c in glyphs.chars]
    bg = "\x1b[48;2;%d;%d;%dm" % BACKDROP[args.theme]
    total = max(0.0, info["duration"] - args.start)
    total = min(total, args.duration) if args.duration else total
    count = int(round(total * FPS)) or None
    sound = None
    sys.stdout.write("\x1b[?25l\x1b[2J")
    try:
        while True:
            if info["audio"] and not args.mute and shutil.which("ffplay"):
                sound = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", f"{args.start}",
                     "-t", f"{total}", str(src)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clock = None
            for i, rgb in enumerate(frames(src, (cols * grid[2], rows * grid[3]), args.start, count, fit)):
                if clock is None:  # start the clock when ffmpeg delivers, not before
                    clock = time.perf_counter()
                lag = time.perf_counter() - (clock + i / FPS)
                if lag > 1 / FPS:
                    continue  # running behind: drop this frame, keep the sound in sync
                if lag < 0:
                    time.sleep(-lag)
                idx, ink = analyze(rgb, grid, glyphs, args)
                sys.stdout.write(ansi(idx, ink, shown, glyphs.space, bg) +
                                 f"\n\x1b[2m glyph_engine · t {i / FPS:5.2f}s · {cols}x{rows}"
                                 f" · ctrl+c to stop\x1b[0m\x1b[K")
                sys.stdout.flush()
            if sound:
                sound.wait()
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if sound and sound.poll() is None:
            sound.terminate()
        sys.stdout.write("\x1b[0m\x1b[?25h\n")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Turn a video into coloured glyph (ASCII) art.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("src", help="input video file")
    ap.add_argument("-o", "--out", help="output file (default: out/<name>.glyph.mp4)")
    ap.add_argument("--layout", choices=["studio", "full"], default="studio",
                    help="studio: 1080x1920 with header + code panel; full: the picture alone")
    ap.add_argument("--ramp", choices=["ascii", "hangul"], default="ascii")
    ap.add_argument("--cols", type=int,
                    help="glyph columns (default: studio 144 ascii / 72 hangul, full 160 / 80)")
    ap.add_argument("--width", type=int, default=1280, help="output width for --layout full")
    ap.add_argument("--theme", choices=["dark", "paper"], default="dark")
    ap.add_argument("--no-edges", dest="edges", action="store_false",
                    help="brightness only, no | / - \\ strokes along edges")
    ap.add_argument("--start", type=float, default=0.0, help="start at this second")
    ap.add_argument("--duration", type=float, help="only this many seconds")
    ap.add_argument("--still", type=float, metavar="T", help="render the frame at T seconds to PNG")
    ap.add_argument("--play", action="store_true", help="play in this terminal instead of rendering")
    ap.add_argument("--loop", action="store_true", help="with --play: loop forever")
    ap.add_argument("--fill", action="store_true",
                    help="with --play: fill the whole terminal, cropping the picture's edges")
    ap.add_argument("--mute", action="store_true", help="no sound")
    look = ap.add_argument_group("look")
    look.add_argument("--gamma", type=float, default=0.9, help="<1 brightens the mids")
    look.add_argument("--black", type=float, default=0.06, help="brightness treated as empty")
    look.add_argument("--sat", type=float, default=1.3, help="colour saturation")
    look.add_argument("--gain", type=float, default=1.3, help="ink brightness")
    look.add_argument("--dither", type=float, default=0.03, help="ordered dither strength")
    look.add_argument("--edge", type=float, default=0.5, help="edge strength needed for a stroke")
    look.add_argument("--edge-share", type=float, default=0.18,
                      help="share of a cell that must lie on an edge")
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"glyph_engine: {tool} not found - install ffmpeg first (macOS: brew install ffmpeg)")
    if not Path(args.src).is_file():
        sys.exit(f"glyph_engine: no such file: {args.src}")
    play(args) if args.play else render(args)


if __name__ == "__main__":
    main()
