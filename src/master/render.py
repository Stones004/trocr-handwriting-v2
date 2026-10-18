"""Draws a label string as a handwritten-style crop. Two sources, picked per crop:
  fonts  : Windows handwriting fonts, one font per crop, each character drawn separately with its own size/baseline/rotation/shear/weight
           (covers every character, incl. a-z, uppercase, brackets, dots)
  EMNIST : real handwritten glyphs (digits 0-9, a-j, plus uppercase/Roman from glyphs_single.npz), brackets and dots drawn as strokes;
           used only when every alphanumeric character of the label exists in the glyph bank
followed by page effects (elastic wobble, perspective, lighting, noise, blur, JPEG, low resolution, clipped edges) from
synth_sequences.render_crop. The label text is drawn exactly as written -- punctuation included.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "emnist"))
import synth_sequences as ss  # noqa: E402

FONT_DIR = Path("C:/Windows/Fonts")
FONTS = {"Inkfree.ttf": 3, "segoepr.ttf": 3, "segoeprb.ttf": 2, "segoesc.ttf": 2, "segoescb.ttf": 1, "LHANDW.TTF": 2, "mvboli.ttf": 3,
         "comic.ttf": 2, "comicbd.ttf": 1, "comici.ttf": 1, "Gabriola.ttf": 1, "SCRIPTBL.TTF": 1, "BRUSHSCI.TTF": 0.7, "FRSCRIPT.TTF": 0.7,
         "segoeui.ttf": 0.4, "segoeuiz.ttf": 0.3}
FONT_SHARE = 0.55            # probability of the font source when the glyph bank could also draw the label
MAX_HEIGHT = 64              # stored crops are shrunk to this height; PARSeq resizes to 32x128 anyway
_cache = {}


def load_bank():
    v5, single = np.load(ROOT / "data/emnist/glyphs_v5.npz"), np.load(ROOT / "data/emnist/glyphs_single.npz")
    have = set(v5["train_labels"].tolist())
    keep = np.array([c not in have for c in single["train_labels"]])
    bank = ss.GlyphBank(np.concatenate([v5["train_images"], single["train_images"][keep]]),
                        np.concatenate([v5["train_labels"], single["train_labels"][keep]]))
    return bank


def _font(name, size):
    if (name, size) not in _cache:
        _cache[(name, size)] = ImageFont.truetype(str(FONT_DIR / name), size)
    return _cache[(name, size)]


Q_DRAWN_SHARE = 0.75         # share of Q characters drawn as strokes (the rest use the font's Q): fonts rarely look like a hurried Q


def q_tile(rng, size, H, base, pad, width):
    """A hand-drawn Q: a loop plus one of several tails, including the 9-like and 2-like shapes that models confuse it with."""
    hd = 0.72 * size
    w, h = hd * rng.uniform(0.7, 0.98), hd * rng.uniform(0.95, 1.2)
    cx, cy = pad + w / 2, base - h / 2
    gap = rng.choice([0.0, 0.0, rng.uniform(0.1, 0.5)])                   # open loop: looks like 0/9/a
    t = np.linspace(0, 2 * np.pi * rng.uniform(0.95, 1.1) - gap, 44) + rng.uniform(-0.8, 0.8)
    wob = 1 + rng.normal(0, 0.03, t.shape)
    pts = np.stack([cx + w / 2 * np.cos(t) * wob, cy + h / 2 * np.sin(t) * wob], 1)
    buf = np.zeros((H, width), np.uint8)
    sw = max(1, int(size * rng.uniform(0.04, 0.09)))
    cv2.polylines(buf, [pts.astype(np.int32)], False, 255, sw, cv2.LINE_AA)
    kind = rng.choice(["diag", "hook", "descender", "flat"], p=[0.35, 0.25, 0.2, 0.2])
    if kind == "diag":                                                     # classic Q: short diagonal across the bottom right
        tail = [(cx + 0.1 * w, cy + 0.15 * h), (cx + 0.6 * w, base + rng.uniform(0.0, 0.3) * h)]
    elif kind == "hook":                                                   # tail with a curl
        tail = [(cx + 0.15 * w, cy + 0.2 * h), (cx + 0.5 * w, base - 0.05 * h), (cx + 0.75 * w, base + 0.1 * h), (cx + 0.6 * w, base + 0.25 * h)]
    elif kind == "descender":                                              # straight descender, like 9 / g / q
        tail = [(cx + 0.5 * w, cy - 0.1 * h), (cx + 0.45 * w, base + rng.uniform(0.3, 0.6) * h)]
    else:                                                                  # flat foot to the right, like a 2
        tail = [(cx - 0.2 * w, base - 0.02 * h), (cx + 0.9 * w, base + rng.uniform(-0.05, 0.1) * h)]
    cv2.polylines(buf, [np.array(tail, np.int32)], False, 255, sw, cv2.LINE_AA)
    return buf.astype(np.float32) / 255.0


def compose_font(rng, text, hd):
    names = [n for n in FONTS if (FONT_DIR / n).exists()]
    w = np.array([FONTS[n] for n in names])
    name = names[rng.choice(len(names), p=w / w.sum())]
    size = max(12, int(hd / 0.72))
    stroke = int(rng.choice([0, 0, 1, 1, 2])) if size > 40 else 0
    H, base = int(size * 2.4), int(size * 1.5)
    width = int(sum(_font(name, size).getlength(c) for c in text) * 1.5 + size * 2)
    ink, x = np.zeros((H, width), np.float32), size * 0.4
    shear_all = rng.uniform(-0.25, 0.25)
    for c in text:
        if c == " ":
            x += size * rng.uniform(0.25, 0.7)
            continue
        pad = int(size * 0.6)
        if c == "Q" and rng.random() < Q_DRAWN_SHARE:
            a = q_tile(rng, size, H, base, pad, int(size * 2.5))
        else:
            tile = Image.new("L", (int(size * 2.5), H), 0)
            ImageDraw.Draw(tile).text((pad, base + int(rng.normal(0, 0.04) * size)), c, font=_font(name, max(10, int(size * rng.uniform(0.88, 1.12)))),
                                      fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
            a = np.asarray(tile, np.float32) / 255.0
        m = cv2.getRotationMatrix2D((pad, base), float(np.clip(rng.normal(0, 5), -12, 12)), 1.0)
        m[0, 1] += shear_all + rng.normal(0, 0.05)
        a = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        xs = np.where(a.max(0) > 0.1)[0]
        if len(xs) == 0:
            x += size * 0.3
            continue
        x0 = int(x) - xs.min()
        lo, hi = max(0, -x0), min(a.shape[1], width - x0)
        if hi > lo:
            ink[:, x0 + lo: x0 + hi] = np.maximum(ink[:, x0 + lo: x0 + hi], a[:, lo:hi])
        x += (xs.max() - xs.min() + 1) + rng.uniform(-0.04, 0.22) * size
    return ss.tight_crop(ink, 0.08)


def _bracket(buf, ch, x, top, bot, sw, rng):
    h, w = bot - top, max(4.0, (bot - top) * rng.uniform(0.15, 0.3))
    pts = {"(": [(x + w, top), (x, (top + bot) / 2), (x + w, bot)], ")": [(x, top), (x + w, (top + bot) / 2), (x, bot)],
           "[": [(x + w, top), (x, top), (x, bot), (x + w, bot)], "]": [(x, top), (x + w, top), (x + w, bot), (x, bot)]}[ch]
    p = np.array(pts, np.float32)
    if ch in "()":
        t = np.linspace(0, 1, 20)[:, None]
        p = (1 - t) ** 2 * p[0] + 2 * t * (1 - t) * (2 * p[1] - 0.5 * (p[0] + p[2])) + t ** 2 * p[2]
    p = p + rng.normal(0, 0.012 * h, p.shape)
    cv2.polylines(buf, [p.astype(np.int32)], False, 255, max(1, int(round(sw * rng.uniform(0.8, 1.4)))), cv2.LINE_AA)
    return w


def compose_glyphs(rng, bank, text, hd):
    sw = max(1.5, hd * rng.uniform(0.05, 0.10))
    height, base = int(hd * 2.6), int(hd * 1.7)
    glyphs = {i: ss.render_glyph(rng, bank, c, hd) for i, c in enumerate(text) if c not in "()[]. "}
    width = int(sum(g.shape[1] for g in glyphs.values()) + hd * (0.5 * len(text) + 2))
    ink, buf, x = np.zeros((height, width), np.float32), np.zeros((height, width), np.uint8), hd * 0.5
    for i, c in enumerate(text):
        gap = rng.uniform(-0.05, 0.22) * hd
        if c == " ":
            x += rng.uniform(0.3, 0.9) * hd
        elif c == ".":
            cv2.circle(buf, (int(x + hd * 0.08), int(base - sw * rng.uniform(0.5, 2.0))), max(1, int(sw * rng.uniform(0.5, 1.4))), 255, -1, cv2.LINE_AA)
            x += hd * 0.2 + gap
        elif c in "()[]":
            tall = hd * rng.uniform(1.1, 1.5)
            x += _bracket(buf, c, x, base - tall * rng.uniform(0.85, 1.0), base + tall * rng.uniform(0.0, 0.15), sw, rng) + gap
        else:
            g = glyphs[i]
            bottom = int(base + ss.DESCENT.get(c, 0.0) * hd + rng.normal(0, 0.03) * hd)
            y0, x0 = bottom - g.shape[0], int(x)
            if y0 >= 0 and x0 + g.shape[1] <= width:
                ink[y0:bottom, x0:x0 + g.shape[1]] = np.maximum(ink[y0:bottom, x0:x0 + g.shape[1]], g)
            x += g.shape[1] + gap
    ink = np.maximum(ink, buf.astype(np.float32) / 255.0)
    m = cv2.getRotationMatrix2D((width / 2, height / 2), float(np.clip(rng.normal(0, 2), -5, 5)), 1.0)
    return ss.tight_crop(cv2.warpAffine(ink, m, (width, height), flags=cv2.INTER_LINEAR), 0.08)


def _elastic(rng, img, alpha, sigma):
    h, w = img.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def page_aug(rng, img):
    h, w = img.shape
    bg, out = float(np.median(img)), img.astype(np.float32)
    if rng.random() < 0.6:
        out = _elastic(rng, out, rng.uniform(1.0, 4.0) * max(h, w) / 100, rng.uniform(6, 14) * max(h, w) / 100)
    if rng.random() < 0.45:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = (src + rng.uniform(-1, 1, (4, 2)) * rng.uniform(0.02, 0.08) * np.array([w, h])).astype(np.float32)
        out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.20:
        y = int(h * rng.uniform(0.85, 0.98))
        cv2.line(out, (int(w * rng.uniform(0, 0.2)), y), (int(w * rng.uniform(0.7, 1.0)), y + int(rng.integers(-3, 4))), float(rng.uniform(40, 140)), int(rng.integers(1, 3)), cv2.LINE_AA)
    if rng.random() < 0.30:
        f = rng.uniform(0.35, 0.7)
        out = cv2.resize(cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.40:
        out = (out - bg) * rng.uniform(0.6, 1.2) + bg + rng.uniform(-15, 15)
    return np.clip(out, 0, 255).astype(np.uint8)


def _compose(rng, bank, text, hd, sep_dot=False):
    can_glyph = all(c in bank.by_char for c in text if c not in "()[]. ")
    return compose_font(rng, text, hd) if (not can_glyph or rng.random() < FONT_SHARE) else compose_glyphs(rng, bank, text, hd)


def _no_clutter(rng, content, hd):
    return np.pad(content, int(hd)), max(1.5, hd * 0.07)


def render_label(text, seed, index, bank):
    """Deterministic: the same (seed, index) always gives the same crop."""
    ss.RECIPE = "v5"
    ss.compose_sequence, ss.draw_clutter_v2 = _compose, _no_clutter
    rng = np.random.default_rng([seed, index])
    img = page_aug(rng, ss.render_crop(rng, bank, text))
    if img.shape[0] > MAX_HEIGHT:
        img = cv2.resize(img, (max(8, round(img.shape[1] * MAX_HEIGHT / img.shape[0])), MAX_HEIGHT), interpolation=cv2.INTER_AREA)
    return img
