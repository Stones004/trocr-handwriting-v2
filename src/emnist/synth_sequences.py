"""Builds synthetic handwritten question-number crops by stitching EMNIST glyphs (digits 0-9, lowercase a-j) into
sequences, with medium-strength augmentation, and writes them to a PARSeq-compatible LMDB.

Why synthesize sequences rather than train on single EMNIST glyphs: PARSeq is a sequence model that reads a whole
crop (variable length, EOS token, spacing/touching characters), and our real crops are question numbers like `2`,
`10`, `b`, `02a`. Single 28x28 glyphs would never exercise the multi-character path.

Sequence templates (D = digit, L = letter a-j; weights are in TEMPLATES_BY_RECIPE):
  D, DD, L, DL, DDL (the `NNa` format, leading zeros allowed: 02a, 70f, 01j) and `ii`/`iii` (repeated 'i').
The real training labels are dominated by short 1-2 character numbers (~65% single digit, ~11% two digits), with the
`NNa` shape almost absent, so the mix covers both the shape we were told to expect and what the real data looks like.

Augmentation ("medium"), applied per glyph and per crop, all randomized per sample:
  glyph   : height/width jitter, rotation (sd 6 deg), shear, stroke thickening/thinning, per-letter height ratio
            (EMNIST tight-crops every glyph, so an 'a' and a 'b' are both full height; ratios restore x-height/ascender
            proportions) and descender drop for g/j
  layout  : variable inter-character gap (slight overlap allowed), baseline jitter, small whole-sequence rotation
  clutter : trailing ')' / '.' or a '( )' pair, a hand-drawn circle around the number, a vertical page-rule line at
            an edge -- all present in the real crops and NOT part of the label (real labels are normalized)
  paper   : light/uneven background, ink darkness, Gaussian noise, blur, occasional JPEG artefacts
  aspect  : canvas aspect ratio sampled to match real crops (~0.86 for single chars, ~1.08 for longer ones)
PARSeq's own RandAugment (data.augment_magnitude) is applied on top at training time.

Determinism: sample i uses rng seeded with (seed, i), so `--n 6000` yields exactly the first 6000 samples of
`--n 24000` with the same seed.

Run with venv_parseq (needs lmdb, cv2, numpy):
    venv_parseq/Scripts/python src/emnist/synth_sequences.py --glyph-split train --n 24000 --seed 1 \
        --out data/emnist_lmdb/emnist_only/train/emnist --preview data/emnist/synth_preview.png
"""
import argparse
from pathlib import Path

import cv2
import lmdb
import numpy as np

DIGITS = "0123456789"
LETTERS = "abcdefghij"

# Glyph height relative to a digit's height (EMNIST tight-crops, so this has to be re-imposed).
REL_H = {**{d: 1.0 for d in DIGITS}, "a": 0.62, "b": 1.0, "c": 0.60, "d": 1.0, "e": 0.62, "f": 1.0, "g": 0.85,
         "h": 1.0, "i": 0.80, "j": 0.95, "v": 0.62, "x": 0.62}  # any other character (e.g. uppercase) defaults to 1.0
DESCENT = {"g": 0.22, "j": 0.20}  # fraction of digit height a glyph hangs below the baseline

# Recipes: v1 = first experiment round; v2 = more varied trailing marks (v1 models read real ')' and '.' as extra
# characters) and more weight on the `NNa` shape, which is the future label format (set with --recipe).
# v3 = v2 without the `ii`/`iii` template: the scanner read real `11i` crops as `iii` (confidence 0.99-1.00), i.e. the
# repeated-'i' sequences taught "bars + dotted i = iii". Its 8% goes to `DD` and `DDL`; everything else is unchanged.
TEMPLATES_BY_RECIPE = {
    "v1": {"D": 0.30, "DD": 0.22, "L": 0.10, "DL": 0.10, "DDL": 0.20, "ii": 0.08},
    "v2": {"D": 0.26, "DD": 0.18, "L": 0.08, "DL": 0.10, "DDL": 0.30, "ii": 0.08},
    "v3": {"D": 0.26, "DD": 0.22, "L": 0.08, "DL": 0.10, "DDL": 0.34},
    "v4": {"D": 0.20, "DD": 0.16, "L": 0.06, "DL": 0.08, "DDL": 0.25, "G9": 0.25},
    "v5": {"D": 0.16, "DD": 0.14, "L": 0.05, "DL": 0.08, "DDL": 0.25, "G9": 0.16, "ONES": 0.16},
}
RECIPES_V2_STYLE = ("v2", "v3", "v4", "v5")  # recipes using the v2 clutter marks and crop widths
# v4 = v3 + a `G9` template for the g-vs-9 confusion. Our real crops contain no `g`, and a single-story `g` looks like a `9`,
# so the two are contrasted in identical contexts: `9` in digit slots, `g` only as the trailing letter of `NNa`-style labels.
# v5 = v4 + `ONES` (hard runs of the digit 1: the scanner dropped or invented a `1` in `11i`, `11.a`, `13.f`) + the dot-separated
# `NN.a` layout seen in some scanner crops (a dot between the number and the letter, NOT part of the label; the letter is written
# larger and further away) + lighter paper / paler ink. Set --glyphs to a bank built with make_g9_glyphs.py --font-share-other.
ONES_PATTERNS = {"11": 0.15, "111": 0.08, "1L": 0.15, "11L": 0.30, "D1L": 0.12, "1DL": 0.12, "1": 0.08}
G9_PATTERNS = {"9": 0.10, "D9": 0.09, "9D": 0.09, "9g": 0.12, "D9g": 0.22, "9Dg": 0.14, "99g": 0.06, "DDg": 0.18}
RECIPE = "v1"
ONLY_TEMPLATE = None  # e.g. "DDL" to build a pure NNa set (see --only-template)


def sample_label(rng):
    kinds, w = zip(*TEMPLATES_BY_RECIPE[RECIPE].items())
    kind = ONLY_TEMPLATE or rng.choice(kinds, p=np.array(w) / sum(w))
    dig = lambda lo=0: str(rng.integers(lo, 10))
    let = lambda: LETTERS[rng.integers(0, len(LETTERS))]

    def ones():
        pat = rng.choice(list(ONES_PATTERNS), p=np.array(list(ONES_PATTERNS.values())))
        return {"11": "11", "111": "111", "1L": "1" + let(), "11L": "11" + let(), "D1L": dig() + "1" + let(),
                "1DL": "1" + dig() + let(), "1": "1"}[pat]

    def g9():
        pat = rng.choice(list(G9_PATTERNS), p=np.array(list(G9_PATTERNS.values())))
        return {"9": "9", "D9": dig(1) + "9", "9D": "9" + dig(), "9g": "9g", "D9g": dig() + "9g", "9Dg": "9" + dig() + "g",
                "99g": "99g", "DDg": dig() + dig() + "g"}[pat]
    return {
        "D": lambda: dig(),
        "DD": lambda: dig(1) + dig(),     # real two-digit numbers have no leading zero
        "L": lambda: let(),
        "DL": lambda: dig() + let(),
        "DDL": lambda: dig() + dig() + let(),  # the `NNa` format: leading zeros allowed (02a, 01j)
        "ii": lambda: "i" * int(rng.choice([2, 3], p=[0.6, 0.4])),
        "G9": g9,
        "ONES": ones,
    }[kind]()


class GlyphBank:
    def __init__(self, images, labels):
        self.by_char = {c: images[labels == c] for c in sorted(set(labels.tolist()))}

    def sample(self, rng, ch):
        arr = self.by_char[ch]
        g = arr[rng.integers(0, len(arr))].astype(np.float32) / 255.0
        return tight_crop(g, 0.15)


def tight_crop(a, thr):
    ys, xs = np.where(a > thr)
    if len(ys) == 0:
        return a
    return a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def render_glyph(rng, bank, ch, hd):
    g = bank.sample(rng, ch)
    h0, w0 = g.shape
    th = max(8, int(round(hd * REL_H.get(ch, 1.0) * rng.uniform(0.9, 1.1))))
    tw = max(4, int(round(w0 * th / h0 * rng.uniform(0.85, 1.15))))
    g = cv2.resize(g, (tw, th), interpolation=cv2.INTER_CUBIC)
    pad = int(0.35 * max(th, tw)) + 2
    g = np.pad(g, pad)
    hh, ww = g.shape
    ang = float(np.clip(rng.normal(0, 6), -14, 14))
    shear = rng.uniform(-0.2, 0.2)
    m = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
    m[0, 1] += shear
    m[0, 2] -= shear * hh / 2
    g = cv2.warpAffine(g, m, (ww, hh), flags=cv2.INTER_LINEAR)
    k = rng.uniform(-1.2, 2.5) * hd / 60  # negative = thin the stroke, positive = thicken it
    if k >= 1:
        n = 2 * int(round(k)) + 1
        g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n)))
    elif k <= -0.6 and hd >= 50:
        g = cv2.erode(g, np.ones((3, 3), np.uint8))
    return np.clip(tight_crop(np.clip(g, 0, 1), 0.08), 0, 1)


def compose_sequence(rng, bank, label, hd, sep_dot=False):
    """sep_dot: `NN.a` layout -- a dot between the number and the trailing letter, which is written larger and set further away."""
    last = len(label) - 1
    glyphs = [render_glyph(rng, bank, ch, hd * (rng.uniform(1.0, 1.35) if sep_dot and k == last else 1.0)) for k, ch in enumerate(label)]
    gaps = [rng.uniform(-0.08, 0.25) * hd for _ in label]
    if sep_dot:
        gaps[last - 1] += rng.uniform(0.35, 0.8) * hd
    height = int(hd * 2.2)
    width = int(sum(g.shape[1] for g in glyphs) + sum(max(0, x) for x in gaps) + hd * 2)
    ink = np.zeros((height, width), np.float32)
    baseline = int(hd * 1.5)
    x = hd * 0.5
    for k, (ch, g, gap) in enumerate(zip(label, glyphs, gaps)):
        bottom = int(baseline + DESCENT.get(ch, 0.0) * hd + rng.normal(0, 0.03) * hd)
        y0, x0 = bottom - g.shape[0], int(x)
        if y0 < 0 or x0 + g.shape[1] > width:
            continue
        ink[y0: bottom, x0: x0 + g.shape[1]] = np.maximum(ink[y0: bottom, x0: x0 + g.shape[1]], g)
        if sep_dot and k == last - 1:
            cv2.circle(ink, (int(x + g.shape[1] + gap * 0.55), int(baseline - rng.uniform(0.05, 0.5) * hd)),
                       max(2, int(hd * rng.uniform(0.035, 0.07))), 1.0, -1, cv2.LINE_AA)
        x += g.shape[1] + gap
    rot = float(np.clip(rng.normal(0, 2), -5, 5))
    m = cv2.getRotationMatrix2D((width / 2, height / 2), rot, 1.0)
    ink = cv2.warpAffine(ink, m, (width, height), flags=cv2.INTER_LINEAR)
    return tight_crop(ink, 0.08)


def draw_clutter(rng, content, hd):
    """Embeds `content` (float ink 0-1) in a padded canvas and draws label-irrelevant marks around it."""
    ch, cw = content.shape
    pad = int(0.9 * hd)
    canvas = np.zeros((ch + 2 * pad, cw + 2 * pad), np.float32)
    canvas[pad: pad + ch, pad: pad + cw] = content
    sw = max(1.5, hd * rng.uniform(0.05, 0.10))
    aa = cv2.LINE_AA
    buf = np.zeros(canvas.shape, np.uint8)
    cy = pad + ch / 2

    def paren(x, opening):
        b = ch * 0.55 * rng.uniform(0.9, 1.2)
        a = hd * rng.uniform(0.10, 0.22)
        lo, hi = (110, 250) if opening else (-70, 70)
        cv2.ellipse(buf, (int(x), int(cy + rng.normal(0, 0.05) * hd)), (int(a), int(b)), 0, lo, hi, 255,
                    max(1, int(round(sw * rng.uniform(0.8, 1.2)))), aa)

    r = rng.random()
    xr = pad + cw + rng.uniform(0.05, 0.30) * hd
    if 0.50 <= r < 0.75:
        paren(xr, False)
    elif 0.75 <= r < 0.90:
        cv2.circle(buf, (int(xr), int(pad + ch - sw)), max(1, int(sw * rng.uniform(0.6, 1.1))), 255, -1, aa)
    elif r >= 0.90:
        paren(xr, False)
        paren(pad - rng.uniform(0.05, 0.30) * hd, True)
    if rng.random() < 0.07:  # hand-drawn circle around the number
        t = np.linspace(0, 2 * np.pi * rng.uniform(1.0, 1.15), 80)
        ax, ay = cw / 2 + rng.uniform(0.15, 0.4) * hd, ch / 2 + rng.uniform(0.15, 0.4) * hd
        wob = 1 + 0.05 * np.sin(rng.integers(2, 5) * t + rng.uniform(0, 6.28))
        pts = np.stack([pad + cw / 2 + ax * wob * np.cos(t), cy + ay * wob * np.sin(t)], 1).astype(np.int32)
        cv2.polylines(buf, [pts], False, 255, max(1, int(round(sw * rng.uniform(0.9, 1.3)))), aa)
    return np.maximum(canvas, buf.astype(np.float32) / 255.0), sw


def draw_clutter_v2(rng, content, hd):
    """v2 marks: varied hand-drawn ')' (height/width/tilt/thickness/wobble, may touch the digit), '.', ':', full '( )'
    wraps and circles, at rates matching what the real crops show (~30% ')' , ~12% '.', ~10% wraps)."""
    ch, cw = content.shape
    pad = int(1.0 * hd)
    canvas = np.zeros((ch + 2 * pad, cw + 2 * pad), np.float32)
    canvas[pad: pad + ch, pad: pad + cw] = content
    sw = max(1.5, hd * rng.uniform(0.05, 0.10))
    buf = np.zeros(canvas.shape, np.uint8)
    cy = pad + ch / 2

    def paren(xc, opening, big=1.0):
        b = ch * 0.5 * rng.uniform(0.7, 1.5) * big
        a = hd * rng.uniform(0.06, 0.35)
        ext = np.deg2rad(rng.uniform(45, 85))
        t = np.linspace(-ext, ext, 24) + (np.pi if opening else 0.0)
        tilt = np.deg2rad(rng.uniform(-15, 15))
        x, y = a * np.cos(t), b * np.sin(t)
        pts = np.stack([xc + x * np.cos(tilt) - y * np.sin(tilt), cy + rng.normal(0, 0.05) * hd + x * np.sin(tilt) + y * np.cos(tilt)], 1)
        pts = pts + rng.normal(0, 0.012 * hd, pts.shape)
        cv2.polylines(buf, [pts.astype(np.int32)], False, 255, max(1, int(round(sw * rng.uniform(0.7, 1.7)))), cv2.LINE_AA)

    def dot(x, y):
        cv2.circle(buf, (int(x), int(y)), max(1, int(sw * rng.uniform(0.5, 1.4))), 255, -1, cv2.LINE_AA)

    xr = pad + cw + rng.uniform(-0.10, 0.35) * hd
    r = rng.random()
    if r < 0.32:
        pass
    elif r < 0.62:
        paren(xr, False)
    elif r < 0.74:
        dot(xr, pad + ch - sw * rng.uniform(0.5, 2.0))
        if rng.random() < 0.2:  # colon
            dot(xr, pad + ch * rng.uniform(0.3, 0.55))
    elif r < 0.78:  # '3.)'
        dot(xr, pad + ch - sw)
        paren(xr + rng.uniform(0.15, 0.4) * hd, False)
    else:  # full '( )' wrap, as in (4) (ii) (iv)
        paren(xr, False, 1.15)
        paren(pad - rng.uniform(0.0, 0.3) * hd, True, 1.15)
    if rng.random() < 0.08:  # hand-drawn circle around the number
        t = np.linspace(0, 2 * np.pi * rng.uniform(1.0, 1.15), 80)
        ax, ay = cw / 2 + rng.uniform(0.15, 0.4) * hd, ch / 2 + rng.uniform(0.15, 0.4) * hd
        wob = 1 + 0.05 * np.sin(rng.integers(2, 5) * t + rng.uniform(0, 6.28))
        pts = np.stack([pad + cw / 2 + ax * wob * np.cos(t), cy + ay * wob * np.sin(t)], 1).astype(np.int32)
        cv2.polylines(buf, [pts], False, 255, max(1, int(round(sw * rng.uniform(0.9, 1.5)))), cv2.LINE_AA)
    return np.maximum(canvas, buf.astype(np.float32) / 255.0), sw


def render_crop(rng, bank, label):
    hd = int(rng.integers(45, 115))
    clutter = draw_clutter_v2 if RECIPE in RECIPES_V2_STYLE else draw_clutter
    sep_dot = RECIPE == "v5" and len(label) >= 2 and label[-1] in LETTERS and rng.random() < 0.35
    ink, sw = clutter(rng, compose_sequence(rng, bank, label, hd, sep_dot), hd)
    ink = tight_crop(ink, 0.05)
    ch, cw = ink.shape
    my, mx = rng.uniform(0.08, 0.45) * ch, rng.uniform(0.08, 0.45) * ch
    if RECIPE in RECIPES_V2_STYLE:  # longer non-'i' sequences (the NNa shape) get wider crops
        mu, sd = (0.86, 0.25) if len(label) == 1 else (1.5, 0.5) if len(label) >= 3 and label[0] != "i" else (1.1, 0.4)
        ar = float(np.clip(rng.normal(mu, sd), 0.5, 3.0))
    else:
        ar = float(np.clip(rng.normal(0.86, 0.25) if len(label) == 1 else rng.normal(1.08, 0.35), 0.5, 2.4))
    h, w = ch + 2 * my, cw + 2 * mx
    if w / h < ar:
        w = h * ar
    else:
        h = w / ar
    h, w = int(round(h)), int(round(w))
    x0 = int(rng.beta(2.5, 2.5) * max(1, w - cw))  # mostly centered, like the real crops
    y0 = int(rng.beta(2.5, 2.5) * max(1, h - ch))
    alpha = np.zeros((h, w), np.float32)
    alpha[y0: y0 + ch, x0: x0 + cw] = ink[: h - y0, : w - x0]
    if rng.random() < 0.12:  # vertical page-rule line at the left or right edge, as in the real crops
        lw = max(2, int(rng.uniform(0.02, 0.06) * w))
        xl = int(rng.uniform(0, 0.06) * w) if rng.random() < 0.5 else w - lw - int(rng.uniform(0, 0.06) * w)
        y1 = int(rng.uniform(0, 0.25) * h)
        alpha[y1:, max(0, xl): xl + lw] = 1.0

    # Photometric: paper, ink, noise, blur, compression.
    bg = rng.uniform(185 if RECIPE == "v5" else 215, 255)
    ramp = rng.uniform(-10, 10) * (np.linspace(-1, 1, w)[None, :] * rng.uniform(-1, 1) + np.linspace(-1, 1, h)[:, None] * rng.uniform(-1, 1))
    ink_lvl = rng.uniform(15, 130 if RECIPE == "v5" else 110)
    img = np.clip(bg + ramp, 0, 255) * (1 - alpha) + ink_lvl * alpha
    img = img + rng.normal(0, rng.uniform(0, 6), img.shape)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.3, 0.9) * max(h, w) / 100)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 91))])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


def write_lmdb(out, images, labels, map_size_mb):
    Path(out).mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=map_size_mb * 1024 * 1024)
    with env.begin(write=True) as txn:
        for n, (img, label) in enumerate(zip(images, labels), 1):
            txn.put(f"image-{n:09d}".encode(), cv2.imencode(".png", img)[1].tobytes())
            txn.put(f"label-{n:09d}".encode(), label.encode())
        txn.put(b"num-samples", str(len(labels)).encode())
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyphs", default="data/emnist/glyphs.npz")
    ap.add_argument("--glyph-split", choices=["train", "test"], default="train",
                    help="which EMNIST split's glyphs to draw from; use `test` for validation so writers never overlap")
    ap.add_argument("--recipe", choices=["v1", "v2", "v3", "v4", "v5"], default="v1")
    ap.add_argument("--only-template", choices=list(TEMPLATES_BY_RECIPE["v1"]), default=None,
                    help="draw every label from one template, e.g. DDL for a pure `NNa` evaluation set")
    ap.add_argument("--labels-csv", default=None,
                    help="render exactly these labels (column `text`, one crop per row, in order) instead of sampling templates; "
                         "overrides --n. Glyphs for every character used must be in --glyphs")
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="LMDB dir, e.g. data/emnist_lmdb/emnist_only/train/emnist")
    ap.add_argument("--dump-dir", default=None,
                    help="also write PNGs + a filename,image_path,text CSV here (for scoring with eval_parseq.py)")
    ap.add_argument("--preview", default=None, help="optional PNG contact sheet of the first 60 samples")
    ap.add_argument("--map-size-mb", type=int, default=1024)
    args = ap.parse_args()
    RECIPE, ONLY_TEMPLATE = args.recipe, args.only_template

    d = np.load(args.glyphs)
    bank = GlyphBank(d[f"{args.glyph_split}_images"], d[f"{args.glyph_split}_labels"])
    images, labels = [], []
    fixed = None
    if args.labels_csv:
        import pandas as pd
        fixed = pd.read_csv(args.labels_csv, keep_default_na=False)["text"].astype(str).tolist()
        args.n = len(fixed)
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, i])
        label = fixed[i] if fixed else sample_label(rng)
        images.append(render_crop(rng, bank, label))
        labels.append(label)
    write_lmdb(args.out, images, labels, args.map_size_mb)
    print(f"wrote {args.n} samples -> {args.out}")
    if args.dump_dir:
        import pandas as pd
        dump = Path(args.dump_dir).resolve()
        dump.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (img, lab) in enumerate(zip(images, labels)):
            cv2.imwrite(str(dump / f"synth_{i:06d}.png"), img)
            rows.append({"filename": f"synth_{i:06d}.png", "image_path": str(dump / f"synth_{i:06d}.png"), "text": lab})
        pd.DataFrame(rows).to_csv(dump / "test.csv", index=False)
    if args.preview:
        cells = []
        for img, lab in zip(images[:60], labels[:60]):
            im = cv2.cvtColor(cv2.resize(img, (110, 100)), cv2.COLOR_GRAY2BGR)
            cv2.putText(im, lab, (3, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cells.append(im)
        rows = [np.concatenate(cells[r * 10: (r + 1) * 10], 1) for r in range(6)]
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.preview, np.concatenate(rows, 0))
