"""When does PARSeq fail on the 226 reviewed crops? Groups its misses by crop source, label, image properties and error type.

    venv_parseq/Scripts/python src/exp_oct/failure_patterns.py

Inputs: outputs/exp_oct/compare_small/per_crop.csv (live-run answers; PARSeq scored vs the characters-only label),
the PARSeq confidence stored in outputs/exp_oct/compare_board/per_crop.csv, and the crop images.
Output: printed tables + outputs/exp_oct/compare_small/parseq_failure_patterns.csv (one row per crop with its features).
Small groups are noisy: read the counts, not just the percentages (Wilson 95% intervals are shown).
"""
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CROPS = Path("D:/OCR_codes/data/roi_watch/input")
OUT = ROOT / "outputs" / "exp_oct" / "compare_small"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def err_type(pred, truth):
    p, t = pred, truth
    if p.lower() == t.lower():
        return "case only"
    if t.startswith(p) and len(p) < len(t):
        return "dropped trailing character(s)"
    if t.endswith(p) and len(p) < len(t):
        return "dropped leading character(s)"
    if p.startswith(t) or p.endswith(t):
        return "extra character(s)"
    if len(p) == len(t):
        d = sum(a != b for a, b in zip(p, t))
        kinds = {("d" if a.isdigit() else "l") + ("d" if b.isdigit() else "l") for a, b in zip(p, t) if a != b}
        return "one character wrong" if d == 1 else "several characters wrong"
    return "different length"


def main():
    s = pd.read_csv(OUT / "per_crop.csv", keep_default_na=False, dtype=str)
    b = pd.read_csv(ROOT / "outputs/exp_oct/compare_board/per_crop.csv", keep_default_na=False, dtype=str)[["filename", "parseq_conf"]]
    d = s.merge(b, on="filename", how="left")
    d["miss"] = d.parseq_ok != "True"
    d["conf"] = pd.to_numeric(d.parseq_conf, errors="coerce")
    d["source"] = np.where(d.filename.str.startswith("Doc"), "document crop", "screenshot")
    d["label_len"] = d.truth.str.len().clip(upper=4).map({1: "1 char", 2: "2 chars", 3: "3 chars", 4: "4+ chars"})
    d["label_kind"] = d.truth.map(lambda t: "digits only" if t.isdigit() else "letters only" if t.isalpha() else "digits + letters")
    d["has_upper"] = d.truth.str.contains("[A-Z]")
    d["confusable"] = d.truth.str.contains(r"[1lIiOo0cCsSvVxXzZuUwWkKpP]")  # letters/digits that look alike or differ only in case
    feats = []
    for fn in d.filename:
        g = cv2.imdecode(np.fromfile(str(CROPS / fn), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        h, w = g.shape
        ink = float((g < 128).mean())
        lo, hi = np.percentile(g, [5, 95])
        # width of the crop relative to its height, and how much of the crop's own width the ink spans
        cols = np.where((g < 128).any(axis=0))[0]
        span = (cols.max() - cols.min() + 1) / w if len(cols) else 0
        feats.append((h, w, w / h, ink, hi - lo, span))
    d[["height", "width", "aspect", "ink_share", "contrast", "ink_span"]] = pd.DataFrame(feats, index=d.index)
    d["height_band"] = pd.qcut(d.height, 4, duplicates="drop").astype(str)
    d["aspect_band"] = pd.qcut(d.aspect, 4, duplicates="drop").astype(str)
    d["ink_band"] = pd.qcut(d.ink_share, 4, duplicates="drop").astype(str)
    d["contrast_band"] = pd.qcut(d.contrast, 4, duplicates="drop").astype(str)

    n, k = len(d), int(d.miss.sum())
    print(f"PARSeq misses {k} of {n} ({100 * k / n:.1f}%)\n")

    def table(col, order=None):
        rows = []
        for key, g in d.groupby(col, sort=True):
            m = int(g.miss.sum())
            lo, hi = wilson(m, len(g))
            rows.append((str(key), len(g), m, 100 * m / len(g), 100 * lo, 100 * hi))
        if order:
            rows.sort(key=lambda r: order.index(r[0]))
        print(f"by {col}:")
        for key, c, m, p, lo, hi in rows:
            print(f"  {key:<28} {m:>3}/{c:<3} miss {p:5.1f}%  (95% {lo:4.1f}-{hi:4.1f})")
        print()

    table("source")
    table("label_len")
    table("label_kind")
    table("has_upper")
    table("confusable")
    for c in ("height_band", "aspect_band", "ink_band", "contrast_band"):
        table(c)
    d["conf_band"] = pd.cut(d.conf, [0, .6, .8, .9, .95, 1.0001], labels=["<0.6", "0.6-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"])
    table("conf_band")

    miss = d[d.miss].copy()
    miss["type"] = [err_type(p, t) for p, t in zip(miss.parseq, miss.truth)]
    print("kinds of error:")
    print(miss.type.value_counts().to_string(), "\n")
    # character-level confusions (same-length one-character errors)
    pairs = {}
    for p, t in zip(miss.parseq, miss.truth):
        if len(p) == len(t):
            for a, c in zip(p, t):
                if a != c:
                    pairs[(c, a)] = pairs.get((c, a), 0) + 1
    print("most common character confusions (truth -> predicted):", sorted(pairs.items(), key=lambda kv: -kv[1])[:12], "\n")
    print("medians, hits vs misses:")
    for f in ("height", "width", "aspect", "ink_share", "contrast", "ink_span"):
        print(f"  {f:<10} hits {d[~d.miss][f].median():8.3f} | misses {d[d.miss][f].median():8.3f}")
    print(f"\nPARSeq confidence: hits median {d[~d.miss].conf.median():.2f}, misses median {d[d.miss].conf.median():.2f}; "
          f"misses with conf >= 0.9: {int((d[d.miss].conf >= 0.9).sum())} of {k}")
    d.drop(columns=["parseq_conf"]).to_csv(OUT / "parseq_failure_patterns.csv", index=False)
    print("\nwrote", OUT / "parseq_failure_patterns.csv")


if __name__ == "__main__":
    main()
