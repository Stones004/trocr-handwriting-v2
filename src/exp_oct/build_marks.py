"""Marked-NNa data: two-digit-plus-letter crops (12a) with the marks . ( ) [ ] drawn at ANY position between and around the characters.

    venv_parseq/Scripts/python src/exp_oct/build_marks.py

The existing master data only draws NNa marks from 9 templates (NN)a NN.a NNa) NNa. NN(a) (NN)a (NNa) NN(a (NNa), never between the two digits and never [ ].
Here the marks sit in four gaps:  gap0 | d1 | gap1 | d2 | gap2 | letter | gap3.
Pattern families (share): single mark in one gap (35%), a bracket pair wrapping a span of characters, '(' with ')' or '[' with ']' (30%, 10% of them mismatched
as sloppy handwriting, 30% also get a dot), two marks at random gaps (25%), three marks (10%). Marks: . ( ) [ ]  (no comma, as requested).
Label = characters only (strip_marks), so every crop's label is NNa. Numbers and letters come from the project's character-balanced samplers (src/master/spec.py),
rendering from src/master/render.py (same fonts, glyph banks, page augmentation).

Outputs: outputs/exp_oct/marks_nna/ (6,000 train crops + 300 held-out eval crops with a different seed, marks_train.csv / marks_eval.csv),
outputs/exp_oct/data/replay30_marks6k/{train,val,test}/handwriting = the replay-30% training set + the 6,000 new rows (val/test copied unchanged).
"""
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import lmdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "master"))
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
import render  # noqa: E402
import spec  # noqa: E402
from progress import log  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct"
IMG = OUT / "marks_nna"
N_TRAIN, N_EVAL = 6000, 300
MARKS = [".", "(", ")", "[", "]"]
CLOSE = {"(": ")", "[": "]"}


def pattern(rng):
    """-> (gaps: list of 4 strings, family)."""
    g = ["", "", "", ""]
    r = rng.random()
    if r < 0.35:
        g[rng.randrange(4)] = rng.choice(MARKS)
        return g, "single mark"
    if r < 0.65:
        o = rng.choice(["(", "["])
        c = CLOSE[o] if rng.random() > 0.10 else rng.choice([m for m in ")]" if m != CLOSE[o]])
        a = rng.randrange(3)
        b = rng.randrange(a + 1, 4)
        g[a] += o
        g[b] += c
        if rng.random() < 0.30:
            g[rng.randrange(4)] += "."
        return g, "bracket pair"
    k, name = (2, "two marks") if r < 0.90 else (3, "three marks")
    for _ in range(k):
        g[rng.randrange(4)] += rng.choice(MARKS)
    return g, name


def make(n, seed):
    rng = random.Random(seed)
    nums, lets = spec.numbers(rng, n), spec.balanced(rng, spec.LETTERS, n)
    rows = []
    for i in range(n):
        g, fam = pattern(rng)
        d1, d2 = nums[i][0], nums[i][1]
        drawn = g[0] + d1 + g[1] + d2 + g[2] + lets[i] + g[3]
        label = spec.strip_marks(drawn)
        assert label == nums[i] + lets[i] and len(label) == 3, (drawn, label)
        rows.append({"label": label, "drawn": drawn, "family": fam, "gap0": g[0], "gap1": g[1], "gap2": g[2], "gap3": g[3]})
    return pd.DataFrame(rows)


def render_set(df, tag, seed):
    bank = render.load_bank()
    names = []
    for i, r in enumerate(df.itertuples()):
        name = f"{tag}_{i:05d}.png"
        cv2.imwrite(str(IMG / name), render.render_label(r.drawn, seed, i, bank))
        names.append(name)
    df.insert(0, "filename", names)
    df.insert(1, "image_path", [str(IMG / n) for n in names])
    df.insert(2, "text", df["label"])
    return df


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    tr = render_set(make(N_TRAIN, 20261007), "mk", 8801)
    ev = render_set(make(N_EVAL, 20261008), "mke", 8802)
    tr.to_csv(IMG / "marks_train.csv", index=False)
    ev.to_csv(IMG / "marks_eval.csv", index=False)
    log(f"marks_nna: {len(tr)} train crops, {len(ev)} held-out eval crops rendered")
    for name, d in (("train", tr), ("eval", ev)):
        print(f"\n{name}: families {d.family.value_counts().to_dict()}")
        pos = Counter()
        for r in d.itertuples():
            for gi, gap in enumerate([r.gap0, r.gap1, r.gap2, r.gap3]):
                for m in gap:
                    pos[(f"gap{gi}", m)] += 1
        t = pd.Series(pos).unstack(fill_value=0)
        print("marks per gap (rows = gap, cols = mark):")
        print(t.to_string())
    print("\nmarks between the two digits (gap1) in train:", int((tr.gap1 != "").sum()), "| crops containing a square bracket:", int(tr.drawn.str.contains(r"[\[\]]").sum()),
          "| crops with 2+ marks:", int((tr.drawn.str.count(r"[.()\[\]]") >= 2).sum()))
    # training set = replay30 + new rows
    src, dst = OUT / "data" / "replay_p30", OUT / "data" / "replay30_marks6k"
    for split in ("val", "test"):
        if not (dst / split).exists():
            shutil.copytree(src / split, dst / split)
    t = dst / "train" / "handwriting"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    senv = lmdb.open(str(src / "train" / "handwriting"), readonly=True, lock=False)
    env = lmdb.open(str(t), map_size=1024 * 1024 * 1024)
    with senv.begin() as stx, env.begin(write=True) as tx:
        n0 = int(stx.get(b"num-samples"))
        for i in range(1, n0 + 1):
            tx.put(f"image-{i:09d}".encode(), stx.get(f"image-{i:09d}".encode()))
            tx.put(f"label-{i:09d}".encode(), stx.get(f"label-{i:09d}".encode()))
        for j, r in enumerate(tr.itertuples(), n0 + 1):
            tx.put(f"image-{j:09d}".encode(), Path(r.image_path).read_bytes())
            tx.put(f"label-{j:09d}".encode(), r.text.encode())
        tx.put(b"num-samples", str(n0 + len(tr)).encode())
    env.close()
    total = n0 + len(tr)
    steps = -(-total // 32)
    log(f"replay30_marks6k: {n0} replay-30 rows + {len(tr)} marked NNa = {total} rows, {steps} steps/epoch, {steps * 15} steps per 15-epoch run")


if __name__ == "__main__":
    main()
