"""Marked data v2: marks . ( ) [ ] at ANY position, on every format of the NNa family (not NNa only).

    venv_parseq/Scripts/python src/exp_oct/build_marks2.py

Formats (share of the new crops): NNa 40% | 1 digit 12% | 2 digits 12% | Na (digit+letter) 8% | 1 letter 14% | 2 letters, Roman only (ii iv vi ix, both cases) 14%.
Marks sit in the gaps around and between the k characters (k+1 gaps), same pattern families as build_marks.py. Label = characters only.
Outputs: outputs/exp_oct/marks2/ (4,000 train crops + 100 held-out crops per format with a different seed) and the LMDB
outputs/exp_oct/data/replay30_marks2/{train,val,test}/handwriting = replay-30% training set + the 4,000 new rows (val/test copied unchanged).
"""
import random
import shutil
import sys
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
IMG = OUT / "marks2"
N_TRAIN, N_EVAL_PER_FORMAT = 4000, 100
MARKS = [".", "(", ")", "[", "]"]
CLOSE = {"(": ")", "[": "]"}
SHARES = [("NNa", 0.40), ("1 digit", 0.12), ("2 digits", 0.12), ("Na", 0.08), ("1 letter", 0.14), ("2 letters (Roman)", 0.14)]
ROMAN2 = ["ii", "iv", "vi", "ix"]
ROMAN2 = ROMAN2 + [r.upper() for r in ROMAN2]


def labels(fmt, n, rng):
    if fmt == "NNa":
        return [a + b for a, b in zip(spec.numbers(rng, n), spec.balanced(rng, spec.LETTERS, n))]
    if fmt == "1 digit":
        return spec.balanced(rng, spec.DIGITS, n)
    if fmt == "2 digits":
        return spec.numbers(rng, n)
    if fmt == "Na":
        return [a + b for a, b in zip(spec.balanced(rng, spec.DIGITS, n), spec.balanced(rng, spec.LETTERS, n))]
    if fmt == "1 letter":
        return spec.balanced(rng, spec.LETTERS, n)
    return spec.balanced(rng, ROMAN2, n)


def pattern(rng, k):
    """k characters -> k+1 gaps. -> (gaps, family)."""
    g = [""] * (k + 1)
    r = rng.random()
    if r < 0.35:
        g[rng.randrange(k + 1)] = rng.choice(MARKS)
        return g, "single mark"
    if r < 0.65:
        o = rng.choice(["(", "["])
        c = CLOSE[o] if rng.random() > 0.10 else rng.choice([m for m in ")]" if m != CLOSE[o]])
        a = rng.randrange(k)
        b = rng.randrange(a + 1, k + 1)
        g[a] += o
        g[b] += c
        if rng.random() < 0.30:
            g[rng.randrange(k + 1)] += "."
        return g, "bracket pair"
    n, name = (2, "two marks") if r < 0.90 else (3, "three marks")
    for _ in range(n):
        g[rng.randrange(k + 1)] += rng.choice(MARKS)
    return g, name


def make(counts, seed):
    rng = random.Random(seed)
    rows = []
    for fmt, n in counts:
        for lab in labels(fmt, n, rng):
            g, fam = pattern(rng, len(lab))
            drawn = g[0] + "".join(c + g[i + 1] for i, c in enumerate(lab))
            assert spec.strip_marks(drawn) == lab, (drawn, lab)
            rows.append({"label": lab, "drawn": drawn, "format": fmt, "family": fam, "marks_at_gaps": "|".join(g)})
    random.Random(seed + 1).shuffle(rows)
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
    counts = [(f, round(N_TRAIN * s)) for f, s in SHARES]
    counts[0] = (counts[0][0], N_TRAIN - sum(n for _, n in counts[1:]))
    tr = render_set(make(counts, 20261101), "m2", 9901)
    ev = render_set(make([(f, N_EVAL_PER_FORMAT) for f, _ in SHARES], 20261102), "m2e", 9902)
    tr.to_csv(IMG / "marks2_train.csv", index=False)
    ev.to_csv(IMG / "marks2_eval.csv", index=False)
    log(f"marks2: {len(tr)} train crops, {len(ev)} held-out eval crops rendered")
    print("train formats:", tr.format.value_counts().to_dict())
    print("train families:", tr.family.value_counts().to_dict())
    print("two-letter labels:", tr[tr.format == "2 letters (Roman)"].label.value_counts().to_dict())
    print("crops with a mark between characters:", int(tr.marks_at_gaps.map(lambda s: any(s.split("|")[1:-1])).sum()))
    src, dst = OUT / "data" / "replay_p30", OUT / "data" / "replay30_marks2"
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
    log(f"replay30_marks2: {n0} replay-30 rows + {len(tr)} marked = {total} rows, {steps} steps/epoch, {steps * 15} steps per 15-epoch run")


if __name__ == "__main__":
    main()
