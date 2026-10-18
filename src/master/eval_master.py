"""Score checkpoints on the REAL test split per bucket, plus a clearly-labelled SYNTHETIC probe for the buckets that have no real test data.

    venv_parseq/Scripts/python src/master/eval_master.py --model master=<final.ckpt> --model baseline=<ckpt> [--out reports/master_eval.md]

Real test = data/master/test.csv (the document-split seed-42 test crops, real only). Rows show n and a 95% Wilson interval.
Synthetic probe = fresh crops of NNa / punctuated-NNa / square-bracket labels drawn with a seed the training set never used; it
shows whether a format was learned, NOT how it performs on real handwriting.
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help="name=checkpoint path (repeatable)")
ap.add_argument("--out", default="reports/master_eval.md")
ap.add_argument("--probe-n", type=int, default=150, help="synthetic probe crops per bucket")
args = ap.parse_args()
models = dict(m.split("=", 1) for m in args.model)
work = ROOT / "data" / "master_eval"
work.mkdir(exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return f"{p * 100:.0f}% [{max(0, c - h) * 100:.0f}-{min(1, c + h) * 100:.0f}]"


def score(name, ckpt, csv, tag):
    pred = work / f"{name}_{tag}.csv"
    if not pred.exists():
        subprocess.run([sys.executable, str(ROOT / "src/parseq/eval_parseq.py"), "--checkpoint", ckpt, "--csv", str(csv), "--output", str(pred)],
                       check=True, capture_output=True, cwd=ROOT)
    return pd.read_csv(pred, keep_default_na=False)


# ---- real test
test = pd.read_csv(ROOT / "data/master/test.csv", keep_default_na=False)
test["text"] = test.text.astype(str)
real = {n: score(n, c, ROOT / "data/master/test.csv", "real").merge(test[["filename", "bucket"]], on="filename") for n, c in models.items()}

# ---- synthetic probe (not real handwriting): held-out seed. Label = characters only; the drawn text carries the marks / Q prefix.
if not (work / "probe.csv").exists():
    import render
    bank = render.load_bank()
    rng = random.Random(999)
    rows = []

    def add(group, label, drawn):
        i = len(rows)
        path = work / f"probe_{i:04d}.png"
        cv2.imwrite(str(path), render.render_label(drawn, 999, i, bank))
        rows.append({"filename": path.name, "image_path": str(path), "text": label, "bucket": group})

    for s in spec.templated(rng, args.probe_n, [("NNa", 1)]):
        add("NNa plain", s, s)
    for s in spec.templated(rng, args.probe_n, spec.MARKED_PATTERNS):
        add("NNa with marks ( ) .", spec.strip_marks(s), s)
    for s in spec.templated(rng, args.probe_n, spec.SQUARE_PATTERNS):
        add("square brackets", spec.strip_marks(s), s)
    for s in spec.templated(rng, args.probe_n, [("NNa", 1), ("NN", 1)]):
        core = s if rng.random() < 0.5 else s[:2]
        add("Q prefix", core, rng.choice(["Q.", "Q", "Q "]) + core)
    pd.DataFrame(rows).to_csv(work / "probe.csv", index=False)
probe = pd.read_csv(work / "probe.csv", keep_default_na=False)
probe["text"] = probe.text.astype(str)
syn = {n: score(n, c, work / "probe.csv", "probe").merge(probe[["filename", "bucket"]], on="filename") for n, c in models.items()}

# ---- held-out new documents: crops from D:/OCR_codes/data/roi_watch/input in documents the model never trained on, labelled by Claude's
# reading of the images (data/roi_labels.csv, holdout=True) -- NOT reviewed ground truth
spot_csv = ROOT / "data/roi_heldout.csv"
spot = {}
lab_csv = ROOT / "data/roi_labels.csv"
if lab_csv.exists():
    st = pd.read_csv(lab_csv, keep_default_na=False)
    st = st[st.holdout.astype(str) == "True"]
    st["text"] = st.text.astype(str)
    st.to_csv(spot_csv, index=False)
    spot = {n: score(n, c, spot_csv, "heldout").merge(st[["filename", "q_prefix", "doc"]], on="filename") for n, c in models.items()}

lines = ["# Master model evaluation\n"]


def table(title, frames, groups, note=""):
    lines.extend([f"## {title}\n", note, "| group | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)])
    for label, mask_fn in groups:
        n = int(mask_fn(next(iter(frames.values()))).sum())
        cells = [wilson(int(f[mask_fn(f)].exact_match.astype(bool).sum()), n) if n else "n/a" for f in frames.values()]
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    lines.append("")


buckets = [b for b in spec.BUCKETS if b != "other"]
groups = [("ALL real test crops", lambda f: f.bucket == f.bucket)] + [(b, (lambda b: lambda f: f.bucket == b)(b)) for b in buckets] + \
         [("other (unclassified)", lambda f: f.bucket == "other")]
table("Real test split, per bucket (document split seed 42, real crops only)", real, groups,
      "95% Wilson interval in brackets. n/a = no real test crops exist for that bucket, so it cannot be evaluated on real data.\n")
roman = ["II", "III", "IV", "VI", "VII", "VIII", "IX"]
table("Sparse Roman numerals, real test crops", real, [(r, (lambda r: lambda f: f.target.astype(str) == r)(r)) for r in roman] +
      [("all seven pooled", lambda f: f.target.astype(str).isin(roman))])
table("SYNTHETIC probe (held-out seed; not real handwriting)", syn,
      [(b, (lambda b: lambda f: f.bucket == b)(b)) for b in ("NNa plain", "NNa with marks ( ) .", "square brackets", "Q prefix")],
      f"{args.probe_n} fresh synthetic crops per bucket. Tells you whether the format was learned, not real-world accuracy.\n")
if spot:
    table("Held-out NEW documents (Doc0729, 0757, 0771, 0773; labels = Claude's reading, unverified, never trained on)", spot,
          [("all held-out crops", lambda f: f.target == f.target), ("with a Q prefix", lambda f: f.q_prefix.astype(str) == "True"),
           ("without Q prefix", lambda f: f.q_prefix.astype(str) == "False")] +
          [(d, (lambda d: lambda f: f.doc == d)(d)) for d in sorted(next(iter(spot.values())).doc.unique())])
for n, f in real.items():
    wrong = f[~f.exact_match.astype(bool)]
    lines.append(f"## Real-test errors: {n} ({len(wrong)})\n")
    lines.append(", ".join(f"{r.target}->{r.prediction}" for r in wrong.itertuples()) + "\n")
out = ROOT / args.out
out.write_text("\n".join(lines), "utf-8")
print("\n".join(lines))
