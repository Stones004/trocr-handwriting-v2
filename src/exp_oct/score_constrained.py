"""Constrained decoding vs plain decoding on the in-vocabulary subsets, for old and in-vocabulary models.

    venv_parseq/Scripts/python src/exp_oct/score_constrained.py
Modes: plain / charset (0-9a-j only) / grammar (\\d{1,2}[a-j]?|[a-j]). Same weights, only the decoding step differs.
Sets: real test (docsplit test.csv, old labels, in-vocab labels only) and the user's reviewed ROI crops (in-vocab truth only).
Writes outputs/exp_oct/roi_check/constrained_scored.csv (per-crop predictions).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
import constrained  # noqa: E402
from eval_suite import TEST_CSV  # noqa: E402
from model_io import get_device, load_model, make_transform  # noqa: E402

E = ROOT / "outputs" / "exp_oct"
PAT = re.compile(r"\d{1,2}[a-j]?|[a-j]")
iv = lambda s: PAT.fullmatch(s) is not None  # noqa: E731
SEEDS = (42, 43, 44)
# the old baseline_s* / replay30_s* checkpoints no longer exist on disk; master_v3 / master_v4 (same recipe, all labels, one seed each) stand in for "old"
GROUPS = {"old master_v3": [("master_v3", "models/parseq_master_v3/parseq_master_v3.ckpt")],
          "old master_v4": [("master_v4", "models/parseq_master_v4/parseq_master_v4.ckpt")],
          "iv control": [(f"ivbase_s{s}", f"outputs/exp_oct/ivbase_s{s}/checkpoints/parseq_ivbase_s{s}.ckpt") for s in SEEDS],
          "iv replay30": [(f"ivreplay30_s{s}", f"outputs/exp_oct/ivreplay30_s{s}/checkpoints/parseq_ivreplay30_s{s}.ckpt") for s in SEEDS]}
MODES = ("plain", "charset", "grammar")
Q = Path("D:/OCR_codes/data/roi_watch/review_queue.csv")


def find(f):
    return next(d / f for d in (Q.parent / "input", Q.parent / "output") if (d / f).exists())


test = pd.read_csv(TEST_CSV, keep_default_na=False, dtype=str)
test = test[test["text"].map(iv)].reset_index(drop=True)
q = pd.read_csv(Q, keep_default_na=False, dtype=str)
q["truth"] = q["corrected_text"].str.strip()
q = q[q["truth"].map(iv)].reset_index(drop=True)
q["set"] = np.where(q["filename"].str.startswith("Doc"), "documents", "screenshots")
sets = {"test": (test["image_path"].tolist(), test["text"].tolist()), "roi": ([str(find(f)) for f in q["filename"]], q["truth"].tolist())}
print(f"in-vocab crops: real test {len(test)}, ROI {len(q)} (documents {(q.set == 'documents').sum()}, screenshots {(q.set == 'screenshots').sum()})")

dev = get_device()
rows, per_crop = [], q[["filename", "truth", "set"]].copy()
for g, runs in GROUPS.items():
    for s, (name, ck) in zip(SEEDS, runs):
        m = load_model(str(ROOT / ck), dev).eval()
        tf = make_transform(m)
        for sname, (paths, labels) in sets.items():
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
            for mode in MODES:
                preds = []
                for i in range(0, len(x), 64):
                    preds += [r[0] for r in constrained.decode(m, x[i:i + 64].to(dev), mode)]
                ok = np.array(preds) == np.array(labels)
                rows.append({"group": g, "seed": s, "set": sname, "mode": mode, "correct": ok.mean() * 100, "oov": np.mean([not iv(p) for p in preds]) * 100,
                             "doc": ok[(q.set == "documents").to_numpy()].mean() * 100 if sname == "roi" else np.nan,
                             "shot": ok[(q.set == "screenshots").to_numpy()].mean() * 100 if sname == "roi" else np.nan})
                if sname == "roi":
                    per_crop[f"{g.replace(' ', '_')}_s{s}_{mode}"] = preds
        print("scored", name, flush=True)
df = pd.DataFrame(rows)
df.to_csv(E / "roi_check" / "constrained_runs.csv", index=False)
per_crop.to_csv(E / "roi_check" / "constrained_scored.csv", index=False)
f = lambda x: f"{x.mean():.1f}" + (f" +- {x.std(ddof=1):.1f}" if len(x) > 1 else "")  # noqa: E731
for sname in ("test", "roi"):
    print(f"\n{sname.upper()} (in-vocabulary crops, exact %, mean +- std over 3 seeds)")
    out = []
    for (g, mode), d in df[df.set == sname].groupby(["group", "mode"], sort=False):
        r = {"model": g, "mode": mode, "exact %": f(d.correct), "pred out of vocab %": f"{d.oov.mean():.1f}"}
        if sname == "roi":
            r["documents %"], r["screenshots %"] = f(d.doc), f(d.shot)
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False))
