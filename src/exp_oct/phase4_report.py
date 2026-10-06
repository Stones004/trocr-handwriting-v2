"""Session 3 report numbers: temperature tables, coverage-vs-precision curves (real test and the user's ROI crops), label-smoothing summary.

    venv_parseq/Scripts/python src/exp_oct/phase4_report.py
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
import summarize as S  # noqa: E402
from model_io import decode, get_device, load_model, make_transform  # noqa: E402

O = ROOT / "outputs" / "exp_oct"
GROUPS = {"baseline": [f"baseline_s{s}" for s in (42, 43, 44)], "replay 30%": [f"replay_p30_s{s}" for s in (42, 43, 44)],
          "label smoothing 0.05": [f"ls005_s{s}" for s in (42, 43, 44)]}
COL = {"baseline": "#222222", "replay 30%": "#1b9e77", "label smoothing 0.05": "#d95f02"}


def curve(conf, ok, grid):
    """precision of the top-k crops by confidence, interpolated on coverage grid."""
    o = np.argsort(-np.asarray(conf), kind="stable")
    c = np.cumsum(np.asarray(ok, float)[o]) / np.arange(1, len(o) + 1)
    cov = np.arange(1, len(o) + 1) / len(o)
    return np.interp(grid, cov, c)


# ---- summary table (real test) for label smoothing, 3 seeds
L = lambda r: S.load(O / r / "eval" / "metrics.json")  # noqa: E731
print(S.table("Label smoothing (mean +- std over seeds)", [(g + (" (3 seeds)"), [L(r) for r in rs]) for g, rs in GROUPS.items()] +
                [("label smoothing 0.1 (seed 42 only)", [L("ls010_s42")])]))

# ---- temperature summary: mean over 3 seeds of before / after
print("\nTemperature scaling (T fitted on val only), mean over 3 seeds, applied to test:")
rows = []
for g, rs in GROUPS.items():
    t = [json.load(open(O / r / "eval" / "temperature.json")) for r in rs]
    for tag, key in (("T=1 (before)", "before"), ("T val-fit, crop NLL", "after_crop"), ("T val-fit, token NLL", "after")):
        rows.append({"model": g, "setting": tag, "T": np.mean([x[key]["T"] for x in t]), "ECE": np.mean([x[key]["test_ece"] for x in t]),
                     "NLL": np.mean([x[key]["test_nll"] for x in t]), "cov@99 %": np.mean([x[key]["cov99"] for x in t]) * 100,
                     "confident errors": np.mean([x[key]["confident_errors"] for x in t])})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---- coverage vs precision, real test
grid = np.linspace(0.1, 1.0, 91)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
print("\nPrecision at fixed coverage on the 379-crop test (mean over 3 seeds):")
for g, rs in GROUPS.items():
    cur = np.mean([curve(d.conf, d.correct.astype(str) == "True", grid) for d in (pd.read_csv(O / r / "eval" / "crops_test.csv", keep_default_na=False) for r in rs)], 0)
    ax[0].plot(grid * 100, cur * 100, color=COL[g], label=g)
    print(f"  {g:22s}", {f"{int(c*100)}%": round(float(np.interp(c, grid, cur)) * 100, 1) for c in (0.5, 0.7, 0.8, 0.9)})
ax[0].set_title("Real test (379 crops): precision at each coverage"); ax[0].set_xlabel("coverage %"); ax[0].set_ylabel("precision of accepted %"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

# ---- coverage vs precision, ROI crops
q = pd.read_csv("D:/OCR_codes/data/roi_watch/review_queue.csv", keep_default_na=False, dtype=str)
truth, folder = q.corrected_text.str.strip().tolist(), Path("D:/OCR_codes/data/roi_watch/input")
dev = get_device()
print("\nPrecision at fixed coverage on the 226 reviewed ROI crops (mean over 3 seeds):")
roi = {}
for g, rs in GROUPS.items():
    curs, accs = [], []
    for r in rs:
        m = load_model(str(O / r / "checkpoints" / f"parseq_{r}.ckpt"), dev)
        tf = make_transform(m)
        out = [decode(m, tf(Image.open(folder / f).convert("RGB")).unsqueeze(0).to(dev))[0] for f in q.filename]
        ok = np.array([o[0] == t for o, t in zip(out, truth)])
        curs.append(curve([o[1] for o in out], ok, grid)); accs.append(ok.mean() * 100)
    cur = np.mean(curs, 0)
    ax[1].plot(grid * 100, cur * 100, color=COL[g], label=f"{g} (acc {np.mean(accs):.1f})")
    print(f"  {g:22s} acc {np.mean(accs):.1f}", {f"{int(c*100)}%": round(float(np.interp(c, grid, cur)) * 100, 1) for c in (0.5, 0.7, 0.8, 0.9)})
ax[1].set_title("Your 226 ROI crops: precision at each coverage"); ax[1].set_xlabel("coverage %"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(O / "coverage_vs_precision.png", dpi=140)
