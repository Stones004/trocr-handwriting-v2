"""Score the in-vocabulary runs (ivbase_s*, ivreplay30_s*) and the older baseline / replay30 models on the IN-VOCABULARY subsets only.

    venv_parseq/Scripts/python src/exp_oct/score_iv.py
In vocabulary = label matches \\d{1,2}[a-j]?|[a-j]. Real test: the 379-crop docsplit test (old labels) filtered by label. ROI: the 226 reviewed crops
filtered by corrected_text. Exact match, case-sensitive. Mean +- std over seeds 42-44. Also prints the output-alphabet error rate (predictions that are
not in vocabulary) for each group.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
from model_io import decode, get_device, load_model, make_transform  # noqa: E402

E = ROOT / "outputs" / "exp_oct"
PAT = re.compile(r"\d{1,2}[a-j]?|[a-j]")
iv = lambda s: PAT.fullmatch(s) is not None  # noqa: E731
GROUPS = {"old baseline": ("baseline_s{}", "baseline_s{}"), "old replay30": ("replay_p30_s{}", "replay30_s{}"),
          "iv control": ("ivbase_s{}", None), "iv replay30": ("ivreplay30_s{}", None)}
SEEDS = (42, 43, 44)

# ---- real test
print("REAL TEST, in-vocabulary crops only")
rows = []
for g, (run, _) in GROUPS.items():
    acc, oov = [], []
    for s in SEEDS:
        f = E / run.format(s) / "eval" / "crops_test.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, keep_default_na=False, dtype=str)
        t = t[t["label"].map(iv)]
        acc.append((t["pred"] == t["label"]).mean() * 100)
        oov.append((~t["pred"].map(iv)).mean() * 100)
    if acc:
        rows.append({"model": g, "n_crops": len(t), "seeds": len(acc), "exact %": f"{np.mean(acc):.1f} +- {np.std(acc, ddof=1) if len(acc) > 1 else 0:.1f}",
                     "pred out of vocab %": f"{np.mean(oov):.1f}", "per-seed": " / ".join(f"{a:.1f}" for a in acc)})
print(pd.DataFrame(rows).to_string(index=False))

# ---- ROI
Q = Path("D:/OCR_codes/data/roi_watch/review_queue.csv")
q = pd.read_csv(Q, keep_default_na=False, dtype=str)
q["truth"] = q["corrected_text"].str.strip()
old = pd.read_csv(E / "roi_check" / "roi_scored.csv", keep_default_na=False, dtype=str)
assert (old["filename"] == q["filename"]).all()
dev = get_device()
preds = {}
for g, (run, col) in GROUPS.items():
    for s in SEEDS:
        if col:
            preds[(g, s)] = old[col.format(s)].tolist()
        else:
            ck = E / run.format(s) / "checkpoints" / f"parseq_{run.format(s)}.ckpt"
            if not ck.exists():
                continue
            m = load_model(str(ck), dev)
            tf = make_transform(m)
            preds[(g, s)] = [decode(m, tf(Image.open(next(d / f for d in (Q.parent / "input", Q.parent / "output") if (d / f).exists())).convert("RGB")).unsqueeze(0).to(dev))[0][0] for f in q["filename"]]
mask = q["truth"].map(iv).to_numpy()
sub = np.where(q["filename"].str.startswith("Doc"), "documents", "screenshots")
print(f"\nROI crops, in-vocabulary truth only: {mask.sum()} of {len(q)} (documents {int((mask & (sub == 'documents')).sum())}, screenshots {int((mask & (sub == 'screenshots')).sum())})")
rows = []
for g in GROUPS:
    r = {"model": g}
    for name, m in (("all", mask), ("documents", mask & (sub == "documents")), ("screenshots", mask & (sub == "screenshots"))):
        a = [np.mean(np.array(preds[(g, s)])[m] == q["truth"].to_numpy()[m]) * 100 for s in SEEDS if (g, s) in preds]
        if a:
            r[name + " %"] = f"{np.mean(a):.1f} +- {np.std(a, ddof=1) if len(a) > 1 else 0:.1f}"
    o = [np.mean([not iv(x) for x in np.array(preds[(g, s)])[mask]]) * 100 for s in SEEDS if (g, s) in preds]
    if o:
        r["pred out of vocab %"] = f"{np.mean(o):.1f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
out = q[["filename", "truth"]].copy()
for (g, s), p in preds.items():
    out[f"{g.replace(' ', '_')}_s{s}"] = p
out["in_vocab"] = mask
out.to_csv(E / "roi_check" / "iv_scored.csv", index=False)
