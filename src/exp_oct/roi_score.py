"""Score models on the user's reviewed ROI queue (D:/OCR_codes/data/roi_watch/review_queue.csv, column corrected_text = ground truth).

    venv_parseq/Scripts/python src/exp_oct/roi_score.py
Writes outputs/exp_oct/roi_check/roi_scored.csv (per-crop predictions of every model) and prints the tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
from eval_suite import wilson  # noqa: E402
from model_io import decode, get_device, load_model, make_transform  # noqa: E402

Q = Path("D:/OCR_codes/data/roi_watch/review_queue.csv")
FOLDER = Q.parent / "input"
E = "outputs/exp_oct"
MODELS = {"master_v3": "models/parseq_master_v3/parseq_master_v3.ckpt",
          **{f"baseline_s{s}": f"{E}/baseline_s{s}/checkpoints/parseq_baseline_s{s}.ckpt" for s in (42, 43, 44)},
          **{f"replay30_s{s}": f"{E}/models/parseq_replay30_s{s}/parseq_replay30_s{s}.ckpt" for s in (42, 43, 44)},
          **{f"replay20_s{s}": f"{E}/replay_p20_s{s}/checkpoints/parseq_replay_p20_s{s}.ckpt" for s in (42, 43, 44)}}
q = pd.read_csv(Q, keep_default_na=False, dtype=str)
q["truth"] = q["corrected_text"].str.strip()
assert (q.truth != "").all() and (q.reviewed == "True").all()
dev = get_device()
out = q[["filename", "truth", "prediction", "confidence"]].rename(columns={"prediction": "pipeline_pred", "confidence": "pipeline_conf"})
out["pipeline_conf"] = out["pipeline_conf"].astype(float)
for name, ck in MODELS.items():
    m = load_model(str(ROOT / ck), dev)
    tf = make_transform(m)
    r = [decode(m, tf(Image.open(FOLDER / f).convert("RGB")).unsqueeze(0).to(dev))[0] for f in q.filename]
    out[name], out[name + "_conf"] = [x[0] for x in r], [x[1] for x in r]
    print("scored", name, flush=True)
out["set"] = np.where(out.filename.str.startswith("Doc"), "doc", "screenshot")
heldout = out.filename.str[:7].isin(["Doc0729", "Doc0757", "Doc0771", "Doc0773"])
out["subset"] = np.where(out.set == "screenshot", "screenshot", np.where(heldout, "doc_heldout", "doc_master_v3_trained"))
out.to_csv(ROOT / E / "roi_check" / "roi_scored.csv", index=False)

names = ["pipeline"] + list(MODELS)
pred = lambda n: out["pipeline_pred" if n == "pipeline" else n]      # noqa: E731
conf = lambda n: out["pipeline_conf" if n == "pipeline" else n + "_conf"]  # noqa: E731


def stats(n, mask=None):
    m = np.ones(len(out), bool) if mask is None else mask.to_numpy()
    ok = (pred(n) == out.truth).to_numpy()[m]
    c = conf(n).to_numpy()[m]
    acc = c >= 0.9
    return {"n": int(m.sum()), "correct": int(ok.sum()), "acc": ok.mean() * 100, "ci_lo": wilson(ok.sum(), m.sum())[0] * 100, "ci_hi": wilson(ok.sum(), m.sum())[1] * 100,
            "ci_acc": (pred(n).str.lower() == out.truth.str.lower()).to_numpy()[m].mean() * 100,
            "conf_err_0.9": int((~ok & acc).sum()), "accepted@0.9": acc.mean() * 100, "prec@0.9": ok[acc].mean() * 100 if acc.any() else float("nan"),
            "mean_conf": c.mean()}


rows = []
for n in names:
    s = stats(n)
    rows.append({"model": n, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}})
print("\nALL 226 reviewed crops (exact, case-sensitive; Wilson 95% on single models)")
print(pd.DataFrame(rows).to_string(index=False))

print("\nBy subset (exact match %, correct/n):")
subs = ["doc_heldout", "doc_master_v3_trained", "screenshot"]
tab = []
for n in names:
    tab.append({"model": n, **{sb: f"{stats(n, out.subset == sb)['correct']}/{(out.subset == sb).sum()} ({stats(n, out.subset == sb)['acc']:.1f})" for sb in subs}})
print(pd.DataFrame(tab).to_string(index=False))

print("\nSeed means (all 226):")
for grp in ("baseline", "replay20", "replay30"):
    a = [stats(f"{grp}_s{s}")["acc"] for s in (42, 43, 44)]
    ce = [stats(f"{grp}_s{s}")["conf_err_0.9"] for s in (42, 43, 44)]
    cv = [stats(f"{grp}_s{s}")["accepted@0.9"] for s in (42, 43, 44)]
    print(f"  {grp:9s} acc {np.mean(a):.1f} +- {np.std(a, ddof=1):.1f}   confident errors(>=0.9) {np.mean(ce):.1f} +- {np.std(ce, ddof=1):.1f}   accepted@0.9 {np.mean(cv):.1f}%")

print("\nWord-like truths (len>=2 letters only):", out.truth[out.truth.str.fullmatch("[A-Za-z]{2,}")].tolist())
w = out[out.truth.str.fullmatch("[A-Za-z]{2,}")]
for n in names:
    print(f"  {n:14s}", list(zip(w.truth, pred(n)[w.index])))
