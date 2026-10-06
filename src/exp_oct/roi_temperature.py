"""Does the val-fitted temperature help on the user's reviewed ROI crops (a third, independent set)?  T comes from <run>/eval/temperature.json (fitted on VAL only).

    venv_parseq/Scripts/python src/exp_oct/roi_temperature.py baseline_s42 ... replay_p30_s44
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
from eval_suite import accepted_stats, conf_from_logits, ece, run_model  # noqa: E402
from model_io import get_device, load_model, make_transform  # noqa: E402

Q = Path("D:/OCR_codes/data/roi_watch/review_queue.csv")
q = pd.read_csv(Q, keep_default_na=False, dtype=str)
df = pd.DataFrame({"filename": q.filename, "image_path": [str(Q.parent / "input" / f) for f in q.filename], "text": q.corrected_text.str.strip()})
dev = get_device()
rows = []
for run in sys.argv[1:]:
    d = ROOT / "outputs" / "exp_oct" / run
    t = json.loads((d / "eval" / "temperature.json").read_text())
    ck = d / "checkpoints" / f"parseq_{run}.ckpt"
    m = load_model(str(ck), dev)
    preds, _, lg = run_model(m, df, make_transform(m), dev)
    ok = np.array([p == t_ for p, t_ in zip(preds, df.text)])
    for tag, T in (("T=1", 1.0), ("T val-fit (crop NLL)", t["after_crop"]["T"]), ("T val-fit (token NLL)", t["after"]["T"])):
        c = conf_from_logits(m, lg, T)
        thr = t["before"]["val_threshold"] if T == 1.0 else t["after_crop" if "crop" in tag else "after"]["val_threshold"]
        cov, prec, n = accepted_stats(c, ok, thr) if thr else (0, float("nan"), 0)
        a9 = c >= 0.9
        rows.append({"run": run, "setting": tag, "T": round(T, 2), "acc": ok.mean() * 100, "ECE": ece(c, ok), "mean_conf": c.mean(),
                     "conf>=0.9: accepted %": a9.mean() * 100, "precision": ok[a9].mean() * 100 if a9.any() else np.nan, "confident errors": int((~ok & a9).sum()),
                     "val-threshold accepted %": cov * 100, "val-threshold precision": prec * 100})
    print("done", run, flush=True)
r = pd.DataFrame(rows)
r.to_csv(ROOT / "outputs" / "exp_oct" / "roi_check" / "roi_temperature.csv", index=False)
g = r.groupby("setting", sort=False)[["acc", "ECE", "mean_conf", "conf>=0.9: accepted %", "precision", "confident errors", "val-threshold accepted %", "val-threshold precision"]].mean().round(3)
print(g.to_string())
print(r.groupby(["setting"], sort=False)["ECE"].agg(["mean", "std"]).round(3))
