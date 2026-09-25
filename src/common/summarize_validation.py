"""Summarize <reports_dir>/validation/*__{train,val,test}.csv into <reports_dir>/model_validation.{csv,md}.
    python src/common/summarize_validation.py [--flag-threshold 0.5] [--rare-max 5]
Defaults come from config.yaml (inference.flag_threshold, validation.rare_max, paths.*). Pure pandas/numpy: works in
either venv.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import cfg, path
from validation_metrics import OUT_DIR, SPLITS, auroc, boot_ci, ece

ap = argparse.ArgumentParser()
ap.add_argument("--flag-threshold", type=float, default=cfg["inference"]["flag_threshold"])
ap.add_argument("--rare-max", type=int, default=cfg["validation"]["rare_max"], help="a test label is 'rare' if it has <= this many training examples")
args = ap.parse_args()

train_freq = pd.read_csv(path("processed_dir") / "train.csv")["text"].astype(str).str.strip().value_counts()
names = sorted({Path(p).name.split("__")[0] for p in glob.glob(f"{OUT_DIR}/*__test.csv")})
rows, preds = [], {}
for n in names:
    d = {s: pd.read_csv(f"{OUT_DIR}/{n}__{s}.csv") for s in SPLITS if os.path.exists(f"{OUT_DIR}/{n}__{s}.csv")}
    if len(d) < 3:
        continue
    meta = json.load(open(f"{OUT_DIR}/{n}__meta.json")) if os.path.exists(f"{OUT_DIR}/{n}__meta.json") else {}
    acc = {s: d[s].exact.mean() for s in SPLITS}
    cer = {s: d[s].edits.sum() / d[s].chars.sum() for s in SPLITS}
    t = d["test"]; wrong = ~t.exact
    flagged = t.conf_min < args.flag_threshold
    lo, hi = boot_ci(t.exact)
    rare = t[t.target.map(train_freq).fillna(0) <= args.rare_max]
    preds[n] = t.exact.to_numpy()
    rows.append({
        "model": n, "params_M": round(meta.get("params_m", np.nan), 1),
        "train_acc": acc["train"], "val_acc": acc["val"], "test_acc": acc["test"], "test_ci_lo": lo, "test_ci_hi": hi,
        "train_cer": cer["train"], "val_cer": cer["val"], "test_cer": cer["test"],
        "gap_train_test": acc["train"] - acc["test"], "gap_val_test": acc["val"] - acc["test"],
                "test_acc_case_insens": t.exact_ci.mean(),
        "flagged_frac": flagged.mean(),
        "flag_precision": wrong[flagged].mean() if flagged.any() else np.nan,
        "error_recall_at_flag": flagged[wrong].mean() if wrong.any() else np.nan,
        "auroc_conf_min": auroc(-t.conf_min.to_numpy(), wrong.to_numpy()),
        "ece_conf_mean": ece(t.conf_mean, t.exact),
        "conf_correct": t.conf_mean[t.exact].mean(), "conf_wrong": t.conf_mean[wrong].mean() if wrong.any() else np.nan,
        "rare_label_acc": rare.exact.mean() if len(rare) else np.nan, "rare_n": len(rare),
        "imgs_per_sec": t.ips.iloc[0],
    })
df = pd.DataFrame(rows).sort_values("test_acc", ascending=False).reset_index(drop=True)

# paired comparison on the same test samples vs the top model (noise check)
ref = df.model.iloc[0]
df["vs_top_only_me_right"] = [int((preds[m] & ~preds[ref]).sum()) if m != ref else 0 for m in df.model]
df["vs_top_only_top_right"] = [int((~preds[m] & preds[ref]).sum()) if m != ref else 0 for m in df.model]
df.to_csv(path("reports_dir") / "model_validation.csv", index=False)

pc = lambda x: f"{x * 100:.1f}%"
lines = [f"# Model validation (test n={len(t)}, flag threshold {args.flag_threshold}, top model = {ref})\n",
         "## 1. Accuracy and overfitting (exact match, case-sensitive)\n",
         "| model | params | train | val | test [95% CI] | test CER | train-test gap | val-test gap | train CER | case-insens. test |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in df.itertuples():
    lines.append(f"| {r.model} | {r.params_M}M | {pc(r.train_acc)} | {pc(r.val_acc)} | {pc(r.test_acc)} [{pc(r.test_ci_lo)}-{pc(r.test_ci_hi)}] | "
                 f"{pc(r.test_cer)} | {r.gap_train_test * 100:+.1f} | {r.gap_val_test * 100:+.1f} | {pc(r.train_cer)} | {pc(r.test_acc_case_insens)} |")
lines += ["", "## 2. Confidence quality and robustness (test set)\n",
          "| model | flagged | flag precision | errors caught | AUROC | ECE | conf correct / wrong | rare-label acc (n) | vs top: only me right / only top right | img/s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for r in df.itertuples():
    lines.append(f"| {r.model} | {pc(r.flagged_frac)} | {pc(r.flag_precision) if r.flag_precision == r.flag_precision else '-'} | {pc(r.error_recall_at_flag)} | "
                 f"{r.auroc_conf_min:.2f} | {r.ece_conf_mean:.3f} | {r.conf_correct:.2f} / {r.conf_wrong:.2f} | {pc(r.rare_label_acc)} ({r.rare_n}) | "
                 f"{r.vs_top_only_me_right} / {r.vs_top_only_top_right} | {r.imgs_per_sec:.0f} |")
(path("reports_dir") / "model_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
