"""Document-level cross-validation for PARSeq: for each seed, build a document-level split, train with the versioned
config, and score the best checkpoint on that split's held-out (unseen-document) test set. Reports mean +- std.
Run from the project root with venv_parseq:
    python src/parseq/docsplit_cv.py --seeds 42 43 44 45 46
Seeds default to docsplit.cv_seeds in config.yaml. Per-fold outputs (folders from config.yaml paths.*):
<docsplit_dir>/seed<N>/, <docsplit_lmdb_dir>/seed<N>/, <outputs_dir>/parseq_docsplit/seed<N>/ (best checkpoint only; the
path must contain 'parseq' for PARSeq's checkpoint loader).
Results: <reports_dir>/docsplit_cv.csv. Tracked in MLflow experiment `parseq_docsplit`. Resumable (finished folds are skipped).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import ROOT, cfg, path, subprocess_env, train_command  # noqa: E402

os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=cfg["docsplit"]["cv_seeds"])
ap.add_argument("--experiment", default="parseq_handwriting", help="experiment config name under configs/parseq/experiment")
ap.add_argument("--tag", default="", help="suffix for output folders/results, e.g. 'fixed'")
ap.add_argument("--use-final", action="store_true", help="score the final-epoch model (checkpoints/final.ckpt) instead of the best-val checkpoint")
args = ap.parse_args()
env = subprocess_env()
py = sys.executable
quiet = dict(env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sfx = f"_{args.tag}" if args.tag else ""
res_path = path("reports_dir") / f"docsplit_cv{sfx}.csv"
res_path.parent.mkdir(parents=True, exist_ok=True)
rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
done = {r["seed"] for r in rows}


def score(ckpt, split_dir, split):
    out = subprocess.run([py, str(ROOT / "src/parseq/eval_parseq.py"), "--checkpoint", ckpt, "--csv", f"{split_dir}/{split}.csv"],
                         env=env, capture_output=True, text=True, check=True).stdout
    return float(re.search(r"Exact-match accuracy: ([\d.]+)", out)[1]), float(re.search(r"CER: ([\d.]+)", out)[1])


for seed in args.seeds:
    if seed in done:
        continue
    sd = (path("docsplit_dir") / f"seed{seed}").as_posix()
    lm = (path("docsplit_lmdb_dir") / f"seed{seed}").as_posix()
    run = (path("outputs_dir") / f"parseq_docsplit{sfx}" / f"seed{seed}").as_posix()
    if not glob.glob(f"{run}/checkpoints/epoch=*.ckpt"):  # skip training if this fold already trained
        print(f"[seed {seed}] split + lmdb + train", flush=True)
        subprocess.run([py, str(ROOT / "src/common/prepare_docsplit.py"), "--seed", str(seed), "--out", sd], **quiet)
        subprocess.run([py, str(ROOT / "src/parseq/prepare_parseq_dataset.py"), "--processed-dir", sd, "--output-dir", lm], **quiet)
        subprocess.run(train_command(args.experiment, [f"data.root_dir={lm}", f"hydra.run.dir={run}",
                                                        f"mlflow_experiment=parseq_docsplit{sfx}", f"seed={seed}"]), **quiet)
    ckpts = glob.glob(f"{run}/checkpoints/epoch=*.ckpt")
    best = f"{run}/checkpoints/final.ckpt" if args.use_final else max(ckpts, key=lambda p: float(re.search(r"val_accuracy=([\d.]+)", p)[1]))
    row = {"seed": seed, "checkpoint": best}
    for split in ("train", "val", "test"):
        row[f"{split}_acc"], row[f"{split}_cer"] = score(best, sd, split)
    for p in ckpts + [f"{run}/checkpoints/last.ckpt", f"{run}/checkpoints/final.ckpt"]:  # keep only the scored checkpoint per fold
        if p != best and os.path.exists(p):
            os.remove(p)
    rows.append(row)
    pd.DataFrame(rows).to_csv(res_path, index=False)
    print(f"[seed {seed}] train {row['train_acc']:.3f} val {row['val_acc']:.3f} test {row['test_acc']:.3f} (CER {row['test_cer']:.3f})", flush=True)

df = pd.DataFrame(rows)
print("\n=== document-level CV (unseen documents) ===")
for k in ("train", "val", "test"):
    print(f"{k:5s} acc {df[f'{k}_acc'].mean()*100:.1f}% +- {df[f'{k}_acc'].std()*100:.1f}   CER {df[f'{k}_cer'].mean()*100:.1f}%")
