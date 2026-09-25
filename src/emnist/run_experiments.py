"""Runs the EMNIST experiments end to end and writes reports/emnist_experiments.md.

Experiments (base PARSeq, parseq_handwriting_fixed.yaml -- only the training data differs):
  base_best        : the CURRENT PRODUCTION RECIPE (base PARSeq, parseq_handwriting_fixed.yaml, seed 42) retrained on the
                     same train split -- the reference every experiment has to beat. (models/parseq_final can't be
                     used: it was trained on all data including this test split.)
  base_bal_s{42,43,44}: base model + real train + 1,305 synthetic single-character crops that top up under-represented
                     classes to 100 each (vs base_best_s{43,44}: the current recipe on the other seeds)
  base_mix_v2      : that same base model trained on real train + 6,000 synthetic sequences (recipe v2: more varied trailing
                     marks and more `NNa` sequences; 8 epochs)
Every model is scored at its FINAL epoch (no checkpoint selection, so nothing is tuned on real data) on the
document-level split data/docsplit/seed42 (train/val/test; no document appears in two of them), plus on a synthetic
pure-`NNa` set (data/emnist/nna_eval, EMNIST *test* glyphs, 1,000 crops like `02a`) that probes the future label format.

Prerequisites (see README "EMNIST experiments"): src/emnist/extract_emnist.py, then the LMDBs under data/emnist_lmdb/.
Run from the project root with venv_parseq; resumable (finished trainings/evaluations are skipped):
    venv_parseq/Scripts/python src/emnist/run_experiments.py [--only base_best base_mix_v2]
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emnist_metrics import VOCAB, kind_of, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src" / "common"))
from project_config import path, subprocess_env, train_command  # noqa: E402

# Experiment definitions (which LMDB, epochs, recipe, seed per run). Folders for outputs/reports/docsplit/MLflow come
# from config.yaml (paths.*); the synthetic-data LMDB folders below are produced by src/emnist/synth_sequences.py.

EXPERIMENTS = {
    "base_best": dict(root="data/emnist_lmdb/mix", train_dir="handwriting", epochs=None, config="parseq_handwriting_fixed"),
    # seeds 43/44 of the current recipe, for a seed-noise-aware comparison against base_bal_s*
    "base_best_s43": dict(root="data/emnist_lmdb/mix", train_dir="handwriting", epochs=None, config="parseq_handwriting_fixed", seed=43),
    "base_best_s44": dict(root="data/emnist_lmdb/mix", train_dir="handwriting", epochs=None, config="parseq_handwriting_fixed", seed=44),
    # base model + real train + synthetic single-character crops topping up under-represented classes to 100
    # (1,305 synthetic crops, plan in data/emnist/balance_plan.csv), same recipe/epochs as base_best
    "base_bal_s42": dict(root="data/emnist_lmdb_bal", train_dir=".", epochs=None, config="parseq_handwriting_fixed", seed=42),
    "base_bal_s43": dict(root="data/emnist_lmdb_bal", train_dir=".", epochs=None, config="parseq_handwriting_fixed", seed=43),
    "base_bal_s44": dict(root="data/emnist_lmdb_bal", train_dir=".", epochs=None, config="parseq_handwriting_fixed", seed=44),
    # production-size model + real data + v2 synthetic data (8 epochs; base-model augmentation)
    "base_mix_v2": dict(root="data/emnist_lmdb_v2/mix", train_dir=".", epochs=8, config="parseq_handwriting_fixed"),
}
SPLIT_DIR = (path("docsplit_dir") / "seed42").as_posix()
SPLITS = {"train": f"{SPLIT_DIR}/train.csv", "val": f"{SPLIT_DIR}/val.csv", "test": f"{SPLIT_DIR}/test.csv",
          "nna": "data/emnist/nna_eval/test.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS),
                help="which runs to train/score; the report always covers every run that has results")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

env = subprocess_env()
py = sys.executable
out_dir = path("reports_dir") / "emnist"
out_dir.mkdir(parents=True, exist_ok=True)
(path("reports_dir") / "logs").mkdir(parents=True, exist_ok=True)


for name in args.only:
    e = EXPERIMENTS[name]
    run = (path("outputs_dir") / "parseq_emnist" / name).as_posix()
    final = f"{run}/checkpoints/final.ckpt"
    if not os.path.exists(final):
        print(f"[{name}] training", flush=True)
        with open(path("reports_dir") / "logs" / f"emnist_{name}_train.log", "w") as log:
            overrides = [f"data.root_dir={e['root']}", f"data.train_dir={e['train_dir']}",
                         *([f"trainer.max_epochs={e['epochs']}"] if e["epochs"] else []),
                         f"hydra.run.dir={run}", "mlflow_experiment=parseq_emnist", f"seed={e.get('seed', args.seed)}"]
            subprocess.run(train_command(e["config"], overrides), cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    for split, split_csv in SPLITS.items():
        csv = out_dir / f"{name}_{split}.csv"
        if not csv.exists():
            print(f"[{name}] scoring {split}", flush=True)
            subprocess.run([py, "src/parseq/eval_parseq.py", "--checkpoint", final, "--csv", split_csv,
                            "--output", str(csv)], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


lines = ["# EMNIST experiments (base PARSeq)", "",
         f"Scored on `{SPLIT_DIR}` (document-level split; no document is in two splits), final-epoch model, exact-match "
         "accuracy (case-sensitive) and CER. `in-vocab` = labels made only of `0-9a-j`, the vocabulary the synthetic data covers.", ""]
rows = []
for name in EXPERIMENTS:
    for split in SPLITS:
        csv = out_dir / f"{name}_{split}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, keep_default_na=False)
        inv = df[df["target"].astype(str).map(lambda s: set(s) <= VOCAB)]
        m, mi = metrics(df), metrics(inv)
        seen = "seen" if split == "train" else "unseen"
        if split == "nna":
            seen = "synthetic NNa"
        rows.append(f"| {name} | {split} ({seen}) | {m['n']} | {m['acc']*100:.1f}% | {m['cer']*100:.1f}% | {mi['n']} | {mi['acc']*100:.1f}% | {mi['cer']*100:.1f}% |")
lines += ["| experiment | split | n | acc | CER | in-vocab n | in-vocab acc | in-vocab CER |", "|---|---|---|---|---|---|---|---|", *rows, ""]

# Error breakdown on the test split by label length / kind.
for name in EXPERIMENTS:
    csv = out_dir / f"{name}_test.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv, keep_default_na=False)
    df["target"] = df["target"].astype(str)
    df["kind"] = df["target"].map(kind_of)
    g = df.groupby("kind").apply(lambda d: pd.Series({"n": len(d), "acc": (d["exact_match"].astype(str) == "True").mean()}), include_groups=False)
    lines += [f"### {name}: test accuracy by label kind", "", "| kind | n | acc |", "|---|---|---|",
              *[f"| {k} | {int(r.n)} | {r.acc*100:.1f}% |" for k, r in g.iterrows()], ""]
report = path("reports_dir") / "emnist_experiments.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("wrote", report)
