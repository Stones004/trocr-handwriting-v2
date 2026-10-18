"""Trains the open-reading model (data/open_read_lmdb) for several seeds, then scores everything with evaluate_open_read.py.
Recipe = config.yaml `train:` (pretrained start, 15 epochs, lr 1.5e-3, stretch resize); only the data and `charset_test` differ:
charset_test = charset_train so validation is case- and punctuation-sensitive, like the labels. Resumable.
    venv_parseq/Scripts/python src/open_read/train_open_read.py [--seeds 42 43 44]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src" / "common"))
from project_config import path, subprocess_env, train_command  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--lmdb", default="data/open_read_lmdb", help="data.root_dir")
ap.add_argument("--tag", default="openread", help="run name prefix: outputs/parseq_<tag>/s<seed>, models named <tag>_s<seed>")
args = ap.parse_args()
env = subprocess_env()
logs = path("reports_dir") / "open_read"
logs.mkdir(parents=True, exist_ok=True)
ckpts = []
for seed in args.seeds:
    run = (path("outputs_dir") / f"parseq_{args.tag}" / f"s{seed}").as_posix()
    final = f"{run}/checkpoints/final.ckpt"
    if not os.path.exists(final):
        print(f"[s{seed}] training", flush=True)
        ov = [f"data.root_dir={args.lmdb}", "data.train_dir=handwriting", f"hydra.run.dir={run}", f"mlflow_experiment=parseq_{args.tag}",
              f"seed={seed}", "model.charset_test=${model.charset_train}"]
        with open(logs / f"{args.tag}_s{seed}_train.log", "w") as log:
            r = subprocess.run(train_command(ov), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if not os.path.exists(final):
            sys.exit(f"[s{seed}] training failed (exit {r.returncode}); see {logs / f'{args.tag}_s{seed}_train.log'}")
    ckpts.append(f"{args.tag}_s{seed}={final}")
base = ["pretrained=pretrained", "base_best=models/parseq_base_best/parseq_base_best.ckpt", "final=models/parseq_final/parseq_final.ckpt",
        "master_v3=models/parseq_master_v3/parseq_master_v3.ckpt"]
subprocess.run([sys.executable, "src/open_read/evaluate_open_read.py", f"--report={args.tag}", *base, "openread_v1=models/parseq_openread_v1/parseq_openread_v1.ckpt", "openread_v2=models/parseq_openread_v2/parseq_openread_v2.ckpt", *ckpts], cwd=ROOT, env=env, check=True)
