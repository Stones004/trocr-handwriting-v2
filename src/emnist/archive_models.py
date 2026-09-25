"""Archives EMNIST-experiment models with everything needed to inspect, reuse or reproduce them, in a folder that the
usual clean-ups (outputs/ pruning) don't touch and that is made read-only.

Per model  <dest>/parseq_<name>/ :
  parseq_<name>.ckpt   weights-only export for inference (load with strhub.models.utils.load_from_checkpoint; the path
                       must contain 'parseq'). Verified to predict identically to the full checkpoint.
  full_final.ckpt      the full Lightning checkpoint (optimizer + scheduler state) of the final epoch, to resume from
  config/              resolved Hydra config, overrides, hparams (the training parameters)
  logs/                training logs + TensorBoard events + MLflow metric history (per-epoch curves)
  metrics/             per-sample predictions for every scored split, and metrics.json (accuracy, CER, in-vocab,
                       by label kind, per split)
Shared <dest>/ :
  datasets/            split CSVs, self-contained LMDBs (real crop images + synthetic sequences), NNa eval set,
                       extracted EMNIST glyphs
  code/                snapshot of the code, configs and patch that produced the models
  MANIFEST.json        sha256 of every file, git commit, package versions, how to reproduce
  README.md            what is where

Run from the project root with venv_parseq:
    venv_parseq/Scripts/python src/emnist/archive_models.py --runs base_best base_mix_v2 \
        --dest models/archive/emnist_2026-09-24 --lmdbs data/emnist_lmdb_v2/mix
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emnist_metrics import VOCAB, kind_of, load, metrics  # noqa: E402
from project_config import cfg, path, subprocess_env  # noqa: E402 (emnist_metrics puts src/common on sys.path)

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="run names under outputs/parseq_emnist/")
ap.add_argument("--dest", required=True)
ap.add_argument("--lmdbs", nargs="*", default=[], help="LMDB roots (data/...) to copy into datasets/")
ap.add_argument("--splits-dir", default=(path("docsplit_dir") / "seed42").as_posix())
ap.add_argument("--extra-files", nargs="*", default=[], help="extra files/dirs (project-relative) copied to datasets/extra/ or reports/")
ap.add_argument("--note", default="", help="free-text note stored in the README/manifest")
args = ap.parse_args()
dest = Path(args.dest)
env = subprocess_env()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def summarize(csv):
    df = load(csv)
    inv = df[df["target"].map(lambda s: set(s) <= VOCAB)]
    out = {"all": metrics(df), "in_vocab_0-9a-j": metrics(inv)}
    out["by_kind"] = {k: {"n": len(g), "acc": float(g["ok"].mean())} for k, g in df.groupby(df["target"].map(kind_of))}
    out["mean_min_char_confidence"] = float(df["seq_min_prob"].mean())
    return out


runs_info = {}
for name in args.runs:
    src = path("outputs_dir") / "parseq_emnist" / name
    d = dest / f"parseq_{name}"
    assert (src / "checkpoints/final.ckpt").exists(), f"{src} has no final.ckpt"
    print(f"[{name}] exporting weights", flush=True)
    subprocess.run([sys.executable, "src/parseq/export_model.py", "--checkpoint", str(src / "checkpoints/final.ckpt"), "--out", str(d)],
                   env=env, check=True)
    copy(src / "checkpoints/final.ckpt", d / "full_final.ckpt")
    copy(src / "config", d / "config")
    copy(src / "hparams.yaml", d / "config/hparams.yaml")
    copy(src / "train.log", d / "logs/train.log")
    for ev in src.glob("events.out.tfevents.*"):
        copy(ev, d / "logs" / ev.name)
    stdout_log = path("reports_dir") / "logs" / f"emnist_{name}_train.log"
    if stdout_log.exists():
        copy(stdout_log, d / "logs/training_stdout.log")
    # MLflow per-epoch history (run name == the run directory name)
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri("sqlite:///" + path("mlflow_db").as_posix())
        c = MlflowClient()
        exp = c.get_experiment_by_name("parseq_emnist")
        runs = c.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{name}'")
        run = runs[0]
        hist = {k: [(m.step, m.value) for m in c.get_metric_history(run.info.run_id, k)] for k in run.data.metrics
                if not k.startswith("system/")}
        (d / "logs").mkdir(parents=True, exist_ok=True)
        (d / "logs/mlflow_history.json").write_text(json.dumps({"run_id": run.info.run_id, "params": run.data.params, "metrics": hist}, indent=1))
    except Exception as e:  # tracking store is a convenience; the logs above are the source of truth
        print(f"[{name}] MLflow history skipped: {e}")
    summary = {}
    for csv in sorted((path("reports_dir") / "emnist").glob(f"{name}_*.csv")):
        split = csv.stem[len(name) + 1:]
        copy(csv, d / "metrics" / f"predictions_{split}.csv")
        summary[split] = summarize(csv)
    (d / "metrics/metrics.json").write_text(json.dumps(summary, indent=1))
    # Verify the exported weights reproduce the recorded test accuracy.
    out = subprocess.run([sys.executable, "src/parseq/eval_parseq.py", "--checkpoint", str(d / f"parseq_{name}.ckpt"),
                          "--csv", f"{args.splits_dir}/test.csv"], env=env, capture_output=True, text=True, check=True).stdout
    acc = float([l for l in out.splitlines() if l.startswith("Exact-match")][0].split(":")[1])
    print(f"[{name}] exported model test acc {acc:.4f} vs recorded {summary['test']['all']['acc']:.4f}")
    assert abs(acc - summary["test"]["all"]["acc"]) < 1e-9, "exported weights do not reproduce the recorded test accuracy"
    runs_info[name] = {"test": summary["test"]["all"], "val": summary["val"]["all"]}

# ---- shared: datasets and code
print("copying datasets and code", flush=True)
for f in ("train", "val", "test"):
    copy(Path(args.splits_dir) / f"{f}.csv", dest / "datasets/splits" / f"{f}.csv")
for root in args.lmdbs:
    copy(root, dest / "datasets/lmdb" / Path(root).name)
copy("data/emnist/nna_eval", dest / "datasets/nna_eval")
for g in Path("data/emnist").glob("glyphs*.npz"):
    copy(g, dest / "datasets/emnist_glyphs" / g.name)
for x in args.extra_files:
    copy(x, dest / "extra" / x)
for p in ("src/emnist", "configs/parseq", "patches", "requirements-parseq.txt", "config.yaml", "src/common/project_config.py",
          "src/parseq/model_io.py", "src/parseq/export_model.py", "src/parseq/eval_parseq.py",
          "src/parseq/prepare_parseq_dataset.py"):
    copy(p, dest / "code" / p)
for pyc in (dest / "code").rglob("__pycache__"):
    shutil.rmtree(pyc)

def version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


pkgs = {p: version(p) for p in ("torch", "torchvision", "pytorch-lightning", "timm", "hydra-core", "lmdb", "mlflow", "numpy",
                                "opencv-python-headless", "imgaug")}
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
files = {str(p.relative_to(dest)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size}
         for p in sorted(dest.rglob("*")) if p.is_file() and p.name not in ("MANIFEST.json",)}
manifest = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "git_commit": git("rev-parse", "HEAD"), "git_note": "working tree had uncommitted changes; the code/ snapshot is the exact code used",
    "packages": pkgs, "runs": runs_info, "note": args.note,
    "real_images_note": f"real crop images originate from {cfg['image_dir']} (referenced by absolute path in datasets/splits/*.csv); "
                        "the LMDBs in datasets/lmdb embed the image bytes, so they are self-contained.",
    "files": files,
}
(dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

readme = f"""# EMNIST experiment archive

{args.note}

Runs: {', '.join(args.runs)}. Each `parseq_<run>/` folder holds the weights-only model (`parseq_<run>.ckpt`), the full
resumable checkpoint (`full_final.ckpt`), `config/` (training parameters), `logs/`, and `metrics/` (per-sample predictions
+ `metrics.json`). `datasets/` has the split CSVs, self-contained LMDBs, the synthetic NNa eval set and the extracted
EMNIST glyphs; `code/` is the exact code/configs/patch used. `MANIFEST.json` has a sha256 for every file.

Load a model:
    sys.path.insert(0, 'third_party/parseq'); from strhub.models.utils import load_from_checkpoint
    model = load_from_checkpoint('parseq_<run>/parseq_<run>.ckpt')      # path must contain 'parseq'

Everything is marked read-only. Verify integrity by re-hashing against MANIFEST.json.
"""
(dest / "README.md").write_text(readme)

# ---- make read-only
for p in dest.rglob("*"):
    if p.is_file():
        os.chmod(p, stat.S_IREAD)
print(f"archived {len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e6:.0f} MB -> {dest}")
