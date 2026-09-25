"""Hyperparameter search for PARSeq fine-tuning (Optuna TPE), one training run per trial.

Each trial launches third_party/parseq/train.py with the versioned experiment config plus the sampled
overrides, so every trial shows up in MLflow (experiment `parseq_sweep`) with system metrics, and its
best val_accuracy is the objective. Trial 0 is the current recipe (the baseline to beat). To save disk,
only the best trial's checkpoint is kept.

The search ranges, the recipe used for every trial and the sweep defaults are in configs/sweep.yaml; the trial-0
baseline is read from the recipe's own yaml (configs/parseq/experiment/), so it cannot drift from it.

Run from the project root with venv_parseq:
    python src/parseq/sweep.py --n-trials 15
    python src/parseq/sweep.py --space aggressive --n-trials 25          # much wider ranges
    python src/parseq/sweep.py --n-trials 1 --max-epochs 1 --study smoke   # quick end-to-end check
Resumable: re-running with the same --study continues where it stopped.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import ROOT, path, subprocess_env, train_command  # noqa: E402

SWEEP_CFG = yaml.safe_load(open(ROOT / "configs" / "sweep.yaml", encoding="utf-8"))
SWEEPS = path("outputs_dir") / "sweeps"


def suggest(trial, name, spec):
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"{name}: unknown parameter type {kind!r} in configs/sweep.yaml")


def recipe_baseline(params):
    """Trial-0 values: each parameter's value in the recipe yaml, unless the sweep config gives an explicit `baseline`."""
    recipe = yaml.safe_load(open(ROOT / "configs" / "parseq" / "experiment" / f"{SWEEP_CFG['recipe']}.yaml", encoding="utf-8"))
    base = {}
    for name, spec in params.items():
        if "baseline" in spec:
            base[name] = spec["baseline"]
            continue
        node = recipe
        for key in spec["override"].split("."):
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"{spec['override']} (baseline of {name!r}) is not set in the {SWEEP_CFG['recipe']} recipe; "
                               "set it there or give the parameter a `baseline:` in configs/sweep.yaml")
            node = node[key]
        base[name] = node
    return base


def best_checkpoint(ckpt_dir):
    """Highest val_accuracy checkpoint (score is encoded in the filename), ignoring last.ckpt."""
    scored = []
    for p in ckpt_dir.glob("epoch=*.ckpt"):
        m = re.search(r"val_accuracy=([\d.]+)-val_NED=([\d.]+)\.ckpt$", p.name)
        if m:
            scored.append((float(m[1]), float(m[2]), p))
    return max(scored) if scored else None


def main():
    d = SWEEP_CFG["defaults"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=list(SWEEP_CFG["spaces"]), default="standard")
    ap.add_argument("--study", default=None, help="default: the space's `study` name in configs/sweep.yaml")
    ap.add_argument("--n-trials", type=int, default=d["n_trials"])
    ap.add_argument("--max-epochs", type=int, default=d["max_epochs"])
    ap.add_argument("--patience", type=int, default=d["patience"], help="early-stopping patience per trial (epochs)")
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--retry-failed", action="store_true", help="re-run trials that failed earlier with the same params")
    ap.add_argument("--min-free-gb", type=float, default=d["min_free_gb"], help="stop before a trial if the system drive has less free space (checkpoint saves fail below this)")
    args = ap.parse_args()
    space = SWEEP_CFG["spaces"][args.space]
    params = space["params"]
    args.study = args.study or space["study"]

    SWEEPS.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{(SWEEPS / 'optuna.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    # Recover from an interrupted/crashed sweep: close stale RUNNING trials and re-queue failed ones with the same params.
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    if args.retry_failed:
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,)):
            if t.params:
                study.enqueue_trial(t.params)
                print(f"Re-queued failed trial {t.number}", flush=True)
    if not study.trials:
        study.enqueue_trial(recipe_baseline(params))

    env = subprocess_env()
    guard_path = (os.environ.get("SystemDrive", "C:") + "\\") if os.name == "nt" else "/"

    def objective(trial):
        free = shutil.disk_usage(guard_path).free / 2**30
        if free < args.min_free_gb:
            study.stop()
            raise RuntimeError(f"system drive has {free:.2f} GB free (< {args.min_free_gb}); Windows grows pagefile.sys on the system drive by ~2 GB during training; stopping. Free space and rerun with --retry-failed")
        hp = {name: suggest(trial, name, spec) for name, spec in params.items()}
        name = f"{args.study}_trial_{trial.number:03d}"
        run_dir = SWEEPS / name
        overrides = [f"hydra.run.dir={run_dir.as_posix()}", f"mlflow_experiment={space['mlflow_experiment']}",
                     f"trainer.max_epochs={args.max_epochs}", f"early_stopping.patience={args.patience}",
                     *[f"{params[k]['override']}={v}" for k, v in hp.items()]]
        cmd = train_command(SWEEP_CFG["recipe"], overrides)
        print(f"[{name}] {hp}", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        best = best_checkpoint(run_dir / "checkpoints")
        if best is None:
            raise RuntimeError(f"{name}: no checkpoint produced")
        acc, ned, ckpt = best
        trial.set_user_attr("val_ned", ned)
        trial.set_user_attr("checkpoint", str(ckpt))
        # Keep disk bounded: this trial's checkpoints survive only if it is the best so far.
        prev = [t.value for t in study.trials if t.value is not None and t.number != trial.number]
        for p in (run_dir / "checkpoints").glob("*.ckpt"):
            if p != ckpt or (prev and acc <= max(prev)):
                p.unlink()
        print(f"[{name}] val_accuracy={acc} val_NED={ned}", flush=True)
        return acc

    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    report = path("reports_dir") / f"sweep_{args.study}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe().sort_values("value", ascending=False)
    df.to_csv(report, index=False)
    print(f"\nBest trial #{study.best_trial.number}: val_accuracy={study.best_value}")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")
    print("Report:", report)


if __name__ == "__main__":
    main()
