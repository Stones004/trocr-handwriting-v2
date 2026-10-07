"""Resumable runner for the October experiments: train one PARSeq run, then evaluate it with eval_suite and record everything.

    venv_parseq/Scripts/python src/exp_oct/run_queue.py baseline_s42 --seed 42 [--lmdb DIR] [--epochs N] [-- hydra overrides ...]
    venv_parseq/Scripts/python src/exp_oct/run_queue.py --queue queue_phase0.txt      (one run per line: same arguments)

Each run lives in outputs/exp_oct/<name>/ and ends by writing done.json (config, seed, steps, wall time, ckpt path, git commit, metrics).
A run with done.json is skipped, so an interrupted queue is simply started again. Progress goes to outputs/exp_oct/progress.log.
"""
import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
from progress import log  # noqa: E402
from project_config import cfg, subprocess_env, train_command  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct"
DEFAULT_LMDB = OUT / "data" / "master_lmdb"


def git_commit():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "patches"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+uncommitted" if dirty else "")


def run_one(name, seed=42, lmdb=DEFAULT_LMDB, epochs=None, overrides=(), est_min=25, note="", resize_mode="stretch", arch="parseq"):
    run_dir = OUT / name
    if (run_dir / "done.json").exists():
        log(f"SKIP {name} (done.json exists)")
        return json.loads((run_dir / "done.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    ov = [f"+seed={seed}", f"data.root_dir={Path(lmdb).as_posix()}", f"hydra.run.dir={run_dir.as_posix()}", "+mlflow_experiment=exp_oct",
          *([f"trainer.max_epochs={epochs}"] if epochs else []), *([f"+data.resize_mode={resize_mode}"] if resize_mode != "stretch" else []), *overrides]
    if arch == "crnn":
        from crnn_support import crnn_command
        cmd = crnn_command(ov)
    else:
        cmd = train_command(ov)
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(c) for c in cmd), encoding="utf-8")
    t0 = time.time()
    ck = run_dir / "checkpoints" / "final.ckpt"
    if ck.exists():
        log(f"RESUME {name}: final.ckpt already exists, skipping training and evaluating")
        p = None
    else:
        log(f"START {name}: seed {seed}, lmdb {Path(lmdb).name}, epochs {epochs or cfg['train']['max_epochs']}, overrides {list(overrides)} (est ~{est_min} min)")
        with open(run_dir / "train.log", "wb") as lf:
            p = subprocess.Popen(cmd, cwd=ROOT, env=subprocess_env(), stdout=lf, stderr=subprocess.STDOUT)
            last = ""
            while p.poll() is None:
                time.sleep(60)
                try:
                    tail = (run_dir / "train.log").read_bytes()[-6000:].decode("utf-8", "ignore")
                except OSError:
                    continue
                m = re.findall("Epoch ([0-9]+): +([0-9]+)%[|][^" + chr(10) + chr(13) + "]*?[|] *([0-9]+)/([0-9]+)", tail)
                if m:
                    e, _, s_, n = m[-1]
                    msg = f"epoch {e}/{(epochs or cfg['train']['max_epochs']) - 1}, step {s_}/{n}"
                    if msg != last:
                        log(f"  {name}: {msg}, {(time.time() - t0) / 60:.1f} min elapsed")
                        last = msg
    wall = time.time() - t0
    if (p is not None and p.returncode != 0) or not ck.exists():
        log(f"FAILED {name}: exit {p.returncode if p else None}, final.ckpt exists={ck.exists()}. See {run_dir / 'train.log'}")
        raise SystemExit(1)
    import torch
    from eval_suite import evaluate_model
    from model_io import get_device, load_model
    steps = int(torch.load(ck, map_location="cpu", weights_only=False).get("global_step", -1))
    dev = get_device()
    alias = ck.with_name(f"{'crnn' if arch == 'crnn' else 'parseq'}_{name}.ckpt")      # PARSeq infers the model class from the path, which must contain 'parseq'
    if not alias.exists():
        shutil.copy(ck, alias)
    res = evaluate_model(load_model(alias, dev), name, run_dir / "eval", dev, resize_mode=resize_mode)
    done = {"name": name, "seed": seed, "epochs": epochs or cfg["train"]["max_epochs"], "overrides": list(overrides), "lmdb": str(lmdb),
            "steps": steps, "wall_minutes": round(wall / 60, 1), "checkpoint": str(ck.relative_to(ROOT)), "git_commit": git_commit(),
            "note": note, "resize_mode": resize_mode, "arch": arch, "metrics": res}
    (run_dir / "done.json").write_text(json.dumps(done, indent=1, default=float), encoding="utf-8")
    log(f"DONE {name}: {steps} steps in {wall / 60:.1f} min")
    return done


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--resize-mode", default="stretch", choices=["stretch", "pad"])
    ap.add_argument("--arch", default="parseq", choices=["parseq", "crnn"])
    ap.add_argument("--queue", default=None)
    ap.add_argument("overrides", nargs="*")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse(sys.argv[1:])
    if a.queue:
        for line in Path(a.queue).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                q = parse(shlex.split(line))
                run_one(q.name, q.seed, q.lmdb, q.epochs, q.overrides, note=q.note, resize_mode=q.resize_mode, arch=q.arch)
    else:
        run_one(a.name, a.seed, a.lmdb, a.epochs, a.overrides, note=a.note, resize_mode=a.resize_mode, arch=a.arch)
