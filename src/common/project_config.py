"""Single loader for the project configuration (config.yaml + config.local.yaml + optional PROJECT_CONFIG file).

Later files override earlier ones, key by key (nested sections are merged, not replaced):
    config.yaml            committed portable defaults
    config.local.yaml      machine-specific overrides, gitignored
    $PROJECT_CONFIG        optional third file, e.g. to run a one-off variant

Usage from any script (pure stdlib + PyYAML, works in both venvs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    from project_config import ROOT, cfg, path, subprocess_env
    processed = path("processed_dir")          # absolute Path built from cfg["paths"]["processed_dir"]

Relative paths in the config are resolved against the project root, so scripts behave the same from any directory.
"""
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(extra=None):
    """Merged config dict. `extra` is an optional additional yaml file layered on top of everything else."""
    files = [(ROOT / "config.yaml", True), (ROOT / "config.local.yaml", False)]  # (file, required)
    if os.environ.get("PROJECT_CONFIG"):
        files.append((Path(os.environ["PROJECT_CONFIG"]), True))
    if extra:
        files.append((Path(extra), True))
    out = {}
    for f, required in files:
        if not f.exists():
            if required:
                raise FileNotFoundError(f"Config file not found: {f}")
            continue
        with open(f, encoding="utf-8") as fh:
            _merge(out, yaml.safe_load(fh) or {})
    return out


cfg = load_config()


def resolve(p):
    """Absolute Path for a config value; relative paths are taken relative to the project root."""
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else ROOT / p


def path(key):
    """Absolute Path of cfg['paths'][key]."""
    try:
        return resolve(cfg["paths"][key])
    except KeyError:
        raise KeyError(f"paths.{key} is not set in config.yaml / config.local.yaml") from None


def subprocess_env():
    """os.environ plus the temp/cache locations from cfg['env'] (creating them), for launching training subprocesses."""
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    e = cfg.get("env") or {}
    if e.get("tmp_dir"):
        tmp = resolve(e["tmp_dir"])
        tmp.mkdir(parents=True, exist_ok=True)
        env["TMP"] = env["TEMP"] = str(tmp)
    if e.get("torch_home"):
        th = resolve(e["torch_home"])
        th.mkdir(parents=True, exist_ok=True)
        env["TORCH_HOME"] = str(th)
    return env


def label_regex(name):
    """Compiled regex from cfg['labels'][name] (clean_pattern, excel_artifact_pattern, doc_id_pattern)."""
    return re.compile(cfg["labels"][name])


def train_command(experiment, overrides=()):
    """argv for one PARSeq training run: third_party/parseq/train.py with the given Hydra experiment yaml plus overrides.

    Project defaults are filled in from config.yaml unless the caller overrides them: data.root_dir (paths.lmdb_dir),
    data.train_dir (lmdb.dataset_name) and mlflow_db (paths.mlflow_db). Run it with cwd=ROOT and env=subprocess_env().
    """
    given = {o.split("=", 1)[0].lstrip("+") for o in overrides}
    defaults = {
        "data.root_dir": path("lmdb_dir").as_posix(),
        "data.train_dir": cfg["lmdb"]["dataset_name"],
        "mlflow_db": path("mlflow_db").as_posix(),
    }
    return [sys.executable, str(ROOT / "third_party" / "parseq" / "train.py"), f"--config-dir={ROOT / 'configs' / 'parseq'}",
            f"+experiment={experiment}", *[f"{k}={v}" for k, v in defaults.items() if k not in given], *overrides]
