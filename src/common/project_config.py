"""Loader for the project configuration: the single file config.yaml.

Usage from any script (pure stdlib + PyYAML, works in both venvs):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    from project_config import ROOT, cfg, path, subprocess_env, train_command
    outputs = path("outputs_dir")              # absolute Path built from cfg["paths"]["outputs_dir"]

Relative paths in the config are resolved against the project root, so scripts behave the same from any directory.
"""
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


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


def train_command(overrides=()):
    """argv for one PARSeq training run: third_party/parseq/train.py with the recipe in config.yaml's `train:` section plus overrides.

    Anything in `overrides` (Hydra syntax, e.g. "seed=7", "data.root_dir=data/other_lmdb") wins over the config. Run with cwd=ROOT and
    env=subprocess_env(). Hydra needs `+key` for keys PARSeq's own defaults do not define; the patch in patches/parseq_train.patch reads them.
    """
    t = cfg["train"]
    interval = t.get("system_metrics_interval")
    defaults = {
        "model": "parseq", "charset": t["charset"], "pretrained": t["pretrained"],
        "+seed": t["seed"], "+swa": str(t["swa"]).lower(), "+mlflow_experiment": t["mlflow_experiment"],
        "+mlflow_db": path("mlflow_db").as_posix(), "+system_metrics_interval": "null" if interval is None else interval,
        "data.root_dir": path("lmdb_dir").as_posix(), "data.train_dir": cfg["lmdb"]["dataset_name"], "data.num_workers": t["num_workers"],
        "+data.augment_magnitude": t["augment_magnitude"], "+data.augment_num_layers": t["augment_num_layers"],
        "model.batch_size": t["batch_size"], "model.lr": t["lr"], "model.dropout": t["dropout"], "model.weight_decay": t["weight_decay"],
        "model.warmup_pct": t["warmup_pct"], "model.perm_num": t["perm_num"], "model.max_label_length": t["max_label_length"],
        "model.img_size": "[" + ",".join(str(v) for v in t["img_size"]) + "]",
        "trainer.max_epochs": t["max_epochs"], "trainer.devices": 1, "trainer.accelerator": "gpu",
        "trainer.gradient_clip_val": t["gradient_clip_val"], "trainer.val_check_interval": 1.0,
    }
    given = {o.split("=", 1)[0].lstrip("+") for o in overrides}
    return [sys.executable, str(ROOT / "third_party" / "parseq" / "train.py"),
            *[f"{k}={v}" for k, v in defaults.items() if k.lstrip("+") not in given], *overrides]
