"""CRNN + CTC support for the experiment runner (strhub's CRNN: 7-layer CNN -> 2-layer BiLSTM -> CTC, 94-character charset + blank).

* crnn_command(overrides): the Hydra command for third_party/parseq/train.py with the project's training recipe (config.yaml `train:`), minus the keys that
  only PARSeq has (perm_num, dropout) and with CRNN's own default learning rate (5.1e-4) and pretrained weights `pretrained=crnn`
  (baudm/parseq release crnn-679d0e31.pt, 32 MB, cached in env.torch_home after the first download).
* ctc_confidence(probs, blank_id): per-crop confidence for a CTC model = the minimum over decoded characters of the best frame probability inside that
  character's run of identical frames (blank frames are ignored). The empty prediction falls back to the minimum frame confidence.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
from project_config import cfg, path  # noqa: E402

CRNN_LR = 5.1e-4


def crnn_command(overrides=(), pretrained=True, lr=CRNN_LR):
    t = cfg["train"]
    interval = t.get("system_metrics_interval")
    d = {
        "model": "crnn", "charset": t["charset"], "pretrained": "crnn" if pretrained else "null",
        "+seed": t["seed"], "+swa": "false", "+mlflow_experiment": t["mlflow_experiment"],
        "+mlflow_db": path("mlflow_db").as_posix(), "+system_metrics_interval": "null" if interval is None else interval,
        "data.root_dir": path("lmdb_dir").as_posix(), "data.train_dir": cfg["lmdb"]["dataset_name"], "data.num_workers": t["num_workers"],
        "+data.augment_magnitude": t["augment_magnitude"], "+data.augment_num_layers": t["augment_num_layers"],
        "model.batch_size": t["batch_size"], "model.lr": lr, "model.weight_decay": t["weight_decay"],
        "model.warmup_pct": t["warmup_pct"], "model.max_label_length": t["max_label_length"],
        "model.img_size": "[" + ",".join(str(v) for v in t["img_size"]) + "]",
        "trainer.max_epochs": t["max_epochs"], "trainer.devices": 1, "trainer.accelerator": "gpu",
        "trainer.gradient_clip_val": t["gradient_clip_val"], "trainer.val_check_interval": 1.0,
    }
    given = {o.split("=", 1)[0].lstrip("+") for o in overrides}
    return [sys.executable, str(ROOT / "third_party" / "parseq" / "train.py"),
            *[f"{k}={v}" for k, v in d.items() if k.lstrip("+") not in given], *overrides]


@torch.no_grad()
def ctc_confidence(probs, blank_id):
    """probs: (N, T, C) softmax. Returns (N,) confidences as defined in the module docstring."""
    mx, ids = probs.max(-1)
    out = []
    for p, i in zip(mx.cpu(), ids.cpu()):
        confs, prev, cur = [], -1, None
        for pj, ij in zip(p.tolist(), i.tolist()):
            if ij != blank_id:
                if ij == prev:
                    cur = max(cur, pj)
                else:
                    if cur is not None:
                        confs.append(cur)
                    cur = pj
            else:
                if cur is not None:
                    confs.append(cur)
                    cur = None
            prev = ij
        if cur is not None:
            confs.append(cur)
        out.append(min(confs) if confs else float(p.min()))
    return torch.tensor(out)
