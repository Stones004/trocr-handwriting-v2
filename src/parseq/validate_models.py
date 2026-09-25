"""Run PARSeq checkpoints listed in configs/validation_models.yaml over train/val/test and save per-sample
predictions + confidences to <reports_dir>/validation/<model>__<split>.csv (venv_parseq).
    python src/parseq/validate_models.py [--only NAME ...]
Then summarize with: python src/common/summarize_validation.py
Data comes from paths.processed_dir, the batch size from validation.batch_size (config.yaml).
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_io import decode, get_device, load_model, make_transform  # noqa: E402
from project_config import cfg, path  # noqa: E402
from validation_metrics import OUT_DIR, SPLITS, sample_row, save_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*")
ap.add_argument("--batch", type=int, default=cfg["validation"]["batch_size"])
args = ap.parse_args()

models = yaml.safe_load(open(ROOT / "configs/validation_models.yaml"))["parseq"]
device = get_device()
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = {s: pd.read_csv(path("processed_dir") / f"{s}.csv") for s in SPLITS}
cache = {}  # PIL images per split, decoded once and reused across models
for s, df in data.items():
    cache[s] = [Image.open(p).convert("RGB") for p in df["image_path"]]

for name, spec in models.items():
    if args.only and name not in args.only:
        continue
    # "pretrained" = the not-fine-tuned reference; anything else is a glob (relative to the project root) to a checkpoint
    ckpt = "pretrained=parseq" if spec == "pretrained" else glob.glob(str(ROOT / spec))[0]
    model = load_model(ckpt, device)
    tf = make_transform(model)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    for split in SPLITS:
        df, imgs, rows = data[split], cache[split], []
        t0 = time.time()
        for i in range(0, len(df), args.batch):
            x = torch.stack([tf(im) for im in imgs[i:i + args.batch]]).to(device)
            for j, (pred, conf_min, conf_mean) in enumerate(decode(model, x)):
                r = df.iloc[i + j]
                rows.append(sample_row(r["filename"], r["text"], pred, conf_min, conf_mean))
        if device.type == "cuda":
            torch.cuda.synchronize()
        save_split(name, split, rows, time.time() - t0)
    pd.Series({"params_m": params, "checkpoint": ckpt}).to_json(OUT_DIR / f"{name}__meta.json")
    print("done", name, f"({params:.1f}M params)", flush=True)
    del model
    torch.cuda.empty_cache()
