"""Read question-number crops with the in-vocabulary model (models/parseq_iv_best) and grammar-constrained decoding.

    venv_parseq/Scripts/python src/exp_oct/predict_iv.py --input "D:\my_crops"                    # a folder of images
    venv_parseq/Scripts/python src/exp_oct/predict_iv.py --input crop.png                         # one image
    venv_parseq/Scripts/python src/exp_oct/predict_iv.py --input "D:\my_crops" --labels labels.csv   # score it: csv with columns filename,text
Options: --mode grammar|charset|plain (default grammar), --output preds.csv (default outputs/exp_oct/iv_predictions.csv), --flag 0.5.

Vocabulary: digits 0-9 and letters a-j; label shape = 1-2 digits + optional letter, or a single letter. Marks (. ( ) [ ]) are not part of the label.
Confidence = probability of the chosen characters after masking (min over characters); rows below --flag are marked REVIEW.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
import constrained  # noqa: E402
from model_io import get_device, load_model, make_transform  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--checkpoint", default=str(ROOT / "models" / "parseq_iv_best" / "parseq_iv_best.ckpt"))
ap.add_argument("--mode", default="grammar", choices=["grammar", "charset", "plain"])
ap.add_argument("--labels", default=None)
ap.add_argument("--output", default=str(ROOT / "outputs" / "exp_oct" / "iv_predictions.csv"))
ap.add_argument("--flag", type=float, default=0.5)
a = ap.parse_args()

inp = Path(a.input)
files = [inp] if inp.is_file() else sorted(f for f in inp.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
dev = get_device()
m = load_model(a.checkpoint, dev).eval()
tf = make_transform(m)
rows = []
for i in range(0, len(files), 64):
    batch = files[i:i + 64]
    x = torch.stack([tf(Image.open(f).convert("RGB")) for f in batch]).to(dev)
    for f, (pred, mn, mean) in zip(batch, constrained.decode(m, x, a.mode)):
        rows.append({"filename": f.name, "prediction": pred, "confidence": round(mn, 4), "review": mn < a.flag or pred == ""})
df = pd.DataFrame(rows)
if a.labels:
    lab = pd.read_csv(a.labels, keep_default_na=False, dtype=str)[["filename", "text"]].rename(columns={"text": "truth"})
    df = df.merge(lab, on="filename", how="left")
    df["correct"] = df["prediction"] == df["truth"]
    k = df["truth"].notna() & (df["truth"] != "")
    print(f"scored {int(k.sum())} labelled crops: {df.loc[k, 'correct'].mean() * 100:.1f}% exact match ({int(df.loc[k, 'correct'].sum())}/{int(k.sum())})")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.output, index=False)
print(df.head(15).to_string(index=False))
print(f"\n{len(df)} crops, {int(df.review.sum())} marked REVIEW -> {a.output}")
