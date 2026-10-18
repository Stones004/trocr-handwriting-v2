"""Minimal noise/clean classification for a single ROI crop, or a whole folder of them.

    python src/noise_classifier/predict.py "D:/path/to/roi.png"
    python src/noise_classifier/predict.py "D:/path/to/folder" [--out predictions.csv]

Also importable directly:
    from predict import predict
    label, confidence = predict("roi.png")   # label is "clean" or "noise"; loads the model once, reused after
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_io import CLASSES, load_model, make_transform  # noqa: E402
from project_config import path  # noqa: E402

DEFAULT_CHECKPOINT = path("models_dir") / "noise_classifier" / "best.pt"

_model = None
_transform = None


def _load(checkpoint):
    global _model, _transform
    if _model is None:
        _model, img_size = load_model(checkpoint)
        _transform = make_transform(img_size, train=False)
    return _model, _transform


def predict(image_path, checkpoint=DEFAULT_CHECKPOINT):
    """Returns (label: "clean"|"noise", confidence: float) for one image."""
    model, transform = _load(checkpoint)
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file or folder of images")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default=None, help="write filename,label,confidence to this CSV (folder mode)")
    args = ap.parse_args()

    target = Path(args.path)
    if target.is_dir():
        import csv
        files = sorted(target.glob("*.png"))
        rows = []
        for f in files:
            label, confidence = predict(f, args.checkpoint)
            rows.append((f.name, label, confidence))
            print(f"{f.name}\t{label}\t{confidence:.3f}")
        if args.out:
            with open(args.out, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["filename", "label", "confidence"])
                w.writerows(rows)
            print(f"wrote {len(rows)} rows to {args.out}")
    else:
        label, confidence = predict(target, args.checkpoint)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.3f}")
