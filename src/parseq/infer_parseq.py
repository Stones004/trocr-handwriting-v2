"""
Run a PARSeq model on a single image or a whole folder of images.

With --checkpoint, loads a fine-tuned Lightning .ckpt from
third_party/parseq/train.py. Without it, uses the pretrained (not fine-tuned)
scene-text weights from torch.hub -- useful for a quick sanity check, but not
what you want once you have a fine-tuned checkpoint.

Must be run with the venv_parseq interpreter (needs strhub importable when
--checkpoint is given):

    venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\\completed\\Doc0485_page_003_roi_1.png"
    venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\\completed" --output predictions.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=None, help="path to a fine-tuned .ckpt; omit for the pretrained zero-shot model")
ap.add_argument("--input", required=True, help="a single image file, or a directory of images")
ap.add_argument("--output", default=None, help="optional CSV to write predictions to (filename, prediction, confidence)")
args = ap.parse_args()

input_path = Path(args.input)
if input_path.is_dir():
    image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No images found in {input_path} (looked for {sorted(IMAGE_EXTS)})")
else:
    image_paths = [input_path]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.checkpoint:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "third_party" / "parseq"))
    from strhub.models.utils import load_from_checkpoint
    model = load_from_checkpoint(args.checkpoint).eval().to(device)
else:
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(device)

img_transform = T.Compose([
    T.Resize(model.hparams.img_size, T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])

rows = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    x = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(x).softmax(-1)
    pred_list, prob_list = model.tokenizer.decode(p)
    prediction = pred_list[0].strip()
    probs = prob_list[0]
    min_conf = probs.min().item() if len(probs) else 0.0
    mean_conf = probs.mean().item() if len(probs) else 0.0
    rows.append({
        "filename": path.name, "prediction": prediction,
        "min_conf": min_conf, "mean_conf": mean_conf,
    })
    if len(image_paths) == 1:
        print("Prediction:", prediction)
        print(f"Confidence (min char prob): {min_conf:.3f}")
        print(f"Confidence (mean char prob): {mean_conf:.3f}")
    else:
        print(f"{path.name}: {prediction!r} (min_conf={min_conf:.3f})")

if args.output:
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Predictions written to", args.output)
