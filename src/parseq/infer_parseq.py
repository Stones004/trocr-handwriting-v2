"""
Run a PARSeq model on a single image or a whole folder of images.

With --checkpoint, loads a fine-tuned Lightning .ckpt from
third_party/parseq/train.py. Without it, uses the pretrained (not fine-tuned)
scene-text weights from torch.hub -- useful for a quick sanity check, but not
what you want once you have a fine-tuned checkpoint.

Predictions below inference.flag_threshold (config.yaml) are marked for review, as are predictions that do not match
the label grammar (labels.clean_pattern).

Must be run with the venv_parseq interpreter (needs strhub importable when
--checkpoint is given):

    venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\\completed\\Doc0485_page_003_roi_1.png"
    venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\\completed" --output reports/predictions.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_grammar import matches_grammar  # noqa: E402 -- flags predictions that don't look like a valid label
from model_io import decode, get_device, load_model, make_transform  # noqa: E402
from project_config import cfg  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=None, help="path to a fine-tuned .ckpt; omit for the pretrained zero-shot model")
ap.add_argument("--input", required=True, help="a single image file, or a directory of images")
ap.add_argument("--output", default=None, help="optional CSV to write predictions to (filename, prediction, confidence)")
ap.add_argument("--flag-threshold", type=float, default=cfg["inference"]["flag_threshold"], help="mark for review if min char confidence is below this")
args = ap.parse_args()

input_path = Path(args.input)
if input_path.is_dir():
    image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No images found in {input_path} (looked for {sorted(IMAGE_EXTS)})")
else:
    image_paths = [input_path]

device = get_device()
model = load_model(args.checkpoint, device)
img_transform = make_transform(model)

rows = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    x = img_transform(image).unsqueeze(0).to(device)
    (prediction, min_conf, mean_conf), = decode(model, x)
    grammar_ok = matches_grammar(prediction)
    review = min_conf < args.flag_threshold
    rows.append({
        "filename": path.name, "prediction": prediction,
        "min_conf": min_conf, "mean_conf": mean_conf, "matches_grammar": grammar_ok, "needs_review": review or not grammar_ok,
    })
    flag = ("" if grammar_ok else "  [WARNING: does not match expected label grammar]") + ("  [REVIEW: low confidence]" if review else "")
    if len(image_paths) == 1:
        print("Prediction:", prediction)
        print(f"Confidence (min char prob): {min_conf:.3f}")
        print(f"Confidence (mean char prob): {mean_conf:.3f}")
        if review:
            print(f"REVIEW: confidence is below the review threshold ({args.flag_threshold})")
        if not grammar_ok:
            print("WARNING: prediction does not match the expected label grammar (digit run, Roman numeral, or subpart letter) -- treat with suspicion regardless of confidence")
    else:
        print(f"{path.name}: {prediction!r} (min_conf={min_conf:.3f}){flag}")

if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Predictions written to", args.output)
