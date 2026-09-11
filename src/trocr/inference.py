"""
Run the fine-tuned TrOCR model on a single image or a whole folder of images.

Usage:
    python src/trocr/inference.py --model checkpoints/best --input "D:\\completed\\Doc0485_page_003_roi_1.png"
    python src/trocr/inference.py --model checkpoints/best --input "D:\\completed" --output predictions.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from constraints import get_closed_vocab_processor, get_logits_processor_from_csvs, transition_confidence
from dataset import resize_with_padding, get_target_size

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="checkpoints/best", help="checkpoint directory (e.g. checkpoints/best)")
ap.add_argument("--input", required=True, help="a single image file, or a directory of images")
ap.add_argument("--output", default=None, help="optional CSV to write predictions to (filename, prediction, confidence)")
ap.add_argument("--num-beams", type=int, default=4)
ap.add_argument("--no-constrain", action="store_true", help="disable constrained decoding entirely")
ap.add_argument("--charmask", action="store_true", help="use the old flat label-token mask instead of the closed-vocabulary trie")
args = ap.parse_args()

input_path = Path(args.input)
if input_path.is_dir():
    image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No images found in {input_path} (looked for {sorted(IMAGE_EXTS)})")
else:
    image_paths = [input_path]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained(args.model)
model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
target_size = get_target_size(processor)

if args.no_constrain:
    logits_processor = None
elif args.charmask:
    logits_processor = get_logits_processor_from_csvs(
        processor, ["data/processed/train.csv", "data/processed/val.csv"]
    )
else:
    logits_processor = get_closed_vocab_processor(
        processor, model, ["data/processed/train.csv", "data/processed/val.csv"]
    )

rows = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    image = resize_with_padding(image, target_size=target_size)
    px = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            px,
            num_beams=args.num_beams,
            max_new_tokens=32,
            logits_processor=logits_processor,
            output_scores=True,
            return_dict_in_generate=True,
        )

    prediction = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    conf = transition_confidence(model, outputs)[0]
    rows.append({
        "filename": path.name, "prediction": prediction,
        "min_conf": conf["seq_min_prob"], "mean_conf": conf["seq_mean_prob"],
    })
    if len(image_paths) == 1:
        print("Prediction:", prediction)
        print(f"Confidence (min token prob): {conf['seq_min_prob']:.3f}")
        print(f"Confidence (mean token prob): {conf['seq_mean_prob']:.3f}")
    else:
        print(f"{path.name}: {prediction!r} (min_conf={conf['seq_min_prob']:.3f})")

if args.output:
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Predictions written to", args.output)
