"""
Quick A/B test: does unsharp masking (same function as test_preprocessing.py)
help on specific failed crops, applied AFTER resize_with_padding? Sharpening
after the upscale directly counteracts the resize's interpolation blur;
sharpening before would partly get undone by the resize itself.

Doesn't retrain or touch your pipeline -- runs both variants through your
existing checkpoint and prints predictions side by side, so you can see on
real failures whether this is worth wiring into dataset.py properly before
committing to a full retrain + re-eval cycle.

Usage:
    # test specific images, with known targets
    python test_unsharp.py --model checkpoints/best --images path1.png path2.png --targets 7 4

    # pull the first N failed samples straight from an error_analysis.csv
    python test_unsharp.py --model checkpoints/best --from-csv error_analysis.csv --n 8

Uses greedy decoding (num_beams=1) for speed on a small diagnostic run --
not the num_beams=4 your evaluate.py/error_analysis.py use for real
numbers. Treat results here as directional, not a final metric.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from constraints import get_closed_vocab_processor, transition_confidence
from dataset import resize_with_padding, get_target_size


def unsharp(gray, sigma=1.0, amount=0.8):
    """Identical to test_preprocessing.py's version."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_unsharp_to_pil(image, sigma=1.0, amount=0.8):
    """
    Sharpens on grayscale (matches test_preprocessing.py's approach), then
    converts back to RGB so the input shape matches what the model was
    trained on (dataset.py loads with .convert("RGB")).
    """
    gray = np.array(image.convert("L"))
    sharpened = unsharp(gray, sigma=sigma, amount=amount)
    return Image.fromarray(sharpened).convert("RGB")


def predict(model, processor, logits_processor, image, device, temperature):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=1,
            max_new_tokens=16,
            logits_processor=logits_processor,
            output_scores=True,
            return_dict_in_generate=True,
        )
    prediction = processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )[0].strip()
    conf = transition_confidence(model, outputs, temperature=temperature)[0]
    return prediction, conf["seq_min_prob"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/best")
    ap.add_argument("--images", nargs="*", default=None, help="specific image paths to test")
    ap.add_argument("--targets", nargs="*", default=None, help="ground-truth labels matching --images, in order (optional)")
    ap.add_argument("--from-csv", default=None, help="pull failed samples from an error_analysis.csv instead of --images")
    ap.add_argument("--n", type=int, default=8, help="how many failed samples to pull from --from-csv")
    ap.add_argument("--sigma", type=float, default=1.0, help="unsharp blur radius")
    ap.add_argument("--amount", type=float, default=0.8, help="unsharp strength")
    ap.add_argument("--temperature", type=float, default=1.0, help="calibration temperature, if you have one fitted")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
    target_size = get_target_size(processor)
    logits_processor = get_closed_vocab_processor(
        processor, model, ["data/processed/train.csv", "data/processed/val.csv"]
    )

    samples = []
    if args.images:
        targets = args.targets or [None] * len(args.images)
        if args.targets and len(args.targets) != len(args.images):
            raise SystemExit("--targets must have the same count as --images, or be omitted")
        samples = list(zip(args.images, targets))
    elif args.from_csv:
        df = pd.read_csv(args.from_csv)
        failed = df[df["exact_match"] == False] if "exact_match" in df.columns else df
        failed = failed.head(args.n)
        samples = list(zip(failed["image_path"], failed["target"]))
    else:
        raise SystemExit("Pass either --images path1 path2 ... or --from-csv error_analysis.csv")

    print(f"{'Target':<8} {'Baseline':<10} {'conf':<6} {'Sharpened':<10} {'conf':<6}  Image")
    print("-" * 90)

    fixed = broke = 0

    for image_path, target in samples:
        raw = Image.open(image_path).convert("RGB")
        padded = resize_with_padding(raw, target_size=target_size)

        base_pred, base_conf = predict(
            model, processor, logits_processor, padded, device, args.temperature
        )

        sharpened = apply_unsharp_to_pil(padded, sigma=args.sigma, amount=args.amount)
        sharp_pred, sharp_conf = predict(
            model, processor, logits_processor, sharpened, device, args.temperature
        )

        target_str = str(target) if target is not None and str(target) != "nan" else "?"
        flag = ""
        if target is not None and str(target) != "nan":
            if sharp_pred == target_str and base_pred != target_str:
                flag = "  <- FIXED"
                fixed += 1
            elif sharp_pred != target_str and base_pred == target_str:
                flag = "  <- BROKE"
                broke += 1

        print(
            f"{target_str:<8} {base_pred:<10} {base_conf:<6.2f} "
            f"{sharp_pred:<10} {sharp_conf:<6.2f}  {image_path}{flag}"
        )

    if fixed or broke:
        print()
        print(f"Net effect on this sample: {fixed} fixed, {broke} broke.")
        print(
            "This is a small, hand-picked sample of failures -- not a "
            "representative eval. If net-positive here, worth wiring "
            "resize_with_padding + unsharp into dataset.py and doing a "
            "real retrain + full error_analysis.py run to confirm on the "
            "whole test set, the same way padding itself was validated."
        )


if __name__ == "__main__":
    main()