"""
Run once to fit a calibration temperature on val.csv. Prints T -- pass it
to --temperature in evaluate.py / error_analysis.py afterward. Re-run this
any time you retrain, since a new checkpoint needs its own T.

Usage: python calibrate_temperature.py --model checkpoints/best
"""
import argparse

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from constraints import fit_temperature

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="checkpoints/best")
ap.add_argument("--csv", default="data/processed/val.csv")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained(args.model)
model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()

fit_temperature(model, processor, args.csv, device)