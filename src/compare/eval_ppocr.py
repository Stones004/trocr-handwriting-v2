"""
Zero-shot PP-OCRv5 recognition baseline on the same test split used for the
fine-tuned TrOCR model (evaluate.py), for a like-for-like comparison.

Uses PaddleOCR's TextRecognition pipeline directly (recognition-only --
no text detection is run, since our crops are already single-token ROIs)
with the pretrained "PP-OCRv5_server_rec" model, NOT fine-tuned on this
dataset. This is a quick sanity check on whether the architecture looks
promising before investing in a full fine-tuning pipeline.

Confidence is PP-OCR's own rec_score, a single sequence-level confidence
(not per-character like TrOCR/PARSeq).

Usage:
    python src/eval_ppocr.py
    python src/eval_ppocr.py --model-name PP-OCRv5_mobile_rec --output ppocr_results.csv
"""
import argparse
import pandas as pd
from paddleocr import TextRecognition


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/processed/test.csv")
ap.add_argument("--model-name", default="PP-OCRv5_server_rec")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--flag-threshold", type=float, default=0.5, help="flag for review if rec_score falls below this")
ap.add_argument("--output", default=None, help="optional per-sample results CSV")
args = ap.parse_args()

model = TextRecognition(model_name=args.model_name)

df = pd.read_csv(args.csv)
results = model.predict(list(df["image_path"]), batch_size=args.batch_size)

exact = edits = chars = flagged = 0
rows = []
for (_, r), res in zip(df.iterrows(), results):
    pred = str(res["rec_text"]).strip()
    conf = float(res["rec_score"])
    truth = str(r["text"]).strip()
    is_flagged = conf < args.flag_threshold
    exact += pred == truth
    edits += lev(pred, truth)
    chars += max(len(truth), 1)
    flagged += is_flagged
    rows.append({
        "filename": r["filename"], "target": truth, "prediction": pred,
        "exact_match": pred == truth, "rec_score": conf,
    })

print(f"Model: {args.model_name} (pretrained, zero-shot)")
print("Samples:", len(df))
print("Exact-match accuracy:", exact / max(len(df), 1))
print("CER:", edits / max(chars, 1))
print(f"Flagged for review (rec_score < {args.flag_threshold}):", flagged, f"({flagged / max(len(df), 1) * 100:.1f}%)")

if args.output:
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Per-sample results written to", args.output)
