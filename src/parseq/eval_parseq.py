"""
PARSeq baseline (pretrained zero-shot, or fine-tuned) on the same test split
used for the TrOCR model (evaluate.py), for a like-for-like comparison.

With no --checkpoint, uses the pretrained scene-text checkpoint from
https://github.com/baudm/parseq via torch.hub -- NOT fine-tuned on this
dataset, a quick sanity check on whether the architecture looks promising.

With --checkpoint pointing at a Lightning .ckpt produced by
third_party/parseq/train.py (see docs/CODEBASE_OVERVIEW.md's "Fine-tuning
PARSeq" section), evaluates that fine-tuned model instead. Must be run with
the venv_parseq interpreter in that case (needs strhub importable):

    venv_parseq/Scripts/python src/eval_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/epoch=20-...ckpt"

Confidence is the per-character softmax probability from PARSeq's own decode
step (min/mean across the predicted sequence, EOS included) -- the PARSeq
analog of constraints.py's transition_confidence() for TrOCR.

Usage:
    python src/eval_parseq.py
    python src/eval_parseq.py --csv data/processed/val.csv --output parseq_results.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd, torch
from PIL import Image
import torchvision.transforms as T


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
ap.add_argument("--checkpoint", default=None, help="path to a fine-tuned .ckpt from third_party/parseq/train.py; omit for the pretrained zero-shot model")
ap.add_argument("--flag-threshold", type=float, default=0.5, help="flag for review if min char confidence falls below this")
ap.add_argument("--output", default=None, help="optional per-sample results CSV")
args = ap.parse_args()

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

df = pd.read_csv(args.csv)
exact = edits = chars = flagged = 0
rows = []
for _, r in df.iterrows():
    image = Image.open(r["image_path"]).convert("RGB")
    x = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(x).softmax(-1)
    pred_list, prob_list = model.tokenizer.decode(p)
    pred = pred_list[0].strip()
    probs = prob_list[0]
    min_conf = probs.min().item() if len(probs) else 0.0
    mean_conf = probs.mean().item() if len(probs) else 0.0
    truth = str(r["text"]).strip()
    is_flagged = min_conf < args.flag_threshold
    exact += pred == truth
    edits += lev(pred, truth)
    chars += max(len(truth), 1)
    flagged += is_flagged
    rows.append({
        "filename": r["filename"], "target": truth, "prediction": pred,
        "exact_match": pred == truth, "seq_min_prob": min_conf, "seq_mean_prob": mean_conf,
    })

print(f"Model: PARSeq (fine-tuned: {args.checkpoint})" if args.checkpoint else "Model: PARSeq (pretrained, zero-shot)")
print("Samples:", len(df))
print("Exact-match accuracy:", exact / max(len(df), 1))
print("CER:", edits / max(chars, 1))
print(f"Flagged for review (min conf < {args.flag_threshold}):", flagged, f"({flagged / max(len(df), 1) * 100:.1f}%)")

if args.output:
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Per-sample results written to", args.output)
