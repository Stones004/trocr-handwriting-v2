"""Re-evaluate a saved noise-classifier checkpoint without retraining.

Scores <paths.noise_dataset_dir>/test.csv by default (the split train.py never trains or tunes on), but --csv
points it at any other labelled CSV (columns: filename, image_path, label) -- e.g. val.csv, or a hand-labelled
sample of images the model has never seen. Writes the same metrics report train.py does, plus a CSV of every
misclassified row (filename, true label, predicted label, confidence) so you can open the actual images and see
what kind of noise/clean crop it's getting wrong.

    python src/noise_classifier/evaluate.py                                    # test split, default checkpoint
    python src/noise_classifier/evaluate.py --csv data/noise_classifier/val.csv --checkpoint models/noise_classifier/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import NoiseDataset  # noqa: E402
from model_io import CLASSES, load_model, make_transform  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import cfg, path, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None, help="default: <paths.noise_dataset_dir>/test.csv")
ap.add_argument("--checkpoint", default=None, help="default: <paths.models_dir>/noise_classifier/best.pt")
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--report", default=None, help="default: <paths.reports_dir>/noise_classifier_eval.md")
ap.add_argument("--errors-csv", default=None, help="default: <paths.reports_dir>/noise_classifier_errors.csv")
args = ap.parse_args()

csv_path = resolve(args.csv) if args.csv else resolve(cfg["paths"]["noise_dataset_dir"]) / "test.csv"
checkpoint = resolve(args.checkpoint) if args.checkpoint else path("models_dir") / "noise_classifier" / "best.pt"
report_path = resolve(args.report) if args.report else path("reports_dir") / "noise_classifier_eval.md"
errors_path = resolve(args.errors_csv) if args.errors_csv else path("reports_dir") / "noise_classifier_errors.csv"

model, img_size = load_model(checkpoint)
device = next(model.parameters()).device
ds = NoiseDataset(csv_path, make_transform(img_size, train=False))
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
print(f"checkpoint={checkpoint}  csv={csv_path}  n={len(ds)}  device={device}")

all_preds, all_labels, all_conf = [], [], []
with torch.no_grad():
    for x, y in dl:
        probs = F.softmax(model(x.to(device)), dim=1)
        all_preds.append(probs.argmax(1).cpu())
        all_conf.append(probs.max(1).values.cpu())
        all_labels.append(y)
preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()
confidences = torch.cat(all_conf).numpy()

precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
acc = (preds == labels).mean()
cm = confusion_matrix(labels, preds, labels=[0, 1])

lines = ["# Noise classifier evaluation", "", f"Checkpoint: `{checkpoint}`  |  Data: `{csv_path}`", "",
         "| class | precision | recall | f1 | support |", "|---|---|---|---|---|"]
for i, name in enumerate(CLASSES):
    support = int((labels == i).sum())
    lines.append(f"| {name} | {precision[i]:.3f} | {recall[i]:.3f} | {f1[i]:.3f} | {support} |")
lines += ["", f"Overall accuracy: {acc:.1%} (n={len(labels)})", "",
          "Confusion matrix (rows=actual, cols=predicted):", "",
          "| | pred clean | pred noise |", "|---|---|---|",
          f"| actual clean | {cm[0, 0]} | {cm[0, 1]} |",
          f"| actual noise | {cm[1, 0]} | {cm[1, 1]} |", ""]
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("wrote", report_path)

errors = ds.df.copy()
errors["true_label"] = [CLASSES[i] for i in labels]
errors["pred_label"] = [CLASSES[i] for i in preds]
errors["confidence"] = confidences
errors = errors.loc[preds != labels].sort_values("confidence", ascending=False)
errors[["filename", "image_path", "true_label", "pred_label", "confidence"]].to_csv(errors_path, index=False)
print(f"wrote {len(errors)} misclassified rows to {errors_path}")
