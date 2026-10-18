"""Train the noise/clean crop classifier (ResNet-18, ImageNet-pretrained, fine-tuned).

Reads <paths.noise_dataset_dir>/{train,val,test}.csv (src/noise_classifier/prepare_dataset.py must be run first).
Saves the checkpoint with the best validation noise-class F1 to <paths.models_dir>/noise_classifier/best.pt, then
scores it on the held-out test split and writes <paths.reports_dir>/noise_classifier.md.

    python src/noise_classifier/train.py [--epochs 15] [--batch-size 64] [--lr 1e-4] [--patience 5]
"""
import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import NoiseDataset, class_weights  # noqa: E402
from model_io import CLASSES, build_model, make_transform, save_checkpoint  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import path, resolve, cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--img-size", type=int, default=128)
ap.add_argument("--patience", type=int, default=5, help="stop if val noise-F1 doesn't improve for this many epochs")
ap.add_argument("--data", default=None, help="default: paths.noise_dataset_dir")
ap.add_argument("--out", default=None, help="default: <paths.models_dir>/noise_classifier/best.pt")
args = ap.parse_args()

data_dir = resolve(args.data) if args.data else resolve(cfg["paths"]["noise_dataset_dir"])
out_ckpt = resolve(args.out) if args.out else path("models_dir") / "noise_classifier" / "best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

train_ds = NoiseDataset(data_dir / "train.csv", make_transform(args.img_size, train=True))
val_ds = NoiseDataset(data_dir / "val.csv", make_transform(args.img_size, train=False))
test_ds = NoiseDataset(data_dir / "test.csv", make_transform(args.img_size, train=False))
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

model = build_model().to(device)
weights = class_weights(data_dir / "train.csv").to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


@torch.no_grad()
def evaluate(dl):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    acc = (preds == labels).mean()
    return {"loss": total_loss / len(labels), "acc": acc, "precision": precision, "recall": recall, "f1": f1,
            "preds": preds, "labels": labels}


best_f1_noise = -1.0
epochs_without_improvement = 0
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y)
    train_loss /= len(train_ds)

    val = evaluate(val_dl)
    scheduler.step(val["f1"][1])
    print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  val_acc={val['acc']:.3f}  "
          f"val_noise_f1={val['f1'][1]:.3f}  val_clean_f1={val['f1'][0]:.3f}")

    if val["f1"][1] > best_f1_noise:
        best_f1_noise = val["f1"][1]
        epochs_without_improvement = 0
        save_checkpoint(model, out_ckpt, img_size=args.img_size)
        print(f"  saved best checkpoint to {out_ckpt} (val_noise_f1={best_f1_noise:.3f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"early stopping: no val_noise_f1 improvement for {args.patience} epochs")
            break

# Score the best checkpoint on the held-out test split.
ckpt = torch.load(out_ckpt, map_location=device)
model.load_state_dict(ckpt["state_dict"])
test = evaluate(test_dl)
cm = confusion_matrix(test["labels"], test["preds"], labels=[0, 1])

lines = ["# Noise classifier", "",
         f"ResNet-18 (ImageNet-pretrained), document-level split ({data_dir}). Best checkpoint by validation "
         f"noise-class F1 ({best_f1_noise:.3f}), scored on the held-out test split.", "",
         "| class | precision | recall | f1 | support |", "|---|---|---|---|---|"]
for i, name in enumerate(CLASSES):
    support = int((test["labels"] == i).sum())
    lines.append(f"| {name} | {test['precision'][i]:.3f} | {test['recall'][i]:.3f} | {test['f1'][i]:.3f} | {support} |")
lines += ["", f"Overall accuracy: {test['acc']:.1%} (n={len(test['labels'])})", "",
          "Confusion matrix (rows=actual, cols=predicted):", "",
          "| | pred clean | pred noise |", "|---|---|---|",
          f"| actual clean | {cm[0, 0]} | {cm[0, 1]} |",
          f"| actual noise | {cm[1, 0]} | {cm[1, 1]} |", ""]
report = path("reports_dir") / "noise_classifier.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("wrote", report)
