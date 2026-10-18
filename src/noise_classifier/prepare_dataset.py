"""Build a labelled, document-level train/val/test split for the noise classifier.

A crop is NOISE (label=1) if either:
  - it has a row in the annotation workbook (labels.annotation_file) whose label_column (Normalize) is blank/NaN, or
  - it has no row in the annotation workbook at all (never annotated -- treated as noise per project decision).
Everything else -- a row with a non-blank Normalize value -- is CLEAN (label=0).

Reuses image_dir / annotation_file / filename_column / label_column / labels.doc_id_pattern from config.yaml, the
same source prepare_dataset.py (the OCR pipeline) reads. The split is document-level (labels.doc_id_pattern, same
`Doc####` prefix as prepare_docsplit.py) so crops from the same scanned document -- which share paper texture,
lighting and scanner artifacts -- don't leak between train and test.

Writes <paths.noise_dataset_dir>/{train,val,test}.csv with columns filename,image_path,label,doc.
    python src/noise_classifier/prepare_dataset.py [--seed 42]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import cfg, label_regex, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=cfg["docsplit"]["seed"])
ap.add_argument("--ratios", type=float, nargs=3, default=[cfg["split"][k] for k in ("train", "val", "test")])
ap.add_argument("--tries", type=int, default=cfg["docsplit"]["tries"])
ap.add_argument("--out", default=None, help="default: paths.noise_dataset_dir")
args = ap.parse_args()
if abs(sum(args.ratios) - 1.0) > 1e-6:
    raise SystemExit(f"--ratios must sum to 1, got {args.ratios}")

image_dir = resolve(cfg["image_dir"])
annotation_file = resolve(cfg["annotation_file"])
filename_col = cfg["filename_column"]
label_col = cfg["label_column"]
doc_id = label_regex("doc_id_pattern")

if not image_dir.exists():
    raise FileNotFoundError(f"Image directory not found: {image_dir}\nSet image_dir in config.yaml (or config.local.yaml).")
if not annotation_file.exists():
    raise FileNotFoundError(f"Annotation file not found: {annotation_file}\nSet annotation_file in config.yaml (or config.local.yaml).")

images = sorted(p.name for p in image_dir.glob("*.png"))
print(f"Images on disk:              {len(images)}")

ann = pd.read_excel(annotation_file)
for col in (filename_col, label_col):
    if col not in ann.columns:
        raise ValueError(f"Missing column {col!r}. Available columns: {list(ann.columns)}")
ann[filename_col] = ann[filename_col].astype("string").str.strip()
ann[label_col] = ann[label_col].astype("string").str.strip()
missing_on_disk = int((~ann[filename_col].isin(images)).sum())

# Duplicate filenames in the workbook (annotator re-entered a row): a filename is clean if ANY of its rows has a
# non-blank label, since that's still evidence the crop was legible enough to transcribe.
ann["is_labelled"] = ann[label_col].notna() & (ann[label_col] != "") & (ann[label_col].str.lower() != "nan")
labelled_lookup = ann.groupby(filename_col)["is_labelled"].any()
duplicate_filenames = int(ann[filename_col].duplicated().sum())

blank_label_count = int((~labelled_lookup).sum())  # in the workbook, but Normalize is blank/NaN

df = pd.DataFrame({"filename": images})
df["image_path"] = df["filename"].map(lambda f: str(image_dir / f))
df["doc"] = df["filename"].str.extract(doc_id)[0]
assert df["doc"].notna().all(), f"filenames without a document id (labels.doc_id_pattern): {df.loc[df.doc.isna(), 'filename'].head().tolist()}"

in_workbook = df["filename"].isin(labelled_lookup.index)
is_clean = in_workbook & df["filename"].map(labelled_lookup).fillna(False)
df["label"] = (~is_clean).astype(int)  # 1 = noise, 0 = clean
unmatched_count = int((~in_workbook).sum())

print(f"Workbook rows:                {len(ann)}")
print(f"Duplicate filenames in wb:    {duplicate_filenames}")
print(f"Workbook rows not on disk:    {missing_on_disk}")
print(f"On disk, blank/NaN Normalize: {blank_label_count}")
print(f"On disk, not in workbook:     {unmatched_count}")
print(f"Noise (label=1):              {int((df.label == 1).sum())}")
print(f"Clean (label=0):              {int((df.label == 0).sum())}")

# Document-level split: greedy assignment to the split furthest below its target size, keeping the one (of --tries
# random document orderings) whose noise ratio is most balanced across train/val/test (mirrors prepare_docsplit.py,
# which instead balances label-vocabulary coverage -- there's no vocabulary here, so we balance the class ratio).
sizes = df.groupby("doc").size()
n = len(df)
targets = [r * n for r in args.ratios]
overall_noise_rate = df["label"].mean()
rng = np.random.default_rng(args.seed)

best = None
for _ in range(args.tries):
    order = rng.permutation(sizes.index.to_numpy())
    assign, load = {}, [0, 0, 0]
    for d in order:
        k = int(np.argmax([targets[i] - load[i] for i in range(3)]))
        assign[d] = k
        load[k] += sizes[d]
    split = df["doc"].map(assign)
    noise_rates = [df.label[split == k].mean() for k in range(3)]
    balance_err = sum(abs(r - overall_noise_rate) for r in noise_rates)
    size_err = sum(abs(load[i] - targets[i]) for i in range(3)) / n
    score = -(balance_err + size_err)
    if best is None or score > best[0]:
        best = (score, split, load, noise_rates)

_, split, load, noise_rates = best
out = resolve(args.out) if args.out else resolve(cfg["paths"]["noise_dataset_dir"])
out.mkdir(parents=True, exist_ok=True)
cols = ["filename", "image_path", "label", "doc"]
for k, name in enumerate(("train", "val", "test")):
    part = df.loc[split == k, cols]
    part.to_csv(out / f"{name}.csv", index=False)
    print(f"{name:5s} rows={len(part):5d} docs={part.doc.nunique():4d} noise_rate={noise_rates[k]:.1%}")
docs = [set(df.doc[split == k]) for k in range(3)]
print("doc overlap train/val/test:", len(docs[0] & docs[1]), len(docs[0] & docs[2]), len(docs[1] & docs[2]))
print("written to", out.resolve())
