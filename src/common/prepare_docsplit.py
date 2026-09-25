"""Build a DOCUMENT-LEVEL train/val/test split from the existing processed splits.

The default split (prepare_dataset.py) is random per crop, so the same document (writer/form) lands in train and
test and test accuracy is optimistic. Here every crop of a document goes to exactly one split, so test measures
generalization to unseen documents. The document id is the first group of labels.doc_id_pattern (default: the
`Doc####` prefix of the filename).
Reads paths.processed_dir and writes <paths.docsplit_dir>/seed<N>/{train,val,test}.csv (same columns), leaving the
processed splits untouched. Defaults (ratios = split.*, seed/tries = docsplit.*) come from config.yaml.
    python src/common/prepare_docsplit.py [--seed 42] [--out data/docsplit/seed42]
"""
import argparse

import numpy as np
import pandas as pd

from project_config import cfg, label_regex, path, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(path("processed_dir")))
ap.add_argument("--out", default=None, help="default: <paths.docsplit_dir>/seed<seed>")
ap.add_argument("--seed", type=int, default=cfg["docsplit"]["seed"])
ap.add_argument("--ratios", type=float, nargs=3, default=[cfg["split"][k] for k in ("train", "val", "test")])
ap.add_argument("--tries", type=int, default=cfg["docsplit"]["tries"], help="random assignments to try; keeps the one with best label coverage")
args = ap.parse_args()
if abs(sum(args.ratios) - 1.0) > 1e-6:
    raise SystemExit(f"--ratios must sum to 1, got {args.ratios}")

doc_id = label_regex("doc_id_pattern")
df = pd.concat([pd.read_csv(f"{args.src}/{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
df["text"] = df["text"].astype(str).str.strip()
df["doc"] = df["filename"].str.extract(doc_id)[0]
assert df["doc"].notna().all(), f"filenames without a document id (labels.doc_id_pattern): {df.loc[df.doc.isna(), 'filename'].head().tolist()}"
sizes = df.groupby("doc").size()
n = len(df)
targets = [r * n for r in args.ratios]
rng = np.random.default_rng(args.seed)

best = None
for _ in range(args.tries):
    order = rng.permutation(sizes.index.to_numpy())
    assign, load = {}, [0, 0, 0]
    for d in order:  # greedy: put each document in the split furthest below its target size
        k = int(np.argmax([targets[i] - load[i] for i in range(3)]))
        assign[d] = k
        load[k] += sizes[d]
    split = df["doc"].map(assign)
    train_labels = set(df.text[split == 0])
    cover = np.mean([df.text[split == k].isin(train_labels).mean() for k in (1, 2)])  # val/test labels seen in train
    size_err = sum(abs(load[i] - targets[i]) for i in range(3)) / n
    score = cover - size_err
    if best is None or score > best[0]:
        best = (score, split, cover, load)

_, split, cover, load = best
out = resolve(args.out) if args.out else path("docsplit_dir") / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
cols = ["filename", "image_path", "text"]
for k, name in enumerate(("train", "val", "test")):
    df.loc[split == k, cols].to_csv(out / f"{name}.csv", index=False)
    part = df[split == k]
    print(f"{name:5s} rows={len(part):5d} docs={part.doc.nunique():4d} labels={part.text.nunique():3d}")
docs = [set(df.doc[split == k]) for k in range(3)]
print("doc overlap train/val/test:", len(docs[0] & docs[1]), len(docs[0] & docs[2]), len(docs[1] & docs[2]))
tl = set(df.text[split == 0])
print(f"val labels seen in train: {df.text[split == 1].isin(tl).mean():.1%} | test: {df.text[split == 2].isin(tl).mean():.1%}")
print("written to", out.resolve())
