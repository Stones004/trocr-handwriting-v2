"""
Converts our data/processed/{train,val,test}.csv (filename, image_path, text)
into the LMDB format PARSeq's training pipeline (third_party/parseq) expects.

PARSeq's strhub.data.dataset.build_tree_dataset recursively globs for any
data.mdb under a root directory -- it doesn't care about folder naming beyond
that, so this script writes one LMDB per split under:

    data/parseq_lmdb/train/handwriting/data.mdb
    data/parseq_lmdb/val/handwriting/data.mdb
    data/parseq_lmdb/test/handwriting/data.mdb

which matches the root_dir/train_dir structure PARSeq's SceneTextDataModule
expects (data.root_dir=data/parseq_lmdb, data.train_dir=handwriting).

Deliberately does NOT reuse third_party/parseq/tools/create_lmdb_dataset.py --
that script hardcodes map_size=1TB, which is harmless on Linux (lazy sparse
mmap) but makes lmdb.open() fail on Windows with "the paging file is too
small for this operation to complete" (Windows tries to back the mapping with
real page-file-committed memory). Our whole dataset is ~9MB, so a 512MB map
size (--map-size-mb) is generous headroom without tripping that.

Must be run with the venv_parseq interpreter (needs lmdb, which the main
project venv doesn't have):

    venv_parseq/Scripts/python src/prepare_parseq_dataset.py
"""
import argparse
import io
from pathlib import Path

import lmdb
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--processed-dir", default="data/processed")
ap.add_argument("--output-dir", default="data/parseq_lmdb")
ap.add_argument("--dataset-name", default="handwriting")
ap.add_argument("--map-size-mb", type=int, default=512)
args = ap.parse_args()

processed_dir = Path(args.processed_dir)
output_dir = Path(args.output_dir)


def create_lmdb(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_dir), map_size=args.map_size_mb * 1024 * 1024)
    cache = {}
    n = 0
    for _, row in df.iterrows():
        image_path = Path(row["image_path"])
        with open(image_path, "rb") as f:
            image_bin = f.read()
        Image.open(io.BytesIO(image_bin)).convert("RGB")  # validity check
        n += 1
        cache[f"image-{n:09d}".encode()] = image_bin
        cache[f"label-{n:09d}".encode()] = str(row["text"]).encode()
    cache[b"num-samples"] = str(n).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    env.close()
    return n


for split in ["train", "val", "test"]:
    csv_path = processed_dir / f"{split}.csv"
    df = pd.read_csv(csv_path)
    lmdb_out = output_dir / split / args.dataset_name
    n = create_lmdb(df, lmdb_out)
    print(f"[{split}] {n} rows -> {lmdb_out}")

print("Done. Point PARSeq's data.root_dir at", output_dir.resolve())
