"""
Converts split CSVs (filename, image_path, text) -- by default <paths.processed_dir>/{train,val,test}.csv --
into the LMDB format PARSeq's training pipeline (third_party/parseq) expects.

PARSeq's strhub.data.dataset.build_tree_dataset recursively globs for any
data.mdb under a root directory -- it doesn't care about folder naming beyond
that, so this script writes one LMDB per split under:

    <output-dir>/train/handwriting/data.mdb
    <output-dir>/val/handwriting/data.mdb
    <output-dir>/test/handwriting/data.mdb

which matches the root_dir/train_dir structure PARSeq's SceneTextDataModule
expects (data.root_dir=<output-dir>, data.train_dir=handwriting). Defaults
(output dir = paths.lmdb_dir, folder name = lmdb.dataset_name, map size =
lmdb.map_size_mb) come from config.yaml.

Deliberately does NOT reuse third_party/parseq/tools/create_lmdb_dataset.py --
that script hardcodes map_size=1TB, which is harmless on Linux (lazy sparse
mmap) but makes lmdb.open() fail on Windows with "the paging file is too
small for this operation to complete" (Windows tries to back the mapping with
real page-file-committed memory). Our whole dataset is ~9MB, so a 512MB map
size (--map-size-mb) is generous headroom without tripping that.

Must be run with the venv_parseq interpreter (needs lmdb, which the main
project venv doesn't have):

    venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py
"""
import argparse
import io
import sys
from pathlib import Path

import lmdb
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import cfg, path, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--processed-dir", default=str(path("processed_dir")))
ap.add_argument("--output-dir", default=str(path("lmdb_dir")))
ap.add_argument("--dataset-name", default=cfg["lmdb"]["dataset_name"])
ap.add_argument("--map-size-mb", type=int, default=cfg["lmdb"]["map_size_mb"])
args = ap.parse_args()

processed_dir = resolve(args.processed_dir)
output_dir = resolve(args.output_dir)


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
