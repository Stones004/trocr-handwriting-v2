"""The master dataset as ONE file: data/master_dataset.mdb (a single-file LMDB holding every real and synthetic crop, images embedded).

    venv_parseq/Scripts/python src/master/dataset_io.py pack      # data/master/*.csv + data/roi_labels.csv  ->  data/master_dataset.mdb
    venv_parseq/Scripts/python src/master/dataset_io.py unpack    # data/master_dataset.mdb  ->  data/master_lmdb/{train,val,test}/handwriting  (PARSeq layout)
    venv_parseq/Scripts/python src/master/dataset_io.py info      # counts per split / source

Keys inside the file: `num-samples`, then for every sample i (1-based) `image-i` (PNG bytes), `label-i` (training label) and `meta-i` (JSON:
filename, split, source, bucket, drawn, style, doc, label_source). Splits: train (14,003 rows = 3,064 real + 10,867 synthetic + 72 repeated
Q-prefix rows), val (384 real), test (379 real). `unpack` writes train/val/test only. NOTE: parseq_master_v3 was trained on a 14,323-row train set that
also contained 64 Claude-read new-document crops (x5); those crops and the 56 held-out ones were removed from this file on 2026-10-04 (docs/MODEL_REPORT.md).
"""
import argparse
import io
import json
import sys
from pathlib import Path

import lmdb
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MDB = ROOT / "data" / "master_dataset.mdb"
META = ["filename", "split", "source", "bucket", "drawn", "style", "doc", "label_source"]


def _rows():
    for s in ("train", "val", "test"):
        for r in pd.read_csv(ROOT / "data" / "master" / f"{s}.csv", keep_default_na=False, dtype=str).itertuples():
            yield r.image_path, r.text, dict(filename=r.filename, split=s, source=r.source, bucket=r.bucket, drawn=r.drawn, style=r.style,
                                              doc=r.filename[:7], label_source="")
    labels = ROOT / "data" / "roi_labels.csv"          # optional: Claude-read crops from new documents (removed from the dataset on 2026-10-04)
    held = pd.read_csv(labels, keep_default_na=False, dtype=str) if labels.exists() else pd.DataFrame(columns=["holdout"])
    for r in held[held.holdout == "True"].itertuples():
        yield r.image_path, r.text, dict(filename=r.filename, split="heldout_newdoc", source="newdoc_heldout", bucket="", drawn="", style="",
                                          doc=r.doc, label_source=r.label_source)


def pack():
    rows = list(_rows())
    size = sum(Path(p).stat().st_size for p, _, _ in rows)
    env = lmdb.open(str(MDB), map_size=int(size * 2 + 64e6), subdir=False, lock=False)
    for start in range(0, len(rows), 2000):
        with env.begin(write=True) as txn:
            for i, (p, label, meta) in enumerate(rows[start:start + 2000], start + 1):
                Image.open(p).convert("RGB")  # validity check
                txn.put(f"image-{i:09d}".encode(), Path(p).read_bytes())
                txn.put(f"label-{i:09d}".encode(), label.encode())
                txn.put(f"meta-{i:09d}".encode(), json.dumps(meta).encode())
    with env.begin(write=True) as txn:
        txn.put(b"num-samples", str(len(rows)).encode())
    env.close()
    pd.DataFrame([{**m, "text": t} for _, t, m in rows])[["filename", "split", "text", *META[2:]]].to_csv(MDB.with_suffix(".csv"), index=False)
    print(f"packed {len(rows)} samples -> {MDB} ({MDB.stat().st_size / 1e6:.0f} MB); manifest {MDB.with_suffix('.csv').name}")


def samples():
    """Yield (split, label, png_bytes, meta) for every sample in the file."""
    env = lmdb.open(str(MDB), readonly=True, subdir=False, lock=False)
    with env.begin() as txn:
        for i in range(1, int(txn.get(b"num-samples")) + 1):
            yield json.loads(txn.get(f"meta-{i:09d}".encode()))["split"], txn.get(f"label-{i:09d}".encode()).decode(), txn.get(f"image-{i:09d}".encode()), \
                json.loads(txn.get(f"meta-{i:09d}".encode()))
    env.close()


def unpack(out=ROOT / "data" / "master_lmdb", name="handwriting"):
    by = {"train": [], "val": [], "test": []}
    for split, label, png, _ in samples():
        if split in by:
            by[split].append((label, png))
    for split, items in by.items():
        d = out / split / name
        d.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(d), map_size=512 * 1024 * 1024)
        with env.begin(write=True) as txn:
            for i, (label, png) in enumerate(items, 1):
                txn.put(f"image-{i:09d}".encode(), png)
                txn.put(f"label-{i:09d}".encode(), label.encode())
            txn.put(b"num-samples", str(len(items)).encode())
        env.close()
        print(f"[{split}] {len(items)} rows -> {d}")


def info():
    df = pd.DataFrame([{"split": s, "source": m["source"]} for s, _, _, m in samples()])
    print(df.groupby(["split", "source"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "unpack", "info"])
    {"pack": pack, "unpack": unpack, "info": info}[ap.parse_args().cmd]()
