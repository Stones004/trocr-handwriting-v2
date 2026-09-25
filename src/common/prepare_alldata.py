"""Build the ALL-DATA set for the final production model: every labeled crop (train+val+test of paths.processed_dir) in
`train`. The `val`/`test` slots reuse val.csv purely so PARSeq's trainer has something to log; those rows are also in
train, so they say nothing about generalization -- rely on the document-level CV (src/parseq/docsplit_cv.py) for that.
Writes paths.all_data_dir; then build its LMDB with:
    python src/common/prepare_alldata.py
    python src/parseq/prepare_parseq_dataset.py --processed-dir <all_data_dir> --output-dir <all_lmdb_dir>
(the defaults for both directories are in config.yaml).
"""
import argparse

import pandas as pd

from project_config import label_regex, path, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(path("processed_dir")))
ap.add_argument("--out", default=str(path("all_data_dir")))
args = ap.parse_args()

src, out = resolve(args.src), resolve(args.out)
out.mkdir(parents=True, exist_ok=True)
parts = {s: pd.read_csv(src / f"{s}.csv") for s in ("train", "val", "test")}
allrows = pd.concat(parts.values(), ignore_index=True)
n_dup = int(allrows.image_path.duplicated().sum())  # duplicates are intentionally preserved (project decision)
print("repeated image paths kept as-is:", n_dup)
allrows.to_csv(out / "train.csv", index=False)
parts["val"].to_csv(out / "val.csv", index=False)
parts["val"].to_csv(out / "test.csv", index=False)
print("all-data train rows:", len(allrows), "| docs:", allrows.filename.str.extract(label_regex("doc_id_pattern"))[0].nunique())
