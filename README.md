# Handwritten question-number recognition (PARSeq)

Reads the handwritten question numbers in the margins of scanned answer sheets (`2`, `11i`, `[2 a].`, `Q.3`, Roman numerals, ...). The model is
PARSeq (scene-text recogniser) fine-tuned on real crops plus synthetic crops. The upstream pipeline that finds and crops the margin regions
lives in `D:\OCR_codes` and loads the model from this repo.

**Models: `models/parseq_master_v3/parseq_master_v3.ckpt` (in use by OCR_codes) and `models/parseq_master_v4/parseq_master_v4.ckpt` (retrained on the label-reviewed dataset; scores the same within noise); `models/parseq_balanced_run/parseq_balanced_run.ckpt` is an experiment trained with synthetic crops only for scarce labels (96% real test, better general words; see MODEL_REPORT 8.13).** Everything about it (architecture, recipe, data, results, how to rebuild it) and a
ledger of every other experiment is in [docs/MODEL_REPORT.md](docs/MODEL_REPORT.md). Raw backing material: `docs/log/` (MLflow export of all runs,
result tables, the training config).

## Layout

```
config.yaml                  the ONE config: data paths, label rules, inference defaults, training recipe (`train:`)
data/master_dataset.mdb      THE dataset: every real + synthetic crop in a single file (images embedded)
data/docsplit/seed42/        the document-level split (train/val/test CSVs) used for all comparisons
data/emnist/glyphs_*.npz     glyph banks for the synthetic renderer
data/noise_classifier/       noise/clean crop CSVs (the noise classifier itself is not in this repo any more)
data/open_read/              evaluation sets of the open-reading experiment (word test, junk, verbatim labels)
models/parseq_master_v3/     the deployable weights
patches/parseq_train.patch   our edits to third_party/parseq (seed, augmentation strength, MLflow, swa switch, final.ckpt)
src/master/                  synthetic data generator (spec, render, build), dataset_io (pack/unpack the single file), eval_master
src/parseq/                  train_parseq, eval_parseq, infer_parseq, predict, export_model, model_io, prepare_parseq_dataset
src/common/                  project_config (reads config.yaml, builds the training command), label_grammar, validation_metrics
src/emnist/                  synth_sequences (used by the renderer)
src/noise_classifier/        ResNet-18 noise/clean classifier (train, evaluate, predict)
src/open_read/               open-reading experiment (not adopted; see the report, section 8.9)
docs/                        MODEL_REPORT.md, PARSEQ_REPLICATION.md (environment setup), log/ (archive)
```

## Setup

Two environments: the main one (`requirements.txt`) and `venv_parseq` (`requirements-parseq.txt`, installs CUDA torch first). `third_party/parseq`
is a clone of baudm/parseq with `patches/parseq_train.patch` applied. Step by step: [docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md).

Venvs, Chandra weights, experiment LMDBs and intermediate checkpoints were deleted on 2026-10-19 to save ~40 GB; how to rebuild each is in [docs/CLEANUP_2026-10-19.md](docs/CLEANUP_2026-10-19.md) (pinned package lists in `docs/env/`).

## Everyday commands (run with `venv_parseq/Scripts/python` from the project root)

```bash
# score a model on any labelled CSV (filename,image_path,text)
python src/parseq/eval_parseq.py --csv data/docsplit/seed42/test.csv --checkpoint models/parseq_master_v3/parseq_master_v3.ckpt

# read one image or a folder (no labels needed)
python src/parseq/infer_parseq.py --checkpoint models/parseq_master_v3/parseq_master_v3.ckpt --input <image_or_folder>

# retrain the kept model from the single dataset file
python src/master/dataset_io.py unpack                       # data/master_dataset.mdb -> data/master_lmdb (PARSeq layout)
python src/parseq/train_parseq.py hydra.run.dir=outputs/parseq_master/run_v3
python src/parseq/export_model.py --checkpoint outputs/parseq_master/run_v3/checkpoints/final.ckpt --out models/parseq_master_v3
```

Training settings are `config.yaml` -> `train:`; any value can be overridden on the command line in Hydra syntax
(`python src/parseq/train_parseq.py model.lr=1e-3 +seed=7`). The saved path must contain `parseq` (PARSeq infers the model class from it).

## Reading the numbers

Expect about 94-96% exact match on the real document-split test and about 94% across other document splits; differences under about one
point are noise (identical-seed reruns differ). Details and caveats: section 5 and 8 of the report.
