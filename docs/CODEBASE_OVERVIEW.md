# Codebase Overview

Reference document describing the `src/` pipeline for this project, intended to give an LLM (or a new contributor) fast context on what each file does. See [README.md](../README.md) for setup/run instructions, current status, and the newer PARSeq-only workflows (final production model, document-level cross-validation, hyperparameter sweeps, cross-model validation report) not covered in depth below.

## What this project is

A fine-tuning pipeline for **PARSeq** (`baudm/parseq`, a permutation-autoregressive scene-text recognition model) on short handwritten text crops. TrOCR (`microsoft/trocr-small-handwritten`) was the original model this project used, but it was retrained on the current data, compared head-to-head against PARSeq (accuracy, overfitting, CPU speed), lost clearly, and was removed from the codebase — see the "Status" section of [README.md](../README.md). Pipeline:

```
annotations.xlsx + images  →  prepare_dataset.py  →  train/val/test.csv
                                                            ↓
                                          prepare_parseq_dataset.py  →  LMDB
                                                            ↓
                                      third_party/parseq/train.py  →  outputs/parseq/<run>/
                                                            ↓
                                    eval_parseq.py / infer_parseq.py / export_model.py
```

## config.yaml

Central config, loaded by every script through `src/common/project_config.py` (config.yaml, then the gitignored config.local.yaml on top). Besides the data source it defines all folders (`paths`), temp/cache settings (`env`), label rules (`labels`), LMDB packing, document-split settings, inference/review thresholds and validation-report settings. See docs/CODEBASE_GUIDE.md. The keys consumed by `src/common/prepare_dataset.py`:

- `image_dir`, `annotation_file`: paths to source images and the Excel annotation workbook (portable defaults are committed; your real paths go in the gitignored `config.local.yaml`)
- `label_column` / `filename_column`: which columns in the annotation file hold the text and filename
- `split`: train/val/test ratios (0.80 / 0.10 / 0.10) and shuffle `seed` (25)

## src/common/prepare_dataset.py

Builds the train/val/test CSVs from the raw annotation workbook.

- Reads the Excel file, renames the configured filename/label columns to `filename`/`text`.
- **Intentionally preserves duplicate rows** — comment explicitly says `DO NOT use drop_duplicates()`.
- Drops rows with missing/empty labels (tracked as `missing_label_count`).
- Separately filters out rows labeled `"ambigous"`/`"ambiguous"` (a reviewer/annotation marker, not real ground truth) into `data/processed/ambiguous.csv`, and rows that don't match the closed symbol grammar (`CLEAN_PATTERN`: digit runs with an optional trailing subpart letter, Roman numerals, bare subpart letters) into `data/processed/freetext.csv` — both excluded from training.
- Verifies every referenced image file exists; missing ones are excluded and logged to `data/processed/missing_images.csv`.
- Shuffles remaining rows with the configured seed, splits by the configured ratios, and writes `data/processed/{train,val,test}.csv` with columns `filename, image_path, text`.
- Prints a full summary (counts at each filtering stage) to stdout.

## src/parseq/eval_parseq.py

Standalone zero-shot baseline for comparing against this project's fine-tuned PARSeq model, using the pretrained (not fine-tuned) weights on `data/processed/test.csv` (or any `--csv`). Not part of the training pipeline; runs in `venv_parseq`. `eval_parseq.py` also doubles as the main evaluation CLI for fine-tuned checkpoints via `--checkpoint`.

- `eval_parseq.py`: with no `--checkpoint`, loads `baudm/parseq` (94-char pretrained checkpoint) via `torch.hub`, replicating the repo's standard preprocessing (resize to the model's `img_size`, normalize) directly rather than importing `strhub.data.module` (which pulls in an `lmdb` dependency not needed for eval). With `--checkpoint <.ckpt>`, loads a fine-tuned Lightning checkpoint via `strhub.models.utils.load_from_checkpoint` instead (needs `venv_parseq`). Confidence is per-character softmax probability from `model.tokenizer.decode` (min/mean across the predicted sequence, EOS included). `--mlflow-run <run>` logs test metrics onto an existing MLflow run.

## src/parseq/prepare_parseq_dataset.py

Converts `data/<split-dir>/{train,val,test}.csv` (default `data/processed`) into the LMDB format PARSeq's own training pipeline (`third_party/parseq`, cloned from `baudm/parseq`, gitignored) expects. Writes one LMDB per split to `<output-dir>/{split}/handwriting/` (image path + label pairs; our `image_path` column is already absolute so no base path is needed). `--processed-dir`/`--output-dir` let it target a different split (e.g. the document-level CV folds, or the all-data set for the final model). Deliberately reimplements the write loop instead of calling `third_party/parseq/tools/create_lmdb_dataset.py`'s `createDataset` directly — that script hardcodes `map_size=1TB`, which is harmless on Linux but makes `lmdb.open()` fail on Windows ("the paging file is too small for this operation to complete", since Windows tries to back the huge mapping with real page-file-committed memory); our script uses a 512MB map size instead (`--map-size-mb`), generous headroom for our ~9MB dataset. Must be run with `venv_parseq` (needs `lmdb`, not part of the main project venv).

## Fine-tuning PARSeq (separate venv)

A dedicated venv (`venv_parseq/`, gitignored) and a vendored clone of `baudm/parseq` (`third_party/parseq/`, gitignored) exist because PARSeq's training stack (Hydra, PyTorch Lightning, `lmdb`, `imgaug`, pinned versions) is heavier than, and would conflict with, the main project venv's data-prep-only dependencies. `venv_parseq` pins `numpy<2` and `opencv-python-headless==4.9.0.80` specifically — `imgaug` (last released 2020, used for PARSeq's built-in `RandAugment`) is incompatible with NumPy 2.x's removed `np.sctypes`, and newer `opencv-python(-headless)` releases require NumPy 2.x, so the two constraints have to be satisfied together. See `requirements-parseq.txt` for the full pinned set (also includes MLflow, Optuna, `psutil`, `nvidia-ml-py` for experiment tracking/sweeps).

Dataset readiness for PARSeq (Hydra + `strhub.data.module.SceneTextDataModule`) requires:

1. **LMDB format, not CSV.** `strhub`'s `LmdbDataset`/`build_tree_dataset` glob for `data.mdb` files under a root dir — `src/parseq/prepare_parseq_dataset.py` handles this conversion from our existing CSVs.
2. **`data.root_dir`/`data.train_dir`** pointed at the converted LMDBs (`data/parseq_lmdb`, `handwriting`).
3. **Charset: leave it at the pretrained default, don't narrow it.** Our labels are a small, case-sensitive, closed set (verified across all rows), and it's tempting to set `model.charset_train`/`charset_test` to exactly that. Don't — the pretrained checkpoint's classification head, token embedding, and positional-query tensors are all sized from the *default* 94-char charset and `max_label_length=25` (confirmed by trying it: shrinking the charset produces `size mismatch for head.weight: ... [95, 384] vs [23, 384]` etc. on `load_state_dict`, since those layers would need to be randomly reinitialized, throwing away the pretrained classifier). Since our label alphabet is a strict subset of the default 94-char charset, leaving `charset_train`/`charset_test`/`max_label_length` at their defaults costs nothing (`LmdbDataset` filters labels against the charset at read time, and every one of our labels already passes) while keeping the pretrained head intact.

Fine-tuning command (from the project root, using `venv_parseq`) — the versioned recipe lives in `configs/parseq/experiment/` (`parseq_handwriting.yaml` for exploratory runs with early stopping, `parseq_handwriting_fixed.yaml` for a reproducible fixed-epoch run), applied via `patches/parseq_train.patch` on top of the vendored `third_party/parseq/train.py`:

```bash
python src/parseq/train_parseq.py
# or directly:
venv_parseq/Scripts/python third_party/parseq/train.py --config-dir=configs/parseq +experiment=parseq_handwriting
```

`data.root_dir` must be an absolute path or resolved against the launch directory — Hydra changes the working directory to its own timestamped output dir at runtime. `model.batch_size` matters beyond throughput: PARSeq scales effective LR linearly with batch size (`lr_scale = batch_size / 256`, see `strhub/models/base.py`'s `configure_optimizers`), so a batch size well below the from-scratch default of 384 brings the effective LR down substantially without touching `model.lr` directly — a sensible fine-tuning rate almost for free. Checkpointing (top-3 by `val_accuracy`, plus `last.ckpt`/`final.ckpt`) and per-epoch TensorBoard + MLflow logging under `outputs/parseq/<date>_<time>/` are handled automatically, no extra plumbing needed.

**Historical note:** the very first fine-tuning run (50 epochs, before any hyperparameter search) already beat the then-existing fine-tuned TrOCR checkpoint on the held-out test set, with no tuning and no encoder freezing (93.1% vs 86.7% exact-match). See `reports/model_validation.md` for the full, current, multi-checkpoint comparison (this project's own PARSeq variants plus the zero-shot baseline) that superseded that single-run number. One infra gotcha hit along the way, worth remembering if it recurs: a run can crash mid-training with `OSError: [Errno 28] No space left on device` while saving a checkpoint even when the drive holding the project has plenty of free space — Windows grows `pagefile.sys` on the *system* drive (`C:`) during training, and that's what actually runs out. Check `C:` free space first if this happens; `scripts\train_parseq.bat` redirects `TMP`/`TEMP`/`TORCH_HOME` off `C:` to reduce the risk, but the pagefile itself isn't relocatable from a script.

## src/emnist/ (EMNIST synthetic-data experiments)

Self-contained experiment tooling; full method and results in [EMNIST_EXPERIMENTS.md](EMNIST_EXPERIMENTS.md). All run with `venv_parseq`.

- `extract_emnist.py`: EMNIST `byclass` zip -> `data/emnist/glyphs*.npz` (chosen characters, un-transposed).
- `synth_sequences.py`: stitches glyphs into crops (templates D/DD/L/DL/DDL/`ii`, per-glyph + per-crop augmentation, clutter marks, paper/ink noise) and writes a PARSeq LMDB; recipes `v1`/`v2`, `--labels-csv` renders exact labels, `--only-template DDL --dump-dir` builds the `NNa` eval set. Deterministic per (seed, index).
- `run_experiments.py`: trains (Hydra, `configs/parseq/experiment/`) and scores every registered run on `data/docsplit/seed42` + the `NNa` set, writes `reports/emnist_experiments.md`. Resumable via `outputs/parseq_emnist/<run>/checkpoints/final.ckpt`.
- `archive_models.py`: exports weights, copies checkpoints/config/logs/predictions/datasets/code into a read-only folder with sha256 `MANIFEST.json`, verifying the export reproduces the recorded test accuracy.
- `emnist_metrics.py`: shared accuracy/CER/label-kind helpers over `eval_parseq.py`'s per-sample CSVs.

Also: `src/parseq/predict.py` is the minimal single-image inference entry point on the production checkpoint (`models/parseq_final`).
