# Codebase Overview

Reference document describing the `src/` pipeline for this project, intended to give an LLM (or a new contributor) fast context on what each file does. See [README.md](../README.md) for setup/run instructions.

## What this project is

A fine-tuning pipeline for **TrOCR** (`microsoft/trocr-small-handwritten`, a Vision Encoder-Decoder model) on short handwritten text crops. Pipeline:

```
annotations.xlsx + images  →  prepare_dataset.py  →  train/val/test.csv
                                                            ↓
                                                       train.py  →  checkpoints/
                                                                          ↓
                                          evaluate.py / error_analysis.py / inference.py
```

## config.yaml

Central config consumed by most scripts via `--config` (default `config.yaml`):

- `model_name`: `microsoft/trocr-small-handwritten`
- `image_dir`, `annotation_file`: paths to source images and the Excel annotation workbook (currently absolute Windows paths outside the repo)
- `label_column` / `filename_column`: which columns in the annotation file hold the text and filename
- `split`: train/val/test ratios (0.80 / 0.10 / 0.10) and shuffle `seed` (42)
- `training`: batch_size 8, eval_batch_size 8, learning_rate 3e-5, weight_decay 0.01, epochs 10, warmup_ratio 0.10, gradient_accumulation_steps 2, `max_target_length: 16`, num_workers 2, fp16 true, seed 42, save_dir `checkpoints`

## src/common/prepare_dataset.py

Builds the train/val/test CSVs from the raw annotation workbook.

- Reads the Excel file, renames the configured filename/label columns to `filename`/`text`.
- **Intentionally preserves duplicate rows** — comment explicitly says `DO NOT use drop_duplicates()`.
- Drops rows with missing/empty labels (tracked as `missing_label_count`).
- Separately filters out rows labeled `"ambigous"`/`"ambiguous"` (a reviewer/annotation marker, not real ground truth) into `data/processed/ambiguous.csv` — these are excluded from training.
- Verifies every referenced image file exists; missing ones are excluded and logged to `data/processed/missing_images.csv`.
- Shuffles remaining rows with the configured seed, splits by the configured ratios, and writes `data/processed/{train,val,test}.csv` with columns `filename, image_path, text`.
- Prints a full summary (counts at each filtering stage) to stdout.

## src/trocr/dataset.py

`HandwritingDataset(Dataset)`: loads one row (`image_path`, `text`) at a time.

- Opens the image, converts to RGB, runs it through `TrOCRProcessor` → `pixel_values`.
- Tokenizes `text` (padded/truncated to `max_target_length`, default 32 in this file), then replaces pad-token ids with `-100` so they're ignored in the loss.
- `collate_fn`: stacks a list of samples into batched tensors for `DataLoader`.

## src/trocr/train.py

Manual PyTorch training loop (does not use HuggingFace `Trainer`).

- `configure_trocr(model, processor)`: sets `decoder_start_token_id` (CLS, falling back to BOS), `pad_token_id`, and `eos_token_id` (SEP) on the model config — required for TrOCR generation/training to work correctly.
- `set_seed`: seeds `random`, `numpy`, and `torch` (+ CUDA) for reproducibility.
- Loads `TrOCRProcessor` / `VisionEncoderDecoderModel` from `model_name`, builds train/val `HandwritingDataset`s from `data/processed/{train,val}.csv`.
- AdamW optimizer, cosine LR schedule with warmup (`get_cosine_schedule_with_warmup`), gradient accumulation, gradient clipping (`max_norm=1.0`), AMP/fp16 autocast + `GradScaler` when CUDA is available.
- Per epoch: trains one pass, computes val loss via `evaluate_loss`, saves a checkpoint to `checkpoints/epoch-N/`, and overwrites `checkpoints/best/` whenever val loss improves.
- Prints device/GPU info, training config, per-epoch metrics, and a final summary.

## src/trocr/evaluate.py

Compact CLI for headline metrics on a CSV split (default `data/processed/test.csv`, model default `checkpoints/best`).

- Runs beam search (`num_beams=4`, `max_new_tokens=32`) per sample.
- Computes **exact-match accuracy** and **CER (character error rate)** using a small hand-rolled Levenshtein distance function (`lev`).
- Very condensed style (single-line loops) compared to the rest of the codebase.

## src/trocr/error_analysis.py

More detailed sibling of `evaluate.py`, same default model/CSV args plus `--output error_analysis.csv` and `--beams`.

- `levenshtein_ops`: full DP edit-distance table with backtracking to classify each edit as a substitution, deletion, or insertion (not just total distance).
- Runs beam search (`max_new_tokens=16`) over every test row, logs per-sample GT vs. prediction to stdout with OK/ERROR markers.
- Saves a full per-sample results table (`error_analysis.csv`) with target, prediction, exact_match, edit_distance, and per-type edit counts.
- Prints a summary: exact-match rate, CER, totals for substitutions/deletions/insertions, and a listing of every failed sample.

## src/trocr/inference.py

Minimal single-image CLI. Loads a checkpoint (default `checkpoints/best`), runs beam search (`--num-beams`, default 4) on one `--image`, prints the decoded prediction. No batching, no metrics.

## src/trocr/sanity_check.py

Debug/smoke-test script — **uses the base pretrained model**, not a fine-tuned checkpoint. Loads 20 rows from `data/processed/train.csv`, builds a tiny batch, and walks through the full mechanics end-to-end: processor load → model load → forward pass → loss → backward pass → `generate()`. Useful for confirming the environment (GPU, dependencies, tensor shapes) works before kicking off a real training run.

## src/trocr/test_preprocessing.py

Experimental image-preprocessing comparison tool, independent of the training/eval pipeline.

- `unsharp`: unsharp-mask sharpening on a grayscale image.
- `resize_with_padding`: aspect-ratio-preserving resize onto a square white canvas (default 384×384) — avoids the distortion of a naive stretch-to-square resize.
- `make_variants`: generates ~14 preprocessing variants of one input image — grayscale, mild/strong sharpen, CLAHE, CLAHE+sharpen, bilateral filter (+sharpen), Otsu threshold, adaptive threshold, plus aspect-ratio-padded versions of several of the above.
- `predict`: runs a given TrOCR model/processor on one BGR image and returns the decoded string.
- `main`: loads an image via `--image`, generates all variants, runs each through the model (`--model`, default `checkpoints/best`), saves every variant to `--save-dir` (default `preprocess_test_results/`), and if `--target` is given, reports which variant(s) produced an exact match. Used to empirically decide which preprocessing to standardize on.

## src/parseq/eval_parseq.py / src/compare/eval_ppocr.py

Standalone zero-shot baselines for comparing against the fine-tuned TrOCR model, using each library's pretrained (not fine-tuned) weights on `data/processed/test.csv` (or any `--csv`). Not part of the training pipeline — require `requirements-ocr-compare.txt` (paddlepaddle, paddleocr, timm, pytorch-lightning, nltk).

- `eval_parseq.py`: loads `baudm/parseq` (94-char pretrained checkpoint) via `torch.hub`, replicates the repo's standard preprocessing (resize to the model's `img_size`, normalize) directly rather than importing `strhub.data.module` (which pulls in an `lmdb` dependency this project doesn't otherwise need). Confidence is per-character softmax probability from `model.tokenizer.decode` (min/mean across the predicted sequence, EOS included) — same shape as `constraints.py`'s `transition_confidence` for TrOCR.
- `eval_ppocr.py`: uses `paddleocr.TextRecognition` (recognition-only — no detection, since crops are already single-token ROIs) with `PP-OCRv5_server_rec` by default. PP-OCR returns one sequence-level `rec_score` per prediction rather than per-character probabilities.
- Both print the same summary format as `evaluate.py` (samples, exact-match accuracy, CER, flagged-for-review count at `--flag-threshold`) and support `--output results.csv` for a per-sample table.

## src/parseq/prepare_parseq_dataset.py

Converts `data/processed/{train,val,test}.csv` into the LMDB format PARSeq's own training pipeline (`third_party/parseq`, cloned from `baudm/parseq`, gitignored) expects. Writes one LMDB per split to `data/parseq_lmdb/{split}/handwriting/` (image path + label pairs; our `image_path` column is already absolute so no base path is needed). Deliberately reimplements the write loop instead of calling `third_party/parseq/tools/create_lmdb_dataset.py`'s `createDataset` directly — that script hardcodes `map_size=1TB`, which is harmless on Linux but makes `lmdb.open()` fail on Windows ("the paging file is too small for this operation to complete", since Windows tries to back the huge mapping with real page-file-committed memory); our script uses a 512MB map size instead (`--map-size-mb`), generous headroom for our ~9MB dataset. Must be run with `venv_parseq` (needs `lmdb`, not part of the main project venv) — see "Fine-tuning PARSeq" below.

## Fine-tuning PARSeq (separate venv)

A dedicated venv (`venv_parseq/`, gitignored) and a vendored clone of `baudm/parseq` (`third_party/parseq/`, gitignored) exist alongside the main TrOCR pipeline, kept isolated because PARSeq's training stack (Hydra, PyTorch Lightning, `lmdb`, `imgaug`, pinned versions) is unrelated to and heavier than what TrOCR fine-tuning needs. `venv_parseq` pins `numpy<2` and `opencv-python-headless==4.9.0.80` specifically — `imgaug` (last released 2020, used for PARSeq's built-in `RandAugment`) is incompatible with NumPy 2.x's removed `np.sctypes`, and newer `opencv-python(-headless)` releases require NumPy 2.x, so the two constraints have to be satisfied together.

Dataset readiness for PARSeq (Hydra + `strhub.data.module.SceneTextDataModule`) requires:

1. **LMDB format, not CSV.** `strhub`'s `LmdbDataset`/`build_tree_dataset` glob for `data.mdb` files under a root dir — `src/parseq/prepare_parseq_dataset.py` handles this conversion from our existing CSVs.
2. **`data.root_dir`/`data.train_dir`** pointed at the converted LMDBs (`data/parseq_lmdb`, `handwriting`).
3. **Charset: leave it at the pretrained default, don't narrow it.** Our labels are a small, case-sensitive, closed set (verified across all 1719 rows: `).0123456789ABIVXabdiv` — parentheses/digits/roman-numeral letters/a handful of lowercase, no spaces, max length 4), and it's tempting to set `model.charset_train`/`charset_test` to exactly that. Don't — the pretrained checkpoint's classification head, token embedding, and positional-query tensors are all sized from the *default* 94-char charset and `max_label_length=25` (confirmed by trying it: shrinking the charset produces `size mismatch for head.weight: ... [95, 384] vs [23, 384]` etc. on `load_state_dict`, since those layers would need to be randomly reinitialized, throwing away the pretrained classifier). Since our label alphabet is a strict subset of the default 94-char charset, leaving `charset_train`/`charset_test`/`max_label_length` at their defaults costs nothing (`LmdbDataset` filters labels against the charset at read time, and every one of our labels already passes) while keeping the pretrained head intact. This mirrors how the TrOCR side handles the same narrow-vocab situation: `constraints.py` restricts *decoding*, not the tokenizer's actual vocab size.

Verified: loading all three converted LMDBs through `strhub.data.dataset.build_tree_dataset` with the default charset recovers all 1375/171/173 train/val/test samples with zero filtering loss, and `pretrained=parseq` loads with no shape mismatches when the charset config is left untouched.

Fine-tuning command (from the project root, using `venv_parseq`), starting from the pretrained checkpoint rather than training from scratch — smoke-tested at 1 epoch (43 steps in ~16s on an RTX 5060 Ti) before committing to a full run:

```bash
venv_parseq/Scripts/python third_party/parseq/train.py \
  +experiment=parseq \
  pretrained=parseq \
  data.root_dir="<absolute path>/data/parseq_lmdb" \
  data.train_dir=handwriting \
  data.num_workers=0 \
  model.batch_size=32 \
  trainer.accelerator=gpu trainer.devices=1 \
  trainer.val_check_interval=1.0 \
  trainer.max_epochs=50
```

Notes on the choices: `data.root_dir` must be an absolute path — Hydra changes the working directory to its own timestamped output dir at runtime, so a relative path resolves against the wrong location. `data.num_workers=0` sidesteps Windows multiprocessing/LMDB-handle-pickling overhead, a non-issue at this dataset size. `trainer.val_check_interval=1.0` overrides the config default of `1000` (steps), which would never trigger within a run this short (~43 steps/epoch) — `1.0` means "once per epoch". `model.batch_size=32` was chosen mainly for its side effect: PARSeq scales effective LR linearly with batch size (`lr_scale = batch_size / 256`, see `strhub/models/base.py`'s `configure_optimizers`), so dropping from the from-scratch default of 384 to 32 brings the default `lr=7e-4` down to an effective ~9e-5 — a sensible fine-tuning rate — without touching `model.lr` directly. Checkpointing (top-3 by `val_accuracy`, plus `last.ckpt`) and per-epoch TensorBoard logging under `outputs/parseq/<date>_<time>/` are handled automatically by `train.py`, no extra plumbing needed.

Result of the first real run (50 epochs, ~2 minutes total on an RTX 5060 Ti, encoder left unfrozen): best checkpoint at epoch 20 (`val_accuracy=95.32`). Evaluated on `data/processed/test.csv` via `eval_parseq.py --checkpoint <that .ckpt>` (same methodology as `evaluate.py`, so directly comparable):

| Model | Exact-match | CER |
|---|---|---|
| PARSeq, fine-tuned (epoch 20) | **93.1%** | **5.9%** |
| TrOCR, fine-tuned (`checkpoints/best`, unconstrained) | 86.7% | 11.3% |

PARSeq's fine-tune beat the fine-tuned TrOCR checkpoint on the same held-out test set on the first attempt, with no hyperparameter search and no encoder freezing. Worth noting one infra gotcha hit along the way: the first attempt at this run crashed mid-training with `OSError: [Errno 28] No space left on device` while saving a checkpoint, despite the project's `G:` drive reporting hundreds of GB free — `C:` was at the time down to ~1.8GB free out of ~476GB, and something in the Windows/CUDA/Lightning stack apparently needs headroom there even when the actual output path is elsewhere (pagefile growth is the leading suspect, unconfirmed). A retry succeeded without any config changes. If this recurs, check `C:` free space first.

## Cross-file notes

- **`max_target_length` / `max_new_tokens` is inconsistent across files**: `config.yaml` and `sanity_check.py` use 16, `dataset.py`'s own default is 32, and `inference.py`/`evaluate.py`'s `generate()` calls use `max_new_tokens=32` while `error_analysis.py`/`test_preprocessing.py` use 16. Worth reconciling if label length handling is touched.
- Checkpoints are self-contained: each `checkpoints/epoch-N/` and `checkpoints/best/` directory holds both the model and the processor (`save_pretrained` for both), so any script can load a checkpoint standalone via `--model <path>`.
- All eval-style scripts (`evaluate.py`, `error_analysis.py`, `inference.py`, `test_preprocessing.py`) default `--model` to `checkpoints/best`.
