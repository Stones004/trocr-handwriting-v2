# Handwriting OCR

A fine-tuned **PARSeq** model for reading short handwritten annotation crops. TrOCR was the original model this
project used; it was retrained on the current data, compared head-to-head against PARSeq, lost clearly (~90% vs
PARSeq's ~94% exact-match, ~6x slower on CPU), and has since been removed from the repo — see `reports/model_validation.md`
for the numbers behind that call before it was removed.

## Status

The production checkpoint is `models/parseq_final/parseq_final.ckpt` — see
["Final production model"](#final-production-model) below for how it was built and what accuracy to expect.
`docs/CODEBASE_GUIDE.md` explains the data flow and what is/isn't configurable; `docs/CODEBASE_OVERVIEW.md` has the deep file-by-file reference; this README has the workflows (overfitting
analysis, hyperparameter sweeps, document-level cross-validation, CPU throughput sizing for a 5 PDF/min target).

## Project layout

```
config.yaml                    Project settings: data source, split, folders, label rules, thresholds (read by every script via src/common/project_config.py).
config.local.yaml              YOUR machine-specific overrides (image folder, annotation file, temp drive) -- gitignored; copy config.example.local.yaml to create it.
requirements.txt                 Main venv: shared data prep only (pandas/openpyxl/pillow/pyyaml/numpy/tqdm).
requirements-parseq.txt            venv_parseq: PARSeq training, sweeps, MLflow (pinned, version-sensitive).

src/
  common/                          Shared / cross-cutting
    project_config.py                 Loads config.yaml + config.local.yaml; helpers for paths, temp/cache env, training command. Imported by every script.
    prepare_dataset.py                Builds data/processed/{train,val,test}.csv from the raw annotations. Run this once, first.
    prepare_docsplit.py               Builds a document-level split (no document split across train/val/test) from data/processed -- used by docsplit_cv.py, one fold per seed
    prepare_alldata.py                Builds data/all_data: every labeled crop in `train`, for training the final production model on everything
    validation_metrics.py, summarize_validation.py
                                       Cross-checkpoint validation: accuracy/CER per split, overfitting gaps, confidence quality -> reports/model_validation.md

  parseq/                          PARSeq (venv_parseq)
    prepare_parseq_dataset.py         Converts any data/<split-dir>/*.csv -> LMDB (PARSeq needs LMDB, not CSV); --processed-dir/--output-dir to point at a different split
    eval_parseq.py                    Batch eval against a labeled CSV; --checkpoint <.ckpt>, --mlflow-run <run> logs test metrics into MLflow
    infer_parseq.py                   Predict on one image or a whole folder (no ground truth needed)
    predict.py                        Minimal single-image inference with the production checkpoint (models/parseq_final): image in, text + confidence out
    validate_models.py                Per-sample train/val/test predictions for the validation report (see src/common/summarize_validation.py)
    sweep.py                          Optuna hyperparameter search (--space standard|aggressive); one MLflow-tracked training run per trial
    docsplit_cv.py                    Document-level cross-validation: N splits where no document appears in two sets, train + score each -> honest accuracy on unseen documents
    train_parseq.py                   Launches one training run with config.yaml's folders/temp settings applied (replaces the old .bat): --experiment <recipe> [hydra overrides]
    model_io.py                       Shared model loading, preprocessing and confidence decode used by eval/infer/predict/validate
    export_model.py                   Strips optimizer/scheduler state from a Lightning checkpoint for deployment (~4x smaller); verifies identical output

  emnist/                          EMNIST synthetic-sequence experiments (venv_parseq) -- see "EMNIST experiments" below
    extract_emnist.py                 EMNIST byclass zip -> data/emnist/glyphs.npz (digits 0-9 + lowercase a-j only)
    synth_sequences.py                Stitches glyphs into question-number crops with medium augmentation -> LMDB
    run_experiments.py                Trains + scores every EMNIST run (base models, seeds), writes reports/emnist_experiments.md
    archive_models.py, emnist_metrics.py   read-only model+data archive / shared metrics
                                       Details: docs/EMNIST_EXPERIMENTS.md

configs/
  validation_models.yaml            Registry of every checkpoint compared in reports/model_validation.md -- add an entry here to include a new checkpoint next run
  sweep.yaml                        Hyperparameter-search ranges and defaults for src/parseq/sweep.py (trial 0 = the recipe's own values)
  parseq/experiment/                 Versioned PARSeq training configs -- the source of truth for hyperparameters
    parseq_handwriting.yaml            Exploratory recipe: up to 50 epochs, early stopping on val_accuracy
    parseq_handwriting_fixed.yaml       Reproducible recipe: fixed 15 epochs, no early stopping, no SWA -- used for the final production model and all document-level CV folds
patches/parseq_train.patch         Our edits to third_party/parseq (seed, early stopping, augmentation strength, MLflow + system metrics, SWA toggle, always-save final.ckpt)
models/parseq_final/               The exported production PARSeq checkpoint (weights-only, ~95MB) -- see "Final production model" below

docs/
  CODEBASE_GUIDE.md                 Start here: script inventory, end-to-end data flow diagram, config-vs-hardcoded audit
  CODEBASE_OVERVIEW.md              Deep reference: what every file does, cross-file gotchas, benchmark results
  PARSEQ_REPLICATION.md              Runbook to set up PARSeq fine-tuning on a fresh machine
  EMNIST_EXPERIMENTS.md              EMNIST synthetic-data experiments: method, results, commands, where the retained models are

Gitignored, generated locally:
  third_party/parseq/               Vendored clone of baudm/parseq
  .venv/ (main), venv_parseq/       The two virtualenvs
  data/                             Splits (data/processed, data/docsplit/seed<N>, data/half_split, data/all_data) and their PARSeq LMDBs (rebuild with prepare_parseq_dataset.py --processed-dir/--output-dir)
  models/                           Exported deployment checkpoints (models/parseq_final/) -- rebuild with export_model.py; models/archive/ = read-only, hash-verified experiment archives (see models/archive/INDEX.md), NOT rebuildable -- back up separately
  outputs/                          New training runs (Hydra) land in outputs/parseq/<run>/ and outputs/sweeps/. The earlier runs there were deleted 2026-09-24 (numbers kept in reports/)
  reports/                          Eval/prediction CSVs, validation report, sweep results, logs/ (raw training logs)
  mlflow.db, mlruns/                MLflow tracking store (experiments: parseq, parseq_sweep, parseq_sweep_aggressive, parseq_docsplit, parseq_docsplit_fixed, parseq_final)
```

## Which venv?

| | `.venv/` (main) | `venv_parseq/` |
|---|---|---|
| Used for | Data prep | PARSeq fine-tuning + evaluating fine-tuned PARSeq checkpoints |
| Install from | `requirements.txt` | `requirements-parseq.txt` (install CUDA torch first; see [docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md)) — heavier, version-sensitive stack (Hydra, Lightning, `lmdb`, pinned NumPy) |
| Scripts | everything under `src/common/` that only needs pandas/numpy (data prep, split building) | everything under `src/parseq/`, plus `third_party/parseq/train.py` directly |

They're kept separate because PARSeq's training stack (Hydra, PyTorch Lightning, `lmdb`, `imgaug`, pinned
versions) is heavier than, and would conflict with, the main venv's lightweight data-prep dependencies — see
[docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md) for exactly why.

## 1. Shared setup (do this once)

Your data:

- Images: the folder `image_dir` in `config.yaml` / `config.local.yaml` points to
- Annotation workbook: `annotation_file` in the same config; columns named by `filename_column` (image filename) and `label_column` (training target)
- Duplicates are intentionally preserved; only rows with missing labels or missing image files are skipped

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy config.example.local.yaml config.local.yaml   # then edit image_dir / annotation_file (and env.* if C: is short on space)
python src/common/prepare_dataset.py
```

### Configuration: one place for everything tunable

All folders, data-source paths, split settings, label rules, review thresholds and validation settings live in
[config.yaml](config.yaml) (committed, portable defaults). Put values specific to your machine in
`config.local.yaml` (gitignored; template: [config.example.local.yaml](config.example.local.yaml)); it is merged on top.
Model-training hyperparameters live in `configs/parseq/experiment/*.yaml`, hyperparameter-search ranges in
`configs/sweep.yaml`, and the models compared in the validation report in `configs/validation_models.yaml`.
Command-line flags still override any of it for a single run. Relative paths are resolved against the project root, so
scripts work from any directory. Details and what remains in code: [docs/CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md).

This produces `data/processed/{train,val,test}.csv` — the input everything below trains and evaluates on.

## 2. PARSeq

One-time environment setup (separate venv, vendored repo, a couple of
version-pinned dependencies, one small patch) — follow
[docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md) top to bottom the
first time. After that's done:

```bash
venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py   # CSV -> LMDB, once

venv_parseq/Scripts/python src/parseq/train_parseq.py    # trains with configs/parseq/experiment/parseq_handwriting.yaml
# extra Hydra overrides pass through, e.g.: python src/parseq/train_parseq.py --experiment parseq_handwriting_fixed model.lr=1e-3 trainer.max_epochs=30

venv_parseq/Scripts/python src/parseq/eval_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt"

venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\completed\Doc0485_page_003_roi_1.png"
venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\completed" --output reports/predictions.csv
```

Both `eval_parseq.py` and `infer_parseq.py` also run with no `--checkpoint`
to use the *pretrained, not-fine-tuned* PARSeq weights (a zero-shot sanity
check, not the real comparison). `eval_parseq.py` needs a labeled CSV
(ground truth to score against); `infer_parseq.py` takes a single image or a
directory with no labels needed and no `--output` requirement.

### Final production model

`models/parseq_final/parseq_final.ckpt` was trained on **every** labeled crop (train+val+test combined, 3,827
rows, `data/all_data/` via `prepare_alldata.py`) using the fixed 15-epoch recipe
(`configs/parseq/experiment/parseq_handwriting_fixed.yaml` — no early stopping, since there's no held-out val
set to stop on when training on everything). It scored 99.45% on its own training data (expected — it's seen
all of it) with the 21 misses all genuine ambiguous cases (`I`/`1`, `II`/`11`, `iv`/`iii`), not corrupted output.

**Expected accuracy on new documents: ~93-94% exact-match, ~5-6% CER** — this comes from `docsplit_cv.py`
run with the *same* fixed recipe on 5 document-level splits (no document in two sets), not from the final
model's own training score (which can't measure generalization, since it saw everything):

```bash
venv_parseq/Scripts/python src/common/prepare_alldata.py
venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py --processed-dir data/all_data --output-dir data/all_lmdb
venv_parseq/Scripts/python third_party/parseq/train.py --config-dir=configs/parseq +experiment=parseq_handwriting_fixed data.root_dir=data/all_lmdb hydra.run.dir=outputs/parseq_final/run seed=42
venv_parseq/Scripts/python src/parseq/export_model.py --checkpoint outputs/parseq_final/run/checkpoints/final.ckpt --out models/parseq_final
```

Load it with `strhub.models.utils.load_from_checkpoint("models/parseq_final/parseq_final.ckpt")` — the path
must contain `parseq` (PARSeq's loader infers the model class from the path string).

### EMNIST experiments (summary)

We tried EMNIST-based synthetic data three ways (synthetic sequences only, real + synthetic sequences, real + class-balancing single
characters), with base PARSeq, to help under-represented characters and prepare for a future `NN`+letter (`02a`) label format.
**None beat the current production recipe on today's data** (base, real only: 96.8% test on the document-level fold vs 91.3-95.3% for the
synthetic-augmented variants; balancing 95.8 +- 0.7% vs 96.4 +- 0.4% over 3 seeds). The one thing synthetic data buys is `NNa` readiness
(~90% on a synthetic `NNa` probe vs ~10% for the current model), and it removes `b->6` / `g->9` there but not `c<->e`. Real `b/g/c/e/6/9` crops are
needed to settle the confusions. Full write-up, commands and tables: [docs/EMNIST_EXPERIMENTS.md](docs/EMNIST_EXPERIMENTS.md); retained models:
`models/archive/INDEX.md`.

### Is it overfitting?

Yes, visibly — and this is expected and was tracked deliberately, not overlooked. Every fine-tuning run in this
project shows the same shape: training loss drops toward ~0.01-0.09 within the first few epochs while
validation loss bottoms out early (epoch 1-5, ~0.8-1.0) and then climbs back up to ~1.0-1.3 by the end of
training. Training exact-match reaches ~99-100%; test exact-match sits at ~93-94%, a consistent ~5-6 point gap
across every run and recipe tried.

What this does and doesn't mean:

- **The rising validation loss is the model becoming overconfident on its own mistakes**, not accuracy
  collapsing — validation *accuracy* keeps improving or plateauing even while validation *loss* rises, because
  loss penalizes confident-wrong predictions much more than accuracy does. Model selection in this project uses
  the best `val_accuracy` checkpoint (or a fixed epoch count for the final model), not the lowest `val_loss`,
  specifically because of this.
- **Two hyperparameter sweeps (24 completed trials total) couldn't close the gap** — dropout, weight decay,
  augmentation strength, learning rate, and batch size were all searched
  and none reduced the train/test gap by more than noise. That's a strong signal the gap is a data-size limit
  (86 distinct labels, ~3,800 crops), not a tunable hyperparameter.
- **It doesn't block deployment.** The ~93-94% test number isn't inflated training-set performance leaking
  through — it comes from `docsplit_cv.py`, which holds out entire documents the model never saw during
  training. That's the honest generalization number, and it's the one to plan around, not the ~99% training
  score or the ~95% same-document validation score.
- **More/better labeled data is the lever that would actually close the gap further** — specifically for the
  rare and visually-confusable labels (`6`/`b`, `1`/`I`, `v`/`V`, Roman numerals) where most of the remaining
  errors concentrate. More sweep trials on the current data won't move this.

### Honest accuracy on unseen documents (document-level CV)

The default split (`data/processed`) is random per crop, so the same document can sit in train and test, which
makes test accuracy optimistic. `docsplit_cv.py` builds 5 splits where every crop of a document goes to exactly
one split, trains on each, and scores on the held-out documents:

```bash
venv_parseq\Scripts\python src\parseq\docsplit_cv.py --seeds 42 43 44 45 46                                                      # exploratory recipe, early stopping
venv_parseq\Scripts\python src\parseq\docsplit_cv.py --experiment parseq_handwriting_fixed --tag fixed --use-final --seeds 42 43 44 45 46   # fixed recipe (matches the final model)
```

| Recipe | Test acc (mean ± std) | Test CER | Results |
|---|---|---|---|
| Exploratory (early-stopped, best-val checkpoint) | 93.1% ± 1.0 | 6.2% | `reports/docsplit_cv.csv` |
| **Fixed 15-epoch (matches the final model)** | **93.9% ± 2.0** | 5.5% | `reports/docsplit_cv_fixed.csv` |

A 50/50 train/test document split (half the training data) was also spot-checked and still generalizes, at a
cost of ~3 points of accuracy — see `reports/parseq_halfsplit_test_results.csv`.

### Experiment tracking (MLflow)

Every PARSeq run logs to `mlflow.db` (experiment `parseq`, or `parseq_sweep`/`parseq_docsplit`/`parseq_final`
etc. for sweeps and other pipelines): hyperparameters, per-epoch loss/accuracy, **system metrics** (CPU, RAM,
disk, GPU utilization/memory/power, sampled every 5 s -- set `system_metrics_interval=null` to turn off), the
resolved config, and the best checkpoint path. View it with:

```bash
venv_parseq\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://127.0.0.1:5001 (hard-refresh with Ctrl+F5 if a new run doesn't appear). To attach test-set
results to a run: `eval_parseq.py --checkpoint <.ckpt> --mlflow-run <run folder name>`.

### Hyperparameter search

```bash
venv_parseq\Scripts\python src\parseq\sweep.py --n-trials 15                       # standard space
venv_parseq\Scripts\python src\parseq\sweep.py --space aggressive --n-trials 25    # much wider ranges
```

Optuna (TPE) searches learning rate, dropout, batch size, augmentation strength/depth, weight decay, warmup,
and (aggressive space only) permutations and gradient clipping. Trial 0 is always
the current config, as the baseline to beat. Each trial is a full training run (early-stopped), tracked in
MLflow; only the best trial keeps its checkpoint. Results land in `reports/sweep_<study>.csv`. Resumable:
rerun with the same `--study` (add `--retry-failed` to re-run trials that failed, e.g. from low disk space —
both sweeps have a `--min-free-gb` guard that stops cleanly rather than crashing mid-trial). **Neither sweep
beat the hand-tuned config by more than noise** — see `reports/model_validation.md`; more/better data is
the lever that will move accuracy further, not more hyperparameter tuning.

### CPU inference speed / throughput sizing

Measured earlier with a CPU benchmark script (since removed; raw numbers are in `reports/cpu_benchmark.jsonl`). On this dev machine (i7-13650HX): base
PARSeq is ~62ms/image single-threaded (16 img/s), ~25ms at 4 threads (40 img/s). For a target of 5 PDFs/min at
25 pages/PDF, the required throughput is only ~6 ROIs/sec (using this project's own data's ROI density as an
estimate) — PARSeq clears that with several times the needed margin on a single CPU thread. The bottleneck at
that target is more likely PDF rasterization / ROI detection upstream of the model, not PARSeq itself.

## Cross-checkpoint validation report

`reports/model_validation.md` (and `.csv`) compares every checkpoint in `configs/validation_models.yaml` on
train/val/test: accuracy, CER, overfitting gap, confidence calibration (ECE, AUROC), flagging quality, rare-label
accuracy, and CPU/GPU speed. Regenerate after adding a checkpoint to the registry:

```bash
venv_parseq\Scripts\python src\parseq\validate_models.py       # scores every entry in configs/validation_models.yaml
venv_parseq\Scripts\python src\common\summarize_validation.py   # writes reports/model_validation.{csv,md}
```

## Reshuffling the split (robustness check)

Regenerates `data/processed/{train,val,test}.csv` with a different random
train/val/test split, so you can retrain and confirm a result wasn't just an
artifact of one particular split. Steps, in order:

```bash
# 1. Back up the current split first -- data/ is gitignored, this is the only copy
mkdir data\processed_seed42_backup
copy data\processed\*.csv data\processed_seed42_backup\

# 2. Edit config.yaml's split.seed to a new number, then regenerate the CSVs
.venv\Scripts\activate.bat
python src\common\prepare_dataset.py

# 3. Rebuild the PARSeq LMDB from the new CSVs (overwrites data/parseq_lmdb in place)
venv_parseq\Scripts\activate.bat
python src\parseq\prepare_parseq_dataset.py

# 4. Retrain PARSeq on the new split (same command as the normal fine-tune above)
python src/parseq/train_parseq.py

# 5. Find the best checkpoint (highest val_accuracy in the filename) in the newest outputs/parseq/<run>/checkpoints/ folder, then evaluate it
python src\parseq\eval_parseq.py --checkpoint "outputs\parseq\<newest-run>\checkpoints\<best>.ckpt" --output reports\my_results.csv

# 6. Sanity-check there's no leakage between the new train/test split
python -c "import pandas as pd; a=pd.read_csv('data/processed/train.csv'); b=pd.read_csv('data/processed/test.csv'); print('overlap:', len(set(a.image_path)&set(b.image_path)))"
```

`overlap` should print `0`. Then open `reports\my_results.csv` and filter to
`exact_match == False` — if the mistakes still look like real character
confusions (`3`/`8`, `v`/`V`, etc.) rather than nonsense, the model is
genuinely working rather than having gotten lucky on one split. A few points
of accuracy swing between different random splits is normal on a dataset
this size and not a sign anything is broken. Prefer `docsplit_cv.py` (above)
over this for a real generalization estimate — a random per-crop reshuffle
still lets the same document appear in both train and test.

## Bonus: zero-shot baseline comparison

Sanity-checks how the *pretrained* (not fine-tuned) PARSeq weights do, mainly to see whether their
built-in confidence scores look useful before investing further, and as a lower-bound reference point against
the fine-tuned numbers in `reports/model_validation.md`. Run with `venv_parseq`:

```bash
venv_parseq/Scripts/python src/parseq/eval_parseq.py
```

It prints a summary and accepts `--output reports/results.csv`. Expect the fine-tuned model to
win by a wide margin — see [docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md) for the historical zero-shot
numbers and `reports/model_validation.md` for the current fine-tuned comparison.
