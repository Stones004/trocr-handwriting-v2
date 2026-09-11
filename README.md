# Handwriting OCR

Two OCR models trained on the same short handwritten annotation crops, so
they can be compared directly: a fine-tuned **TrOCR** (the original model)
and a fine-tuned **PARSeq** (added later, currently the stronger of the two —
see [docs/CODEBASE_OVERVIEW.md](docs/CODEBASE_OVERVIEW.md) for the numbers).
A third, [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR), is wired up
only as a zero-shot comparison baseline, not as a model this project trains.

## Project layout

```
config.yaml                    Shared: data paths + split ratios. Its `training:` block is TrOCR-only.
requirements.txt                 Main venv: TrOCR + shared data prep.
requirements-ocr-compare.txt       Main venv, optional: zero-shot PARSeq/PP-OCRv5 comparison scripts.

src/
  common/                          SHARED by both models
    prepare_dataset.py                Builds data/processed/{train,val,test}.csv from the raw annotations. Run this once, first, regardless of which model you use.
    check_pdf_resolution.py           Upstream diagnostic for the raw PDF→crop pipeline. Not part of either model.

  trocr/                           TrOCR-only (main venv)
    dataset.py                        HandwritingDataset, resize_with_padding
    constraints.py                    Closed-vocab decoding + confidence scoring
    train.py                          Fine-tunes microsoft/trocr-small-handwritten
    evaluate.py                       Batch eval on a CSV split -> exact-match, CER
    inference.py                      Predict on one image or a whole folder (no ground truth needed)
    error_analysis.py                 Per-sample error breakdown
    calibrate_temperature.py          Confidence calibration
    sanity_check.py, test_preprocessing.py, test_unsharp.py, check_resolution.py, class_balance_vs_errors.py
                                       Debug/analysis utilities, not part of the main train/infer path

  parseq/                          PARSeq-only (separate venv_parseq — see below)
    prepare_parseq_dataset.py         Converts data/processed/*.csv -> LMDB (PARSeq needs LMDB, not CSV)
    eval_parseq.py                    Batch eval against a labeled CSV, zero-shot OR --checkpoint <fine-tuned .ckpt>
    infer_parseq.py                   Predict on one image or a whole folder (no ground truth needed), same flags as eval_parseq.py

  compare/                         Zero-shot baseline only, not trained here (main venv)
    eval_ppocr.py                     PP-OCRv5 recognition, pretrained weights only

third_party/parseq/               Vendored clone of baudm/parseq (gitignored) -- PARSeq's own train.py/configs live here
venv/, venv_parseq/                Two separate virtualenvs (gitignored) -- see "Which venv?" below
checkpoints/                      TrOCR checkpoints (gitignored)
outputs/parseq/                   PARSeq checkpoints + logs, Hydra-managed (gitignored)
docs/
  CODEBASE_OVERVIEW.md              Deep reference: what every file does, cross-file gotchas, benchmark results
  PARSEQ_REPLICATION.md              Copy-pasteable runbook to set up PARSeq fine-tuning on a fresh machine
```

## Which venv?

| | `venv/` (main) | `venv_parseq/` |
|---|---|---|
| Used for | TrOCR train/infer/eval, data prep, zero-shot comparison scripts | PARSeq fine-tuning + evaluating fine-tuned PARSeq checkpoints |
| Install from | `requirements.txt` (+ `requirements-ocr-compare.txt` optional) | see [docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md) — heavier, version-sensitive stack (Hydra, Lightning, `lmdb`, pinned NumPy) |
| Scripts | everything under `src/common/`, `src/trocr/`, `src/compare/` | everything under `src/parseq/`, plus `third_party/parseq/train.py` directly |

They're kept separate because PARSeq's training dependencies conflict with
(and are irrelevant to) TrOCR's — see [docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md)
for exactly why.

## 1. Shared setup (do this once, either way)

Your data:

- Images: `D:\completed` (or whatever `image_dir` in `config.yaml` points to)
- Annotation workbook: `data/raw/annotations.csv.xlsx`, columns `filename` (image filename) and `Normalize` (training target)
- Duplicates are intentionally preserved; only rows with missing labels or missing image files are skipped

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# install a CUDA-enabled PyTorch build matching your GPU if the default CPU wheel gets installed

python src/common/prepare_dataset.py
```

This produces `data/processed/{train,val,test}.csv` — the input every model
below trains and evaluates on.

## 2. Option A: TrOCR

```bash
python src/trocr/train.py
python src/trocr/inference.py --model checkpoints/best --input "D:\completed\Doc0485_page_003_roi_1.png"
python src/trocr/inference.py --model checkpoints/best --input "D:\completed" --output predictions.csv
python src/trocr/evaluate.py --model checkpoints/best
```

`inference.py`'s `--input` takes either a single image file or a whole
directory of images (predictions print to stdout either way; `--output`
additionally saves a CSV when running on a directory). `evaluate.py` reports
exact-match accuracy, CER, and a flagged-for-review count (based on
per-token confidence — see `constraints.py`) against a labeled CSV split,
for when you have ground truth to check against.

## 3. Option B: PARSeq

One-time environment setup (separate venv, vendored repo, a couple of
version-pinned dependencies, one small patch) — follow
[docs/PARSEQ_REPLICATION.md](docs/PARSEQ_REPLICATION.md) top to bottom the
first time. After that's done:

```bash
venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py   # CSV -> LMDB, once

venv_parseq/Scripts/python third_party/parseq/train.py \
  +experiment=parseq pretrained=parseq \
  data.root_dir="<ABSOLUTE PATH>/data/parseq_lmdb" data.train_dir=handwriting \
  data.num_workers=0 model.batch_size=32 \
  trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1.0 \
  trainer.max_epochs=50

venv_parseq/Scripts/python src/parseq/eval_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt"

venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\completed\Doc0485_page_003_roi_1.png"
venv_parseq/Scripts/python src/parseq/infer_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" --input "D:\completed" --output predictions.csv
```

Both `eval_parseq.py` and `infer_parseq.py` also run with no `--checkpoint`
to use the *pretrained, not-fine-tuned* PARSeq weights (a zero-shot sanity
check, not the real comparison). `eval_parseq.py` needs a labeled CSV
(ground truth to score against); `infer_parseq.py` takes a single image or a
directory with no labels needed, same `--input`/`--output` interface as
TrOCR's `inference.py` above.

## Reshuffling the split (robustness check)

Regenerates `data/processed/{train,val,test}.csv` with a different random
train/val/test split, so you can retrain and confirm a result wasn't just an
artifact of one particular split. Steps, in order:

```bash
# 1. Back up the current split first -- data/ is gitignored, this is the only copy
mkdir data\processed_seed42_backup
copy data\processed\*.csv data\processed_seed42_backup\

# 2. Edit config.yaml's split.seed to a new number, then regenerate the CSVs
venv\Scripts\activate.bat
python src\common\prepare_dataset.py

# 3. Rebuild the PARSeq LMDB from the new CSVs (overwrites data/parseq_lmdb in place)
venv_parseq\Scripts\activate.bat
python src\parseq\prepare_parseq_dataset.py

# 4. Retrain PARSeq on the new split (same command as the normal fine-tune above)
python third_party\parseq\train.py +experiment=parseq pretrained=parseq data.root_dir="G:/trocr-handwriting-v2/data/parseq_lmdb" data.train_dir=handwriting data.num_workers=0 model.batch_size=32 trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1.0 trainer.max_epochs=50

# 5. Find the best checkpoint (highest val_accuracy in the filename) in the newest outputs/parseq/<run>/checkpoints/ folder, then evaluate it
python src\parseq\eval_parseq.py --checkpoint "outputs\parseq\<newest-run>\checkpoints\<best>.ckpt" --output my_results.csv

# 6. Sanity-check there's no leakage between the new train/test split
python -c "import pandas as pd; a=pd.read_csv('data/processed/train.csv'); b=pd.read_csv('data/processed/test.csv'); print('overlap:', len(set(a.image_path)&set(b.image_path)))"
```

`overlap` should print `0`. Then open `my_results.csv` and filter to
`exact_match == False` — if the mistakes still look like real character
confusions (`3`/`8`, `v`/`V`, etc.) rather than nonsense, the model is
genuinely working rather than having gotten lucky on one split. A few points
of accuracy swing between different random splits is normal on a dataset
this size and not a sign anything is broken.

**Caveat:** an existing TrOCR checkpoint (e.g. `checkpoints/best`) was
trained on whichever split was active *when it was trained*. Comparing it
against a `test.csv` from a *different* split risks leakage (some of its
training rows may now sit in the new test set) — retrain TrOCR too
(`python src/trocr/train.py`) before trusting a TrOCR-vs-PARSeq comparison
on a reshuffled split.

## Bonus: zero-shot baseline comparison

Sanity-checks how the *pretrained* (not fine-tuned) PARSeq and PP-OCRv5
weights do against a fine-tuned TrOCR, mainly to see whether their built-in
confidence scores look useful before investing further. Main venv:

```bash
pip install -r requirements-ocr-compare.txt
python src/parseq/eval_parseq.py            # needs venv_parseq instead if you followed option B above
python src/compare/eval_ppocr.py
python src/trocr/evaluate.py --model microsoft/trocr-small-handwritten --no-constrain
```

All three print the same summary format and accept `--output results.csv`.
Expect the fine-tuned models to win by a wide margin — see
[docs/CODEBASE_OVERVIEW.md](docs/CODEBASE_OVERVIEW.md) for the actual
numbers from all five (2 fine-tuned + 3 zero-shot) configurations.

## Current TrOCR architecture

```text
Handwritten crop
      ↓
TrOCRProcessor
      ↓
ViT image encoder
      ↓
Transformer text decoder
      ↓
Predicted text
```
