# TrOCR Handwriting OCR

Standalone TrOCR-small experiment for short handwritten annotations.

## Your data

Images stay in:

`D:\completed`

The supplied annotation workbook is:

`data/raw/annotations.csv.xlsx`

Expected columns:

- `filename` — image filename
- `Normalize` — training target

**Duplicates are intentionally preserved.**

Only rows with missing/empty labels or missing image files are skipped.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your NVIDIA GPU if necessary.

Check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Prepare

```bash
python src/prepare_dataset.py
```

This produces:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

## Train

```bash
python src/train.py
```

The script uses GPU when CUDA is available and otherwise falls back to CPU.

## Inference

```bash
python src/inference.py --model checkpoints/best --image "D:\completed\Doc0485_page_003_roi_1.png"
```

## Evaluation

```bash
python src/evaluate.py --model checkpoints/best
```

Metrics:

- Exact-match accuracy
- Character Error Rate (CER)

## Current architecture

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

This version intentionally uses vanilla TrOCR decoding. Grammar-constrained decoding and an explicit ACCEPT/REVIEW path come later, after the baseline is validated.
