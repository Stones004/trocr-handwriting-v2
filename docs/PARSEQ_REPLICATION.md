# Replicating the PARSeq fine-tuning setup

Step-by-step runbook to reproduce, on another machine, everything done in this
project to (1) benchmark zero-shot PARSeq/PP-OCRv5 against the fine-tuned
TrOCR model, and (2) fine-tune PARSeq itself. Written so someone with no
memory of the original session can follow it end to end. See
[CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) for the *why* behind each
decision — this file is the *how*, in order.

Tested on: Windows 11, Python 3.10.11, NVIDIA GPU with CUDA 12.8-capable
driver (RTX 5060 Ti). Adjust the CUDA wheel index if your GPU/driver differs.

## Prerequisites

- `data/processed/{train,val,test}.csv` already exist (via this project's own
  `src/common/prepare_dataset.py`), with columns `filename, image_path, text`.
- A working main project venv is not required for any of the steps below —
  everything here is self-contained in a new, separate venv.

## 1. Zero-shot comparison scripts (PARSeq + PP-OCRv5, no fine-tuning)

These run in the **main project venv** (the one `requirements.txt` installs
into), not a separate environment.

```bash
pip install -r requirements-ocr-compare.txt
# installs: paddlepaddle, paddleocr, timm, pytorch-lightning, nltk

python src/parseq/eval_parseq.py                       # pretrained PARSeq, zero-shot
python src/compare/eval_ppocr.py                         # PP-OCRv5_server_rec, zero-shot
python src/trocr/evaluate.py --model microsoft/trocr-small-handwritten --no-constrain
                                                  # pretrained TrOCR, zero-shot, for a fair 3-way comparison
```

All three print the same summary format (samples, exact-match accuracy, CER,
flagged-for-review count) and accept `--output results.csv` for a per-sample
breakdown. Results we got (173-row test set):

| Model | Exact-match | CER |
|---|---|---|
| TrOCR (fine-tuned, `checkpoints/best`) | 86.7% | 11.3% |
| PP-OCRv5_server_rec (zero-shot) | 56.1% | 68.9% |
| PARSeq (zero-shot) | 40.5% | 71.2% |
| TrOCR-small-handwritten (zero-shot) | 7.5% | 426.1% |

## 2. Set up the PARSeq fine-tuning environment

A **separate venv** — PARSeq's training stack (Hydra, Lightning, `lmdb`,
`imgaug`) is heavier than, and version-conflicts with, the main project's.

```bash
python -m venv venv_parseq
```

Install GPU-enabled PyTorch matching your CUDA driver (check with
`nvidia-smi`; the example below is for CUDA 12.8):

```bash
venv_parseq/Scripts/python -m pip install --upgrade pip
venv_parseq/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU support before continuing:

```bash
venv_parseq/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.11.0+cu128 True
```

Install the rest of the dependencies. **Versions matter here** —
`imgaug` (last released 2020, used by PARSeq's built-in `RandAugment`) breaks
under NumPy 2.x (`np.sctypes` was removed), and newer `opencv-python(-headless)`
releases require NumPy 2.x, so NumPy and OpenCV have to be pinned down
*together*:

```bash
venv_parseq/Scripts/python -m pip install pytorch-lightning timm nltk hydra-core omegaconf lmdb imgaug scikit-image matplotlib tensorboardx fire pandas
venv_parseq/Scripts/python -m pip install "numpy==1.26.4" "opencv-python-headless==4.9.0.80" "scikit-image<0.23" "tifffile<2024.3"
```

Exact versions that were verified working together (`pip freeze`):

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
pytorch-lightning==2.6.5
torchmetrics==1.9.0
timm==1.0.29
nltk==3.10.3
hydra-core==1.3.6
omegaconf==2.3.1
lmdb==2.3.0
imgaug==0.4.0
numpy==1.26.4
opencv-python-headless==4.9.0.80
scikit-image==0.22.0
tifffile==2024.2.12
matplotlib==3.10.9
tensorboardX==2.6.5
fire==0.7.1
pandas==2.3.3
pillow==12.3.0
```

If you have both `opencv-python` and `opencv-python-headless` installed,
remove the non-headless one — having both can leave a broken merged `cv2`
package (`AttributeError: module 'cv2' has no attribute 'INTER_NEAREST'`):

```bash
venv_parseq/Scripts/python -m pip uninstall -y opencv-python
```

Clone PARSeq itself and install it (editable, no auto-deps since we just
installed them all manually above):

```bash
git clone --depth 1 https://github.com/baudm/parseq.git third_party/parseq
venv_parseq/Scripts/python -m pip install -e third_party/parseq --no-deps
```

Add to `.gitignore` (both are external/generated, not project source):

```
venv_parseq
third_party
```

### Patch: fix the missing random seed

PARSeq's `train.py` never calls `seed_everything`, so every training run
shuffles data / applies augmentation / initializes dropout differently, even
with identical CLI hyperparameters — the specific accuracy you land on is
**not** reproducible run-to-run without this patch. Apply it once, right
after cloning:

In `third_party/parseq/train.py`:

1. Change the import line:
   ```python
   from pytorch_lightning import Trainer, seed_everything
   ```
2. Add as the first line inside `def main(config: DictConfig):`:
   ```python
   seed_everything(config.get('seed', 42), workers=True)
   ```

Verify GPU access and the patch both work with a fast 1-epoch smoke test
before committing to a full run (see step 4 below for the full command; just
change `trainer.max_epochs=1`).

## 3. Convert the dataset to LMDB

PARSeq's `strhub.data.dataset` only reads LMDB, not CSV. Run this with
`venv_parseq` (needs `lmdb`, which the main project venv doesn't have):

```bash
venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py
```

This writes `data/parseq_lmdb/{train,val,test}/handwriting/data.mdb`. Notably
**does not** use PARSeq's own `tools/create_lmdb_dataset.py` — that script
hardcodes `map_size=1TB`, which is harmless on Linux (lazy sparse mmap) but
fails on Windows with `lmdb.Error: ... the paging file is too small for this
operation to complete` (Windows tries to back the huge mapping with real
page-file-committed memory). `src/parseq/prepare_parseq_dataset.py` uses a 512MB map
size instead (`--map-size-mb`, generous for a ~9MB dataset).

Do **not** override `model.charset_train` / `charset_test` / `max_label_length`
to match your narrower label alphabet, even though it seems like the "correct"
thing to do — the pretrained checkpoint's classification head, token
embedding, and positional-query tensors are sized for the *default* charset
(94 chars) and `max_label_length=25`. Narrowing them causes
`load_state_dict` shape mismatches (`head.weight: [95, 384] vs [23, 384]`
etc.), forcing those layers to be randomly reinitialized instead of using the
pretrained weights. As long as your label alphabet is a subset of the default
94-char charset (verify with the snippet below), leaving everything at
default costs nothing.

```bash
python -c "import pandas as pd; d=pd.concat(pd.read_csv(f'data/processed/{s}.csv') for s in ['train','val','test']); print(''.join(sorted(set(''.join(d['text'].astype(str))))))"
```

## 4. Fine-tune

```bash
venv_parseq/Scripts/python third_party/parseq/train.py \
  +experiment=parseq \
  pretrained=parseq \
  data.root_dir="<ABSOLUTE PATH>/data/parseq_lmdb" \
  data.train_dir=handwriting \
  data.num_workers=0 \
  model.batch_size=32 \
  trainer.accelerator=gpu trainer.devices=1 \
  trainer.val_check_interval=1.0 \
  trainer.max_epochs=50
```

| Parameter | Value | Why |
|---|---|---|
| `pretrained` | `parseq` | Start from the pretrained scene-text checkpoint |
| `model.charset_train` / `charset_test` | *(left at default)* | See "don't narrow it" note above |
| `model.max_label_length` | *(left at default, 25)* | Same reasoning |
| `model.batch_size` | `32` | PARSeq scales effective LR linearly with batch size (`lr_scale = batch_size/256`, see `strhub/models/base.py`'s `configure_optimizers`) — dropping from the from-scratch default of 384 to 32 brings the default `lr=7e-4` down to an effective ~9e-5, a sensible fine-tuning rate, without touching `model.lr` |
| `data.num_workers` | `0` | Avoids Windows multiprocessing/LMDB-handle overhead; irrelevant at this dataset size |
| `trainer.accelerator` / `devices` | `gpu` / `1` | Single local GPU |
| `trainer.val_check_interval` | `1.0` | Validate once per epoch — the config default (`1000` steps) never triggers in a run this short (~43 steps/epoch on 1375 samples) |
| `trainer.max_epochs` | `50` | Generous ceiling; PARSeq's built-in `ModelCheckpoint(monitor='val_accuracy', save_top_k=3)` saves the best checkpoints regardless of where training ends, so overshooting costs time, not quality |
| `data.root_dir` | **must be absolute** | Hydra changes the working directory to its own timestamped output dir at runtime, so a relative path resolves against the wrong location |

Everything else (dropout, warmup_pct, weight_decay, architecture dims) is
left at PARSeq's defaults.

Checkpoints and TensorBoard logs land under
`outputs/parseq/<date>_<time>/checkpoints/`, named
`{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}.ckpt`. Top-3 by
`val_accuracy` plus `last.ckpt` are kept automatically — no extra config
needed.

**If a run crashes with `OSError: [Errno 28] No space left on device`
while saving a checkpoint**, even though the drive holding the project
reports plenty of free space: check `C:` (or your system drive) specifically.
We hit this with `C:` down to ~1.8GB free out of ~476GB while `G:` (holding
the project) had ~400GB free — something in the Windows/CUDA/Lightning stack
needs headroom on the system drive regardless of where the actual output path
is (pagefile growth is the leading suspect, unconfirmed). Freeing space on
the system drive and retrying with no config changes fixed it.

## 5. Evaluate the fine-tuned checkpoint

Run with `venv_parseq` (needs `strhub` importable):

```bash
venv_parseq/Scripts/python src/parseq/eval_parseq.py \
  --checkpoint "outputs/parseq/<run>/checkpoints/<best>.ckpt" \
  --output parseq_finetuned_results.csv
```

Same summary format/methodology as `evaluate.py` and the zero-shot scripts
in step 1, so the numbers are directly comparable. `--flag-threshold 0.5`
(default) flags any prediction whose minimum per-character confidence falls
below that value for manual review — same semantics and default as
`evaluate.py`'s `--flag-threshold`.

Result from our run (best checkpoint, epoch 20/50, `val_accuracy=95.32`),
evaluated on the same held-out `data/processed/test.csv`:

| Model | Exact-match | CER |
|---|---|---|
| PARSeq, fine-tuned | **93.1%** | **5.9%** |
| TrOCR, fine-tuned (`checkpoints/best`, unconstrained) | 86.7% | 11.3% |

Verified this wasn't a fluke: reran the eval from the same checkpoint (got
the identical 93.06%), checked for train/test image-path overlap (zero — no
leakage despite the dataset intentionally preserving duplicate rows), and
inspected the 12 misses (all plausible handwriting confusions — `v`/`V` case,
`3`/`7`, `5`/`b`, `VII`/`VI`, digit shape mixups — not degenerate output).

## Known sources of run-to-run variance

Even with the seed patch applied, don't expect bit-for-bit identical numbers
across machines/GPUs — CUDA/cuDNN mixed-precision training isn't fully
deterministic across hardware. What the seed patch guarantees is that data
shuffling order, augmentation, and dropout masks are fixed *given the same
environment*; it removes the single biggest source of variance, not all of
it. A few points of swing in `val_accuracy` between otherwise-identical runs
on a dataset this small (1375 training rows, 43 steps/epoch) is expected and
not a sign of a broken setup.
