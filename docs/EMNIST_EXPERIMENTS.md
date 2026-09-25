# EMNIST experiments

Detailed record of the experiments that tried to use EMNIST (public handwritten digits/letters) to help our PARSeq models. Summary and conclusions are in the README; everything needed to reproduce is below.

## Synthetic sequences (base model)

Question: can EMNIST (public handwritten digits/letters) replace or supplement our labeled crops, and prepare the
model for a future `NNa` label format (two digits + a letter a-j, e.g. `02a`, `70f`)? EMNIST only has single 28x28
glyphs, so `src/emnist/synth_sequences.py` stitches digits 0-9 and lowercase a-j (from EMNIST `byclass`, the only
split that keeps case) into crops, with per-glyph and per-crop augmentation, then PARSeq's RandAugment on top.
Recipe `v1` was the first round; `v2` adds more varied trailing marks (tilted/thick/touching `)`, dots, colons, full
`( )` wraps, circles) and raises `NNa` from 20% to 30% of sequences with wider crops. Validation crops use EMNIST
*test* glyphs so writers don't overlap with training.

```bash
curl -L -o data/emnist/gzip.zip https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip      # 562MB
venv_parseq/Scripts/python src/emnist/extract_emnist.py
venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py --processed-dir data/docsplit/seed42 --output-dir data/emnist_lmdb_v2/mix
venv_parseq/Scripts/python src/emnist/synth_sequences.py --recipe v2 --glyph-split train --n 6000  --seed 11 --out data/emnist_lmdb_v2/mix/train/emnist
# pure-NNa evaluation set (EMNIST test glyphs; a probe of the future format, NOT real handwriting)
venv_parseq/Scripts/python src/emnist/synth_sequences.py --recipe v2 --only-template DDL --glyph-split test --n 1000 --seed 777 --out data/emnist/_nna_lmdb --dump-dir data/emnist/nna_eval
venv_parseq/Scripts/python src/emnist/run_experiments.py      # trains + scores every run below; resumable
```

Scored on the document-level split `data/docsplit/seed42` (test = 379 crops from documents never seen in training),
final-epoch model, no checkpoint selection. `base_best` is the current production recipe (base PARSeq,
`parseq_handwriting_fixed`) retrained on this split (`models/parseq_final` can't be scored here: it trained on the
test crops); it reproduces the 96.8% this fold got in the CV report. Full tables: `reports/emnist_experiments.md`.

| model | training data | test acc | test CER | synthetic `NNa` acc |
|---|---|---|---|---|
| **base_best (current)** | real | **96.8%** | 2.9% | 10.7% |
| base mix_v2 | real + 6k synthetic | 91.3% | 8.6% | 91.1% |

- **No model beats the current one on today's data.** Adding synthetic data cost accuracy on real crops
  (96.8% -> 91.3%).
- **`NNa` readiness is the only thing synthetic data buys:** models never shown `NNa` (base_best) score ~10%
  on the synthetic `NNa` set (the base model mostly drops the trailing letter); models trained with it score ~90%.
  Caveat: that set is synthetic and in-distribution for those models, so ~90% is an upper bound for real handwriting.
- Caveats: one seed, one fold (n = 379); EMNIST's own `g`/`9` and `i`/`1` glyph ambiguities carry into synthetic labels.
  Untried, and the standard next step: synthetic *pre*-training then real fine-tuning, or a much smaller synthetic share.

## Class-balancing with synthetic single characters (base PARSeq, 3 seeds)

Our single-character labels are very skewed (train: `2` 406, `3` 327, ... `9` 74, `a` 44, `I` 34, `X` 8, `0` 5, `J` 5, `c` 3,
`d` 1, `e` 1), so this run tops every single-character class up to 100 examples with EMNIST-based synthetic crops
(1,305 crops, +43% on the real train set; the balance plan (`data/emnist/balance_plan.csv`) was built from the real *train* split only; the floor of 100 is
the digit 7-9 tier, not a tuned value). Glyphs cover all 25 classes incl. uppercase/roman (`extract_emnist.py --vocab`),
rendered with recipe v2. Same base model and `parseq_handwriting_fixed` recipe, seeds 42/43/44 for both sides.

```bash
venv_parseq/Scripts/python src/emnist/extract_emnist.py --vocab 0123456789abcdeABCDIJVXiv --out data/emnist/glyphs_single.npz
venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py --processed-dir data/docsplit/seed42 --output-dir data/emnist_lmdb_bal
venv_parseq/Scripts/python src/emnist/synth_sequences.py --recipe v2 --glyphs data/emnist/glyphs_single.npz --glyph-split train --labels-csv data/emnist/balance_plan.csv --seed 21 --out data/emnist_lmdb_bal/train/emnist_bal
venv_parseq/Scripts/python src/emnist/run_experiments.py --only base_bal_s42 base_best_s43 base_bal_s43 base_best_s44 base_bal_s44
```

The scripts that produced the balance plan (`make_balance_plan.py`) and the seed-aware comparison
(`compare_balanced.py`) have been removed from the repo; their outputs (`data/emnist/balance_plan.csv`,
`reports/emnist_balanced.md`) remain as records.

| | val acc | test acc | test CER |
|---|---|---|---|
| current recipe (real only), mean +- std | 91.6 +- 0.3% | **96.4 +- 0.4%** | 3.3% |
| + balancing synthetic single chars | 88.5 +- 0.8% | 95.8 +- 0.7% | 4.4% |

Balancing did **not** help: val is lower on all three seeds (paired p = 0.002-0.05 on pooled val+test), test is within
noise but lower on 2 of 3 seeds, and accuracy on the topped-up rare classes did not rise (83.5% vs 86.3%, n = 168 pooled
val+test crops). Multi-character labels got worse (89.0% vs 93.6%) -- extra single-character-only data seems to bias the
model toward short outputs. Per-class rare-class counts in val+test are tiny (1-16), so per-class rows are anecdotal.
Full tables: `reports/emnist_balanced.md`. Models and data for every EMNIST run are archived read-only under
`models/archive/emnist_2026-09-24/` (base_best, base_mix_v2) and `models/archive/emnist_balanced_2026-09-24/`
(seed runs), each with `MANIFEST.json` sha256 hashes; `src/emnist/archive_models.py` builds them.

## Real-data coverage of the confusable characters

Checked afterwards from the saved per-sample predictions: the reported confusions `b->6`, `g->9` and `c<->e` cannot be measured on our real
data. Crops whose label contains each character -- train: `b` 88, `c` 3, `e` 1, `g` 0; val + test: `b` 8, `c` 1, `e` 0, `g` 0. On the
synthetic `NNa` set (letter in the last position), models trained with synthetic letters no longer read `b` as `6` or `g` as `9`, but `c<->e` still
confuses (3-6% each way); the current production model mostly drops a trailing letter and shows all three confusions when it does emit one.
This is synthetic evidence only (EMNIST's own `g` glyphs often look like `9`); real labeled `b/g/c/e/6/9` crops are needed to settle it.

## Where things are

- Code: `src/emnist/` -- `extract_emnist.py` (EMNIST zip -> glyphs), `synth_sequences.py` (sequence/crop synthesizer, recipes v1/v2, `--labels-csv`
  exact-label mode), `run_experiments.py` (train + score every run, writes `reports/emnist_experiments.md`),
  `archive_models.py`, `emnist_metrics.py` (shared metrics).
- Config: all runs use `configs/parseq/experiment/parseq_handwriting_fixed.yaml`.
- Retained models: `models/archive/INDEX.md` (base_best, base_mix_v2, base_bal_s42; sha256-verified, read-only). Each archive is self-contained
  (weights, full checkpoint, config, logs, predictions, datasets, code snapshot).
- Working data (regenerable, gitignored): `data/emnist/` (zip, glyphs, balance plan, NNa eval set), `data/emnist_lmdb{,_v2,_bal}/`.
- Previews: `reports/emnist/previews/`. Per-sample predictions of every scored run: `reports/emnist/<run>_<split>.csv`.
- The base runs live in the archive, not in `outputs/parseq_emnist/`. Re-running `run_experiments.py` for a run whose
  `final.ckpt` is gone retrains it.
