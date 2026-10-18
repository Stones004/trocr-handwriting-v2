# Model lineage: what each model was trained on

## Models kept in the repo (as of 2026-10-02)

| folder | trained on | real test acc (doc split seed 42, 379 unseen-document crops) |
|---|---|---|
| `models/parseq_final/` | all 3,827 real crops, no synthetic data (production) | not measurable: it saw the test crops |
| `models/parseq_base_best/` | the 3,064 real train crops of doc split seed 42, no synthetic data | 96.8% |
| `models/parseq_master_v2/` | master set v2: 3,064 real + 10,897 capped synthetic crops. Labels are characters only; brackets, dots and a `Q.`/`Q` prefix are drawn in the image but stripped from the label (7 real marked labels stripped too). Retrained from base PARSeq, `parseq_handwriting_fixed` recipe, 15 epochs. | 96% [94-98] (15 errors; base_best 11). Synthetic probes of the new formats 95-99%. |
| `models/parseq_master_v3/` | master set v3 = v2 recipe + more Q-prefix crops (23% of synthetic, hand-drawn Q with 2-/9-like tails) + the 18 real Q-prefix crops x5 + 64 newly labelled new-document crops x5 (labels = Claude's reading, unverified; Doc0729/0757/0771/0773 held out). | 95% [92-97] real test (19 errors); held-out new documents 86% vs 82% (v2); Q-prefix crops there 76% vs 59% (17 crops). |
| `models/parseq_openread_v1/` | open-reading set v1: 2,250 real crops relabelled verbatim (brackets/dots/`Q`/`Ans` kept; 814 crops without evidence dropped) + 800 real junk crops labelled `~` + 1,500 synthetic words. From pretrained PARSeq, seed 43, 15 epochs, `charset_test` = full 94. | real test (old labels) 93.4% (3 seeds: 91.8 / 93.4 / 90.8; `base_best` 96.8); words 85.7% (`base_best` 40.3); junk rejected 99.1%. Reads `Add:`, `Less:`, `(a)`, `1]b]` on unseen documents. Not adopted: plain numbers regressed. See `reports/open_read_eval.md`. |
| `models/noise_classifier/` | ResNet-18 noise/clean classifier, not PARSeq | n/a |

`parseq_master_v2` replaced the first master model (v1: marks kept in the label, hallucinated digits for the `Q` prefix, 94% real test), which was deleted on 2026-10-02. Full tables: `reports/master_eval.md`.

**Deleted on 2026-10-02:** `models/archive/` (all EMNIST archives, with their dataset/code snapshots) and `outputs/parseq_emnist/`, which held
`base_best` seeds 43/44, `base_bal` s42-44, `base_mix_v2`, `base_mix_v3` (and variants), `v3nr`, `v4`, `v5` stage 1/2 and mix. Their scores are
in the tables below and in `reports/emnist_experiments.md`; the weights are gone. Also removed: the master run's training checkpoints (only the
exported weights above remain). `parseq_finetuned_epoch21_acc93.57_ned94.93.ckpt` (94.2% on this test) lives outside the repo in
`D:\OCR_codes\ocr_dataset_builder\models` and `Downloads`, and was not touched.

Sections 2 and 3 below describe those deleted models; they are kept as a record of what was tried.

Every model below starts from the same place: **PARSeq's pretrained scene-text weights** (baudm/parseq, `pretrained: parseq`),
fine-tuned with the recipe in `configs/parseq/experiment/parseq_handwriting_fixed.yaml` (15 epochs unless stated, lr 1.5e-3,
RandAugment magnitude 9, seed 42, batch 32). "Real" = your handwritten crops (`data/raw`, 3,827 labelled crops from ~100 documents).
"Synthetic" = images we generated ourselves (EMNIST glyphs stitched into sequences, or fonts).

How the real data is split (these are the only real-data sets used anywhere):

| name | folder | what it is | train / val / test |
|---|---|---|---|
| random split | `data/processed` | random per-crop split of all real crops (a document can appear in train and test, so scores are optimistic) | 3,061 / 382 / 384 |
| all data | `data/all_data` | every real crop in `train` (val/test reuse val, so they say nothing about generalization) | 3,827 / 382 / 382 |
| **document split, seed 42** | `data/docsplit/seed42` | no document appears in two sets: the honest test. **All comparisons below use this one.** | 3,064 / 384 / 379 |

## 1. Production and standalone models (`models/`)

| model | trained on | notes |
|---|---|---|
| `models/parseq_final/parseq_final.ckpt` | **all 3,827 real crops** (`data/all_data`), no synthetic data. 15 epochs. | The production model. Its own training score (99.45%) says nothing about new documents; document-level CV estimated ~93-94% on unseen documents. |
| `models/noise_classifier/best.pt` | Real crops labelled noise/clean from the annotation workbook (`data/noise_classifier`), document-level split. ResNet-18, **not PARSeq**. | Separate model: flags non-label crops. |

## 2. Archived EMNIST experiments (`models/archive/`, read-only)

All scored on the document split seed 42 (test = 379 unseen-document crops).

| model | trained on (train set) | test acc |
|---|---|---|
| `base_best` (+ seeds 43, 44) | real doc-split train, 3,064 crops. No synthetic data. | 96.8% |
| `base_mix_v2` | 3,064 real + 6,000 synthetic EMNIST sequences (recipe v2), 8 epochs | 91.3% |
| `base_bal_s42` (+ 43, 44) | 3,064 real + 1,305 synthetic single-character EMNIST crops that top up rare classes to 100 | 95.3% |

## 3. EMNIST experiment runs (`outputs/parseq_emnist/`)

All train on the document-split real crops plus synthetic data, except stage-1 runs (synthetic only). 8 epochs unless stated.

| run | training data | other |
|---|---|---|
| `base_mix_v3` (+ `_s43`, `_mag6_s42`, `_lr5e4_s42`) | 3,064 real + 6,000 synthetic v3 (digits + letters a-j, no `ii`/`iii`) | seed / augmentation strength / lr variants of the same data |
| `base_mix_v3nr` | 2,596 real **without Roman numerals** + 6,000 synthetic v3 | |
| `base_mix_v4` | 2,596 real without Roman numerals + 6,000 synthetic v4 (adds g/9 crops) | |
| `v5_mix` | 3,064 real + 6,000 synthetic v5 (dot-separated `NN.a`, runs of `1`) | one stage |
| `v5_stage1` | **8,000 synthetic v5 only** | no real data |
| `v5_stage2_lr5e4`, `v5_stage2_lr15e4` | 3,064 real, starting from `v5_stage1` weights | two-stage; lr 5e-4 / 1.5e-3, 10 epochs |

## 4. Runs made on 2026-10-01 / 02 (augmented-label work)

Labels were generated by `augment_labels.py` (brackets, dots, spacing, case) and drawn as images. A crop drawn as `[2 a].` is
labelled `2a` (punctuation and spacing stripped, case kept). **None of these beat the real-only baseline.**

| run | trained on | doc-split test acc (379 crops) |
|---|---|---|
| `parseq_augcmp/base_seed42` | 3,064 real only (same recipe as `base_best` above, a duplicate of it) | **96.6%** |
| `parseq_augcmp/aug_seed42` | 3,064 real + 5,394 synthetic variants (EMNIST glyphs, 72 labels), one stage | 94.5% |
| `parseq_aug/run` | 3,827 real (all data) + 5,394 synthetic variants, one stage | not measurable (val is inside train) |
| `parseq_diverse/stage1` | 32,571 synthetic only: full label grammar (0-9, 10-99, N/NN + letter, single letters, Roman 1-20) with brackets/dots anywhere, 16 fonts + EMNIST; 5 epochs | 59.4% |
| `parseq_diverse/stage2` | 3,064 real, from `stage1` weights, lr 5e-4, 10 epochs | 93.9% |

Why the diverse set lost: every label got at least 8 crops, so ~half the set was rare `NN`+letter combinations and the model
learned to append suffix letters (stage 1 reads `2` as `2j`, `7` as `7j`, `1` as `11i`). Stage 2's remaining errors are mostly Roman
numerals (`V`, `v`, `X`, `VI`) and a few digits.

## How to verify any of this yourself

Each run's exact settings: `outputs/<experiment>/<run>/config/config.yaml` (`data.root_dir`, `data.train_dir`, `init_ckpt`,
`trainer.max_epochs`, `model.lr`). Training set sizes: `outputs/<experiment>/<run>/train.log`, lines `dataset root: ... num samples: N`.
The production model's recipe comes from the README ("Final production model"); its run folder no longer exists.
