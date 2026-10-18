# Datasets: everything on disk, with counts

Counted 2026-10-02. Full per-label counts for every set below: [dataset_label_counts.csv](dataset_label_counts.csv).
Which model used which set: [MODEL_LINEAGE.md](MODEL_LINEAGE.md).

## The source: real handwritten crops

* Images: `D:/all_roi_images/` has **7,005** crop images from **219 documents**. Names are `Doc0692_page_002_roi_1.png`.
* Labels: `D:/annotations.xlsx` (path set in `config.local.yaml`). `data/raw/annotations.csv.xlsx` is an old copy with only 1,962 rows; do not use it.
* **3,827 crops carry a usable label** (86 distinct labels, 182 documents). The other ~3,180 images are noise (see noise classifier below), ambiguous, free text or unlabelled.

### Shape of the 3,827 labelled crops

| shape | crops | share |
|---|---|---|
| single digit (`0-9`) | 2,508 | 66% |
| Roman numeral (`i ii iii iv v vi ... IX`, either case) | 619 | 16% |
| two digits (`10-26`...) | 411 | 11% |
| single letter (`a b c d e A B C D J X`) | 221 | 6% |
| number + letter (`2a 7a 6b 17a ...`) | 61 | 1.6% |
| other / with punctuation (`III.`, `2)`, `A.`) | 7 | 0.2% |

Most common: `2` 497, `3` 406, `1` 381, `4` 371, `5` 342, `6` 164, `7` 124, `8` 114, `10` 111, `9` 103, `ii` 86, `b` 75.
**Labels with 5 or fewer examples (34 of 86):** `6b` 5, `J` 5, `c` 4, `2b` 4, `4a` 4, `4b` 4, `1b` 4, `25` 3, `17a` 3, `5a` 3, `5b` 2,
`III.` 2, `28` 2, `2A` 2, `1a` 2, `d` 2, and 18 labels with a single example (`e`, `Vii`, `2)`, `7i`, `4A`, `4)`, `8a`, `9i`, `iV`, `3b`,
`II)`, `3)`, `1A`, `3i`, `38`, `A.`, `41`, `16b`). Labels keep their case: `II` and `ii`, `A` and `a`, `2A` and `2a` are different.

## Real-data splits (all are cuts of the same 3,827 crops)

| set | folder | train / val / test | how it is cut | used for |
|---|---|---|---|---|
| random split | `data/processed` | 3,061 / 382 / 384 | random per crop, so a document can be in train and test (optimistic) | original TrOCR/PARSeq work, `parseq_lmdb` |
| all data | `data/all_data` | 3,827 / 382 / 382 | everything in train; val/test are a copy of val | `models/parseq_final` |
| document split seed 42 | `data/docsplit/seed42` | 3,064 / 384 / 379 | no document in two sets | **every honest comparison**, EMNIST experiments, archive |
| document splits 43-46 | `data/docsplit/seed43..46` | ~3,060 / ~383 / ~384 | same idea, other seeds | 5-fold cross-validation |
| no Roman numerals | `data/docsplit_noroman/seed42` | 2,596 / 384 / 379 | seed 42 with Roman rows removed from train | `base_mix_v3nr`, `base_mix_v4` |
| half split | `data/half_split` | 1,913 / 191 / 1,723 | train on half the documents, test on the rest | overfitting analysis |
| side files | `data/processed/` | ambiguous 2, free text 170, missing image 3 | rows the cleaner kept out of training | nothing (excluded) |

## Noise classifier data (`data/noise_classifier`)

All **7,005** images labelled noise (`0`: 3,993) or clean (`1`: 3,012), split by document: train 5,607 / val 712 / test 686.
Used only by `models/noise_classifier/best.pt` (ResNet-18). The 3,012 clean crops are the superset of the question-number crops.

## PARSeq training LMDBs (the same crops, packed for PARSeq)

| folder | contents |
|---|---|
| `data/parseq_lmdb/{train,val,test}/handwriting` | the random split: 3,061 / 382 / 384 |
| `data/emnist_lmdb_v3/mix` | train = `handwriting` 3,064 real + `emnist` 6,000 synthetic (1,066 distinct labels); val 384, test 379 |
| `data/emnist_lmdb_v3nr/mix` | 2,596 real (no Roman) + 6,000 synthetic v3 |
| `data/emnist_lmdb_v4/mix` | 2,596 real (no Roman) + 6,000 synthetic v4 (adds g/9) |
| `data/emnist_lmdb_v5/mix` | 3,064 real + 6,000 synthetic v5 |
| `data/emnist_lmdb_v5/stage1` | 8,000 synthetic v5 only (val 384 real) |
| `data/emnist_lmdb_v5/stage2` | 3,064 real only (val 384, test 379) |

Synthetic sets only contain digits `0-9` and letters `a-j` (EMNIST glyphs); no uppercase, no Roman numerals except `i`.

## Synthetic building blocks (`data/emnist`)

| file | contents |
|---|---|
| `glyphs.npz` | EMNIST digits 0-9 + lowercase a-j |
| `glyphs_v5.npz` | same, larger: 522,146 train / 69,725 test single-character images |
| `glyphs_single.npz` | uppercase and Roman characters: each digit 31,000-38,000, `e` 24,631, `I` 11,946, `d` 10,177, `C` 10,094, `a` 10,033, `A` 6,407, `b` 5,159, `V` 4,637, `D` 4,562, `B` 3,878, `J` 3,762, `v` 2,910, `c` 2,854, `X` 2,771, `i` 2,725 |
| `glyphs_g9.npz` | glyph bank for the g-versus-9 confusion |
| `nna_eval/` | 1,000 synthetic `NN`+letter crops used only as a probe, never for training |
| `balance_plan.csv` | plan for 1,305 single-character crops topping rare classes up to 100 (the `base_bal` models) |

The archived models carry their own copies of their datasets in `models/archive/*/datasets/`.

## What a master dataset could contain

| ingredient | rows | role |
|---|---|---|
| labelled real crops | 3,827 | the truth; always included |
| real crops with a noise label | ~3,180 | not label data; keep out, or use as a "reject" class |
| synthetic EMNIST sequences | 6,000-8,000 per recipe (previous runs) | only helped the `NNa` probe, hurt real accuracy |
| synthetic variants with brackets/dots (made Oct 1-2, deleted) | 5,394 and 32,571 | did not beat real-only (see lineage doc) |

Test sets must stay real and document-level (seed 42 or CV), and every ingredient needs a source column so a master set can be cut
back apart.
