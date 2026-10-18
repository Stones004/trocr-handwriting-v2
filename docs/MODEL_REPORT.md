# Model report: `parseq_master_v3` (the model we keep) and the log of everything else

Written 2026-10-04 when the repo was cleaned up. This file is meant to make every training run below reproducible **without re-running
the experiments**. Raw material that backs it up lives in `docs/log/` (MLflow export of all 80 runs, every result table, the master_v3
training config).

Contents: 1 the model - 2 architecture and parameters - 3 training recipe - 4 data - 5 results - 6 using it - 7 reproducing it - 8 everything
else we tried (ledger) - 9 findings that should shape the next step - 10 `parseq_master_v4` (retrain on the reviewed labels) - 11 what was deleted and how to get it back

---

## 1. The model

| | |
|---|---|
| File | `models/parseq_master_v3/parseq_master_v3.ckpt` (95 MB, weights-only export; sha256 starts `9fa923225d96bb6f`) |
| What it is | PARSeq (scene-text recogniser, baudm/parseq) fine-tuned from its pretrained weights on the **master v3** dataset: real handwritten question-number crops + synthetic crops (NNa formats, brackets, dots, `Q` prefix, Roman numerals) |
| Trained | 2026-10-02, seed 42, RTX 4050 Laptop GPU (6 GB), 15 epochs |
| Used by | `D:\OCR_codes` loads this exact file for OCR (`parseq_worker.py`); `inference.checkpoint` in `config.yaml` points at it. **A newer retrain, `parseq_master_v4` (section 10), was added on 2026-10-04 and scores about the same; `inference.checkpoint` still points at v3.** |
| Chosen because | on 2026-10-04 it was identified as the best model we have. Note: on the real document-split test it is **not** the highest scorer (94.7% vs 96.8% for the real-only `base_best`); it is the only model that reads the NNa / bracketed / `Q`-prefix formats (97-99% on the synthetic probe vs 11-57% for the others) and it did best on unseen new documents (86% vs 79-82%) |

## 2. Architecture and parameters

PARSeq (permutation language modelling, autoregressive decode with 1 refinement iteration). Values are from the checkpoint's own hyper-parameters.

| parameter | value |
|---|---|
| input | `img_size` = [32, 128] (height, width); every crop is **stretched** to this (bicubic), normalised with mean 0.5 / std 0.5 |
| encoder | ViT, patch size [4, 8], `embed_dim` 384, 12 layers, 6 heads, MLP ratio 4 |
| decoder | 1 layer, 12 heads, MLP ratio 4 |
| parameters | 23.8 M |
| `max_label_length` | 25 characters |
| vocabulary | `charset_train` = 94 printable ASCII characters `0-9a-zA-Z` + punctuation, plus 3 special tokens (end, begin, pad) = 97 token ids. The order fixes the token ids, so it must not change (pretrained weights depend on it) |
| `charset_test` | `0-9a-z` (36). Only affects validation scoring: labels and predictions are lowercased and stripped of other characters during validation. It does not change what the model can read |
| decoding | `decode_ar` true, `refine_iters` 1, `perm_num` 6 (permutations per sample, mirrored, forward) |

## 3. Training recipe (identical to `config.yaml` -> `train:`)

Verified: the Hydra config composed from `config.yaml` matches the saved run config (`docs/log/master_v3_train_config.yaml`) on all 53 keys; the
only difference is the MLflow experiment label.

| parameter | value |
|---|---|
| start weights | `pretrained: parseq` (baudm/parseq scene-text weights) |
| epochs | 15, fixed. No early stopping, no SWA. **The last epoch is the model** (it is saved as `final.ckpt`) |
| batch size | 32 (14,323 samples -> 448 steps/epoch, 6,720 steps) |
| learning rate | 1.5e-3 (PARSeq default 7e-4), 7.5% warm-up, annealed to zero at the last epoch |
| dropout | 0.2 (default 0.1) |
| weight decay | 0.0 |
| gradient clip | 20 |
| augmentation | PARSeq RandAugment, magnitude 9 (default 5), 3 layers |
| seed | 42 |
| precision / hardware | single GPU, `num_workers` 0 |
| patches to PARSeq | `patches/parseq_train.patch` (seed, augmentation strength, MLflow logging, `swa` switch, `final.ckpt`, `+init_ckpt`, optional resize mode, `SaveFinal` callback) |
| outcome | final-epoch validation accuracy 90.6% (lenient 36-char metric; real val crops), val NED 92.3%, val loss 1.57. Best val epoch was 10 (90.9%), not used |

## 4. Data: master v3

Source: `data/master_dataset.mdb`, a single file holding every crop (image bytes embedded) with its label and metadata
(`src/master/dataset_io.py`; manifest `data/master_dataset.csv`).

**Change on 2026-10-04: the new-document crops were removed from the dataset file.** `parseq_master_v3` itself was trained on 14,323 train rows that
included 64 new-document crops (repeated x5 = 320 rows, source `newdoc` / `newdoc_repeat`, from Doc0750, 0752, 0755, 0767, 0769, 0775; labels = Claude's
reading, unverified). Those rows and the 56 held-out crops (Doc0729, 0757, 0771, 0773) were taken out of the file (376 rows). **The file therefore no longer
reproduces master_v3's exact training set**; the trained weights still contain the influence of those 64 crops, and the held-out figures in section 5 were measured with
the 56 crops that are now gone. The crop list and my labels are archived in `docs/log/newdoc_labels/`. Before the removal the file unpacked to the original
training database byte for byte (0 label and 0 image differences over 14,323 / 384 / 379 rows).

| split | rows | content |
|---|---|---|
| train | 14,003 | 3,057 real (was 3,064) + 10,867 synthetic + 72 (real Q-prefix crops repeated, x5 total). master_v3 itself saw 14,323 = these + 64 new-document crops + 256 repeats (now removed) |
| val | 382 | real crops only (document split, seed 42); was 384 before the label review |
| test | 378 | real crops only (document split, seed 42); was 379 before the label review |
| junk | 10 | real crops the label review found not to be question numbers (label `~`, original label kept in the file); excluded from train/val/test |
| ~~heldout_newdoc~~ | ~~56~~ | removed 2026-10-04 (crops from Doc0729, 0757, 0771, 0773, labels = Claude's reading, unverified); list kept in `docs/log/newdoc_labels/` |

**Label review (2026-10-04).** The real crops were checked with a 5-fold, document-level cross-check: five real-only models (same recipe) each read the documents they had
not seen; 298 of 3,821 crops (7.8%) were flagged where that reading and the original pretrained PARSeq both, or only the cross-check model, disagreed with the label (out-of-fold
exact match 89.9 / 92.9 / 93.3 / 92.3 / 93.3%, overall 92.4%). The flagged crops were reviewed by hand: 177 `ok`, 111 `fix`, 10 `junk`. Applied to `master_dataset.mdb`: 109 label fixes
(113 rows, because some crops are repeated: 90 train, 18 val, 5 test) and 10 crops moved to a `junk` split (label `~`). Two edited labels outside the usual format were confirmed by the reviewer as correct for the image and applied: `JQ` (Doc0609_page_004_roi_4) and `I 1` (Doc0532_page_003_roi_7, stored as `I1` because PARSeq strips whitespace);
the `labels.clean_pattern` inference check will flag predictions of that shape. Audit trail of every change: `docs/log/label_fixes_2026-10-04.csv`; the review CSV and sheets: `data/label_check/`.
The random audit sample of 150 trusted crops (`review_audit.csv`) was reviewed by hand: **0 wrong labels in 150**, i.e. an error rate of 0% with a 95% interval of 0-2.5% (Wilson), about 88 wrong labels at the very worst across the 3,523 trusted crops. The trusted labels are therefore reliable; the remaining label noise is concentrated in the 298 flagged crops, which were reviewed and corrected.
Most common corrections: `v`<->`V` (16 crops, case), `I`->`1` (8), `II`/`ii`->`11` (9), and crops labelled `1` that show other numbers. The cross-check also showed that the data held crops that are not
numbers (page-edge slivers, strokes, blobs) labelled `1`, which probably explains the model answering `1` for junk. The models were NOT retrained after the fixes.
Effect on `parseq_master_v3` (trained on the old labels): real test 378 crops, **96.3% (364/378) against the corrected labels** vs 95.2% (360/378) on the same crops with the old labels.
The numbers in section 5 were measured on the old 379-crop test set and old labels, so they understate the model by about 1.1 points.

Real data: 3,827 labelled crops from about 100 documents (images in `D:\all_roi_images`, labels in `D:\annotations.xlsx`, column `Normalize`).
Document-level split seed 42: 155 / 17 / 10 documents = 3,064 / 384 / 379 crops; no document appears in two sets.

**Label convention:** the label is the characters only (digits and letters, case kept). Brackets, dots, spaces and a `Q.`/`Q` prefix are drawn in
the image but stripped from the label, matching the workbook's `Normalize` column (`Q.3` -> `3`, `2a)` -> `2a`). 7 real marked labels were
stripped the same way (`III.` -> `III`, `A.` -> `A`, `2)` -> `2`).

**Synthetic generator** (`src/master/spec.py`, `render.py`, `build_master.py`; seed 0; fonts are Windows handwriting fonts; glyphs from EMNIST
banks `data/emnist/glyphs_v5.npz` + `glyphs_single.npz`):

| bucket | content | notes |
|---|---|---|
| B1 NNa | two-digit number + one letter (`12a`, `47k`, `05z`), balanced by character, about 5,000 NNa crops incl. 2,000 with marks | only 4 real two-digit+letter crops existed (`16b` x1, `17a` x3), so this is purely synthetic |
| B2 single digit / letter | 0-9, a-z; target 450 per digit, 150 per letter | |
| B3 legacy floor | `NN` alone (target 25 each), Roman numerals i-x both cases (target 100 each), `Na` (400) | |

Image styles: plain; trailing mark `3)` `3.` `3.)` (18%); wrapped `(3)` (8%); `Q` prefix `Q.3` `Q16` (30%); interior marks `2.a` (4%); square
brackets `[3]` (2%). Synthetic crops per label are capped at 5x the real count (floor 20). Page augmentation: elastic warp, perspective, rule line,
blur, contrast. Distribution report: `docs/log/reports/master_distribution.md`; label counts: `docs/log/master_label_counts.csv`.

## 5. Results of `parseq_master_v3`

All on unseen documents. Real test = 379 crops (document split seed 42). Scoring in `master_eval.md` is exact match on the label; 95% Wilson
intervals in brackets. `docs/log/reports/master_eval.md` has the full tables.

| measure | master_v3 | master_v2 | base_best (real only) | production `parseq_final`* |
|---|---|---|---|---|
| real test, all (379) | **95% [92-97]** (19 errors) | 96% [94-98] (15) | 97% [95-98] (11) | 100%* |
| single digit (293) | 97% | 98% | 98% | 100%* |
| single letter (16) | 88% | 81% | 88% | |
| `NN` alone (21) | 95% | 95% | 100% | |
| Roman (49) | 84% | 90% | 92% | |
| synthetic NNa plain (150) | **97%** | 98% | 13% | 13% |
| synthetic NNa with marks (150) | **97%** | 97% | 14% | 11% |
| synthetic square brackets (150) | **97%** | 95% | 13% | 13% |
| synthetic Q prefix (150) | **99%** | 97% | 57% | 47% |
| held-out new documents (56, unverified labels; crops since removed from the dataset) | **86% [74-93]** | 82% | 80% | 79% |
| ... of which with Q prefix (17) | **76%** | 59% | 59% | 53% |

\* the production model was trained on all 3,827 real crops including this test set, so its real-test numbers are not a measurement.

Re-scored on 2026-10-04 with `eval_parseq.py` (case-sensitive exact match, strict): **94.7% exact, 5.5% CER, 2 crops flagged for review at min
confidence < 0.5** (359/379). The 19-error figure above uses the master evaluation's own scoring; the two differ by one crop.

Real-test errors (19): C->c, I->1, 4->1, 2->3, VI->III, A->1, iii->m, ii->iii, I->1, 2->1, II->V, 6->4, 10->4, V->I, 2->1, 7->2, ii->i, 2->v, 2->2j.
Errors concentrate in Roman numerals and in `1`/`I`/`i`-type confusions.

Other facts: base PARSeq runs about 62 ms/image single-threaded on the dev CPU (i7-13650HX), about 25 ms at 4 threads (`docs/log/reports/cpu_benchmark.jsonl`).

Behaviour that matters for the pipeline (measured on unseen documents Doc0729-Doc0775 by eye; no ground truth):
- It answers `1` for stray strokes, lines and arcs, because nothing in its training data teaches "nothing to read".
- Words and formats outside the numeric grammar (`Add:`, `Less:`, `Ans (3)`) come out as number-like strings (`Less:` -> `iii`).
- On random synthetic words it scores 39% exact (the original pretrained PARSeq: 63%), see 8.6.

## 6. Using it

```python
# run with venv_parseq; the path must contain 'parseq' (PARSeq infers the model class from it)
from strhub.models.utils import load_from_checkpoint
model = load_from_checkpoint("models/parseq_master_v3/parseq_master_v3.ckpt").eval()
```

Or use `src/parseq/model_io.py` (load_model / make_transform / decode), `src/parseq/infer_parseq.py` (one image or a folder),
`src/parseq/predict.py` (CSV with review flags), `src/parseq/eval_parseq.py --csv <file with filename,image_path,text>`.
Preprocessing must match training: stretch to 32x128 (bicubic), `Normalize(0.5, 0.5)`. The review score is the minimum per-character softmax
probability (end token included); below `inference.flag_threshold` (0.5) a prediction is flagged. `labels.clean_pattern` flags predictions that do
not look like a valid label.

## 7. Reproducing it

```bash
# 0. environment: docs/PARSEQ_REPLICATION.md (venv_parseq, third_party/parseq clone, apply patches/parseq_train.patch)
# 1. the dataset is data/master_dataset.mdb (14,766 rows; 14,003 train). Make the PARSeq training layout from it:
venv_parseq/Scripts/python src/master/dataset_io.py unpack            # -> data/master_lmdb/{train,val,test}/handwriting
# 2. train (the recipe is config.yaml -> train:; roughly 20-30 minutes on an RTX 4050, extrapolated from 3,064-sample runs that took about 5 minutes)
venv_parseq/Scripts/python src/parseq/train_parseq.py hydra.run.dir=outputs/parseq_master/run_v3
# 3. export the deployable weights
venv_parseq/Scripts/python src/parseq/export_model.py --checkpoint outputs/parseq_master/run_v3/checkpoints/final.ckpt --out models/parseq_master_v3
# 4. evaluate (real test crops are the `test` split of the dataset; or use data/docsplit/seed42/test.csv)
venv_parseq/Scripts/python src/master/eval_master.py --model master_v3=models/parseq_master_v3/parseq_master_v3.ckpt
```

Retraining on the dataset as it is now gives a model close to, but not identical to, master_v3 (it lacks the 64 new-document crops; expect results within noise, about 1 point).
To rebuild the dataset itself from scratch (needs the workbook and the real crop folder; pass `--extra-real <csv>` to `build_master.py` to add new-document crops again):
`src/master/build_master.py --render`, then `src/master/dataset_io.py pack`. Training results vary by about
+-1 point between seeds and even between identical reruns on this GPU (see 8.7), so expect 94-96% on the real test, not exactly 94.7%.

---

## 8. Everything else we tried (ledger)

Every run in MLflow is exported to `docs/log/mlflow_runs.csv` (80 runs, 15 experiments: seed, lr, dropout, epochs, data folder, final metrics).
Result tables are in `docs/log/reports/`. All PARSeq runs start from the same pretrained weights and use the recipe in section 3 unless noted.

### 8.1 The other models (deleted on 2026-10-04)

| model | trained on | result | why it lost / notes |
|---|---|---|---|
| `parseq_final` (production until now) | all 3,827 real crops (train+val+test), no synthetic, 15 epochs | its own training score 99.45% (21 misses, all genuine ambiguity: I/1, II/11, iv/iii). Expected on new documents about 93-94% (5-fold document CV) | cannot be measured on our test (it saw it); reads no NNa (13% on the synthetic probe) |
| `parseq_base_best` | the 3,064 real train crops of document split seed 42, no synthetic | **96.8%** on the 379-crop test (seeds 43/44 similar; another rerun 96.6%) | best on plain real crops, but 13% on NNa formats and 40% on random words |
| `parseq_master_v2` | master v2: 3,064 real + 10,897 capped synthetic crops, characters-only labels | 96% [94-98] real test; synthetic probes 95-99%; held-out new documents 82% | v3 added more Q-prefix crops and did better on new documents |
| `parseq_openread_v1` | open-reading set v1 (see 8.9) | plain-number accuracy 93.4%, words 85.7%, junk rejected 99.1% | plain numbers regressed |
| `parseq_openread_v2` (seed 42 saved) | open-reading set v2 | plain 96.8%, words 84.0%, junk 98.2%, verbatim 77.9% | not adopted: unfinished (labels inconsistent for 814 crops) |
| `noise_classifier` | ResNet-18, 5,607 / 712 / 686 crops (train/val/test, document level), classes clean/noise from the annotation workbook | test accuracy **92.9%** (n=686): clean P 0.928 / R 0.949, noise P 0.929 / R 0.900; confusion 376 / 20 / 29 / 261 | not PARSeq; the production copy lives in `D:\OCR_codes\models\noise_classifier\best.pt` (byte-identical). Code and CSVs were kept |
| the first master model (v1) | marks kept in the label | 94% real test, hallucinated digits for the `Q` prefix | deleted 2026-10-02 |

### 8.2 Honest accuracy on unseen documents (document-level CV)

5 splits, no document in two sets, fixed 15-epoch recipe, real data only (`docs/log/reports/docsplit_cv_fixed.csv`):
test accuracy 96.8 / 93.5 / 92.4 / 92.0 / 94.8 -> **93.9% +- 2.0**, CER 5.5%. The exploratory recipe (early stopping, best-val checkpoint) gave 93.1% +- 1.0,
CER 6.2% (`docsplit_cv.csv`). A 50/50 document split cost about 3 points. Seed 42 is the easiest fold, which is why the single test split reads
96-97% while the real unseen-document figure is about 94%. Training loss goes to about 0.01-0.09 within a few epochs while validation loss
bottoms out at epoch 1-5 and rises: overconfident, not collapsing (accuracy plateaus), a data-size limit, not a tuning problem.

### 8.3 Hyper-parameter sweeps (Optuna TPE)

Two sweeps, 24 completed trials (learning rate, dropout, batch size, augmentation strength/depth, weight decay, warm-up; the aggressive space
also permutations and gradient clipping). Trial 0 was the hand-tuned config. **Neither beat it by more than noise** (`sweep_parseq_hparams.csv`,
`sweep_best_test_results.csv`, `model_validation.md`). The recipe in section 3 is that hand-tuned config.

### 8.4 EMNIST synthetic data (3 ways), base PARSeq, document split seed 42

Real only (`base_best`) 96.8% vs real + 6,000 stitched EMNIST sequences (`base_mix_v2`) 91.3%; real + 1,305 single-character top-ups
(`base_bal`) 95.3% (3 seeds 95.8 +- 0.7 vs 96.4 +- 0.4 for real only); two-stage (synthetic pretraining then real) and v3-v5 recipes were no better. What
synthetic data bought: NNa readiness (about 90% on a synthetic NNa probe vs about 10%) and it fixed `b->6` / `g->9` there, not `c<->e`. Details:
`docs/log/reports/emnist_experiments.md`, `emnist_balanced.md`. The glyph banks used by the master renderer came from this work.

### 8.5 Augmented-label and "diverse" synthetic sets (2026-10-01/02)

Labels varied with brackets, dots, spacing, case and drawn as images (label = characters only). Real + 5,394 synthetic variants: 94.5% vs 96.6%
for the matching real-only run. A 32,571-crop synthetic-only "diverse" set (full grammar, 16 fonts + EMNIST) stage 1: 59.4%; then fine-tuned on real
(stage 2): 93.9%. It learned to append suffix letters (`2` -> `2j`), because rare `NN`+letter labels made up half the set. None beat real-only.

### 8.6 General-text reading was lost in fine-tuning (word test, 2026-10-04)

300 synthetic word crops (common words, Capitalized, random lowercase, random mixed case) rendered in handwriting fonts, words unseen in training:

| model | exact | case-insensitive | CER | common | Capitalized | random | mixed-case |
|---|---|---|---|---|---|---|---|
| original pretrained PARSeq | **63.3%** | 73.3% | 15.8% | 85% | 80% | 49% | 39% |
| `base_best` | 40.3% | 45.0% | 33.6% | 59% | 60% | 25% | 17% |
| `parseq_final` | 37.0% | 41.7% | 38.9% | 59% | 57% | 17% | 15% |
| `master_v3` | 39.0% | 44.7% | 32.3% | 67% | 43% | 35% | 12% |

Fine-tuning on question numbers cost about 23-26 points of general reading. The pretrained model answers stray strokes with punctuation (`-`, `|`,
`)`), ours with `1`. Synthetic words are fonts, not real handwriting, so the absolute numbers are indicative only.

### 8.7 Resize: stretch vs aspect-preserving pad (2026-10-04)

Crops are nearly square (median width/height 0.86, 95th percentile 1.43, max 2.5), so stretching to 32x128 widens them about 4-5x. Pad = scale to fit
inside 32x128 keeping the aspect ratio, pad right/bottom with the median colour. Real train only, 3 seeds, final epoch:

| mode | test acc (379) | test CER | val acc (384) |
|---|---|---|---|
| **stretch** (current) | **96.5% +- 0.5** (97.1 / 96.6 / 95.8) | 3.3% | 91.3% |
| pad | 95.1% +- 0.1 (95.3 / 95.0 / 95.0) | 4.8% | 86.7% |

Pad was worse on every seed (val gap 4.6 points). Likely causes (not isolated): the pretrained weights expect text spread across the width; about 80%
of the padded input is blank; fewer patches per character. Kept stretch. The `resize_mode` option remains in the PARSeq patch (default `stretch`).

### 8.8 `charset_test` and run-to-run noise (2026-10-04)

Setting `model.charset_test` to all 94 characters changes only validation scoring (case/punctuation become significant). Same recipe, seed 42, stretch:

| run | val acc | test acc | test CER |
|---|---|---|---|
| default `charset_test` (36 chars) | 90.1% | 97.1% | 2.7% |
| default, repeated | 90.9% | 96.8% | 2.9% |
| full 94 characters | 90.6% | 96.3% | 3.3% |

**Two identical-seed reruns differ** (GPU nondeterminism), so a gap under about 1 point on the 379-crop test set is noise. `charset_test` made no measurable difference.

### 8.9 Open reading (read what is written; train on words and junk) - not adopted

Idea: label every crop with what is written, teach general text with synthetic words, and teach `~` = "nothing readable" with real junk crops.
Scripts are in `src/open_read/` (data builder, trainer, evaluator); the build is deterministic.

- **Data v1/v2/v3:** real crops from the document split (relabelled verbatim, e.g. `2` -> `2)`, `3a` -> `3a)`), plus 800 real junk crops labelled `~`
  and 1,500 synthetic words (35% common words, 20% Capitalized, 20% random lowercase, 15% random mixed case, 10% markers `Add:` `Less:` `Ans (n)`; vocabulary
  disjoint from the word-test list). Junk came from the workbook's `noise` crops (3,012), filtered to 1,175 where the original pretrained PARSeq reads no letter or
  digit (61% of "noise" crops turned out to be readable marks such as `6.` or `9.3` and were excluded). Junk was never taken from validation/test documents.
- **Verbatim relabelling** accepted a candidate (workbook `text`, then the pretrained model's reading limited to `( ) [ ] . :`) only when stripping punctuation and a
  leading `Q`/`Ans` gave back the existing label: 2,250 of 3,064 train crops kept (1,160 gained marks). v2 kept the other 814 with their old stripped labels;
  v3 (never trained; interrupted) would have relabelled 564 of them from the v1 model's readings and dropped 249 that disagreed (a sample showed apparent label errors such as
  an image of `1.` labelled `2`, an image of `5` labelled `3`).
- **Held-out measures** (same sets for every row: old-label real test 379 crops; word test 300; junk 112 crops from validation documents):

| model | real test, old labels | words | junk rejected | real test verbatim (307) |
|---|---|---|---|---|
| `base_best` | **96.8** | 40.3 | 3.6 | 44.0 |
| original pretrained PARSeq | 60.7 | 63.3 | 100 | 65.1 |
| open-reading v1, seeds 42/43/44 | 91.8 / 93.4 / 90.8 | 85.7 / 85.7 / 86.3 | 99.1 / 99.1 / 98.2 | 87.6 / 88.9 / 86.3 |
| open-reading v2, seeds 42/43/44 | 96.8 / 95.3 / 93.9 | 84.0 / 85.7 / 85.7 | 98.2 / 96.4 / 96.4 | 77.9 / 74.3 / 75.2 |

(`parseq_final`'s 99.5% is invalid: it trained on the test crops. The "verbatim" column is not like-for-like for models that cannot output brackets.)
On unseen new documents v1 read `Add:`, `Less:` (x3), `(a)`, `(A)`, `(B)`, `Ans(5)` and `1]b]` where `base_best` gave `i`, `Less`, `a`, `A`, `B`, `5`, `b`, and returned `~` for 83 of
189 crops (`base_best`: 2 rejections). Errors that remain: mark misreads (`2)b7`), words (`Aela:` for `Add:`), clipped digits.
- **Why not adopted:** v1 loses about 4 points on plain numbers; v2 recovers them but its labels are inconsistent (814 crops keep stripped labels).
  Finishing it needs consistent verbatim labels for those 814 crops (v3 plan above), about 35 minutes per three-seed run.

### 8.10 What the unseen-document analysis showed (Doc0729-Doc0775, 2026-10-04)

The recogniser is not the main source of failures on new documents:
1. **Junk crops read as `1`:** the ROI detector cuts stray strokes, strike-throughs, arcs and page-edge shadows (e.g. 15 ROIs from a few dashes on one page); the model has to answer something.
2. **Clipped ink:** by a rough pixel test about 12% of ROIs (40 of 329) have ink touching the left edge, 85 the right edge (mostly the printed separator line, by design `crop.right_padding: 5`).
   `1]b]` loses its `1` because the digit sits at x=0 of the scan; padding the box cannot recover ink that is not in the image. Decision: no edge-touch handling for now.
3. **Formats outside the numeric grammar:** `Add:`, `Less:`, `Ans (n)`, `(A)`.

### 8.11 Label facts worth knowing

- Real labels: 1,994 single-digit, 351 multi-digit, 171 single letter a-j, 548 other (Roman/upper/parens) of 3,064 train crops; only 123 labels have 3+ characters; only 59 are NNa-shaped (1.9%).
- The annotation workbook `D:\annotations.xlsx` has 5,765 rows (`text` verbatim, `Normalize` = training label); the copy kept in `data/raw/` is an older 1,962-row version.
  In that older copy 424 of 1,021 matched crops had a verbatim text differing from the trained label (brackets, dots, `Q`, `Ans`). One crop had verbatim `V` but trained label `VI`: expect some label noise.
- Pretrained PARSeq on the real test with our metric: 60.7% (it reads brackets and words that our labels strip).

---

### 8.12 Freezing encoder layers to keep general reading (2026-10-05) - not adopted

Idea: lock the early (image-reading) layers so fine-tuning cannot erase general reading (8.6). Same recipe and data as `master_v4` (15 epochs, seed 42, corrected data), one run each.
Scored on the corrected real test (378 crops) and the word test (300 unseen words), exact match:

| model | real test | words |
|---|---|---|
| `master_v4` (all layers, full learning rate) | 96.8% | 37.0% |
| pretrained, no fine-tuning | 28.0% | 63.3% |
| `+freeze=encoder` (whole encoder locked, decoder only learns) | 90.2% | 19.0% |
| `+freeze=half` (patch embedding + first 6 of 12 blocks locked) | 94.7% | 38.3% |
| `+early_lr_mult=0.1` (those early layers learn at 10% of the learning rate) | 94.2% | 42.0% |

Result: every variant is 2-6 points below `master_v4` on real crops (more than the ~1-point run noise) and only the 10% learning-rate variant moved the word score (+5 points, still far from 63%).
So the loss of general reading is not mainly early-layer drift; it comes from the training labels/data (digit and code labels, ~78% synthetic rows). Next ideas: mix general word crops into training,
a real-heavy mix, more real documents. The options exist in `third_party/parseq/train.py` / `strhub/models/base.py` (`+freeze=none|encoder|half`, `+early_lr_mult=<f>`) but are not in `patches/parseq_train.patch`
(`third_party/` is untracked). The three run folders (about 1 GB), the unpacked `data/master_lmdb` and `mlflow.db` were deleted afterwards.

### 8.13 `parseq_balanced_run`: synthetic crops only where real data is scarce (2026-10-05) - saved, not the default

Idea: look at the label counts in the real training crops and keep synthetic crops only where real data is scarce. Real train has 3,057 crops (+72 repeated Q-prefix) over 84 labels:
digits 1-5 have 270-407 each, 6-10 73-135, `ii` 64, `b` 61, `a` 44, most Romans 14-48, letters `A`-`D` 11-23, two-digit numbers 40 (`11`) down to 1, NNa-shaped labels 1-13 (59 crops), and 2,551 label strings that never occur.

**Data rule** (subset of the existing synthetic pool in `master_dataset.mdb`, nothing re-rendered; subset seed 42): a label with fewer than 40 real crops is topped up with synthetic crops up to 40, as far as the pool allows;
the NNa family (2,265 distinct labels, 5,182 synthetic crops in master) is held to a random 600; labels with 40+ real crops get no synthetic crops.
Train = **6,505 rows** (3,129 real incl. 72 repeated Q-prefix + 3,376 synthetic = 52% synthetic vs 78% in master); val 382, test 378 (identical to master). Per-label counts: `docs/log/balanced_run_label_counts.csv`.

**Model:** `models/parseq_balanced_run/parseq_balanced_run.ckpt` (95 MB, sha256 prefix `7b59af6f7d2b5808`; the folder name contains `parseq` because PARSeq infers the model class from the path).
Recipe identical to section 3 (pretrained start, 15 epochs, lr 1.5e-3, dropout 0.2, RandAugment 9/3, batch 32, seed 42, stretch resize, 94-char charset); full resolved config: `docs/log/balanced_run_train_config.yaml`; MLflow row in `docs/log/mlflow_runs.csv` (82 runs).
The first run of this recipe (weights deleted by mistake during cleanup) scored 95.5% / 44.3% on the same sets; this is the retrained copy, so figures are the retrained model's.
Subset code: about 25 lines on top of `dataset_io.samples()` (not kept in the repo; the rule above reproduces it).

| | `master_v4` | `parseq_balanced_run` |
|---|---|---|
| real test (378), exact match | **96.8%** (12 errors) | 95.5% (17 errors), CER 4.7% (v4 3.4%) |
| words (300) | 37.0% | **43.7%** |
| synthetic probe, 150 each: NNa plain / marks / square brackets / Q prefix | 98 / 97 / 97 / 99% | 92 / 95 / 91 / 97% |
| real single digits (293) / single letters (16) / Roman (49) | 99 / 88 / 90% | 99 / 81 / 80% |
| val accuracy at last epoch (382) | 92.1% | 92.9% |

balanced_run errors (label->read): C->c, I->1, 1->2, V->II, i->j, 9->91, X->I, VI->VII, A->1, iii->u, JQ->V, ii->iii, V->IV, 7->2, ii->i, a->i, ii->iii.

Result: less synthetic data kept real digits equal (99%) and gave back about 7 points of general word reading, but cost about 1.3 points on real crops, 2-6 points on the format probes and 10 points on Romans
(more look-alike mixups `V->IV`, `VI->VII`). The NNa / bracket formats need their volume. `master_v4` remains the best on our real test; `parseq_balanced_run` is the one to try when general reading matters more.
`config.yaml` still points at `master_v3`.

## 9. Findings that should shape the next step

1. More distinct handwriting (documents), not tuning, is what moves unseen-document accuracy: sweeps, resize, augmentation and `charset_test` all stayed within noise.
2. Plain real-only fine-tuning wins on plain numbers; synthetic data is what buys the NNa / bracket / `Q` formats. `master_v3` is the compromise we kept.
3. The failures flagged on new documents are mostly junk crops, clipped ink and out-of-grammar formats. A reject path (`~` class or a pre-filter) and open reading address the first and third; neither fixes clipping.
4. Compare models only with a margin above about 1 point and on more than one seed.
5. Open reading with consistent verbatim labels (finish the 814 crops) is the most promising unfinished direction.

## 10. `parseq_master_v4` (added 2026-10-04): the same recipe on the corrected data

| | |
|---|---|
| File | `models/parseq_master_v4/parseq_master_v4.ckpt` (95 MB weights-only; sha256 starts `8778e99cb50f001f`) |
| Recipe | identical to master_v3 (section 3): pretrained start, 15 epochs, lr 1.5e-3, dropout 0.2, RandAugment 9, batch 32, seed 42; config `docs/log/master_v4_train_config.yaml` |
| Data | `data/master_dataset.mdb` after the label review: train 13,996 rows (3,057 real + 72 repeated Q-prefix + 10,867 synthetic), val 382, test 378. Differences from v3: 109 corrected real labels, 10 junk crops removed, **no new-document crops** (v3 had 64, x5) |
| Final-epoch validation | 92.1% (lenient 36-char metric), NED 93.2%, val loss 1.64 (MLflow row in `docs/log/mlflow_runs.csv`) |

Results on the corrected 378-crop real test (same crops, same labels for both models; scripts `eval_parseq.py` and `eval_master.py`, table in `docs/log/reports/master_eval_v3_v4.md`):

| measure | master_v3 | master_v4 |
|---|---|---|
| real test exact match (case-sensitive) | 96.3% (364/378) | **96.8% (366/378)** |
| CER | 4.0% | 3.4% |
| single digit (293) | 98% | 99% |
| single letter (16) | 88% | 88% |
| `NN` alone (20) | 95% | 95% |
| Roman (49) | 88% | 90% |
| synthetic NNa plain / with marks / square brackets / Q prefix (150 each) | 97 / 97 / 97 / 99% | 98 / 97 / 97 / 99% |

v4 was right where v3 was wrong on 5 crops, the reverse on 3, both wrong on 9: the difference of 2 crops is **within the roughly one-point run-to-run noise**, so treat v3 and v4 as equal on the real test.
v4 errors (label->read): C->c, 4->5, 3->5, X->I, A->1, iii->m, JQ->12, ii->iii, V->I, 7->2, ii->i, 2->1. It keeps the NNa / bracket / Q-prefix formats (no loss on the synthetic probes).
What v4 gives that v3 does not: a dataset and a model that are consistent with the reviewed labels, and no dependence on the unverified new-document labels.
Note the earlier numbers in section 5 (old labels, 379 crops) are not comparable with this table.

## 11. What was deleted on 2026-10-04 and how to recover it

- **Models:** `parseq_final`, `parseq_base_best`, `parseq_master_v2`, `parseq_openread_v1/v2`, `noise_classifier` (the OCR_codes copy of the last one remains). Not recoverable except by retraining
  from the recipes above (`parseq_final`: `data/all_data` was all 3,827 real crops, 15 epochs, seed 42; `base_best`: real doc-split train only, same recipe).
- **Datasets:** the per-experiment sets (random split, all-data, 4 of 5 document splits, half split, EMNIST sequence LMDBs, noise datasets in LMDB form, open-read v1-v3 training images,
  `master_lmdb`, the `master_eval` outputs, assorted `roi_*` comparison files) were redundant. Everything the kept model needs is in `data/master_dataset.mdb` (unpack gives `master_lmdb` back).
  Also deleted: `glyphs.npz` (the raw EMNIST glyph bank, 123 MB) and `glyphs_g9.npz`; the renderer only needs `glyphs_v5.npz` and `glyphs_single.npz`, which were kept. Rebuilding those two would need the EMNIST
  download again (https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip, 562 MB).
  Kept: `data/master_dataset.*`, `data/docsplit/seed42`, `data/emnist/glyphs_{v5,single}.npz`, `data/master/test.csv` (eval_master),
  `data/noise_classifier` CSVs, `data/raw`, evaluation sets under `data/open_read` (word test, junk and verbatim val/test CSVs, relabel drafts).
- **Config:** `configs/` (Hydra experiment files, sweep and validation lists), `config.local.yaml` and `config.example.local.yaml` are merged into the single `config.yaml`.
  The machine-specific values (`D:/all_roi_images`, `D:/annotations.xlsx`, `D:/tmp`, `D:/cache/torch`) are now inside it.
- **Code (tracked in git, restore with `git checkout HEAD -- <path>`):** `src/common/{prepare_dataset,prepare_alldata,prepare_docsplit,summarize_validation}.py`,
  `src/emnist/{run_experiments,archive_models,emnist_metrics,extract_emnist}.py`, `src/parseq/{docsplit_cv,sweep,validate_models}.py`, `docs/{CODEBASE_GUIDE,CODEBASE_OVERVIEW,EMNIST_EXPERIMENTS}.md`.
  **Deleted and NOT in git** (untracked, results are in this report): `src/parseq/compare_resize.py` (stretch-vs-pad driver, 8.7), `src/emnist/make_g9_glyphs.py`.
  **Deliberately kept although not needed for `master_v3`:** `src/noise_classifier/` (the production noise classifier is trained here; untracked), `src/open_read/` (the unfinished open-reading
  direction, 8.9; untracked). Say the word to delete them too.
- **After the v4 retrain:** its training outputs (`outputs/parseq_master/run_v4`, about 1.4 GB of checkpoints and TensorBoard events), the unpacked `data/master_lmdb` and the new `mlflow.db` were deleted; only the exported `models/parseq_master_v4` remains (config in `docs/log/master_v4_train_config.yaml`, MLflow row appended to `docs/log/mlflow_runs.csv`, now 81 runs).
- **Tracking and logs:** `mlflow.db`, `mlruns/` (exported to `docs/log/mlflow_runs.csv`), `outputs/` (checkpoints, TensorBoard events), `reports/` (summaries archived in `docs/log/reports/`), all `*.log`.
- **Archived instead of deleted:** `docs/log/` holds the old README, MODEL_LINEAGE.md, DATASETS.md, the label-count tables, the MLflow export and all result tables.
