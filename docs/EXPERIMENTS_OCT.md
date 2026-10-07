# EXPERIMENTS_OCT: controlled experiment suite on `parseq_master_v3`

Living report. Branch `exp_oct`. Plan: [EXPERIMENTS_OCT_PLAN.md](EXPERIMENTS_OCT_PLAN.md). Outputs: `outputs/exp_oct/`.
Live log (updates every minute while a run is going): `Get-Content outputs\exp_oct\progress.log -Wait -Tail 30`

## Status

| Session | Phase | State |
|---|---|---|
| 0 | Infrastructure, baseline reproduction gate | **DONE, gate passed** |
| 1 | P0 baseline x3 seeds, P1 WiSE-FT, soups | **DONE** (baseline 94.6 +- 0.8; WiSE-FT a=0.9 recommended; soup no gain) |
| 2 | P2 Replay | **DONE** (30% replay recommended: real 94.5, words 89.4) |
| 3 | P4 Temperature scaling + label smoothing | **DONE** (both negative / not adopted; see Phase 4) |
| 4 | P5 Learning curve | not started |
| 5 | P3 L2-SP (moved to optional) | not started |
| 6 | Combination, SSL plan, final report | not started |

Exported models for ROI testing: `outputs/exp_oct/models/parseq_replay30_s{42,43,44}/` (primary s44). See the revised plan in EXPERIMENTS_OCT_PLAN.md.

## Session 0: infrastructure and gate

Built: [src/exp_oct/eval_suite.py](../src/exp_oct/eval_suite.py) (all 7 metrics), [tests/test_eval_suite.py](../tests/test_eval_suite.py)
(coverage/precision/tier/ECE/Wilson on hand-made data: passes), [src/exp_oct/run_queue.py](../src/exp_oct/run_queue.py) (resumable train + eval + provenance).

**Checks**
- No empty labels in train/val/test (0 / 0 / 0). Documents: train 155, val 17, test 10; val and test, train and test, train and val share **no document**.
- Test = `data/docsplit/seed42/test.csv` (379 crops), val = same folder `val.csv` (384), words = `data/open_read/word_eval.csv` (300).

**Gate 0: existing `master_v3.ckpt` on the 379-crop test (case-sensitive, charset_test = 94)**

| metric | value |
|---|---|
| exact match | **359/379 = 94.7%** (Wilson 95% CI 92.0-96.6) -> matches the report, **gate passed** |
| word test | 39.0% (report: 39%) |
| coverage at 99% precision (T from val) | 45.4% of test accepted, precision 99.4% |
| confident errors (conf > 0.9) | 6 |
| ECE / NLL | 0.033 / 0.151 |

Details: `outputs/exp_oct/eval/master_v3_existing/`.

**Things to know before reading any later number**
1. **The baseline retrain is not the original master_v3.** `data/master_dataset.mdb` was edited on 2026-10-04 (109 label fixes, 10 junk crops removed, the 64 new-document
   crops taken out). Unpacked it gives 13,996 train rows (master_v3 saw 14,323). Our `baseline_s{seed}` runs use this current dataset; they are the reference for
   all comparisons, as specified. The existing `master_v3.ckpt` is reported separately.
2. Test/val CSVs in `data/docsplit/seed42/` still carry the **old labels** (379/384 crops), as instructed. The label review found about 1 point of test labels were wrong, so all
   absolute numbers understate the models by roughly that much; comparisons between methods are unaffected.
3. Speed: about 66 s/epoch, so about 17 min per 15-epoch run (faster than the 20-30 min assumed). Time estimates in the plan are therefore conservative.

## Phase 0: baseline (master_v3 recipe retrained on the current dataset, seeds 42/43/44)

| Phase 0 baseline | real test % | ignoring marks % | words % | coverage@99 % | precision on accepted % | confident errors | ECE | NLL |
|---|---|---|---|---|---|---|---|---|
| seed 42 | 94.2 | 94.5 | 37.3 | 38.5 | 99.3 | 8.0 | 0.035 | 0.154 |
| seed 43 | 94.2 | 94.5 | 42.3 | 55.4 | 99.5 | 3.0 | 0.033 | 0.149 |
| seed 44 | 95.5 | 95.8 | 40.7 | 54.6 | 99.0 | 7.0 | 0.034 | 0.129 |
| **mean +- std (3 seeds)** | 94.6 +- 0.8 | 94.9 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 99.3 +- 0.2 | 6.0 +- 2.6 | 0.034 +- 0.001 | 0.144 +- 0.013 |

Noise floor measured: real test std 0.8 points (range 94.2-95.5), words std 2.5, **coverage at 99% std 9.5 points (38.5 to 55.4)**. Coverage is far noisier than accuracy, so
a coverage gain is only believable if it is well above about 10 points or holds in all three seeds. The existing `master_v3.ckpt` (94.7 / 39.0 / 45.4) sits inside this spread.

## Phase 1: WiSE-FT and soups (no training)

State-dict check: pretrained PARSeq and each fine-tuned checkpoint have the same 175 keys with identical shapes (passed). Sanity (every model): a=1.0 reproduces the
checkpoint and a=0 reproduces the pretrained model **exactly** (predictions identical, confidences within 1e-4). The pretrained model scores 60.7% real / 63.3% words as in the
report, but **only under the "ignore marks" metric** (alphanumerics only, as in `evaluate_open_read.py`); under strict case-sensitive match it is 27.4%, because it writes the
brackets and dots our labels strip (`2.`, `4)`, `III.`). Both are shown; strict is the headline.

| WiSE-FT (mean +- std over the 3 baseline seeds) | real test % | ignoring marks % | words % | coverage@99 % | precision on accepted % | confident errors | ECE | NLL |
|---|---|---|---|---|---|---|---|---|
| a = 0.0 | 27.4 +- 0.0 | 60.7 +- 0.0 | 63.3 +- 0.0 | 0.0 +- 0.0 | nan +- nan | 80.0 +- 0.0 | 0.474 +- 0.000 | 1.987 +- 0.000 |
| a = 0.25 | 47.0 +- 1.4 | 76.2 +- 0.9 | 68.7 +- 0.9 | 8.9 +- 1.7 | 100.0 +- 0.0 | 56.0 +- 1.0 | 0.312 +- 0.010 | 0.970 +- 0.024 |
| a = 0.5 | 80.7 +- 0.7 | 86.1 +- 0.3 | 67.7 +- 0.3 | 28.1 +- 1.0 | 100.0 +- 0.0 | 10.3 +- 2.5 | 0.068 +- 0.014 | 0.324 +- 0.029 |
| a = 0.75 | 92.0 +- 1.1 | 92.3 +- 1.1 | 61.2 +- 0.8 | 36.1 +- 7.6 | 99.3 +- 0.1 | 5.3 +- 1.5 | 0.031 +- 0.007 | 0.172 +- 0.013 |
| a = 0.9 | 94.5 +- 1.2 | 94.7 +- 1.2 | 49.3 +- 1.7 | 43.1 +- 10.7 | 99.2 +- 0.2 | 5.3 +- 1.5 | 0.029 +- 0.001 | 0.144 +- 0.011 |
| a = 1.0 | 94.6 +- 0.8 | 94.9 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 99.3 +- 0.2 | 6.0 +- 2.6 | 0.034 +- 0.001 | 0.144 +- 0.013 |

| Soup | real test % | ignoring marks % | words % | coverage@99 % | precision on accepted % | confident errors | ECE | NLL |
|---|---|---|---|---|---|---|---|---|
| soup of 3 seeds | 94.7 | 95.0 | 42.7 | 53.6 | 99.0 | 7.0 | 0.030 | 0.150 |
| individual seeds (mean +- std) | 94.6 +- 0.8 | 94.9 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 99.3 +- 0.2 | 6.0 +- 2.6 | 0.034 +- 0.001 | 0.144 +- 0.013 |

Single-model version on the existing `master_v3.ckpt` (the earlier preliminary run): a=0.9 gives 94.2% real / 49.0% words; a=0.75 gives 92.6% / 63.3%. Same picture.

**Trade-off (real vs words, 3-seed means):** a=1.0 94.6 / 40.1 - a=0.9 94.5 / 49.3 - a=0.75 92.0 / 61.2 - a=0.5 80.7 / 67.7. Words peak at about 68% (a=0.25-0.5), above the pretrained 63%,
but real accuracy collapses below a=0.75.

**Decision (Session 1 gate)**
- **WiSE-FT is a word-recovery method, not an accuracy method**: it does not beat the baseline on real test (best 94.6 at a=1.0, a=0.9 is 94.5) and coverage at 99% *falls* as a drops
  (49.5 -> 43.1 at a=0.9, but that drop is inside the 9.5-point noise; at a=0.75 and below it is clearly lower). Under the decision rule it does not count as "better" on real test or coverage.
- **Recommended point: a = 0.9.** It keeps real test within 1 point of baseline (-0.1) and lifts words by +9.2 points (40.1 -> 49.3, about 3.7 std). a=0.75 gets +21 words but costs 2.6 points on real test, which breaks the 1-point limit.
- It costs nothing to train, so a=0.9 goes forward as a **combination candidate** (applied last, to whichever recipe wins).
- **Soup of the 3 seeds: no clear gain.** 94.7 real (+0.1), 42.7 words (+2.6, about 1 std), coverage 53.6 (inside the noise). Recorded as a negative/neutral result.
- Baseline std (0.8) confirms the spec's rule of thumb: a real-test gap below about 1 point is noise.

## Phase 2: replay (word crops mixed into training)

**Set-up.** Words from the open_read generator (same word kinds and renderer, new seed 777), literal labels, no `~` class, no number labels changed. N word rows **replace** N randomly
chosen *synthetic* rows so every run keeps 13,996 rows = 438 steps/epoch = **6,570 steps** (identical to baseline). Real rows are never removed. Sets are nested (10% is inside 20% inside 30%).
Counts: 10% = 1,400 words (9,467 synthetic kept), 20% = 2,799 (8,068), 30% = 4,199 (6,668); real rows 3,129 throughout.

**Leakage check (asserted in code, printed): overlap = 0** between replay labels and (a) the 198 distinct word-test words, (b) the 2,619 distinct labels of the master dataset
(master_v3 trained on numbers, letters, Roman numerals only, no words). The generator's own word list *did* overlap the word test on `poverty`, `state`, `total`; those three were removed
(73 words left). Comparison is case-insensitive. Replay uses 4,199 rows but only 1,707 distinct strings (short vocabulary, many renderings).

Screening (seed 42): real/words 10% = 93.7/87.3, 20% = 93.9/89.3, 30% = 93.9/89.0. All were at least baseline mean - 1 (93.6), so by the pre-set rule the top 2 by words, **20% and 30%**, were confirmed with seeds 43 and 44.

| Phase 2 replay (mean +- std over seeds) | real test % | ignoring marks % | words % | coverage@99 % | precision on accepted % | confident errors | ECE | NLL |
|---|---|---|---|---|---|---|---|---|
| baseline 0% (3 seeds) | 94.6 +- 0.8 | 94.9 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 99.3 +- 0.2 | 6.0 +- 2.6 | 0.034 +- 0.001 | 0.144 +- 0.013 |
| replay 10% (seed 42 only, screened out) | 93.7 | 93.9 | 87.3 | 60.2 | 99.1 | 7.0 | 0.038 | 0.150 |
| replay 20% (3 seeds) | 93.6 +- 0.6 | 93.8 +- 0.6 | 88.3 +- 0.9 | 54.8 +- 9.6 | 99.6 +- 0.4 | 7.7 +- 2.1 | 0.037 +- 0.003 | 0.151 +- 0.018 |
| replay 30% (3 seeds) | 94.5 +- 0.7 | 94.8 +- 0.7 | 89.4 +- 0.5 | 50.9 +- 10.7 | 99.5 +- 0.4 | 9.0 +- 1.0 | 0.036 +- 0.001 | 0.143 +- 0.016 |

![real vs word accuracy](../outputs/exp_oct/tradeoff_real_vs_words.png)

(Plot: `outputs/exp_oct/tradeoff_real_vs_words.png`; WiSE-FT curve and replay points on the same axes.)

**Reading**
- **Words: +48 to +49 points** (40.1 -> 88.3 at 20%, 89.4 at 30%), consistent across all seeds (std 0.5-0.9). Far outside noise.
- **Real test:** 20% = 93.6 (-1.0 vs baseline, at the limit; seed 43 alone was 92.9); **30% = 94.5 (-0.1, indistinguishable from baseline)**. The 20% vs 30% real-test difference is within seed noise, so
  there is no evidence that more replay hurts real test; 10% (single seed) was also slightly low (93.7).
- **Coverage at 99%:** 54.8 / 50.9 vs 49.5, all inside the +-9.5 baseline spread: no coverage claim. Confident errors 7.7-9.0 vs 6.0 (small, within spread). ECE/NLL unchanged.
- Replay does not improve real-test accuracy (it was not expected to); it is purely word recovery.

**Decision (Session 2 gate)**
- Not "better" under the strict rule (no real-test gain, no clear coverage gain) but it is a word-recovery method and the rule for those is the trade-off curve. **Recommended point: 30% replay** - real test 94.5 (within 1 point of baseline, in fact -0.1) and words 89.4. 20% sits exactly at the 1-point limit, so 30% is safer.
- vs WiSE-FT: replay reaches ~89% words at ~94.5% real; WiSE-FT a=0.9 only 49% words at 94.5% real, and a=0.75 reaches 61% only by losing 2.6 real points. **Replay is much better on the trade-off curve than WiSE-FT.** It costs a full training run, which WiSE-FT does not.
- Combination candidate: replay 30% (and WiSE-FT would add little on top, to be checked in the combined run if it is one of the two methods selected).

**Caveats**
- **The word test is probably easy for replay models.** Replay words and the word-test words both come from the same synthetic generator (same fonts, glyph bank and page augmentation); only the vocabulary is disjoint. The 89% shows the model reads *these* kinds of words, not that it reads real handwritten words. We have no real handwritten word test.
- **Confound:** replay also removes synthetic number rows (at 30%, 39% of them). Real test was not hurt, but this was not separated from the effect of adding words.
- 10% was only screened with one seed.

## ROI check on the user's own crops (226 crops, reviewed ground truth)

Truth = `corrected_text` of `D:/OCR_codes/data/roi_watch/review_queue.csv`, reviewed by the user for all 226 crops (142 document crops, 84 screenshots). Script `src/exp_oct/roi_score.py`;
per-crop predictions of every model: `outputs/exp_oct/roi_check/roi_scored.csv`. Exact match, case-sensitive. "pipeline" = the `prediction` column of the user's own run (which weights it used is unknown).
Document crops of Doc0750/0752/0755/0767/0769/0775 were in master_v3's training set (and probably the pipeline's); the baseline/replay models (retrained on the current dataset) never saw any of these crops.

| model | correct / 226 | exact % [Wilson 95%] | confident errors (conf >= 0.9) | accepted at conf >= 0.9 | precision of accepted |
|---|---|---|---|---|---|
| pipeline (user's run) | 185 | 81.9 [76.3-86.3] | 24 | 83% | 87.2% |
| master_v3 | 181 | 80.1 [74.4-84.8] | 27 | 84% | 85.8% |
| baseline (3 seeds) | 181 / 183 / 184 | **80.8 +- 0.7** | 16.7 +- 1.5 | 80% | 90-92% |
| replay 20% (3 seeds) | 187 / 187 / 191 | **83.3 +- 1.0** | 13.0 +- 2.6 | 76% | 91-94% |
| replay 30% (3 seeds) | 188 / 188 / 189 | **83.3 +- 0.3** | 12.7 +- 2.1 | 74% | 91-94% |

By subset (correct/n, exact %):

| model | held-out documents (68; never in any training) | documents master_v3 trained on (74) | screenshots (84) |
|---|---|---|---|
| pipeline | 59 (86.8) | 69 (93.2) | 57 (67.9) |
| master_v3 | 56 (82.4) | 66 (89.2, leaky) | 59 (70.2) |
| baseline s42/s43/s44 | 58 / 60 / 59 (85.3 / 88.2 / 86.8) | 67 / 65 / 63 | 58 / 59 / 59 |
| replay 30% s42/s43/s44 | 60 / 62 / 62 (88.2 / 91.2 / 91.2) | 69 / 66 / 68 | 60 / 60 / 58 |
| replay 20% s42/s43/s44 | 61 / 58 / 59 | 67 / 66 / 68 | 63 / 63 / 60 |

**Reading**
- Replay is **about 2.5 points better than the retrained baseline on real new crops** (83.3 vs 80.8) and has fewer confident errors (12.7 vs 16.7). The gain is the same sign in every seed (all 6 replay runs >= 82.7; all 3 baselines <= 81.4),
  but it is 3-6 crops of 226, a single reviewed set, and mostly in the held-out documents (91.2 vs 86.8-88.2). A suggestive result that agrees with the word results; not a proven accuracy gain, and unlike the 379-crop test (where replay was -0.1 to -1.0), here it helps.
- **Real handwritten words are NOT solved.** The only word-like truths are `or` and two `Add`. `or` is read as `1` / `01` by every model. `Add` becomes `Aelo`, `Aadi`, `Adds`, `Aad`, `Addi` (replay) versus `42d`, `20`, `ii` (baseline): replay moves the
  output toward the right letters and the right kind of string, but **no model gets any of them exactly right**. The 89% word-test score is therefore on synthetic words only; on real ones the evidence is 0 of 3.
- **Screenshots (84) are the weak spot for every model (68-75%)**, and the errors are mostly a **dropped or wrong leading character** that all models share: `12 -> 2`, `19 -> 1`, `45 -> 5`, `15 -> 75`, `10c -> 0c`, `17d -> d`, `1a -> a`, `15e -> 5e`.
  This points at the crops (left edge cut or tightly cropped, wide aspect squeezed by the 32x128 stretch) rather than at any method tested; not investigated further.
- **The 99%-precision operating point does not transfer.** Applying each model's val-derived threshold (T = 0.999-1.000) accepts only 12-31% of these crops, with precision 96-100% (0-2 errors accepted). Simply accepting conf >= 0.9 gives precision of only 91-94% for the
  retrained models and 86-87% for master_v3 / the pipeline: on new documents confidence is clearly over-optimistic (consistent with the ECE and the 94% unseen-document CV in the model report). This strengthens the case for Session 3 (calibration).
- Caveat: the labels are the user's corrections of the model outputs, so they may lean toward the model's reading in ambiguous cases; the held-out/screenshot sets are small and from few writers.

## Phase 4: label smoothing and temperature scaling (Session 3)

**Label smoothing patch** (`patches/exp_oct.patch`, on top of `parseq_train.patch`, applied to `third_party/parseq`; off by default): `F.cross_entropy(..., ignore_index=pad, label_smoothing=eps)` in `PARSeq.training_step`.
Tests (`tests/test_label_smoothing.py`): with eps=0 the loss equals the original to 1e-6 on a real batch (0.052418 vs 0.052418); logits at pad positions do not affect the smoothed loss; EOS handling unchanged.
Note torch spreads the smoothing mass over all 97 classes including the special tokens (standard behaviour). Seed 42 screening of eps 0.05 and 0.1; eps 0.05 confirmed with seeds 43/44; **0.1 was not confirmed** (clearly worse: 3.4% coverage, 93.9% real; confirming it is 2 more runs on request).

| (mean +- std) | real test % | words % | coverage@99 % | precision on accepted % | confident errors | ECE | NLL |
|---|---|---|---|---|---|---|---|
| baseline (3 seeds) | 94.6 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 99.3 +- 0.2 | 6.0 +- 2.6 | 0.034 | 0.144 |
| replay 30% (3 seeds) | 94.5 +- 0.7 | 89.4 +- 0.5 | 50.9 +- 10.7 | 99.5 +- 0.4 | 9.0 +- 1.0 | 0.036 | 0.143 |
| **label smoothing 0.05 (3 seeds)** | 94.6 +- 0.3 | **25.6 +- 2.0** | **9.9 +- 6.3** | 98.5 +- 1.5 | 4.0 +- 0.0 | 0.047 | 0.153 |
| label smoothing 0.1 (seed 42) | 93.9 | 18.7 | 3.4 | 92.3 | 4 | 0.074 | 0.197 |

**Temperature scaling** (`src/exp_oct/phase4_temperature.py`). One scalar T per model, **fitted on the 384 val crops only** (no test or ROI data used for fitting), applied to test. Two objectives: A = token NLL (the literal spec), B = crop-level
binary NLL of "crop correct" (added after seeing A; both give nearly the same T). Accuracy is unchanged by construction. Means over 3 seeds:

| model | setting | T | test ECE | test NLL | coverage@99 % | confident errors |
|---|---|---|---|---|---|---|
| baseline | before | 1 | 0.034 | 0.144 | 49.6 | 6.0 |
| baseline | T fitted on val (A) | 2.08 | 0.060 | 0.145 | 42.4 | 2.0 |
| replay 30% | before | 1 | 0.036 | 0.143 | 50.9 | 9.0 |
| replay 30% | T fitted on val (A) | 2.05 | 0.063 | 0.148 | 54.6 | 2.0 |
| label smoothing 0.05 | before | 1 | 0.047 | 0.153 | 9.2 | 4.0 |
| label smoothing 0.05 | T fitted on val (A) | 1.00 | 0.048 | 0.154 | 9.3 | 4.0 |

**Why temperature scaling did not help on test.** The val set (17 documents) is much harder than the test set: models score about 86% on val (ECE 0.10, overconfident) but 94.5% on test (ECE 0.034, well calibrated).
A T fitted on val (about 2) therefore makes test *under*-confident and raises test ECE. This is document-to-document variation, not a bug (the same T from two different objectives; the logic is checked by `tests/test_eval_suite.py` and by the identical argmax).
Label smoothing already does the calibration job on val: its fitted T is about 1.0.

**On the user's 226 reviewed ROI crops** (a third, independent set where models are overconfident; `src/exp_oct/roi_temperature.py`, mean of 6 baseline/replay30 models): T fitted on val lowers ECE 0.105 -> 0.082-0.092, confident errors (conf >= 0.9) 14.7 -> 1.6, and at conf >= 0.9 it
accepts 37-39% of crops at 98% precision instead of 77% at 91.6%. That is mostly the same ranking with every confidence moved down, so it does not improve the coverage/precision trade-off; it makes the number "0.9" mean about 0.9 on new documents.

**Coverage vs precision** (`outputs/exp_oct/coverage_vs_precision.png`; precision of the accepted crops at fixed coverage, mean of 3 seeds)

| coverage | 50% | 70% | 80% | 90% |
|---|---|---|---|---|
| real test: baseline | 99.1 | 99.2 | 99.2 | 98.4 |
| real test: replay 30% | 99.5 | 99.2 | 99.2 | 98.0 |
| real test: label smoothing 0.05 | 98.9 | 99.1 | 99.1 | 98.5 |
| ROI crops: baseline (acc 80.8) | 96.5 | 92.8 | 91.0 | 85.6 |
| ROI crops: replay 30% (acc 83.3) | 97.6 | 94.3 | 90.6 | 88.4 |
| ROI crops: label smoothing 0.05 (acc 82.9) | 96.2 | 94.9 | 92.1 | 88.5 |

![coverage vs precision](../outputs/exp_oct/coverage_vs_precision.png)

**Decisions (Session 3 gate)**
- **Label smoothing: negative result.** Real test unchanged (94.6), words fall 40.1 -> 25.6, and coverage at 99% falls 49.5 -> 9.9. Its ranking of crops is no better: precision at fixed coverage matches the baseline on the test set (98.9-99.1 vs 99.1-99.2),
  and on ROI crops it is within 1-3 points (ECE 0.06 vs 0.105 there, but that is the confidence shift, not a better ordering). The low "coverage at 99%" comes from capped confidences meeting a val-chosen threshold, not from worse accuracy. **Not adopted.**
- **Temperature scaling with one T fitted on this val set: not adopted as a default.** It worsens test calibration because val and test differ. The calibration data has to resemble the deployment documents; the ROI result shows it can help there.
  Recommendation: refit T on a reviewed sample of the actual production documents, and re-pick any accept threshold on the same sample.
- **The "coverage at 99% precision with a threshold from val" metric is fragile**: it depends on how hard the 17 val documents happen to be (val accuracy 86% vs test 94.5%), which explains the 9.5-point seed spread and the large swings above. Treat coverage claims with caution; precision-at-fixed-coverage is more stable.
- Nothing from this session goes into the combination except possibly temperature scaling refit on in-domain data.

## PARSeq pipeline vs Chandra OCR on the 226 reviewed ROI crops

Script `src/exp_oct/chandra_compare.py`; per-crop table `outputs/exp_oct/roi_check/chandra_vs_parseq.csv`; Chandra raw text (transcribed from the user's Colab log) `outputs/exp_oct/roi_check/chandra_raw.py`.
PARSeq = the `prediction` column of the user's review queue (their pipeline model; exact match). Truth = the user's `corrected_text`. Chandra writes the marks (`a.1.`, `(b)`, `6)`, `Q 16]`) and sometimes LaTeX, words or other scripts,
while the labels are characters only, so Chandra is scored after a stated normalisation: **strict** = Unicode NFKC, keep only A-Za-z0-9, case-sensitive; **lenient** = strict, then drop a leading Q/q/A/a directly before a digit and ignore case.
Transcription of the pasted log into the file was done by hand; 226/226 crops matched.

| method | all 226 (Wilson 95%) | document crops (142) | screenshots (84) |
|---|---|---|---|
| PARSeq pipeline, exact | **185 = 81.9%** [76.3-86.3] | **128 = 90.1%** | **57 = 67.9%** |
| Chandra, raw text exactly as printed | 30 = 13.3% | | |
| Chandra, strict normalisation | 99 = 43.8% [37.5-50.3] | 69 = 48.6% | 30 = 35.7% |
| Chandra, lenient normalisation | 128 = 56.6% [50.1-62.9] | 86 = 60.6% | 42 = 50.0% |
| Chandra, lenient + circled/Phi marker before digits ignored (sensitivity) | 132 = 58.4% | 86 = 60.6% | 46 = 54.8% |

**Agreement (PARSeq exact vs Chandra lenient):** both right 113, only PARSeq right 72, only Chandra right 15, both wrong 26 (oracle either-right 200/226 = 88.5%). The two give the same string on 116 crops and are right 97.4% of the time there; on the 110 where they differ, PARSeq is right 72 times, Chandra 15, neither 23.

**How they differ**
- Chandra reads *what is written*, marks included: 141 of 226 outputs contain dots/brackets/Q prefix, and its raw output matches the label exactly on only 30 crops. PARSeq was trained to return the characters only, so it matches by design.
- Chandra's failure modes: 22 outputs are not label-like at all (13 non-Latin or symbol characters such as `س`, `司`, `ج`, `⊙`; 6 LaTeX/HTML such as `\partial_1`, `\mathcal{B}`, `<math>`; 3 sentences or runaway repetition, one crop returned about 150 copies of `Ⓒ`). It confuses the small letter labels with digits (`b` -> `6`, `6)`, `(6)`; `c` -> `(5)`; 13 of the 52 single-letter crops lost this way), and on documents it often returns pieces of neighbouring text (`1.363`, `4.65`, `R.J. a.`).
- Where Chandra beats PARSeq (15 crops) it is mostly on **bracketed or prefixed letters PARSeq reads as digits**: `(b)` vs `3`, `(A)` vs `4`, `A.` vs `9`; on the marker word `Add:` (PARSeq `Mad`); and on screenshots where it keeps the leading digit PARSeq drops (`Q 75` vs `15`, `Q 34)` vs `24`, `(16)` vs `1b`, `1. a)` vs `a`).
- Both fail on: the word `or` (PARSeq `1`, Chandra LaTeX), the second `Add` (`Mad` vs `ddq ::`), the letters/digits on tight screenshot crops (`12`, `19`, `45`, `46`, `18a`, `10c`, `15e`, `17d`).
- Screenshots: PARSeq 67.9% vs Chandra 50-55%; the leading-character problem is partly shared and partly PARSeq-specific (Chandra recovers about 8 of PARSeq's lost-leading-digit crops).

**Using them together (post-hoc, one set, not a validated recipe):** routing the crops PARSeq is unsure about to Chandra (PARSeq if conf >= 0.9 else Chandra) gives 178/226 = 78.8% and sends 17% of crops to Chandra: **lower than PARSeq alone (81.9%)**, because Chandra's answer is worse than PARSeq's even on its low-confidence crops. Agreement between the two is a much stronger signal than either confidence: when both agree (116 crops) the answer is correct 97.4% of the time.

Caveats: 226 crops, one reviewer, few writers; Chandra's cost, prompt and settings were not controlled (API via Colab), and the normalisation choices above favour Chandra; the PARSeq model is the user's pipeline model whose weights are unknown to me (it scores 185 vs 181 for master_v3 here).

## Extra experiment: padding instead of stretching (user-approved, outside the original grid)

`+data.resize_mode=pad` (aspect-preserving resize to fit 32x128, padding right and bottom), baseline recipe and data otherwise identical, seeds 42/43/44; evaluated with the same padded resize
(`run_queue.py --resize-mode pad`; `eval_suite.evaluate_model(resize_mode=...)`). Wall time 25-39 min per run (the GPU was shared for part of the time).

| (mean +- std, 3 seeds) | real test % | words % | coverage@99 % | confident errors | ECE | NLL |
|---|---|---|---|---|---|---|
| stretch baseline | 94.6 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 6.0 +- 2.6 | 0.034 | 0.144 |
| **pad** | 94.1 +- 0.3 | **12.6 +- 1.7** | 42.1 +- 1.9 | 6.7 +- 0.6 | 0.031 | 0.185 |

On the user's 226 reviewed crops (`outputs/exp_oct/roi_check/roi_pad.py`):

| | all 226 | document crops (142) | screenshots (84) |
|---|---|---|---|
| stretch baseline (s42/43/44: 183/184/181) | **80.8 +- 0.7%** | 87.3% | 69.8% |
| pad (s42/43/44: 170/171/166) | **74.8%** (166-171) | 81.7% (115-118) | 63.1% (51-56) |

**Decision: padding is worse; not adopted.** Real test is within noise (-0.5), but words collapse (40.1 -> 12.6) and the user's crops lose about 6 points on both documents and screenshots, far outside the 0.7 seed spread.
It does not fix the screenshot leading-digit problem (63% vs 70%). Likely cause: the user's crops are roughly square (median 101x104 px documents, 172x100 screenshots) so padding gives a digit only about 32x32 pixels of the 128-wide canvas
(about 4x less horizontal detail than stretching), and the wide word crops shrink the most. Not tested: a larger canvas (e.g. 64x256), which needs different position embeddings and a separate run.
It also corrects an earlier guess: screenshots are *closer* to 4:1 than the document crops (aspect 1.8 vs 1.0), so aspect distortion does not explain why they are the weak set.

## Pixel density analysis of all crops

`src/exp_oct/analyze_density.py` (read-only). 19,191 crops: the master dataset (3,129 real train, 382 val, 378 test, 10 junk, 10,867 synthetic), 4,199 replay words, 226 ROI crops.
Per-crop table `outputs/exp_oct/density/density_per_crop.csv`, group medians `summary_by_group.csv`, plot `density_histograms.png`, ROI crops joined to model correctness `roi_density_vs_accuracy.csv`.
Ink = Otsu threshold; stroke width = 2 x ink area / ink perimeter; "after stretch" = scaled by the geometric mean of the x and y scale factors to 32x128.

| group (median) | n | w x h | aspect | contrast | ink fraction | stroke px | stroke px after stretch | ink touches L / R / top / bottom % |
|---|---|---|---|---|---|---|---|---|
| real train | 3,129 | 79 x 89 | 0.86 | 155 | 0.146 | 5.1 | 3.8 | 8 / 32 / 25 / 27 |
| real val | 382 | 82 x 91 | 0.85 | 148 | 0.141 | 5.2 | 3.6 | 5 / 24 / 21 / 20 |
| real test | 378 | 74 x 84 | 0.88 | 166 | 0.144 | 5.3 | 4.2 | 9 / 22 / 18 / 19 |
| **synthetic train** | 10,867 | 87 x 64 | 1.36 | **110** | **0.099** | **3.4** | **2.8** | 8 / 7 / 2 / 14 |
| replay words | 4,199 | 95 x 64 | 1.48 | 102 | 0.081 | 2.5 | 2.2 | 16 / 13 / 2 / 13 |
| ROI document crops | 142 | 101 x 104 | 0.98 | 161 | 0.171 | 5.5 | 3.5 | 14 / 39 / 25 / 28 |
| **ROI screenshots** | 84 | 173 x 100 | 1.80 | 150 | **0.054** | **3.1** | **1.6** | **0 / 0 / 7 / 5** |

**Findings**
1. **Real crops and the user's document crops look alike** (same size, stroke width 5.1-5.5 px, ink fraction 0.15-0.17, contrast 150-165), so the document ROIs are in-domain.
2. **Synthetic crops differ from real ones in four ways:** strokes are about 1.5x thinner (3.4 vs 5.1 px; 2.8 vs 3.8 after the stretch), contrast is much lower (110 vs 155), they are wider (aspect 1.36 vs 0.86), and
   they are almost never cut by the crop edge (7%/2% touch right/top, against 32%/25% for real). Real crops are tight and often clipped; synthetic crops always sit inside a margin. Replay words are thinner and lower contrast still (2.5 px, contrast 102).
   78% of the training rows are synthetic, so the model is trained mostly on thin, faint, well-margined ink.
3. **The screenshots are out of domain, and not because the left edge is cut.** Ink touches the left border in 0 of 84 screenshots (real train: 8%). Their ink fraction is 0.054 (a third of real, 0.146), strokes are 3.1 px (real 5.1), and after the 32x128 stretch a stroke is only
   1.6 px wide (real 3.8 px), because the crops are 100 px tall and are cut down 3x. So the leading-digit loss is more likely a thin, faint, large-crop domain gap than clipping. This refutes the earlier hypothesis that the left edge was being cut.
4. **Within the ROI sets, density alone does not predict errors well.** Document crops with ink touching the left border score 85% against 91-92% without (20 crops); screenshots with ink fraction < 0.05 score 62-63% (pipeline, replay) against 68-71% overall; the differences are small and on small n, so no causal claim.

**Not done / suggestions (outside the planned grid):** (a) thicken or darken the synthetic strokes and add edge clipping so they match real crops; (b) augment real/synthetic training crops with thin, wide, margined screenshot-style crops; (c) test whether darkening or dilating the screenshots before inference improves them (a no-training test on the 84 crops).

## Run ledger

Every run: config, seed, steps, wall time, checkpoint, git commit (from each run's `done.json`).

| run | seed | steps | wall (min) | real test | words | cov@99 | checkpoint | commit |
|---|---|---|---|---|---|---|---|---|
| baseline_s42 | 42 | 6570 | 18.0 (eval resumed after a crash) | 94.2% | 37.3% | 38.5% | `outputs\exp_oct\baseline_s42\checkpoints\final.ckpt` | cd6ef72+uncommitted |
| baseline_s43 | 43 | 6570 | 18.0 | 94.2% | 42.3% | 55.4% | `outputs\exp_oct\baseline_s43\checkpoints\final.ckpt` | cd6ef72+uncommitted |
| baseline_s44 | 44 | 6570 | 19.0 | 95.5% | 40.7% | 54.6% | `outputs\exp_oct\baseline_s44\checkpoints\final.ckpt` | cd6ef72+uncommitted |
| master_v3_existing (not trained here) | 42 | 6720 | n/a | 94.7% | 39.0% | 45.4% | `models/parseq_master_v3/parseq_master_v3.ckpt` | n/a |
| replay_p10_s42 | 42 | 6570 | 18.0 | 93.7% | 87.3% | 60.2% | `outputs\exp_oct\replay_p10_s42\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| replay_p20_s42 | 42 | 6570 | 18.0 | 93.9% | 89.3% | 55.4% | `outputs\exp_oct\replay_p20_s42\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| replay_p30_s42 | 42 | 6570 | 18.0 | 93.9% | 89.0% | 61.7% | `outputs\exp_oct\replay_p30_s42\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| replay_p20_s43 | 43 | 6570 | 19.0 | 92.9% | 88.0% | 44.9% | `outputs\exp_oct\replay_p20_s43\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| replay_p20_s44 | 44 | 6570 | 20.0 | 93.9% | 87.7% | 64.1% | `outputs\exp_oct\replay_p20_s44\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| replay_p30_s43 | 43 | 6570 | 19.0 | 95.3% | 90.0% | 40.4% | `outputs\exp_oct\replay_p30_s43\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| replay_p30_s44 | 44 | 6570 | 19.0 | 94.5% | 89.3% | 50.7% | `outputs\exp_oct\replay_p30_s44\checkpoints\final.ckpt` | 3dfea8b+uncommitted |
| ls005_s42 | 42 | 6570 | 20.0 | 95.0% | 23.3% | 16.6% | `outputs\exp_oct\ls005_s42\checkpoints\final.ckpt` | ead9580+uncommitted |
| ls010_s42 | 42 | 6570 | 20.0 | 93.9% | 18.7% | 3.4% | `outputs\exp_oct\ls010_s42\checkpoints\final.ckpt` | ead9580+uncommitted |
| ls005_s43 | 43 | 6570 | 20.0 | 94.5% | 26.0% | 4.2% | `outputs\exp_oct\ls005_s43\checkpoints\final.ckpt` | ead9580+uncommitted |
| ls005_s44 | 44 | 6570 | 20.0 | 94.5% | 27.3% | 8.7% | `outputs\exp_oct\ls005_s44\checkpoints\final.ckpt` | ead9580+uncommitted |
| pad_s42 (resize pad) | 42 | 6570 | 25.0 | 93.9% | 11.0% | 40.1% | `outputs\exp_oct\pad_s42\checkpoints\final.ckpt` | 3ffeeca+uncommitted |
| pad_s43 (resize pad) | 43 | 6570 | 20.0 | 93.9% | 14.3% | 43.8% | `outputs\exp_oct\pad_s43\checkpoints\final.ckpt` | 3ffeeca+uncommitted |
| pad_s44 (resize pad) | 44 | 6570 | 39.0 | 94.5% | 12.3% | 42.5% | `outputs\exp_oct\pad_s44\checkpoints\final.ckpt` | 3ffeeca+uncommitted |

All runs: 15 epochs, batch 32, lr 1.5e-3, dropout 0.2, RandAugment 9, stretch 32x128 (config.yaml `train:`), data `outputs/exp_oct/data/master_lmdb` (unpacked from `data/master_dataset.mdb`, 13,996 train rows). Hash `+uncommitted` means the working tree had changes.

## What failed / could not verify

- The first evaluation of `baseline_s42` crashed (PARSeq infers the model class from the checkpoint path, which must contain 'parseq'). Training was fine; fixed by loading a copy named `parseq_<run>.ckpt`; no retraining.
- Seeds are not bit-reproducible on this GPU (noted in the model report); the 3-seed spread above is the real noise level.
- Soup and WiSE-FT results on the 379-crop test have Wilson intervals of roughly +-2.5 points per single model.
- Labels that look wrong (flagged, NOT relabelled): see "Possible label problems" below.

## Possible label problems (flagged, not relabelled)

- `Doc0544_page_005_roi_2.png` test label `III.` (trailing dot) while the model predicts `III` with conf 0.9999 - labels elsewhere strip dots.
- `Doc0592_page_004_roi_2.png` label `I`, predicted `1` (I/1 ambiguity). `Doc0609_page_004_roi_7.png` label `A`, predicted `1`.
- Crop images for all confident errors: `outputs/exp_oct/eval/*/confident_errors/`.
