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

## CRNN + CTC vs PARSeq (user-requested, outside the original grid)

strhub CRNN (7-layer CNN -> 2-layer BiLSTM, hidden 256, CTC over the 94 characters + blank, 8.4M parameters), initialised from the pretrained `crnn-679d0e31.pt` (downloaded once from the PARSeq GitHub release, cached in `D:/cache/torch`;
training itself needs no internet). Same data (13,996 rows), 15 epochs, batch 32, RandAugment 9, 32x128 stretch, seeds 42/43/44; learning rate = CRNN's own default 5.1e-4 (**not tuned**; PARSeq's recipe lr 1.5e-3 was tuned for PARSeq).
Code: `src/exp_oct/crnn_support.py`; `train.py` needed a guard for models without a ViT `.encoder` (in `patches/exp_oct.patch`). CTC confidence = minimum over decoded characters of the best frame probability inside that character's run (blanks ignored): not the same definition as PARSeq's,
so ECE/coverage are only roughly comparable; the NLL column is per-character CTC NLL, **not comparable** with PARSeq's token NLL; temperature scaling not applied to CTC.

| (mean +- std, 3 seeds) | real test % | words % | coverage@99 % | confident errors | ECE | train time | params | CPU speed (4 threads) |
|---|---|---|---|---|---|---|---|---|
| PARSeq stretch baseline | **94.6 +- 0.8** | **40.1 +- 2.5** | 49.5 +- 9.5 | 6.0 +- 2.6 | 0.034 | ~18 min | 23.8M | 58 ms/crop |
| CRNN + CTC | 88.1 +- 0.3 | 23.8 +- 0.5 | 39.5 +- 1.6 | 6.7 +- 1.2 | 0.042 | **~9 min** | **8.4M** | **13 ms/crop** |

On the user's 226 reviewed crops (`roi_pad.py`): CRNN **68.9%** (160/153/154 correct) vs PARSeq baseline 80.8 +- 0.7; document crops 77.3% vs 87.3%; screenshots 54.8% vs 69.8%.

**Reading:** CRNN is clearly less accurate: -6.5 points on the 379-crop test, about -12 points on the user's crops, and worse on words and screenshots; seed-to-seed spread is small (0.3), so the gap is not noise.
Its confidence is usable (precision of accepted crops 99.6%, similar confident-error count), and it is 2x smaller, about 4.5x faster on CPU and trains in half the time.
Caveats: untuned lr, no replay words, a different confidence definition, one architecture setting (hidden 256). The gap could narrow with tuning but a 6-12 point gap is unlikely to close completely; the speed would only matter if CPU latency is a real constraint.
**Decision: PARSeq stays the model; CRNN is not adopted for accuracy.** Worth considering only as a fast first-pass or a second opinion (its errors differ from PARSeq's), which was not tested.

## Marked-NNa data: marks at every position (user request)

**Motivation.** The master data drew NNa marks from only 9 templates (`NN)a`, `NN.a`, `NNa)`, `NNa.`, `NN(a)`, `(NN)a`, `(NNa)`, `NN(a`, `(NNa`): never between the two digits and no `[ ]` inside the label.
**New data** (`src/exp_oct/build_marks.py`): 6,000 synthetic NNa crops (same fonts, glyph banks and page augmentation as `src/master/render.py`) with the marks `. ( ) [ ]` (no comma, as requested) in any of four gaps (before the number, between the two digits,
between digits and letter, after the letter): 35% single mark, 30% bracket pair wrapping a span (10% mismatched, 30% also a dot), 25% two marks, 10% three marks; 2,400 crops have a mark between the digits, 3,219 contain a square bracket.
Labels remain characters only (`12a`). Plus 300 held-out evaluation crops (different seed). Contact sheet: `outputs/exp_oct/marks_nna/contact_sheet.png`. Training set = replay-30% set (13,996 rows) + the 6,000 new rows = 19,996 rows,
625 steps/epoch, **9,375 steps per run (replay-30 control: 6,570)**; seeds 42/43/44, ~27 min each. Val/test unchanged.

| (mean +- std, 3 seeds) | real test % | words % | coverage@99 % | confident errors | ECE | your 226 crops | your doc crops | your screenshots | **held-out marked NNa (300, synthetic)** |
|---|---|---|---|---|---|---|---|---|---|
| stretch baseline | 94.6 +- 0.8 | 40.1 +- 2.5 | 49.5 +- 9.5 | 6.0 | 0.034 | 80.8 +- 0.7 | 87.3% | 69.8% | 92.4% |
| replay 30% (control) | 94.5 +- 0.7 | 89.4 +- 0.5 | 50.9 +- 10.7 | 9.0 | 0.036 | **83.3 +- 0.3** | **90.8%** | 70.6% | 87.7% |
| replay 30% + 6k marked NNa | 94.2 +- 0.7 | 89.4 +- 0.4 | 57.1 +- 6.6 | 5.3 | 0.031 | 81.4% (185/181/186) | 87.8% | 70.6% | **96.8%** (97.0/96.7/96.7) |

Held-out marked set by gap (mean of 3 seeds): the weakest case before, a mark between the two digits, rises from 79-84% (replay 30%) to about 95-96%; all five mark types are 93-100%; three-mark crops 94-100% (replay 30%: 74-82%).
`outputs/exp_oct/eval_marks` output per run: `<run>/marks_eval_preds.csv`.

**Reading**
- The model **learns the new formats** (87.7% -> 96.8% on held-out marked NNa) and replay's loss on this format (87.7 vs baseline 92.4, replay removed 39% of the synthetic rows) is more than repaid.
- Plain numbers and words are unchanged within noise (real test 94.2 vs 94.5, words 89.4 both); coverage and confident errors are slightly better (57.1 vs 50.9, 5.3 vs 9.0) but inside the seed noise.
- **On the user's own 226 crops it is not better: 81.4% vs 83.3% for replay 30%, with the whole difference on document crops (87.8 vs 90.8)**; screenshots are the same (70.6). Three seeds (181-186 vs 188-189 correct) so this is probably real but small (4-5 crops). The user's crops contain few marked NNa labels, so the new data could not help there, and
  adding 6,000 synthetic rows may shift the model away from real documents. Not investigated further.
- Caveats: the held-out marked set is **synthetic from the same generator**, so it proves the model learned the format but not that it reads students' real marked NNa; the compute differs (9,375 vs 6,570 steps); real marked NNa crops from students are the missing evaluation.

**Decision:** keep both. Use **replay 30% (no marks)** if the documents resemble the user's 226 crops; use **replay 30% + marked NNa** if real marked NNa labels (`1.2a`, `(12)a`, `[1]2a`, ...) are common in production. The choice needs real examples of marked NNa labels to settle.

## Marked data on every format of the NNa family (user request, outside the original grid)

Why: the 6k marked-NNa set made NNa 46% of the training rows and plain 1-digit/letter crops on the user's 226 got worse (1 digit 97.8 -> 94.2). New set `replay30_marks2`: replay-30% rows + 4,000 marked crops (`src/exp_oct/build_marks2.py`): NNa 1,600, 1 letter 560, 2 letters 560 (Roman only: ii iv vi ix, both cases), 1 digit 480, 2 digits 480, Na 320. Marks `. ( ) [ ]` at every gap, labels = characters only. NNa is now about 27% of the training set. 17,996 rows, 8,445 steps per run, 3 seeds (42-44). Held-out marked set: 100 crops per format, different seed (`outputs/exp_oct/marks2/marks2_eval.csv`). Scoring script: `outputs/exp_oct/roi_check/score_rm2.py`, all numbers are the mean over 3 seeds.

Real test (379), words, coverage: rm2 94.6 (94.5, 93.9, 95.3), words 89.2, coverage@99 58.7, confident errors 7.7: no change versus replay 30% (94.5) beyond noise.

Held-out marked crops (synthetic, same generator as the training marks):

| Format | n | replay 30% | +6k marked NNa | +4k all formats |
|---|---|---|---|---|
| NNa | 100 | 90.7 | 98.7 | 97.0 |
| Na | 100 | 73.3 | 80.3 | 93.7 |
| 1 digit | 100 | 86.0 | 90.3 | 94.7 |
| 2 digits | 100 | 83.3 | 93.3 | 96.0 |
| 1 letter | 100 | 67.0 | 78.3 | 89.7 |
| 2 letters (Roman) | 100 | 57.3 | 74.0 | 94.3 |
| all | 600 | 76.3 | 85.8 | 94.2 |
| old marked-NNa set | 300 | 87.7 | 96.8 | 95.2 |

The user's 226 reviewed crops (n per format varies; small groups are noisy):

| Format | n | replay 30% | +6k marked NNa | +4k all formats |
|---|---|---|---|---|
| 1 digit | 46 | 97.8 | 94.2 | 92.0 |
| 1 letter | 52 | 93.6 | 92.3 | 89.7 |
| 2 digits | 30 | 67.8 | 63.3 | 71.1 |
| NNa | 55 | 77.6 | 77.0 | 78.2 |
| Na | 31 | 77.4 | 74.2 | 77.4 |
| 2 letters (Roman) | 5 | 93.3 | 100 | 100 |
| other | 7 | 42.9 | 47.6 | 42.9 |
| all | 226 | 83.3 | 81.4 | 82.0 |

Real test by format (379): 1 digit 97.5 / 97.0 / 96.8, 2 digits 95.2 / 93.7 / 100 (n=21), 1 letter 80.5 / 80.5 / 81.6 (n=29), 2 letters 79.7 / 81.2 / 82.6 (n=23), same column order.

Findings:
- The mix fixed what it was built for: marked crops of every format are now read at 90-97% (all 600: 94.2% vs 76.3% for replay 30% and 85.8% for the NNa-only set). NNa kept 97.0% (old marked set 95.2% vs 96.8%, about 1 crop in 100, noise).
- The prior-shift worry is only partly resolved. On the user's 226 crops, 1-digit (97.8 -> 92.0, about 3 crops) and 1-letter (93.6 -> 89.7, about 2 crops) are still below replay 30%. On the real test they are within noise (97.5 vs 96.8; 80.5 vs 81.6). Errors that replay 30% did not make: `3 -> 93`, `6 -> 26`, `9 -> 99`, `2 -> 12`, `3 -> Aus`, `b -> 5`, `C -> 5`. They look like extra or substituted characters, not NNa-ification as such. Sample sizes are small (46 and 52 crops), so this is suggestive, not proven. The overall 226 score (82.0 vs 83.3) is inside the noise band.
- Decision rule: rm2 does not beat replay 30% by more than 1 point on the real test (94.6 vs 94.5), so it is not "better" under the rule. It improves marked-crop robustness a lot and loses nothing measurable on real test. Marked real crops are the missing evidence, as before.

## Screenshot stroke test, Step A (no training, user-approved)

Question: are screenshots weak (about 70%) because their strokes are thin and faint? Test: before inference, darken and thicken the strokes of the 226 ROI crops (84 screenshots, 142 documents) and re-score replay 30% and rm2, 3 seeds each (`outputs/exp_oct/roi_check/stroke_test.py`, per-crop results in `stroke_test.csv`). Variants: s1 = contrast stretch (2-98 percentile) + gamma 0.7; s2/s3/s4 = s1 + stroke thickening with a kernel of 3 / 6 / 9% of the crop height.

| Model, variant | Screenshots (84) | Documents (142) |
|---|---|---|
| replay 30%, none | 70.6 +- 1.4 | 90.8 +- 0.7 |
| replay 30%, s1 contrast | 72.2 +- 1.8 | 90.1 |
| replay 30%, s2 thick 3% | 67.5 | 89.7 |
| replay 30%, s3 thick 6% | 57.1 | 82.9 |
| replay 30%, s4 thick 9% | 45.6 | 70.9 |
| rm2, none | 73.8 +- 4.1 | 86.9 |
| rm2, s1 contrast | 73.8 | 85.9 |
| rm2, s2 thick 3% | 69.0 | 87.6 |
| rm2, s3 thick 6% | 60.7 | 83.1 |
| rm2, s4 thick 9% | 48.4 | 70.2 |

Result: negative. Thickening makes screenshots worse at every strength (70.6 -> 67.5 -> 57.1 -> 45.6 for replay 30%), and the damage is largest on 2-digit labels. Contrast alone gives +1.6 points for replay 30% (about 1.3 crops of 84) and 0.0 for rm2, which is inside the noise (seed spread 1.4-4.1; 84 crops; the rule needs about 4 crops). Documents are unchanged or worse. Screenshots contain only 2-digit, NNa, Na and other labels (no 1-digit or letter crops among the 84).

Conclusion: stroke thinness alone does not explain the screenshot weakness, so Step B (screenshot-style synthetic strokes) is not justified by this test. Other candidate causes, untested: low resolution/blur, the different background and padding, no left-border contact, and too few real screenshots in training. Caveat: thickening was a simple morphological filter at native resolution, not a rendering that matches how screenshot strokes look.

## LoGo-HTR setup (inference only, user request; nothing trained)

LoGo-HTR (WACV 2026, self-supervised HTR, CoMER-style DenseNet encoder + bidirectional transformer decoder) set up for zero-shot inference. Repo https://github.com/shreemitra/LoGoHTR (MIT licence, commit 9eaea5f) in `third_party/LoGoHTR` (patched by `patches/logohtr.patch`); weights `Downstream_Full_Model_Best.ckpt` from https://huggingface.co/Shree10/LoGo-HTR (52,019,943 bytes, sha256 `25db3a51...c432`) in `third_party/LoGoHTR_weights/`; own venv `venv_logohtr` (torch 2.11+cu128, Lightning 2.6.6, current versions worked, no old pins needed). Script: `src/exp_oct/logo_htr_infer.py`.

Verified: 6.39M parameters (about 6.4M as expected), `load_state_dict` matched all keys, output layer 97 rows -> dictionary of 94 lines. Two repo dictionaries have 94 lines (`SSL_HWD_Dict.txt`, `dictionary.txt`); tested both on 60 labelled word crops (`data/open_read/word_eval.csv`, synthetic): SSL_HWD_Dict.txt 12/60 exact with sensible text (Given, First read correctly), dictionary.txt 0/60 (garbage), so SSL_HWD_Dict.txt is the correct one. The first dictionary line is a space; the script maps the stripped empty token back to a space.

Checkpoint safety: the pickle references only OrderedDict, tensor rebuild helpers and Lightning's `ModelCheckpoint` class (checked with `pickletools`, nothing executed). `torch.load(weights_only=True)` fails on that one class, so the script allow-lists it instead of falling back to unsafe loading.

First sanity check on the 4 non-numeric crops in the user's 226 (`or`, `Add`, `Add`, `091`): it read `a`, `ladi`, `sadi`, `oar`, so it does not read these crops. 4 crops is not an evaluation; the full scoring follows.

### LoGo-HTR zero-shot scores (`src/exp_oct/logo_htr_score.py`, beam 10, case-sensitive exact match; `outputs/exp_oct/logohtr/`)

| Set | Raw | Marks stripped | For comparison |
|---|---|---|---|
| User's 226 reviewed crops | 13.7% | 39.8% | PARSeq replay 30%: 81.9-83.3%; Chandra 56.6%; Mistral 59.7% (lenient) |
| of which documents (142) | 12.0% | 45.1% | PARSeq 90.8% |
| of which screenshots (84) | 16.7% | 31.0% | PARSeq 70.6% |
| Real test (379) | 30.9% | 64.6% | PARSeq 94.5% |
| Words (300, synthetic) | 18.3% | 18.3% | PARSeq replay 30% 89.4% |

Words by kind: common 36.0%, Capitalized 28.0%, random strings 6.7%, mixed case 2.7%. ROI by format (marks stripped): 1 letter 75.0%, NNa 40.0%, Na 38.7%, 1 digit 28.3%, 2 digits 3.3%.

Findings:
- It does not read the project's labels. Roughly half of its outputs on the number crops contain a mark (168 of 379 on test, 119 of 226 on ROI), and stripping them still leaves 39.8% / 64.6%.
- The failures look like a language prior, not a reading error: single digits come out as common words (`8 -> of`, `6 -> of`, `5 -> sy`, `9 -> a`), digits get extra digits (`1 -> 17`, `3 -> 37`, `4 -> 47`), and random strings become real words (`Vfdmwi -> veamm`, `UnpyjB -> more`). It was trained on running text in scientific documents.
- It also loses on the synthetic words (18.3%), including when the word is a real one with a capital (`Poverty -> poverty`, case is part of the label), and on our style of rendering. Zero-shot is not a fair test of its ability on its own domain; it only shows it cannot be dropped in for question numbers.
- Decision: not a candidate. It would need training on this project's labels, which is a different project, and its one potential strength (real handwritten words) is not visible here: on the 4 real words in the ROI set it read none.


## Real SSL-HWD crops added to PARSeq training (user request, outside the original grid)

Question: do real handwritten digits and letters from another dataset improve PARSeq? Data: SSL-HWD-Words (Labeled), https://huggingface.co/datasets/Shree10/SSL-HWD-Words (Apache-2.0 per the dataset card; labels auto-annotated by Amazon Textract at >=99% confidence, then reportedly checked by language experts, not verified here). Only the 835 label CSVs were read (zip central directory + range requests, `src/exp_oct/ssl_hwd_scan.py`, no images): 2,089,929 labels. Counts by project format (plain labels): 1 letter 109,317 (66% of them the words a/A/I), 1 digit 45,056 (35% are 2 or 3), 2 digits 42,356 (mostly counts and fractions), Na 1,361, Roman pairs 503, NNa only 116. Marked forms such as `(2)`, `3)`, `(a)`, `a)`, `(i)` exist in the thousands.

Pilot set (`src/exp_oct/ssl_hwd_sample.py`, seed 20261109): labels of the form `[ ( [ ]  CORE  [ ) ] . ]` only (no fractions, decimals, percentages), plain a/A/I excluded, per-label caps, at most 15 crops per writer. 3,062 sampled, 3,056 kept after dedupe/size filter: 826 one-digit, 982 two-digit, 881 one-letter, 154 Roman pairs, 213 digit+letter; 607 carry marks; 481 writers; median crop 56x62 px, colour photos. Training label = characters only. Training set `replay30_ssl3k` = replay 30% + these 3,056 rows (17,052 rows, 7,995 steps per run). 3 seeds (42-44), same recipe. Crops: `outputs/exp_oct/ssl_hwd/crops/`, contact sheet `contact_sheet.png`.

| 3 seeds | Real test (379) | Words | Coverage@99 | Confident errors | Your 226 crops |
|---|---|---|---|---|---|
| Replay 30% | 94.5 +- 0.7 | 89.4 | 50.9 | 9.0 | 83.3 |
| + 4k marks, all formats (rm2) | 94.6 | 89.2 | 58.7 | 7.7 | 82.0 |
| + 3,056 real SSL-HWD crops (ss3k) | 93.4 (92.9, 93.7, 93.7) | 88.5 (88.3, 88.3, 89.0) | 44.2 (39.6, 48.3, 44.6) | 9.0 | 82.9 |

By format (mean of 3 seeds):

| Format | Real test: replay 30% / ss3k | Your 226 crops: replay 30% / ss3k |
|---|---|---|
| 1 digit | 97.5 / 96.2 (n=293) | 97.8 / 92.0 (n=46) |
| 1 letter | 80.5 / 78.2 (n=29) | 93.6 / 93.6 (n=52) |
| 2 digits | 95.2 / 93.7 (n=21) | 67.8 / 68.9 (n=30) |
| NNa | n/a | 77.6 / 80.6 (n=55) |
| Na | n/a | 77.4 / 74.2 (n=31) |
| 2 letters (Roman) | 79.7 / 82.6 (n=23) | 93.3 / 100 (n=5) |

Held-out synthetic marked crops (600, all formats): replay 30% 76.3, ss3k 74.8, rm2 94.2; old marked-NNa set: 87.7 / 86.1 / 95.2 (rm2). The real crops, which have no synthetic marks, do not help on these, as expected.

Findings:
- No gain. Real test falls 1.1 points (94.5 -> 93.4), just past the 1-point noise band and in the wrong direction; words fall 0.9, coverage@99 falls 6.7 (inside its +-9.5 noise); your 226 crops are flat (82.9 vs 83.3). By the decision rule, the real SSL-HWD crops are not better than replay 30%.
- The formats the data was meant to strengthen did not improve: 1-digit crops on your 226 are 92.0 (replay 30%: 97.8), and the real test 1-digit is 96.2 (97.5). Small per-format differences (n=5 to 52) are within noise.
- Likely reasons (untested): domain gap (coloured photos of digits written in running text vs your grey exam-margin crops), label noise from automatic annotation, and a different digit style from the question-number writing. Adding 3,056 crops of that kind appears to pull the model slightly away from the target.
- Decision: do not adopt. This closes the "more data" direction that does not come from the target documents: replay words, synthetic marks, and real out-of-domain digits have each left the real test at 93-95%.

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
| crnn_s42 (CRNN+CTC) | 42 | 6570 | 9.0 | 88.4% | 24.3% | 41.2% | `outputs\exp_oct\crnn_s42\checkpoints\final.ckpt` | 3e86353+uncommitted |
| crnn_s43 (CRNN+CTC) | 43 | 6570 | 9.0 | 88.1% | 23.3% | 39.3% | `outputs\exp_oct\crnn_s43\checkpoints\final.ckpt` | 3e86353+uncommitted |
| crnn_s44 (CRNN+CTC) | 44 | 6570 | 8.0 | 87.9% | 23.7% | 38.0% | `outputs\exp_oct\crnn_s44\checkpoints\final.ckpt` | 3e86353+uncommitted |
| rm6k_s42 (replay30 + 6k marked NNa) | 42 | 9375 | 27.0 | 94.7% | 89.0% | 62.3% | `outputs\exp_oct\rm6k_s42\checkpoints\final.ckpt` | 5f227df+uncommitted |
| rm6k_s43 (replay30 + 6k marked NNa) | 43 | 9375 | 27.0 | 93.4% | 89.7% | 59.4% | `outputs\exp_oct\rm6k_s43\checkpoints\final.ckpt` | 5f227df+uncommitted |
| rm6k_s44 (replay30 + 6k marked NNa) | 44 | 9375 | 28.0 | 94.5% | 89.7% | 49.6% | `outputs\exp_oct\rm6k_s44\checkpoints\final.ckpt` | 5f227df+uncommitted |
| rm2_s42 (replay30 + 4k marked, all formats) | 42 | 8445 | 23.0 | 94.5% | 90.3% | 58.8% | `outputs\exp_oct
m2_s42\checkpointsinal.ckpt` | c5dd623+uncommitted |
| rm2_s43 (replay30 + 4k marked, all formats) | 43 | 8445 | 23.0 | 93.9% | 89.0% | 57.5% | `outputs\exp_oct
m2_s43\checkpointsinal.ckpt` | c5dd623+uncommitted |
| rm2_s44 (replay30 + 4k marked, all formats) | 44 | 8445 | 23.0 | 95.3% | 88.3% | 59.9% | `outputs\exp_oct
| ss3k_s42 (replay30 + 3,056 real SSL-HWD crops) | 42 | 7995 | 21-22 | 92.9% | 88.3% | 39.6% | `outputs\exp_oct\ss3k_s42\checkpoints\final.ckpt` | 6938c31+uncommitted |
| ss3k_s43 (replay30 + 3,056 real SSL-HWD crops) | 43 | 7995 | 21-22 | 93.7% | 88.3% | 48.3% | `outputs\exp_oct\ss3k_s43\checkpoints\final.ckpt` | 6938c31+uncommitted |
| ss3k_s44 (replay30 + 3,056 real SSL-HWD crops) | 44 | 7995 | 21-22 | 93.7% | 89.0% | 44.6% | `outputs\exp_oct\ss3k_s44\checkpoints\final.ckpt` | 6938c31+uncommitted |
m2_s44\checkpointsinal.ckpt` | c5dd623+uncommitted |

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
