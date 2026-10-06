# EXPERIMENTS_OCT: controlled experiment suite on `parseq_master_v3`

Living report. Branch `exp_oct`. Plan: [EXPERIMENTS_OCT_PLAN.md](EXPERIMENTS_OCT_PLAN.md). Outputs: `outputs/exp_oct/`.
Live log (updates every minute while a run is going): `Get-Content outputs\exp_oct\progress.log -Wait -Tail 30`

## Status

| Session | Phase | State |
|---|---|---|
| 0 | Infrastructure, baseline reproduction gate | **DONE, gate passed** |
| 1 | P0 baseline x3 seeds, P1 WiSE-FT, soups | **DONE** (baseline 94.6 +- 0.8; WiSE-FT a=0.9 recommended; soup no gain) |
| 2 | P2 Replay | **DONE** (30% replay recommended: real 94.5, words 89.4) |
| 3 | P4 Temperature scaling + label smoothing (moved up) | not started (PAUSED, see plan) |
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
