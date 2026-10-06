# EXPERIMENTS_OCT: Execution Plan (session-by-session)

## REVISED PLAN (2026-10-06, after Sessions 0-2)

**Done:** Session 0 (tools, gate passed), Session 1 (baseline 94.6 +- 0.8 real / 40.1 words; WiSE-FT a=0.9 and soups), Session 2 (replay: 30% recommended, 94.5 real / 89.4 words).
**Status: PAUSED** while the 30% replay model is tried on real ROIs. Resume on request.

Why the order changes: real-test accuracy did not move with any method (noise floor +-0.8), and replay already solved word recovery, so L2-SP (same goal) is now low value.
The open goals are trustworthy confidence (coverage at 99% precision, noisy at +-9.5) and whether more data would help real accuracy.

| New order | What | Runs | ~Time | Why now |
|---|---|---|---|---|
| **Next: Session 3 (was 4)** | Temperature scaling on existing models first (no training, minutes); then label smoothing 0.05 / 0.1 screening + confirm | 6 | ~2 h | Targets confidence, the main open goal |
| Session 4 (was 5) | Learning curve: 25/50/75/100% of train documents | 12 (7.5 run-equivalents) | ~2.5 h | Says whether real accuracy is data-limited or plateaued |
| Session 5 (was 3) | L2-SP (optional: run only if still wanted) | 7 + calibration | ~2.3 h | Same goal as replay / WiSE-FT |
| Session 6 | Combination (replay 30% + best of the above, 3 seeds), SSL plan only, final report | 3 | ~1.5 h | unchanged |

Model for ROI testing (exported 2026-10-06): `outputs/exp_oct/models/parseq_replay30_s{42,43,44}/parseq_replay30_s{seed}.ckpt` (95 MB each, weights-only). Primary: **s44**
(94.5% real, 89.3% words; closest to the 3-seed mean). Checked: the exported file reproduces 358/379 on the test set.

---

(Original plan below; session numbers there refer to the original order.)


Companion to the task spec. This file says **what runs in which order, how long each
session takes, and what decision follows each experiment**. Results go in
`docs/EXPERIMENTS_OCT.md`; outputs in `outputs/exp_oct/`; branch `exp_oct`.

Per-run cost: ~20-30 min (RTX 4050). Estimates below use 25 min/full run.
Total: ~38 runs, about 33 full-run equivalents, ~14 h GPU, spread over 7 sessions of 2-4 h.

## Ground rules for splitting work across days

1. **Resumable runner.** One script (`src/exp_oct/run_queue.py`) takes a queue of run
   configs. Each run writes `outputs/exp_oct/<run>/done.json` (config, seed, steps,
   wall time, ckpt path, git commit, eval metrics). Re-running skips finished runs.
   Closing the laptop mid-session loses at most one run.
2. **Every session ends with a gate**: write the results table into
   `docs/EXPERIMENTS_OCT.md`, state the decision, commit. Next session starts from that
   decision, not from memory.
3. **Print run count and time estimate before each phase** (spec requirement).
4. Order is cheapest and most informative first, so an early stop still leaves a usable
   report.
5. No run beyond the stated grid without asking.

---

## Session 0: Infrastructure (~3 h, almost no GPU)

| Step | Output |
|---|---|
| Create branch `exp_oct`; create `outputs/exp_oct/` | branch |
| Write `src/exp_oct/eval_suite.py` (all 7 metrics) | module |
| Unit test coverage/precision on a tiny hand-made dataset | `tests/test_eval_suite.py` |
| Data checks: no NaN/empty labels, val/test share no document | printed assertions |
| Evaluate existing `master_v3.ckpt` | metrics + per-crop CSV |
| Write `run_queue.py` (resumable) | runner |

**Gate 0 (hard stop):** master_v3 must give 359/379 (94.7%) within 1 crop.
- Pass: continue to Session 1.
- Fail: STOP. Debug preprocessing (resize, charset_test=94, normalisation, case handling)
  and ask before anything else.

## Session 1: Baseline + WiSE-FT + soups (~2.5 h)

1. **Phase 0:** train baseline seeds 42, 43, 44 (3 runs, ~1.5 h). Evaluate each. Report
   mean +- std for every metric. This is the reference for everything after.
2. **Phase 1 (no training, ~30 min):**
   - Check pretrained vs master_v3 state_dict keys and shapes. STOP if they mismatch.
   - Interpolate a in {0, 0.25, 0.5, 0.75, 0.9, 1.0}. Sanity: a=1.0 equals master_v3,
     a=0 gives about 60.7% real / 63.3% words.
   - Soup of baseline seeds 42/43/44 vs the individual seeds.

**Decision after Session 1**
- If some a in (0,1) keeps real test within 1 point of baseline and raises word accuracy,
  record it as the **cheap word-recovery candidate**. It is free, so it goes into the
  combination later.
- If the soup beats individual seeds by more than 1 point, record "soup helps". It
  becomes a free final step for any winning recipe.
- If the baseline std is large (more than about 1 point), the 1-point noise rule gets
  stricter. State the measured std in the report and use it for all later comparisons.

## Session 2: Replay (Phase 2, ~4 h)

1. Build replay words with the open_read generator. Assert vocabulary overlap is 0 against
   the word-test list and against master_v3 training words. Print the overlap.
2. Build 10/20/30% mixes with total size capped. Report steps per run.
3. **Screen:** 3 runs, seed 42 (~1.5 h). Evaluate on real test, word test, coverage@99%.
4. **Confirm:** best 2 ratios with seeds 43, 44 (4 runs, ~2 h).

**Decision after Session 2**
- Plot real vs word accuracy (trade-off curve).
- Recommended point = the highest word accuracy that stays within 1 point of baseline real
  test (3-seed mean).
- If no ratio does that, record replay as a negative or trade-off result. Do not tune
  further.
- If replay hurts real test by more than 1 point at every ratio, drop it from the
  combination.

## Session 3: L2-SP (Phase 3, ~4 h)

1. Implement `l2sp_lambda` in `patches/exp_oct.patch` (default 0 = unchanged). Check that
   the penalty is 0 at step 0 and grows, and log it every N steps.
2. **Calibrate:** 1-epoch run (~2 min). Report CE vs penalty. Choose 3 lambdas giving
   penalty/CE ~ 0.01, 0.1, 1 at end of training.
3. **Screen:** 3 runs, seed 42 (~1.5 h).
4. **Confirm:** best 2 with seeds 43, 44 (4 runs, ~2 h).
5. Note in the report: EWC not run because it needs Fisher information from the original
   task data, which we lack.

**Decision after Session 3**
- Apply the decision rule against the 3-seed baseline (more than 1 point on real test, or
  clear coverage@99% gain, without losing more than 1 point).
- L2-SP most likely helps by staying near pretrained weights. If it improves word accuracy
  with real test unchanged, it is a combination candidate. Otherwise record it as a
  negative result.

## Session 4: Label smoothing + temperature scaling (Phase 4, ~4 h)

1. Patch the loss in `strhub/models/parseq/system.py` via `exp_oct.patch`
   (label_smoothing, ignore_index=pad). Check that eps=0 matches baseline loss and that
   pad/EOS handling is unchanged.
2. **Screen:** eps 0.05 and 0.1, seed 42 (2 runs, ~1 h).
3. **Confirm:** both, seeds 43, 44 (4 runs, ~2 h).
4. **Temperature scaling (no training):** for baseline and each smoothing model, fit one T
   on VAL logits only, apply to TEST. Report ECE, NLL, coverage@99% before and after, and
   confident errors (conf > 0.9) remaining.

**Decision after Session 4**
- Temperature scaling is free. If it improves coverage@99% or ECE without changing
  accuracy, it is **adopted by default** for any deployed model.
- Label smoothing counts as a win only if accuracy or coverage@99% improves per the
  decision rule. Smoothing can lower confidence on correct predictions, so check that
  coverage does not drop.

## Session 5: Learning-curve diagnostic (Phase 5, ~3.5 h)

1. Split the 155 train documents by document id (never by crop).
2. 25/50/75%: 3 random document subsets each, 9 runs. Smaller sets are faster, about 2.25
   full-run equivalents.
3. 100%: seeds 42-44 on real-only base_best recipe (3 runs, ~1.25 h).
4. Plot test accuracy and coverage@99% vs document fraction (mean +- std).

**Decision after Session 5**
- State the slope from 50% to 100%.
- Gain above about 1 point and still rising -> **"variety still limiting"**: more
  documents or writers will help, which strengthens the case for the SSL pilot or for
  collecting data.
- Otherwise -> **"plateau"**: more of the same data will not help. Prioritise
  method-level fixes (WiSE-FT, L2-SP, calibration).

## Session 6: Combination + SSL plan + report (~3 h)

1. **Combination, only if two or more methods each passed the decision rule.** Run ONE
   combined config (for example WiSE-FT or replay + label smoothing + temperature), 3 seeds
   (~1.5 h). No other combinations.
2. **Phase 6, plan only:** locate unlabelled crops, count them, count what survives the
   noise classifier, count distinct documents and writers, estimate compute for continued
   masked pretraining on this GPU. **Do not train. Ask for approval.**
3. Finalise `docs/EXPERIMENTS_OCT.md`: one table per phase, trade-off plot,
   coverage-vs-precision curves, tier tables, confident-error lists with image folders,
   learning-curve plot, per-run provenance, "what failed / could not verify" section, and
   any crops whose labels look wrong (I/1, ii/iii), flagged but not relabelled.

---

## Summary

| Session | Phase | Runs | Time | Ends with |
|---|---|---|---|---|
| 0 | Infra + baseline check | 0 | ~3 h | Gate 0: reproduce 94.7% |
| 1 | P0 baseline, P1 WiSE-FT, soups | 3 | ~2.5 h | noise floor, free-win candidates |
| 2 | P2 Replay | 7 | ~4 h | trade-off curve, recommended ratio |
| 3 | P3 L2-SP | 7 (+1 short) | ~4 h | keep or drop L2-SP |
| 4 | P4 Label smoothing, temp scaling | 6 | ~4 h | calibration decision |
| 5 | P5 Learning curve | 12 | ~3.5 h | "variety limiting" vs "plateau" |
| 6 | Combination, P6 plan, report | 3 | ~3 h | final report, SSL approval request |

## Stop-and-ask triggers (any session)
- Baseline does not reproduce (Gate 0).
- State-dict mismatch in Phase 1.
- Any run outside the stated grid.
- Any change to data or labels.
- Any SSL training.
- Missing document ids for the Phase 5 split.
