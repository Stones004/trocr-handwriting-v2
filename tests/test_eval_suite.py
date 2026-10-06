"""Unit tests for the coverage / precision / calibration code, on tiny hand-made data.
Run: venv_parseq/Scripts/python tests/test_eval_suite.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "exp_oct"))
from eval_suite import accepted_stats, coverage_threshold, ece, tier_table, wilson  # noqa: E402

# 10 crops. Sorted by conf: .99 ok, .97 ok, .96 ok, .93 ok, .90 ok, .80 WRONG, .75 ok, .60 WRONG, .40 ok, .20 WRONG
conf = [0.60, 0.99, 0.80, 0.97, 0.20, 0.93, 0.96, 0.75, 0.90, 0.40]
ok = [False, True, False, True, False, True, True, True, True, True]
# the list above has .60/.80/.20 wrong; precision of "conf >= t" is 1.0 down to t=.90 (5/5), then .80 gives 5/6

T = coverage_threshold(conf, ok, 0.99)
assert T == 0.90, T
cov, prec, n = accepted_stats(conf, ok, T)
assert (cov, prec, n) == (0.5, 1.0, 5), (cov, prec, n)

# target 0.80: .75 gives 6/7 = .857 ok; .60 gives 6/8 = .75; .40 gives 7/9 = .78; .20 gives 7/10 = .70 -> lowest qualifying is .75
assert coverage_threshold(conf, ok, 0.80) == 0.75
# impossible target -> inf, coverage 0
assert math.isinf(coverage_threshold([0.5, 0.6], [False, False], 0.99))
assert accepted_stats([0.5], [False], math.inf)[0] == 0.0

# apply the val threshold to a different test set: accepts .95 and .92 -> 2 of 4, one right
tc, tp, tn = accepted_stats([0.95, 0.92, 0.85, 0.5], [True, False, True, True], T)
assert (tc, tp, tn) == (0.5, 0.5, 2)

# tiers: >0.95 | 0.70-0.95 (0.7 itself is in here? bounds are (0.70, 0.95]) | <0.70
tt = tier_table([0.99, 0.96, 0.9, 0.71, 0.5], [True, False, True, True, False])
assert [(r["crops"], r["errors"]) for r in tt] == [(2, 1), (2, 0), (1, 1)], tt

# ECE: confidently wrong 50% of the time -> 0.5; perfectly calibrated -> 0
assert abs(ece([1.0, 1.0], [1, 0], 15) - 0.5) < 1e-9
assert ece([0.5, 0.5], [1, 0], 15) < 1e-9

# Wilson: 359/379 -> about [0.92, 0.97]; 0/0 safe
lo, hi = wilson(359, 379)
assert 0.917 < lo < 0.925 and 0.962 < hi < 0.970, (lo, hi)
assert wilson(0, 0) == (0.0, 0.0)
print("eval_suite unit tests: all passed")
