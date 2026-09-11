"""
Directly tests the "rare classes fail more" hypothesis we've been assuming
throughout this pipeline: joins per-label TRAINING frequency (train.csv)
against actual ERROR RATE on that label (error_analysis.csv), so you can
see the correlation with real numbers instead of eyeballing it.

No retraining needed -- just reads your existing files.

Usage:
    python class_balance_vs_errors.py --error-csv error_analysis.csv
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", default="data/processed/train.csv")
ap.add_argument("--error-csv", default="error_analysis.csv")
args = ap.parse_args()

train = pd.read_csv(args.train_csv)
errors = pd.read_csv(args.error_csv)

train_freq = train["text"].astype(str).value_counts()

errors["target"] = errors["target"].astype(str)
total_per_label = errors.groupby("target").size()
wrong_per_label = errors[~errors["exact_match"]].groupby("target").size()

report = pd.DataFrame({
    "train_freq": train_freq,
    "test_appearances": total_per_label,
    "test_errors": wrong_per_label,
}).fillna(0)

report["test_appearances"] = report["test_appearances"].astype(int)
report["test_errors"] = report["test_errors"].astype(int)
report["train_freq"] = report["train_freq"].astype(int)

# Only labels that actually showed up in the test split are meaningful here
report = report[report["test_appearances"] > 0].copy()
report["error_rate"] = report["test_errors"] / report["test_appearances"]

report = report.sort_values("train_freq")

print(f"{'Label':<8} {'train_freq':<11} {'test_n':<7} {'errors':<7} {'error_rate':<10}")
print("-" * 50)
for label, row in report.iterrows():
    print(
        f"{label:<8} {int(row['train_freq']):<11} {int(row['test_appearances']):<7} "
        f"{int(row['test_errors']):<7} {row['error_rate']:<10.2f}"
    )

# Split into low-frequency vs high-frequency halves (by training count) and
# compare their pooled error rates -- a simple, honest way to see if the
# imbalance hypothesis holds without over-interpreting single-label noise
# (some labels only appear once or twice in test, so their individual
# error_rate is a coin flip, not a reliable per-label number).
median_freq = report["train_freq"].median()
low = report[report["train_freq"] <= median_freq]
high = report[report["train_freq"] > median_freq]

def pooled_rate(df):
    n = df["test_appearances"].sum()
    e = df["test_errors"].sum()
    return e / n if n else float("nan"), n, e

low_rate, low_n, low_e = pooled_rate(low)
high_rate, high_n, high_e = pooled_rate(high)

print()
print(f"Below-median training frequency (<= {median_freq:.0f} train examples):")
print(f"  Pooled error rate: {low_rate:.2%}  ({low_e}/{low_n} test samples)")
print(f"Above-median training frequency (> {median_freq:.0f} train examples):")
print(f"  Pooled error rate: {high_rate:.2%}  ({high_e}/{high_n} test samples)")

if low_n >= 5 and high_n >= 5:
    print()
    if low_rate > high_rate * 1.3:
        print("Rare-in-training labels are meaningfully worse -- supports "
              "the class-imbalance hypothesis. More data/oversampling on "
              "the low end is likely to help.")
    elif high_rate > low_rate * 1.3:
        print("Common-in-training labels are actually doing WORSE -- "
              "imbalance isn't your main driver here. Look elsewhere "
              "(e.g. resolution, specific confusable shapes) instead of "
              "assuming more balancing will fix it.")
    else:
        print("No strong split by training frequency -- errors are "
              "roughly as likely on common labels as rare ones. This "
              "points AWAY from class imbalance being the dominant cause "
              "and TOWARD something else (shape confusability, "
              "resolution) affecting specific characters regardless of "
              "how often they appear in training.")
else:
    print()
    print("Too few test samples on one side of the split to draw a "
          "reliable conclusion -- read the per-label table above instead.")