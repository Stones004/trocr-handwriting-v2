"""Metric helpers shared by the EMNIST experiment scripts (run_experiments.py, archive_models.py).

Operates on the per-sample CSVs written by src/parseq/eval_parseq.py (columns: filename, target, prediction,
exact_match, seq_min_prob, seq_mean_prob). Accuracy is exact-match and case-sensitive; CER is total edit distance over
total target characters (an empty target counts as 1 character).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from validation_metrics import lev  # noqa: E402,F401 -- the project's single edit-distance implementation
VOCAB = set("0123456789abcdefghij")


def load(csv):
    df = pd.read_csv(csv, keep_default_na=False)
    df["target"] = df["target"].astype(str)
    df["prediction"] = df["prediction"].astype(str)
    df["ok"] = df["exact_match"].astype(str) == "True"
    return df


def metrics(df):
    t, p = df["target"].astype(str), df["prediction"].fillna("").astype(str)
    return dict(n=len(df), acc=float((t == p).mean()),
                cer=sum(lev(a, b) for a, b in zip(p, t)) / max(1, sum(max(len(a), 1) for a in t)))


def kind_of(s):
    return ("single digit" if len(s) == 1 and s.isdigit() else
            "multi digit" if s.isdigit() else
            "single letter a-j" if len(s) == 1 and s in "abcdefghij" else
            "other (roman/upper/parens/etc.)")
