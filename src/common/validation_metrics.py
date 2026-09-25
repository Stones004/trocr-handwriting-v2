"""Shared helpers for the model-validation pipeline (pandas/numpy only, usable from either venv).

Also the project's single `lev` (edit distance) implementation, imported by eval_parseq.py and the EMNIST scripts.
Defaults for the statistics come from config.yaml (validation.*), the output folder from paths.reports_dir.
"""
import numpy as np
import pandas as pd

from project_config import cfg, path

SPLITS = ("train", "val", "test")
OUT_DIR = path("reports_dir") / "validation"


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sample_row(filename, target, prediction, conf_min, conf_mean):
    t, p = str(target).strip(), str(prediction).strip()
    return {"filename": filename, "target": t, "prediction": p, "exact": p == t, "exact_ci": p.lower() == t.lower(),
            "edits": lev(p, t), "chars": max(len(t), 1), "conf_min": conf_min, "conf_mean": conf_mean}


def save_split(name, split, rows, secs):
    df = pd.DataFrame(rows)
    df["ips"] = len(df) / max(secs, 1e-9)
    df.to_csv(OUT_DIR / f"{name}__{split}.csv", index=False)


def auroc(scores, positives):
    """P(score of a random positive > score of a random negative); ties count half."""
    s = pd.Series(scores).rank().to_numpy()
    n1 = int(np.sum(positives)); n0 = len(s) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (s[np.asarray(positives, bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(conf, correct, bins=None):
    bins = bins or cfg["validation"]["ece_bins"]
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def boot_ci(exact, n=None, seed=0):
    n = n or cfg["validation"]["bootstrap_n"]
    rng = np.random.default_rng(seed)
    e = np.asarray(exact, float)
    m = rng.choice(e, (n, len(e))).mean(1)
    return np.percentile(m, [2.5, 97.5])
