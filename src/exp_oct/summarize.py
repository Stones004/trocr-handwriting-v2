"""Aggregate metrics.json files into mean +- std markdown tables.  Usage: python src/exp_oct/summarize.py baseline wise"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "outputs" / "exp_oct"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def row(ms):
    """Metrics of several runs (list of metrics dicts) -> list of 'mean +- std' strings."""
    def f(get, pct=True, d=1):
        v = np.array([get(m) for m in ms], float) * (100 if pct else 1)
        return f"{np.nanmean(v):.{d}f} +- {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"
    return [f(lambda m: m["test"]["exact"]), f(lambda m: m["test"]["exact_alnum"]), f(lambda m: m["words"]["exact"]),
            f(lambda m: m["coverage99"]["test_coverage"]), f(lambda m: m["coverage99"]["test_precision"]),
            f(lambda m: m["confident_errors"]["count"], False, 1), f(lambda m: m["test"]["ece"], False, 3), f(lambda m: m["test"]["nll"], False, 3)]


HEAD = "| {} | real test % | ignoring marks % | words % | coverage@99 % | precision on accepted % | confident errors | ECE | NLL |\n|---|---|---|---|---|---|---|---|---|"


def table(label, groups):
    lines = [HEAD.format(label)]
    for name, ms in groups:
        lines.append(f"| {name} | " + " | ".join(row(ms)) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    seeds = (42, 43, 44)
    base = [load(OUT / f"baseline_s{s}" / "eval" / "metrics.json") for s in seeds]
    print(table("Phase 0 baseline", [(f"seed {s}", [m]) for s, m in zip(seeds, base)] + [("**mean +- std (3 seeds)**", base)]))
    print()
    alphas = ["0.0", "0.25", "0.5", "0.75", "0.9", "1.0"]
    groups = [(f"a = {a}", [load(OUT / "phase1" / f"wise_baseline_s{s}_a{a}" / "metrics.json") for s in seeds]) for a in alphas]
    print(table("WiSE-FT (mean +- std over the 3 baseline seeds)", groups))
    print()
    soup = load(OUT / "phase1" / "soup_baseline" / "metrics.json")
    print(table("Soup", [("soup of 3 seeds", [soup]), ("individual seeds (mean +- std)", base)]))
