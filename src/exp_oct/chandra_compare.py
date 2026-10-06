"""Compare the user's PARSeq pipeline output (review_queue.csv `prediction`) with Chandra OCR raw output (outputs/exp_oct/roi_check/chandra_raw.py),
both scored against the user's reviewed truth (`corrected_text`).

    venv_parseq/Scripts/python src/exp_oct/chandra_compare.py

Chandra writes the marks (dots, brackets, Q.) and sometimes LaTeX / words / other scripts, while our labels are characters only. So Chandra is scored after a
documented normalisation:
  strict   : Unicode NFKC (circled digits, Roman numeral glyphs -> ASCII), drop everything that is not A-Z a-z 0-9, compare case-sensitively
  lenient  : strict, then also drop a leading Q / q / A / a that is directly followed by a digit (the `Q.` / `a.` question prefix), compare case-insensitively
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "exp_oct" / "roi_check"))
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
from chandra_raw import RAW  # noqa: E402
from eval_suite import wilson  # noqa: E402

q = pd.read_csv("D:/OCR_codes/data/roi_watch/review_queue.csv", keep_default_na=False, dtype=str)
assert set(RAW) == set(q.filename) and len(RAW) == 226, (len(RAW), set(q.filename) ^ set(RAW))
d = pd.DataFrame({"filename": q.filename, "truth": q.corrected_text.str.strip(), "parseq": q.prediction, "parseq_conf": q.confidence.astype(float)})
d["chandra_raw"] = d.filename.map(RAW)


def strict(s):
    return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKC", s))


def lenient(s):
    t = strict(s)
    return re.sub(r"^[QqAa](?=[0-9])", "", t).lower()


def junk_kind(s):
    if re.search(r"\\[a-z]+|<(sub|math)|\$", s):
        return "LaTeX / HTML markup"
    if len(s) > 25:
        return "sentence or runaway repetition" if "Ⓒ" not in s else "runaway repetition"
    if re.search(r"[^\x00-\x7F]", unicodedata.normalize("NFKC", s)):
        return "non-Latin / symbol characters"
    return ""


d["chandra_strict"] = d.chandra_raw.map(strict)
d["chandra_lenient"] = d.chandra_raw.map(lenient)
d["junk"] = d.chandra_raw.map(junk_kind)
d["parseq_ok"] = d.parseq == d.truth
d["parseq_ok_ci"] = d.parseq.str.lower() == d.truth.str.lower()
d["chandra_raw_ok"] = d.chandra_raw == d.truth
d["chandra_strict_ok"] = d.chandra_strict == d.truth
d["chandra_len_ok"] = d.chandra_lenient == d.truth.str.lower()
d["subset"] = np.where(d.filename.str.startswith("Doc"), "document crops", "screenshots")
out = ROOT / "outputs" / "exp_oct" / "roi_check" / "chandra_vs_parseq.csv"
d.to_csv(out, index=False)


def row(name, ok, m=None):
    m = np.ones(len(d), bool) if m is None else m
    k, n = int(ok[m].sum()), int(m.sum())
    lo, hi = wilson(k, n)
    return f"| {name} | {k}/{n} = {k/n*100:.1f}% [{lo*100:.1f}-{hi*100:.1f}] |"


print("ALL 226 crops (truth = user-reviewed labels)")
print("| method | correct | ")
for nm, col in (("PARSeq pipeline, exact", "parseq_ok"), ("PARSeq pipeline, case-insensitive", "parseq_ok_ci"),
                ("Chandra RAW text, exact (no cleaning)", "chandra_raw_ok"), ("Chandra, strict normalisation", "chandra_strict_ok"),
                ("Chandra, lenient normalisation (drop Q./a. prefix, ignore case)", "chandra_len_ok")):
    print(row(nm, d[col]))
for sb in ("document crops", "screenshots"):
    m = (d.subset == sb).to_numpy()
    print(f"\n{sb} (n={m.sum()})")
    for nm, col in (("PARSeq", "parseq_ok"), ("Chandra strict", "chandra_strict_ok"), ("Chandra lenient", "chandra_len_ok")):
        print(row(nm, d[col], m))

print("\nChandra outputs that are not label-like:", d.junk.replace("", np.nan).value_counts().to_dict(), "| total", int((d.junk != "").sum()))
print("Chandra outputs containing marks (any of . ) ( [ ] Q): ", int(d.chandra_raw.str.contains(r"[.\)\(\[\]]|^Q").sum()), "of 226")

# agreement analysis (lenient Chandra vs PARSeq exact)
a, b = d.parseq_ok.to_numpy(), d.chandra_len_ok.to_numpy()
both, only_p, only_c, none = (a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()
print(f"\nAgreement table (PARSeq exact vs Chandra lenient): both right {both}, only PARSeq right {only_p}, only Chandra right {only_c}, both wrong {none}")
agree = (d.parseq.str.lower() == d.chandra_lenient).to_numpy()
print(f"PARSeq and Chandra give the same string: {agree.sum()}/226; when they agree it is correct {np.mean(d.parseq_ok[agree]) * 100:.1f}% of the time (n={agree.sum()}); "
      f"when they disagree PARSeq right {d.parseq_ok[~agree].sum()}, Chandra right {d.chandra_len_ok[~agree].sum()}, neither {(~d.parseq_ok[~agree] & ~d.chandra_len_ok[~agree]).sum()} (n={(~agree).sum()})")
print("Oracle (either right):", int((a | b).sum()), "/226 =", f"{(a | b).mean() * 100:.1f}%")

# PARSeq confidence as a router: where PARSeq conf < t, use Chandra
for t in (0.9, 0.95, 0.99):
    use_c = (d.parseq_conf < t).to_numpy()
    ok = np.where(use_c, b, a)
    print(f"Router: PARSeq if conf >= {t} else Chandra: {ok.sum()}/226 = {ok.mean() * 100:.1f}%  (sends {use_c.mean() * 100:.0f}% of crops to Chandra)")

print("\nWhere they differ (first 40): filename | truth | PARSeq | Chandra raw")
x = d[~agree][["filename", "truth", "parseq", "chandra_raw"]].copy()
x["chandra_raw"] = x.chandra_raw.str.slice(0, 40)
print(x.head(40).to_string(index=False))
print("\nWords: ", d[d.truth.str.fullmatch(r"[A-Za-z]{2,}")][["filename", "truth", "parseq", "chandra_raw"]].to_string(index=False))
