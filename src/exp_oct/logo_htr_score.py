"""Score LoGo-HTR (zero-shot, no training) on the project's three evaluation sets.

    venv_logohtr/Scripts/python src/exp_oct/logo_htr_score.py [--beam 10]

Sets: user's 226 reviewed ROI crops (truth = corrected_text), real test (data/docsplit/seed42/test.csv, 379), words (data/open_read/word_eval.csv, 300).
Accuracy is case-sensitive exact match. For the two number sets the prediction is also scored with marks stripped (. ( ) [ ] etc. removed, as the labels are
characters only); words are scored on the raw text and with marks stripped. Output: outputs/exp_oct/logohtr/ (predictions per set, summary.csv).
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
import logo_htr_infer as L  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct" / "logohtr"
CROPS = Path("D:/OCR_codes/data/roi_watch/input")


def norm(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s))


def fmt(t):
    if re.fullmatch(r"\d", t): return "1 digit"
    if re.fullmatch(r"\d\d", t): return "2 digits"
    if re.fullmatch(r"\d\d[A-Za-z]", t): return "NNa"
    if re.fullmatch(r"\d[A-Za-z]", t): return "Na"
    if re.fullmatch(r"[A-Za-z]", t): return "1 letter"
    if re.fullmatch(r"[ivxIVX]{2}", t): return "2 letters (Roman)"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=10)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, _ = L.build_model(L.DEFAULT_CKPT)
    model.to(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv("D:/OCR_codes/data/roi_watch/review_queue.csv", keep_default_na=False, dtype=str)
    t = pd.read_csv(ROOT / "data/docsplit/seed42/test.csv", keep_default_na=False, dtype=str)
    w = pd.read_csv(ROOT / "data/open_read/word_eval.csv", keep_default_na=False, dtype=str)
    sets = {
        "roi226": pd.DataFrame({"filename": q.filename, "path": [str(CROPS / f) for f in q.filename], "truth": q.corrected_text.str.strip(),
                                "subset": ["documents" if f.startswith("Doc") else "screenshots" for f in q.filename]}),
        "test379": pd.DataFrame({"filename": t.filename, "path": t.image_path, "truth": t.text, "subset": "test"}),
        "words300": pd.DataFrame({"filename": w.filename, "path": w.image_path, "truth": w.text, "subset": w.kind}),
    }
    rows = []
    for name, d in sets.items():
        d["pred"] = L.predict(model, vocab, d.path.tolist(), a.beam, dev, batch=16)
        d["ok_raw"] = d.pred == d.truth
        d["ok_marks_stripped"] = d.pred.map(norm) == d.truth
        d["format"] = d.truth.map(fmt)
        d.to_csv(OUT / f"preds_{name}.csv", index=False)
        n = len(d)
        rows.append((name, "ALL", n, d.ok_raw.mean() * 100, d.ok_marks_stripped.mean() * 100))
        for col in ("subset", "format"):
            for k, g in d.groupby(col):
                if k != "test":
                    rows.append((name, f"{col}={k}", len(g), g.ok_raw.mean() * 100, g.ok_marks_stripped.mean() * 100))
        print(f"{name}: {d.ok_raw.sum()}/{n} raw = {d.ok_raw.mean() * 100:.1f}% | marks stripped {d.ok_marks_stripped.mean() * 100:.1f}%", flush=True)
    s = pd.DataFrame(rows, columns=["set", "group", "n", "acc_raw", "acc_marks_stripped"]).round(1)
    s.to_csv(OUT / "summary.csv", index=False)
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
