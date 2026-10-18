"""Scores PARSeq checkpoints on the open-reading criteria and writes reports/open_read_eval.md (+ per-sample CSVs).

  real test (old labels)  data/open_read/test.csv: 379 unseen-document crops, labels with brackets/dots/Q stripped, so every earlier
                          model is comparable. A prediction counts as correct if its alphanumerics (minus a leading Q / Ans) equal the label.
  real test (verbatim)    307 of those crops with verbatim labels (brackets/dots kept); exact string match.
  words                   data/open_read/word_eval.csv: 300 synthetic random-word crops (words unseen in training); exact match.
  junk rejected           112 junk crops from validation documents; "rejected" = the prediction contains no letter or digit (`~`, nothing, punctuation).
  real val (verbatim)     265 relabelled real val crops, exact match.

Usage: venv_parseq/Scripts/python src/open_read/evaluate_open_read.py name=ckpt [name=ckpt ...]   (ckpt `pretrained` = original PARSeq)
"""
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "parseq"))
from model_io import decode, get_device, load_model, make_transform  # noqa: E402

D = ROOT / "data" / "open_read"
norm = lambda s: re.sub(r"^(?:Ans|ans|Q|q)", "", re.sub(r"[^A-Za-z0-9]", "", str(s)))  # noqa: E731
rd = lambda f: pd.read_csv(D / f, keep_default_na=False, dtype=str)  # noqa: E731
val = rd("val.csv")
SETS = {"test": rd("test.csv"), "test_verbatim": rd("test_verbatim.csv"), "words": rd("word_eval.csv"),
        "junk": val[val.kind == "junk"], "val_verbatim": val[val.kind == "real"]}
OUT = ROOT / "reports" / "open_read"
OUT.mkdir(parents=True, exist_ok=True)
dev = get_device()
report = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--report=")), "eval")
rows = []
for spec in [a for a in sys.argv[1:] if not a.startswith("--")]:
    name, ck = spec.split("=", 1)
    m = load_model("pretrained=parseq" if ck == "pretrained" else ck, dev)
    tf, res = make_transform(m), {"model": name}
    for sname, df in SETS.items():
        preds = [decode(m, tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))[0][0] for p in df.image_path]
        d = df.assign(prediction=preds)
        if sname == "test": ok = [norm(p) == t for p, t in zip(preds, d.text)]
        elif sname == "junk": ok = [re.search(r"[A-Za-z0-9]", p) is None for p in preds]
        else: ok = [p == t for p, t in zip(preds, d.text)]
        d["correct"] = ok
        d.to_csv(OUT / f"{report}_{name}_{sname}.csv", index=False)
        res[sname] = 100 * sum(ok) / len(ok)
    rows.append(res)
    print(res, flush=True)
t = pd.DataFrame(rows).set_index("model").round(1)
t.columns = ["real test, old labels (379)", "real test, verbatim (307)", "words (300)", "junk rejected (112)", "real val, verbatim (265)"]
md = ["| model | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
md += [f"| {n} | " + " | ".join(f"{v:.1f}" for v in r) + " |" for n, r in t.iterrows()]
table = chr(10).join(md)
(ROOT / "reports" / f"open_read_{report}.md").write_text("# Open-reading evaluation (% correct)" + chr(10) * 2 + table + chr(10), encoding="utf-8")
print(table)
