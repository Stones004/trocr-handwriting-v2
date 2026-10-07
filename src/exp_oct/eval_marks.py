"""Score checkpoints on the held-out marked-NNa set (300 synthetic crops, different seed from training; outputs/exp_oct/marks_nna/marks_eval.csv).
Breaks accuracy down by pattern family, by which gap holds a mark, and by mark type. Labels are the characters only (e.g. 12a).

    venv_parseq/Scripts/python src/exp_oct/eval_marks.py replay_p30_s42 replay_p30_s43 ... rm6k_s42 ...   (run names under outputs/exp_oct/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "parseq"))
from model_io import decode, get_device, load_model, make_transform  # noqa: E402

ev = pd.read_csv(ROOT / "outputs/exp_oct/marks_nna/marks_eval.csv", keep_default_na=False, dtype=str)
dev = get_device()
res = {}
for run in sys.argv[1:]:
    m = load_model(str(ROOT / "outputs/exp_oct" / run / "checkpoints" / f"parseq_{run}.ckpt"), dev)
    tf = make_transform(m)
    out = [decode(m, tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))[0] for p in ev.image_path]
    ok = np.array([o[0] == t for o, t in zip(out, ev.text)])
    d = ev.assign(ok=ok, pred=[o[0] for o in out], conf=[o[1] for o in out])
    d.to_csv(ROOT / "outputs/exp_oct" / run / "marks_eval_preds.csv", index=False)
    row = {"overall": ok.mean() * 100}
    for fam, g in d.groupby("family"):
        row[f"fam:{fam}"] = g.ok.mean() * 100
    for gi in range(4):
        g = d[d[f"gap{gi}"] != ""]
        row[f"gap{gi} has mark"] = g.ok.mean() * 100
    for mk, name in ((".", "dot"), ("(", "open paren"), (")", "close paren"), ("[", "open bracket"), ("]", "close bracket")):
        g = d[d.drawn.str.contains(mk, regex=False)]
        row[f"has {name}"] = g.ok.mean() * 100
    res[run] = row
t = pd.DataFrame(res).round(1)
print(t.to_string())
for grp in sorted({r.rsplit("_s", 1)[0] for r in res}):
    cols = [c for c in res if c.rsplit("_s", 1)[0] == grp]
    print(f"\n{grp} mean over {len(cols)} seed(s): overall {t[cols].loc['overall'].mean():.1f}")
