"""Compare models on the user's ROI folder (D:/OCR_codes/data/roi_watch/input): predictions, confidence, agreement, and (for the Doc crops)
agreement with the archived UNVERIFIED reading in docs/log/newdoc_labels/roi_labels.csv.

    venv_parseq/Scripts/python src/exp_oct/roi_compare.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq"):
    sys.path.insert(0, str(ROOT / p))
from label_grammar import matches_grammar  # noqa: E402
from model_io import decode, get_device, load_model, make_transform  # noqa: E402

FOLDER = Path("D:/OCR_codes/data/roi_watch/input")
MODELS = {"master_v3": "models/parseq_master_v3/parseq_master_v3.ckpt",
          "replay30_s44": "outputs/exp_oct/models/parseq_replay30_s44/parseq_replay30_s44.ckpt",
          "baseline_s44": "outputs/exp_oct/baseline_s44/checkpoints/parseq_baseline_s44.ckpt"}
dev = get_device()
files = sorted(FOLDER.glob("*.png"))
labels = pd.read_csv(ROOT / "docs/log/newdoc_labels/roi_labels.csv", keep_default_na=False, dtype=str).set_index("filename")["text"]
rows = pd.DataFrame({"filename": [f.name for f in files]})
for name, ck in MODELS.items():
    m = load_model(str(ROOT / ck), dev)
    tf = make_transform(m)
    out = [decode(m, tf(Image.open(f).convert("RGB")).unsqueeze(0).to(dev))[0] for f in files]
    rows[name] = [o[0] for o in out]
    rows[name + "_conf"] = [o[1] for o in out]
rows["claude_reading"] = rows.filename.map(labels).fillna("")
rows["set"] = ["doc" if f.startswith("Doc") else "screenshot" for f in rows.filename]
out_dir = ROOT / "outputs" / "exp_oct" / "roi_check"
out_dir.mkdir(parents=True, exist_ok=True)
rows.to_csv(out_dir / "roi_compare.csv", index=False)

for name in MODELS:
    ok_grammar = rows[name].map(lambda s: bool(matches_grammar(s)))
    print(f"\n== {name}")
    for st, d in rows.groupby("set"):
        g = ok_grammar[d.index]
        print(f"  {st:10s} n={len(d):3d}  mean conf {d[name + '_conf'].mean():.3f}  conf<0.5 {(d[name + '_conf'] < 0.5).sum():3d}  "
              f"conf<0.9 {(d[name + '_conf'] < 0.9).sum():3d}  not-valid-label-shape {(~g).sum():3d}")
    d = rows[rows.claude_reading != ""]
    same = (d[name].str.lower() == d.claude_reading.str.lower())
    print(f"  agreement with Claude's UNVERIFIED reading on {len(d)} doc crops: {same.sum()}/{len(d)} = {same.mean() * 100:.1f}% (case-insensitive)")
a, b = "master_v3", "replay30_s44"
diff = rows[rows[a] != rows[b]]
print(f"\n{a} vs {b}: {len(diff)} of {len(rows)} predictions differ")
print(diff[["filename", "claude_reading", a, a + "_conf", b, b + "_conf"]].round(3).to_string(index=False))
