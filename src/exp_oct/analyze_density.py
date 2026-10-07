"""Pixel density analysis of every crop we have: real train/val/test, junk, synthetic (by bucket and style), replay words, and the user's 226 ROI crops.
Read-only on all data. Outputs go to outputs/exp_oct/density/.

    venv_parseq/Scripts/python src/exp_oct/analyze_density.py

Per crop: size and aspect ratio, background/ink contrast, ink fraction (Otsu), stroke width (2*area/perimeter), where the ink sits (bounding box, touching the
left/right/top/bottom border), sharpness, and what the 32x128 resize does to it (scale factors, stroke width and ink pixels after stretch and after pad).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "master"))
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
import dataset_io  # noqa: E402
from progress import log  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct" / "density"
OUT.mkdir(parents=True, exist_ok=True)
H, W = 32, 128


def stats(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = g.shape
    thr, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # ink = dark = 255
    ink = mask > 0
    n = int(ink.sum())
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(border)); ink_level = float(g[ink].mean()) if n else bg
    r = {"w": w, "h": h, "aspect": w / h, "bg_level": bg, "ink_level": ink_level, "contrast": bg - ink_level, "otsu": float(thr),
         "ink_fraction": n / (w * h), "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var())}
    if n == 0:
        return {**r, "stroke_w": np.nan, "bx0": np.nan, "bx1": np.nan, "by0": np.nan, "by1": np.nan, "touch_l": False, "touch_r": False, "touch_t": False, "touch_b": False}
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    ys, xs = np.where(ink)
    r.update({"stroke_w": 2 * n / max(per, 1), "bx0": xs.min() / w, "bx1": (xs.max() + 1) / w, "by0": ys.min() / h, "by1": (ys.max() + 1) / h,
              "touch_l": bool(ink[:, :2].any()), "touch_r": bool(ink[:, -2:].any()), "touch_t": bool(ink[:2].any()), "touch_b": bool(ink[-2:].any())})
    return r


def add_resize(r):
    sx, sy = W / r["w"], H / r["h"]
    s_pad = min(sx, sy)
    r.update({"scale_x": sx, "scale_y": sy, "distortion": sx / sy,                         # stretch: x and y scaled differently
              "stroke_w_stretch": r["stroke_w"] * (sx * sy) ** 0.5, "ink_px_stretch": r["ink_fraction"] * W * H,
              "stroke_w_pad": r["stroke_w"] * s_pad, "ink_px_pad": r["ink_fraction"] * r["w"] * r["h"] * s_pad ** 2,
              "canvas_used_pad": (r["w"] * s_pad * r["h"] * s_pad) / (W * H)})
    return r


rows = []
log("density analysis: reading master dataset (real + synthetic) ...")
for split, label, png, meta in dataset_io.samples():
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    grp = ("real_" + split if meta["source"].startswith("real") else "synthetic_" + split) if split != "junk" else "junk_real"
    rows.append(add_resize({"group": grp, "source": meta["source"], "bucket": meta.get("bucket", ""), "style": meta.get("style", ""), "label": label,
                            "filename": meta["filename"], **stats(img)}))
log(f"  master dataset: {len(rows)} crops")
rw = pd.read_csv(ROOT / "outputs/exp_oct/replay_words/replay_words.csv", keep_default_na=False, dtype=str)
for r in rw.itertuples():
    img = cv2.imread(str(ROOT / "outputs/exp_oct/replay_words" / r.filename))
    rows.append(add_resize({"group": "replay_words", "source": "replay_word", "bucket": r.kind, "style": "", "label": r.text, "filename": r.filename, **stats(img)}))
q = pd.read_csv("D:/OCR_codes/data/roi_watch/review_queue.csv", keep_default_na=False, dtype=str)
for r in q.itertuples():
    img = cv2.imread(str(Path("D:/OCR_codes/data/roi_watch/input") / r.filename))
    rows.append(add_resize({"group": "roi_doc" if r.filename.startswith("Doc") else "roi_screenshot", "source": "user_roi", "bucket": "", "style": "",
                            "label": r.corrected_text, "filename": r.filename, **stats(img)}))
d = pd.DataFrame(rows)
d.to_csv(OUT / "density_per_crop.csv", index=False)
log(f"  total {len(d)} crops analysed -> {OUT / 'density_per_crop.csv'}")

cols = ["w", "h", "aspect", "contrast", "ink_fraction", "stroke_w", "distortion", "stroke_w_stretch", "stroke_w_pad", "canvas_used_pad", "sharpness"]
med = d.groupby("group")[cols].median().round(3)
med.insert(0, "n", d.groupby("group").size())
for c in ("touch_l", "touch_r", "touch_t", "touch_b"):
    med[c + " %"] = (d.groupby("group")[c].mean() * 100).round(1)
med.to_csv(OUT / "summary_by_group.csv")
print("\nMEDIANS BY GROUP")
print(med.to_string())
syn = d[d.group.str.startswith("synthetic")].groupby(["bucket"])[["w", "h", "aspect", "ink_fraction", "stroke_w", "stroke_w_stretch", "contrast"]].median().round(3)
syn.insert(0, "n", d[d.group.str.startswith("synthetic")].groupby("bucket").size())
print("\nSYNTHETIC BY BUCKET (medians)")
print(syn.to_string())
st = d[d.group.str.startswith("synthetic")].groupby(["style"])[["aspect", "ink_fraction", "stroke_w", "touch_l"]].median().round(3)
print("\nSYNTHETIC BY STYLE (medians)")
print(st.to_string())

# ---- ROI: does ink touching the left border / low density go with errors? uses the scored table from roi_score.py
sc = pd.read_csv(ROOT / "outputs/exp_oct/roi_check/roi_scored.csv", keep_default_na=False, dtype=str)
roi = d[d.group.str.startswith("roi")].merge(sc[["filename", "truth", "pipeline_pred", "baseline_s42", "baseline_s43", "baseline_s44", "replay30_s42", "replay30_s43", "replay30_s44"]], on="filename")
roi["pipe_ok"] = roi.pipeline_pred == roi.truth
roi["base_ok"] = np.mean([roi[f"baseline_s{s}"] == roi.truth for s in (42, 43, 44)], axis=0)
roi["rep_ok"] = np.mean([roi[f"replay30_s{s}"] == roi.truth for s in (42, 43, 44)], axis=0)
print("\nROI CROPS: accuracy by ink position / density (pipeline | mean of 3 baseline seeds | mean of 3 replay30 seeds)")
for name, m in (("ink touches LEFT border", roi.touch_l), ("does NOT touch left", ~roi.touch_l), ("ink touches RIGHT border", roi.touch_r),
                ("ink touches TOP or BOTTOM", roi.touch_t | roi.touch_b), ("no border contact", ~(roi.touch_l | roi.touch_r | roi.touch_t | roi.touch_b)),
                ("ink fraction < 0.05", roi.ink_fraction < 0.05), ("ink fraction >= 0.15", roi.ink_fraction >= 0.15),
                ("stroke after stretch < 1.5 px", roi.stroke_w_stretch < 1.5), ("stroke after stretch >= 2.5 px", roi.stroke_w_stretch >= 2.5)):
    for g in ("roi_doc", "roi_screenshot"):
        x = roi[m & (roi.group == g)]
        if len(x):
            print(f"  {g:15s} {name:30s} n={len(x):3d}  pipeline {x.pipe_ok.mean()*100:5.1f}%  baseline {x.base_ok.mean()*100:5.1f}%  replay30 {x.rep_ok.mean()*100:5.1f}%")
roi.to_csv(OUT / "roi_density_vs_accuracy.csv", index=False)

# ---- plots
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

G = {"real_train": "#1b9e77", "synthetic_train": "#d95f02", "replay_words": "#7570b3", "roi_doc": "#e7298a", "roi_screenshot": "#222222"}
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for i, (c, t, rng) in enumerate([("ink_fraction", "ink fraction (Otsu)", (0, 0.5)), ("stroke_w", "stroke width, original px", (0, 12)), ("stroke_w_stretch", "stroke width after 32x128 stretch, px", (0, 6)),
                                 ("aspect", "aspect ratio w/h", (0, 5)), ("h", "crop height, px", (0, 300)), ("contrast", "ink contrast (background - ink gray level)", (0, 255))]):
    a = ax.flat[i]
    for g, col in G.items():
        v = d[d.group == g][c].dropna()
        a.hist(v.clip(*rng), bins=50, range=rng, density=True, histtype="step", color=col, label=f"{g} (n={len(v)})", lw=1.6)
    a.set_title(t); a.grid(alpha=.25)
ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "density_histograms.png", dpi=130)
log("density analysis done: " + str(OUT))
