"""Which of the crops we actually have are clearly visible, and which are compromised?

    venv_parseq/Scripts/python src/exp_oct/crop_visibility.py

Two sources, kept separate on purpose:
  1. MANUAL: a visual read of every crop in D:/OCR_codes/data/roi_watch/input (216 crops, shown WITHOUT their labels so the
     judgement does not depend on what the answer should be). It is subjective: a crop is flagged only when a character is
     cut off, overwritten, struck through or crossed by a stray line, so a person would also hesitate. Faint-but-legible
     thin strokes were NOT flagged. Sheets used: outputs/exp_oct/compare_small/visibility_sheets/.
  2. MEASURED: objective numbers per crop (ink darkness gap, ink share, stroke width, share of ink touching the crop border).
     None of them separated the crops a person finds hard; they are reported so the flags can be checked, not used to decide.
The flag is independent of the model results; model accuracy by flag is reported afterwards, not used to set it.
Output: outputs/exp_oct/compare_small/crop_visibility.csv and flagged_crops.png, plus a printed summary.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CROPS = Path("D:/OCR_codes/data/roi_watch/input")
OUT = ROOT / "outputs" / "exp_oct" / "compare_small"
LIVE = ROOT / "demo" / "out" / "live_run_226crops.csv"

CUT = "character partly cut off at the crop edge"
MESSY = "overwritten, struck through, or crossed by a stray line"
FLAGS = {  # filename -> reason (visual read, labels hidden)
    **{f: CUT for f in [
        "Doc0752_page_005_roi_2.png", "Doc0755_page_004_roi_1.png", "Doc0755_page_005_roi_1.png", "Doc0757_page_007_roi_1.png",
        "Doc0767_page_001_roi_1.png", "Doc0767_page_003_roi_3.png", "Doc0767_page_005_roi_1.png", "Doc0769_page_001_roi_1.png",
        "Doc0769_page_005_roi_1.png"]},
    **{f: MESSY for f in [
        "Doc0729_page_002_roi_3.png", "Doc0729_page_003_roi_5.png", "Doc0729_page_007_roi_4.png", "Doc0771_page_002_roi_5.png",
        "Doc0773_page_007_roi_3.png", "Doc0775_page_018_roi_4.png", "Doc0775_page_018_roi_6.png",
        "Screenshot 2026-10-02 230000.png", "Screenshot 2026-10-02 230002.png", "Screenshot 2026-10-02 230006.png",
        "Screenshot 2026-10-02 230023.png", "Screenshot 2026-10-02 230029.png", "Screenshot 2026-10-05 112258.png",
        "Screenshot 2026-10-05 112305.png", "Screenshot 2026-10-05 112418.png"]},
}


def measure(path):
    g = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    flat = np.sort(g.ravel())
    paper = float(np.median(g))
    dark = float(flat[:max(1, int(.03 * flat.size))].mean())
    ink = g < 128
    edge = np.zeros_like(ink)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return {"height": g.shape[0], "width": g.shape[1], "ink_darkness_gap": round(paper - dark, 1),
            "ink_share": round(float(ink.mean()), 4), "stroke_px": round(2 * float(np.median(dt[ink])) if ink.any() else 0, 1),
            "border_ink_share": round(float((ink & edge).sum() / max(ink.sum(), 1)), 3)}


def main():
    q = pd.read_csv("D:/OCR_codes/data/roi_watch/review_queue.csv", keep_default_na=False, dtype=str)
    label = dict(zip(q.filename, q.corrected_text.str.strip()))
    names = [f for f in q.filename if (CROPS / f).exists()]
    unknown = [f for f in FLAGS if f not in names]
    if unknown:
        print("NOTE: flagged crops no longer present:", unknown)
    rows = []
    for f in names:
        r = {"filename": f, "label": label[f], "visibility": "compromised" if f in FLAGS else "clear", "reason": FLAGS.get(f, "")}
        r.update(measure(CROPS / f))
        rows.append(r)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "crop_visibility.csv", index=False, encoding="utf-8")
    n_bad = int((d.visibility == "compromised").sum())
    print(f"{len(d)} crops present: {len(d) - n_bad} clear, {n_bad} compromised")
    print(d[d.visibility == "compromised"].reason.value_counts().to_string(), "\n")

    # do the measured numbers back the 'cut off' flags? (ink touching the crop border)
    cut = d[d.reason == CUT]
    print(f"border_ink_share: cut-off flagged median {cut.border_ink_share.median():.3f} vs others {d[d.reason != CUT].border_ink_share.median():.3f}; "
          f"{(cut.border_ink_share > 0.02).sum()}/{len(cut)} flagged crops exceed 0.02 vs {(d[d.reason != CUT].border_ink_share > 0.02).sum()}/{len(d) - len(cut)} others")

    if LIVE.exists():
        live = pd.read_csv(LIVE, keep_default_na=False, dtype=str).set_index("filename")
        print("\nmodel accuracy by visibility (live run, current scoring):")
        for vis, g in d.groupby("visibility"):
            sub = live.loc[g.filename]
            cells = []
            for m in ("parseq", "mistral", "chandra"):
                k = int((sub[m + "_ok"] == "True").sum())
                cells.append(f"{m} {k}/{len(sub)} = {100 * k / len(sub):.0f}%")
            print(f"  {vis:<12} {len(g):>3} crops | " + " | ".join(cells))

    # contact sheet of the flagged crops
    fl = d[d.visibility == "compromised"]
    f16 = ImageFont.truetype("arial.ttf", 14)
    cols, cw, ch = 3, 430, 100
    rows_n = (len(fl) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows_n * ch), "white")
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(fl.itertuples()):
        x, y = (i % cols) * cw, (i // cols) * ch
        im = Image.open(CROPS / r.filename).convert("RGB")
        s = min(240 / im.width, 80 / im.height)
        sheet.paste(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS), (x + 6, y + 6))
        dr.text((x + 256, y + 8), f"label {r.label}", fill="black", font=f16)
        dr.text((x + 256, y + 30), "cut off" if r.reason == CUT else "overwritten / struck", fill=(179, 38, 30), font=f16)
        dr.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(215, 215, 215))
    sheet.save(OUT / "flagged_crops.png")
    print("\nwrote", OUT / "crop_visibility.csv", "and flagged_crops.png")


if __name__ == "__main__":
    main()
