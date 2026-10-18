"""Small side-by-side of the crops that were actually run through all three models live (demo/live_compare.py).

    python src/exp_oct/compare_small.py [--csv demo/out/live_run_226crops.csv]

Input: the per-crop answers saved by the live demo (PARSeq replay30_s44, Pixtral 12B via API, Chandra 5B 4-bit local),
all given the ROI-enhanced crop. Scoring is the demo's own `judge` (single source of truth):
  PARSeq  vs the characters-only label;
  Mistral and Chandra vs the as-written transcript (demo/verbatim_truth.csv), both sides stripped to letters and digits
  (answers of 10+ characters are not stripped).
Output (outputs/exp_oct/compare_small/): small_comparison.html, summary.csv, per_crop.csv, and a printed verdict.
"""
import argparse
import base64
import csv
import html
import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
import live_compare as L  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct" / "compare_small"
MODELS = ["parseq", "mistral", "chandra"]
NAME = {"parseq": "PARSeq", "mistral": "Mistral (Pixtral 12B)", "chandra": "Chandra (5B)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def why_wrong(m, txt, c):
    """Short reason for a miss, from the answer vs the as-written transcript and the label."""
    a, t, lab = L.norm_answer(m, txt), L.key(c["verbatim"]), L.key(c["truth"])
    if not str(txt).strip() or str(txt).strip() == "?":
        return "no answer"
    if a == lab and a != t:
        return "dropped the Q/prefix the writer wrote"
    if a and (t.endswith(a) or t.startswith(a) or a.endswith(t) or a.startswith(t)):
        return "partial or extra characters"
    return "misread"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "demo" / "out" / "live_run_226crops.csv"))
    a = ap.parse_args()
    truth, verb = L.truth_map(), L.verbatim_map()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    rows = [r for r in rows if (L.CROPS_DIR / r["filename"]).exists()]  # skip crops removed from the input folder
    recs = []
    for r in rows:
        c = {"truth": truth[r["filename"]], "verbatim": verb.get(r["filename"])}
        rec = {"filename": r["filename"], "truth": c["truth"], "as_written": c["verbatim"] or ""}
        for m in MODELS:
            txt = r[m]
            answered = r[m + "_seconds"] != ""  # a crop where the run was stopped before this model answered is not scored
            ok = L.judge(m, txt, c)[0] if answered else None
            rec[m], rec[m + "_ok"], rec[m + "_s"] = txt, ok, float(r[m + "_seconds"]) if answered else None
            rec[m + "_why"] = "" if ok is None or ok else why_wrong(m, txt, c)
        recs.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    common = [x for x in recs if all(x[m + "_ok"] is not None for m in MODELS)]
    summ = []
    for m in MODELS:
        sc = [x for x in recs if x[m + "_ok"] is not None]
        k, n = sum(x[m + "_ok"] for x in sc), len(sc)
        kc = sum(x[m + "_ok"] for x in common)
        lo, hi = wilson(kc, len(common))
        secs = sorted(x[m + "_s"] for x in sc)
        summ.append({"model": NAME[m], "correct_all_scored": f"{k}/{n}", "correct_common": f"{kc}/{len(common)}",
                     "accuracy_common_pct": round(100 * kc / len(common), 1), "wilson95_lo": round(100 * lo, 1),
                     "wilson95_hi": round(100 * hi, 1), "median_seconds": secs[len(secs) // 2]})
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    with open(OUT / "per_crop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)

    # paired comparison on the common crops
    def pair(x, y):
        both = sum(1 for r in common if r[x + "_ok"] and r[y + "_ok"])
        ox = sum(1 for r in common if r[x + "_ok"] and not r[y + "_ok"])
        oy = sum(1 for r in common if r[y + "_ok"] and not r[x + "_ok"])
        return both, ox, oy, len(common) - both - ox - oy
    print(f"crops run through all three: {len(recs)} | scored for all three: {len(common)}")
    for s in summ:
        print(f"  {s['model']:<24} {s['correct_common']:>6} = {s['accuracy_common_pct']:>5}%  (95% CI {s['wilson95_lo']}-{s['wilson95_hi']})  median {s['median_seconds']}s/crop")
    for x, y in (("parseq", "chandra"), ("parseq", "mistral"), ("chandra", "mistral")):
        b, ox, oy, n = pair(x, y)
        print(f"  {NAME[x]} vs {NAME[y]}: both right {b}, only {x} {ox}, only {y} {oy}, neither {n}")
    for m in ("mistral", "chandra", "parseq"):
        why = {}
        for r in recs:
            if r[m + "_why"]:
                why[r[m + "_why"]] = why.get(r[m + "_why"], 0) + 1
        print(f"  why {NAME[m]} missed: {why}")
    allwrong = [r["filename"] for r in common if not any(r[m + "_ok"] for m in MODELS)]
    print("  crops no model got right:", allwrong)
    only_other = [r["filename"] for r in common if not r["parseq_ok"] and (r["mistral_ok"] or r["chandra_ok"])]
    print("  crops PARSeq missed but Mistral or Chandra got:", only_other)

    # html
    esc = html.escape

    def img(fn):
        p = L.OUT / "enhanced" / (Path(fn).stem + ".png")
        im = __import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB")
        h = 56
        im = im.resize((max(1, round(im.width * h / im.height)), h))
        b = io.BytesIO(); im.save(b, "PNG")
        return base64.b64encode(b.getvalue()).decode()

    def cell(r, m):
        ok = r[m + "_ok"]
        cls = "n" if ok is None else ("g" if ok else "r")
        mark = "&ndash;" if ok is None else ("&#10003;" if ok else "&#10007;")
        return f'<td class="{cls}"><span>{mark}</span> {esc(r[m]) or "(empty)"}</td>'
    body = "".join(
        f'<tr><td><img src="data:image/png;base64,{img(r["filename"])}"></td><td><b>{esc(r["truth"])}</b></td><td>{esc(r["as_written"])}</td>'
        + "".join(cell(r, m) for m in MODELS) + "</tr>" for r in sorted(recs, key=lambda r: sum(bool(r[m + "_ok"]) for m in MODELS)))
    sumhtml = "".join(f"<tr><th>{esc(s['model'])}</th><td>{s['correct_common']}</td><td><b>{s['accuracy_common_pct']}%</b></td>"
                      f"<td>{s['wilson95_lo']}-{s['wilson95_hi']}%</td><td>{s['median_seconds']} s</td></tr>" for s in summ)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Small comparison</title><style>
:root{{--bg:#fff;--fg:#1c1c1c;--mut:#666;--line:#ddd;--g:#e4f4e8;--gt:#1b7a3a;--r:#fbe7e5;--rt:#b3261e}}
@media(prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--mut:#9a9a9a;--line:#333;--g:#17301f;--gt:#6fd48a;--r:#3a1a18;--rt:#ff8a80}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,Segoe UI,sans-serif}}main{{max-width:1000px;margin:0 auto;padding:18px 16px 50px}}
h1{{font-size:22px;margin:0 0 4px}}p{{color:var(--mut);margin:0 0 14px}}table{{border-collapse:collapse;width:100%;margin-bottom:18px}}
th,td{{border:1px solid var(--line);padding:5px 9px;text-align:left}}td img{{display:block;background:#fff;border-radius:3px;max-width:160px}}
td.g{{background:var(--g);color:var(--gt)}}td.r{{background:var(--r);color:var(--rt)}}td.n{{color:var(--mut)}}td span{{font-weight:700;margin-right:4px}}
@media(max-width:700px){{table{{font-size:12px}}}}</style></head><body><main>
<h1>PARSeq vs Mistral vs Chandra on the {len(recs)} crops run live through all three</h1>
<p>Each model saw the same ROI-enhanced crop. PARSeq is scored against the characters-only label; Mistral and Chandra against the as-written transcript, both sides stripped to letters and digits (case-sensitive, exact match). {len(common)} crops have an answer from all three and are used for the accuracy; hardest crops first.</p>
<table><thead><tr><th>Model</th><th>Correct</th><th>Accuracy</th><th>95% interval</th><th>Median time per crop</th></tr></thead><tbody>{sumhtml}</tbody></table>
<table><thead><tr><th>Crop</th><th>Label</th><th>As written</th><th>PARSeq</th><th>Mistral (Pixtral 12B)</th><th>Chandra (5B)</th></tr></thead><tbody>{body}</tbody></table>
<p>Small sample ({len(common)} crops, one reviewer): treat the intervals as the real uncertainty. PARSeq is fine-tuned on this project's data and returns characters only; Mistral and Chandra are general models used without fine-tuning, and the Pixtral prompt told it to ignore marks and the Q prefix, which counts against it when scored as written.</p>
</main></body></html>"""
    (OUT / "small_comparison.html").write_text(page, encoding="utf-8")
    print("wrote", OUT / "small_comparison.html")


if __name__ == "__main__":
    main()
