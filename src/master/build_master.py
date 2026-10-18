"""Build the master training dataset: real crops (document split) + capped synthetic crops, per src/master/spec.py.

    venv_parseq/Scripts/python src/master/build_master.py              # plan + distribution report only (no images)
    venv_parseq/Scripts/python src/master/build_master.py --render     # also render the synthetic crops and write train/val/test.csv

Labels are the characters only: brackets, dots and a Q prefix are drawn in synthetic images but stripped from the label. Real labels
are put through the same stripping (the 6 real crops labelled III. II) A. 2) 4) 3) become III II A 2 4 3). Train = all real train
crops + synthetic; val/test = the real document-split val/test only. Outputs in --out (default data/master): plan.csv, train/val/test.csv
(filename,image_path,text,bucket,source,drawn,style), images/; report in reports/master_distribution.md.
"""
import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec  # noqa: E402

ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--split", default="data/docsplit/seed42", help="document-level real split (train/val/test.csv)")
ap.add_argument("--out", default="data/master")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--extra-real", default="", help="optional CSV of newly labelled new-document crops (rows with holdout=False are trained on); none by default")
ap.add_argument("--annotations", default=None, help="annotation workbook (default: config annotation_file); crops whose raw text starts with Q are repeated")
ap.add_argument("--render", action="store_true")
args = ap.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)

real, changed = {}, 0
for s in ("train", "val", "test"):
    df = pd.read_csv(ROOT / args.split / f"{s}.csv", keep_default_na=False)
    stripped = df.text.map(spec.strip_marks)
    changed += int((stripped != df.text.astype(str)).sum())
    real[s] = df.assign(text=stripped)
# ---- real training rows: originals, Q-prefix crops repeated, newly labelled new-document crops repeated
sys.path.insert(0, str(ROOT / "src" / "common"))
from project_config import cfg  # noqa: E402

real["train"] = real["train"].assign(source="real")
wb = pd.read_excel(args.annotations or cfg["annotation_file"])
q_files = set(wb[wb["text"].astype(str).str.strip().str.match(r"^[Qq]")].filename)
q_rows = real["train"][real["train"].filename.isin(q_files)]
parts = [real["train"]] + [q_rows.assign(source="real_q_repeat")] * (spec.Q_REAL_REPEAT - 1)
n_new = 0
if args.extra_real and (ROOT / args.extra_real).exists():
    nd = pd.read_csv(ROOT / args.extra_real, keep_default_na=False)
    nd = nd[nd.holdout.astype(str) == "False"][["filename", "image_path", "text"]].assign(text=lambda d: d.text.astype(str).map(spec.strip_marks))
    n_new = len(nd)
    parts += [nd.assign(source="newdoc")] + [nd.assign(source="newdoc_repeat")] * (spec.NEWDOC_REPEAT - 1)
unique_real = pd.concat([real["train"], *([nd.assign(source="newdoc")] if n_new else [])], ignore_index=True)
real_counts = unique_real.text.value_counts().to_dict()                  # per-label cap uses UNIQUE real crops, not repeats
real_train_full = pd.concat(parts, ignore_index=True)
rows, notes, truncated = spec.plan_synthetic(real_counts, args.seed)
synth_counts = collections.Counter(r[0] for r in rows)

# ---------------------------------------------------------------- distribution report
lab = pd.DataFrame({"real": pd.Series(real_counts), "synthetic": pd.Series(synth_counts)}).fillna(0).astype(int)
lab["bucket"] = [spec.bucket_of(l) for l in lab.index]
lab["total"] = lab.real + lab.synthetic
lab["cap"] = [spec.cap(l, real_counts) for l in lab.index]
assert (lab.synthetic <= lab.cap).all(), "per-label cap violated"

bk = lab.groupby("bucket").agg(labels=("total", "size"), real=("real", "sum"), synthetic=("synthetic", "sum"), total=("total", "sum"))
bk["synth/real"] = (bk.synthetic / bk.real.where(bk.real > 0)).round(2)
bk["% of train"] = (bk.total / bk.total.sum() * 100).round(1)
bk = bk.reindex([b for b in spec.BUCKETS if b in bk.index])
bk.loc["TOTAL"] = [bk.labels.sum(), bk.real.sum(), bk.synthetic.sum(), bk.total.sum(), round(bk.synthetic.sum() / bk.real.sum(), 2), 100.0]

styles = pd.Series([r[2] for r in rows]).value_counts().reindex(spec.STYLES).fillna(0).astype(int)
styles = pd.DataFrame({"synthetic crops": styles, "% of synthetic": (styles / styles.sum() * 100).round(1)})


def char_table(cr, cs, chars):
    t = pd.DataFrame({"real": pd.Series(cr), "synthetic": pd.Series(cs)}).reindex(list(chars)).fillna(0).astype(int)
    t["total"] = t.real + t.synthetic
    t["% of total"] = (t.total / t.total.sum() * 100).round(1)
    return t


def chars_of(counts, fold):
    c = collections.Counter()
    for label, n in counts.items():
        for ch in label:
            c[ch.lower() if fold else ch] += n
    return c


digit_t = char_table(chars_of(real_counts, False), chars_of(synth_counts, False), spec.DIGITS)
nr = lambda counts: {l: n for l, n in counts.items() if spec.bucket_of(l) != "B3 Roman"}
letter_nr = char_table(chars_of(nr(real_counts), True), chars_of(nr(synth_counts), True), spec.LETTERS)
nna = [r[0] for r in rows if spec.bucket_of(r[0]) == "B1 NNa"]
pos = pd.DataFrame({"tens digit": pd.Series(collections.Counter(l[0] for l in nna)), "units digit": pd.Series(collections.Counter(l[1] for l in nna))}).reindex(list(spec.DIGITS)).fillna(0).astype(int)
pos_letters = pd.Series(collections.Counter(l[2] for l in nna)).reindex(list(spec.LETTERS)).fillna(0).astype(int)

dn = pd.DataFrame(notes)
flags = [f"- {changed} real labels (train+val+test) had marks stripped to match the convention (III. -> III, A. -> A, 2) -> 2 ...).",
         f"- NNa: **{int(lab[lab.bucket == 'B1 NNa'].real.sum())} real crops** ({', '.join(f'{l} x{c}' for l, c in lab[(lab.bucket == 'B1 NNa') & (lab.real > 0)].real.items())}): "
         "no real signal to judge 'balanced' against; balanced by character only; cannot be evaluated on real data."]
for b, title in (("B2 single digit", "digits"), ("B2 single letter", "single letters"), ("B3 Roman", "Roman numerals"), ("B3 NN alone", "two-digit numbers")):
    sub = dn[dn.bucket == b]
    sparse, capped = sub[sub.real < spec.SPARSE_REAL], sub[sub.capped]
    flags.append(f"- **{title}**: {len(sparse)} of {len(sub)} labels have < {spec.SPARSE_REAL} real crops ({int((sparse.real == 0).sum())} have none); "
                 f"{len(capped)} cannot reach their target because of the cap.")
    keep = capped if b != "B3 NN alone" else capped[capped.real > 0]
    if len(keep):
        flags.append("    - cap-limited: " + ", ".join(f"`{r.label}` ({r.real} real, total {r.real + r.granted}/{r.real + r.wanted})" for r in keep.itertuples())
                     + (f"; plus {int((capped.real == 0).sum())} two-digit numbers with 0 real crops, each held at {spec.CAP_FLOOR}" if b == "B3 NN alone" else ""))
flags.append(f"- {truncated} tail-bucket samples were dropped by the per-label cap.")

text = "\n".join([
    "# Master dataset distribution (plan)\n",
    f"Real part: `{args.split}` train ({len(real['train'])} crops; {len(q_rows)} have a Q prefix and are used {spec.Q_REAL_REPEAT}x) + {n_new} newly labelled "
    f"new-document crops (my reading, unverified; used {spec.NEWDOC_REPEAT}x) = {len(real_train_full)} real rows after repeats; val {len(real['val'])}, test {len(real['test'])} (real only). "
    f"Cap = max({spec.CAP_MULTIPLIER} x real, {spec.CAP_FLOOR}) per label (spec.py). Labels are characters only; marks and the Q prefix are image-only.\n",
    "## Per bucket (training set, by label)\n", bk.to_string(), "\n## Synthetic image styles (what is drawn around the label)\n", styles.to_string(),
    "\n## Digit frequency (characters in labels, all positions)\n", digit_t.to_string(),
    "\n## Letter frequency excluding the Roman bucket (case-folded)\n", letter_nr.to_string(),
    "\n## NNa synthetic bucket, per position\n", pos.to_string(), "\nletters:\n", pos_letters.to_string(), "\n## Flags\n", *flags,
    "\n## Largest synthetic/real ratios per label (labels with real crops)\n",
    lab[lab.real > 0].assign(ratio=lambda d: (d.synthetic / d.real).round(2)).sort_values("ratio", ascending=False).head(10)[["bucket", "real", "synthetic", "ratio"]].to_string()])
report = ROOT / "reports" / "master_distribution.md"
report.parent.mkdir(exist_ok=True)
report.write_text(text, "utf-8")
lab.sort_values(["bucket", "total"], ascending=[True, False]).to_csv(out / "plan.csv", index_label="label")
print(text)

# ---------------------------------------------------------------- render + manifests
if args.render:
    import cv2
    import render

    bank = render.load_bank()
    (out / "images").mkdir(exist_ok=True)
    recs = []
    for i, (label, drawn, style) in enumerate(rows):
        name = f"master_{i:06d}.png"
        cv2.imwrite(str(out / "images" / name), render.render_label(drawn, args.seed, i, bank))
        recs.append({"filename": name, "image_path": str(out / "images" / name), "text": label, "bucket": spec.bucket_of(label),
                     "source": "synthetic", "drawn": drawn, "style": style})
        if (i + 1) % 2000 == 0:
            print(f"  rendered {i + 1}/{len(rows)}", flush=True)
    syn = pd.DataFrame(recs)
    for s in ("train", "val", "test"):
        base = real_train_full if s == "train" else real[s].assign(source="real")
        r = base.assign(bucket=[spec.bucket_of(t) for t in base.text], drawn="", style="real")
        (pd.concat([r, syn], ignore_index=True) if s == "train" else r).to_csv(out / f"{s}.csv", index=False)
    print(f"wrote {out}/train.csv ({len(real_train_full) + len(syn)} rows), val.csv, test.csv")
