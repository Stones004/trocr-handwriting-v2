"""Builds the "open reading" training set: real number crops + real junk crops labelled `~` + synthetic random words.

Why: fine-tuning on question-number crops alone made PARSeq forget how to read general text (63% -> 37-40% exact match on random
words) and made it answer `1` for stray strokes. This set keeps the number data but also teaches (a) general text (random words in
handwriting fonts, rendered with the master-set renderer), (b) `~` = "nothing readable" for junk crops.

Label convention: the label is what is written in the image (case-sensitive, brackets and dots kept, whitespace removed:
`Add:`, `Ans(3)`, `2)`, `(i)`). Junk -> `~`. Earlier models trained on labels with brackets/dots/Q prefix stripped, which made the
label disagree with the image; here the real crops are relabelled verbatim where there is evidence (see relabel()).

Outputs data/open_read/{train,val,test}.csv (+ images/); turn them into LMDBs with
    venv_parseq/Scripts/python src/parseq/prepare_parseq_dataset.py --processed-dir data/open_read --output-dir data/open_read_lmdb
The unseen documents (Doc0729-Doc0775) are NOT used here; they are only for scoring.
"""
import sys
from pathlib import Path

import argparse
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "master"))
import render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--keep-unverified", action="store_true",
                help="v2: real TRAIN crops with no verbatim evidence are kept with their old (stripped) labels instead of dropped; written to data/open_read_v2")
ap.add_argument("--use-v1-draft", action="store_true",
                help="v3: also accept the readings of models/parseq_openread_v1 (relabel_v1_draft.csv) as verbatim labels; crops still without "
                     "evidence are DROPPED (their old labels often disagree with the image); written to data/open_read_v3")
ARGS = ap.parse_args()
DRAFT_DIR = ROOT / "data" / "open_read"                                     # relabel_draft.csv lives here
OUT = ROOT / "data" / ("open_read_v3" if ARGS.use_v1_draft else "open_read_v2" if ARGS.keep_unverified else "open_read")
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)
N_JUNK_TRAIN, N_WORDS_TRAIN, N_JUNK_VAL, N_WORDS_VAL = 800, 1500, 150, 200
NAN_LIKE = {"nan", "null", "none", "na", "n/a", "nil"}

# vocabulary for TRAINING words; deliberately different from the evaluation word list (scratchpad wordeval.py) so the word score
# measures reading of unseen words, not memorisation.
VOCAB = ("question marks write list name draw label mark under over above below before after between during within without about "
         "because therefore however although example numbers figure diagram section chapter paragraph page total amount balance account "
         "interest budget policy market economy trade labour capital wage savings investment consumption inflation income poverty "
         "inequality distribution measure indicator average median variance sample survey census region country state centre rural urban "
         "female male child adult education health housing water energy transport").split()
docsplit = {s: pd.read_csv(ROOT / "data" / "docsplit" / "seed42" / f"{s}.csv") for s in ("train", "val", "test")}
doc_of = lambda f: f[:7]  # noqa: E731
docs = {s: set(d.filename.map(doc_of)) for s, d in docsplit.items()}


def norm(s):
    """What the OLD labels kept: alphanumerics only, minus a leading Q / Ans marker."""
    return re.sub(r"^(?:Ans|ans|Q|q)", "", re.sub(r"[^A-Za-z0-9]", "", str(s)))


def relabel(df, keep_unverified=False):
    """Verbatim label for each real crop. Candidates, in order: the workbook's raw `text` column, then the original pretrained PARSeq's
    reading (cached in relabel_draft.csv). A candidate is accepted only if norm(candidate) equals the existing trained label, so it can
    only ADD the brackets/dots/prefix the old label dropped, never change the characters. Crops with no such candidate are dropped
    (a wrong label hurts more than a missing one)."""
    draft = pd.read_csv(DRAFT_DIR / "relabel_draft.csv", keep_default_na=False, dtype=str).drop_duplicates("filename").set_index("filename")
    v1 = pd.read_csv(DRAFT_DIR / "relabel_v1_draft.csv", keep_default_na=False, dtype=str).set_index("filename").v1 if ARGS.use_v1_draft else {}
    keep, labels = [], []
    for r in df.itertuples():
        raw = draft.at[r.filename, "raw"] if r.filename in draft.index else ""
        pre = draft.at[r.filename, "pretrained"] if r.filename in draft.index else ""
        raw = re.sub(r"_x000[0-9A-Fa-f]_", "", raw or "")
        pre = re.sub(r"\s+", "", pre)
        if not re.fullmatch(r"[A-Za-z0-9()\[\].:]+", pre):   # the model's own marks are only trusted when they are brackets / dots / colons
            pre = ""
        p1 = re.sub(r"\s+", "", v1.get(r.filename, "")) if ARGS.use_v1_draft else ""
        if not re.fullmatch(r"[A-Za-z0-9()\[\].:]+", p1):
            p1 = ""
        cands = [re.sub(r"\s+", "", raw), pre, p1]
        ok = next((c for c in cands if c and c.isascii() and len(c) <= 25 and norm(c) == str(r.text)), None)
        if ok is None and keep_unverified:
            ok = str(r.text)                                   # no evidence for a verbatim label: keep the old stripped one
        keep.append(ok is not None); labels.append(ok)
    out = df.assign(text=labels)[keep]
    print(f"relabel: kept {keep.count(True)} of {len(df)}; label changed (brackets/dots/prefix added) in {(out.text != df[keep].text).sum()}")
    return out


def rand_word(rng):
    kind = rng.choice(["common", "Capitalized", "random", "mixed", "marker"], p=[0.35, 0.2, 0.2, 0.15, 0.10])
    if kind == "common":
        w = str(rng.choice(VOCAB)); return w, w, kind
    if kind == "Capitalized":
        w = str(rng.choice(VOCAB)).capitalize(); return w, w, kind
    if kind == "random":
        w = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), int(rng.integers(3, 9)))); return w, w, kind
    if kind == "mixed":
        w = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), int(rng.integers(3, 8)))); return w, w, kind
    m = rng.integers(0, 3)  # section markers seen in the unseen documents' format: Add: / Less: / Ans (n)
    if m == 0: d = str(rng.choice(["Add:", "Add :", "Add"]))
    elif m == 1: d = str(rng.choice(["Less:", "Less :", "Less"]))
    else: d = f"Ans ({int(rng.integers(1, 10))})"
    return d, d.replace(" ", ""), kind


def make_words(n, seed, tag):
    bank, rows, rng = render.load_bank(), [], np.random.default_rng(seed)
    i = 0
    while len(rows) < n:
        draw, label, kind = rand_word(rng)
        if label.lower() in NAN_LIKE:
            continue
        p = IMG / f"{tag}_{len(rows):05d}.png"
        cv2.imwrite(str(p), render.render_label(draw, seed, i, bank)); i += 1
        rows.append((p.name, str(p), label, f"word:{kind}"))
    return rows


def unreadable(paths):
    """The workbook's `noise` crops are not all empty: some hold readable marks (a `6.`, a `9.3`) that were noise for another reason.
    Labelling those `~` would teach the model to refuse real digits, so keep only crops the ORIGINAL pretrained PARSeq reads as no
    letter or digit at all (strokes, lines, arcs -> punctuation or nothing)."""
    sys.path.insert(0, str(ROOT / "src" / "parseq"))
    from model_io import decode, get_device, load_model, make_transform
    dev = get_device(); m = load_model("pretrained=parseq", dev); tf = make_transform(m)
    keep = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            keep.append(False); continue
        keep.append(re.search(r"[A-Za-z0-9]", decode(m, tf(im).unsqueeze(0).to(dev))[0][0]) is None)
    return np.array(keep)


rng = np.random.default_rng(42)
noise = pd.concat([pd.read_csv(ROOT / "data" / "noise_classifier" / f"{s}.csv") for s in ("train", "val", "test")])
noise = noise[noise.label == 1]
n0 = len(noise)
noise = noise[unreadable(noise.image_path)]
print(f"junk pool: {n0} workbook-noise crops -> {len(noise)} kept (pretrained PARSeq reads no letter/digit in them)")
val_docs, test_docs = docs["val"], docs["test"]
junk_pool = noise[~noise.doc.isin(val_docs | test_docs)]          # never from a document in the real val/test split
junk_val_pool = noise[noise.doc.isin(val_docs)]
junk_tr = junk_pool.sample(N_JUNK_TRAIN, random_state=1)
junk_va = junk_val_pool.sample(min(N_JUNK_VAL, len(junk_val_pool)), random_state=1)
asrows = lambda df, kind: [(r.filename, r.image_path, "~", kind) for r in df.itertuples()]  # noqa: E731
real = lambda s, kind, verbatim=True, keep=False: [(r.filename, r.image_path, str(r.text), kind) for r in (relabel(docsplit[s], keep) if verbatim else docsplit[s]).itertuples()]  # noqa: E731

train = real("train", "real", keep=ARGS.keep_unverified) + asrows(junk_tr, "junk") + make_words(N_WORDS_TRAIN, 111, "wtr")
val = real("val", "real") + asrows(junk_va, "junk") + make_words(N_WORDS_VAL, 222, "wva")
test = real("test", "real", verbatim=False)          # old (stripped) labels: comparable with every earlier model
test_verbatim = real("test", "real")                  # same crops with verbatim labels (only those with evidence)
for name, rows in (("train", train), ("val", val), ("test", test), ("test_verbatim", test_verbatim)):
    pd.DataFrame(rows, columns=["filename", "image_path", "text", "kind"]).to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(rows), pd.Series([r[3].split(":")[0] for r in rows]).value_counts().to_dict())
