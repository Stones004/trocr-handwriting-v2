"""Phase 2: build the replay training sets (word crops mixed into the master training set at 10 / 20 / 30%).

    venv_parseq/Scripts/python src/exp_oct/build_replay.py

Words come from the open_read synthetic word generator (src/open_read/build_open_read.py: same VOCAB, same four word kinds, same renderer
src/master/render.py). Labels are the literal text (letters and case kept, no spaces). Section markers (Add:/Less:/Ans (n)) are NOT used:
`Less` would collide (case-insensitively) with the word-test word `less`.

LEAKAGE CHECK (asserted, overlap printed): no replay label equals (case-insensitively) any word in data/open_read/word_eval.csv, and none equals any label
of the master dataset (what master_v3 trained on: numbers, letters, Roman numerals), so a random 3-letter string like `vii` is rejected.

Size cap: the total training rows stay at the baseline's 13,996 (438 steps/epoch). N word rows replace N randomly chosen SYNTHETIC rows (real rows are
never removed). The 10% set is a subset of the 20% set, which is a subset of the 30% set (nested), and the removed rows are nested the same way.
Output: outputs/exp_oct/data/replay_p{10,20,30}/{train,val,test}/handwriting (PARSeq LMDB layout) and outputs/exp_oct/replay_words/.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "master"))
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
import dataset_io  # noqa: E402
import render  # noqa: E402
from progress import log  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct"
IMG = OUT / "replay_words"
SEED = 777                       # open_read used 111/222, so these crops differ from every earlier word set
TRAIN_TOTAL = 13996              # baseline train rows
RATIOS = (10, 20, 30)
# copied from src/open_read/build_open_read.py (that script cannot be imported: it parses arguments and writes files at import time)
VOCAB = ("question marks write list name draw label mark under over above below before after between during within without about "
         "because therefore however although example numbers figure diagram section chapter paragraph page total amount balance account "
         "interest budget policy market economy trade labour capital wage savings investment consumption inflation income poverty "
         "inequality distribution measure indicator average median variance sample survey census region country state centre rural urban "
         "female male child adult education health housing water energy transport").split()
KINDS, P = ["common", "Capitalized", "random", "mixed"], np.array([0.35, 0.2, 0.2, 0.15]) / 0.9
LOW = "abcdefghijklmnopqrstuvwxyz"


def rand_word(rng):
    kind = str(rng.choice(KINDS, p=P))
    if kind == "common":
        return str(rng.choice(VOCAB)), kind
    if kind == "Capitalized":
        return str(rng.choice(VOCAB)).capitalize(), kind
    if kind == "random":
        return "".join(rng.choice(list(LOW), int(rng.integers(3, 9)))), kind
    return "".join(rng.choice(list(LOW + LOW.upper()), int(rng.integers(3, 8)))), kind


def main():
    test_words = pd.read_csv(ROOT / "data" / "open_read" / "word_eval.csv", keep_default_na=False, dtype=str)["text"]
    master = pd.read_csv(ROOT / "data" / "master_dataset.csv", keep_default_na=False, dtype=str)
    banned = set(test_words.str.lower()) | set(master["text"].str.lower())
    log(f"leakage lists: {len(set(test_words.str.lower()))} distinct word-test words, {master['text'].str.lower().nunique()} distinct master labels")
    global VOCAB
    removed = sorted(w for w in VOCAB if w.lower() in banned)
    print(f"vocabulary words removed because they are in the word test or master labels: {removed}")
    VOCAB = [w for w in VOCAB if w.lower() not in banned]
    v_over = {w for w in VOCAB if w.lower() in banned} | {w.capitalize() for w in VOCAB if w.capitalize().lower() in banned}
    print(f"vocabulary overlap after removal: {len(v_over)} (must be 0); {len(VOCAB)} words kept")
    assert not v_over, f"VOCAB overlaps the banned lists: {sorted(v_over)}"
    log(f"VOCAB: removed {removed} (found in the word test); {len(VOCAB)} words left, overlap 0")

    n_max = round(max(RATIOS) / 100 * TRAIN_TOTAL)
    bank, rng, rows, rejected, i = render.load_bank(), np.random.default_rng(SEED), [], 0, 0
    IMG.mkdir(parents=True, exist_ok=True)
    while len(rows) < n_max:
        label, kind = rand_word(rng)
        if label.lower() in banned:
            rejected += 1
            continue
        png = cv2.imencode(".png", render.render_label(label, SEED, i, bank))[1].tobytes()
        i += 1
        (IMG / f"rw_{len(rows):05d}.png").write_bytes(png)
        rows.append({"filename": f"rw_{len(rows):05d}.png", "text": label, "kind": kind})
    words = pd.DataFrame(rows)
    words.to_csv(IMG / "replay_words.csv", index=False)
    # independent re-check on the final list
    gen = set(words["text"].str.lower())
    ov_test, ov_master = gen & set(test_words.str.lower()), gen & set(master["text"].str.lower())
    print(f"generated {len(words)} words, {len(gen)} distinct; rejected during generation: {rejected}")
    print(f"OVERLAP replay vs word test: {len(ov_test)}   replay vs master_v3 labels: {len(ov_master)}   (must be 0)")
    assert not ov_test and not ov_master
    assert words["text"].map(lambda t: t.isascii() and " " not in t and 0 < len(t) <= 25).all()
    log(f"replay words: {len(words)} generated ({len(gen)} distinct), overlap with word test 0, with master labels 0; kinds {words['kind'].value_counts().to_dict()}")

    # master training rows
    train = [(label, png, meta) for split, label, png, meta in dataset_io.samples() if split == "train"]
    assert len(train) == TRAIN_TOTAL, len(train)
    synth = [k for k, (_, _, m) in enumerate(train) if m["source"] == "synthetic"]
    order = np.random.default_rng(0).permutation(synth)          # first n of this order are dropped (nested across ratios)
    for pct in RATIOS:
        n = round(pct / 100 * TRAIN_TOTAL)
        drop = set(order[:n].tolist())
        items = [(l, p) for k, (l, p, _) in enumerate(train) if k not in drop]
        items += [(r.text, (IMG / r.filename).read_bytes()) for r in words.iloc[:n].itertuples()]
        assert len(items) == TRAIN_TOTAL
        d = OUT / "data" / f"replay_p{pct}"
        for split in ("val", "test"):
            if not (d / split).exists():
                shutil.copytree(OUT / "data" / "master_lmdb" / split, d / split)
        t = d / "train" / "handwriting"
        if t.exists():
            shutil.rmtree(t)
        t.mkdir(parents=True)
        env = lmdb.open(str(t), map_size=512 * 1024 * 1024)
        with env.begin(write=True) as txn:
            for k, (label, png) in enumerate(items, 1):
                txn.put(f"image-{k:09d}".encode(), png)
                txn.put(f"label-{k:09d}".encode(), label.encode())
            txn.put(b"num-samples", str(len(items)).encode())
        env.close()
        real = sum(1 for k, (_, _, m) in enumerate(train) if m["source"] != "synthetic")
        info = {"ratio_pct": pct, "word_rows": n, "synthetic_rows_kept": len(synth) - n, "real_rows": real, "total_rows": len(items),
                "steps_per_epoch": -(-len(items) // 32), "total_steps_15_epochs": -(-len(items) // 32) * 15}
        (d / "info.json").write_text(json.dumps(info, indent=1))
        log(f"replay_p{pct}: {info}")


if __name__ == "__main__":
    main()
