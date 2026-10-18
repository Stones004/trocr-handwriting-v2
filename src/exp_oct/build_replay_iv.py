"""In-vocabulary replay: restrict the master dataset to the closed vocabulary (digits 0-9, letters a-j) and mix extra in-vocabulary strings in at 30%.

    venv_parseq/Scripts/python src/exp_oct/build_replay_iv.py

Vocabulary rule (labels are characters only, marks are drawn but not in the label): a label is in vocabulary when it matches  \\d{1,2}[a-j]?  or  [a-j]
(NNa with the number and letter each optional, at most two digits). Roman numerals (ii, iii, vi...), capitals, `Q` prefixes and `~` junk are out.
Applied to train, val and test of data/master_dataset.mdb.

Outputs (PARSeq LMDB layout) in outputs/exp_oct/data/:
  iv_base/       in-vocabulary master data, no replay (control)
  iv_replay30/   same, with 30% of the rows replaced: N extra strings replace N randomly chosen SYNTHETIC rows (real rows are never removed), so the total
                 and the number of steps equal the control. Same mechanism as build_replay.py.
Replay strings: random a-j strings (3-8 characters, 60%) and random 0-9a-j strings (3-6 characters, 40%), rendered with the same renderer as the
master synthetic crops. Words outside a-j cannot be used (out of vocabulary). Strings equal to any master label are rejected (leakage rule).
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
IMG = OUT / "replay_iv_strings"
PATTERN = r"\d{1,2}[a-j]?|[a-j]"
SEED = 888
PCT = 30
LET, ALN = "abcdefghij", "0123456789abcdefghij"


def in_vocab(label):
    import re
    return re.fullmatch(PATTERN, label) is not None


def write_lmdb(path, items):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    env = lmdb.open(str(path), map_size=512 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for k, (label, png) in enumerate(items, 1):
            txn.put(f"image-{k:09d}".encode(), png)
            txn.put(f"label-{k:09d}".encode(), label.encode())
        txn.put(b"num-samples", str(len(items)).encode())
    env.close()


def main():
    rows = [(s, l, p, m) for s, l, p, m in dataset_io.samples() if s in ("train", "val", "test")]
    keep = [(s, l, p, m) for s, l, p, m in rows if in_vocab(l)]
    by = {s: [(l, p, m) for ss, l, p, m in keep if ss == s] for s in ("train", "val", "test")}
    counts = {s: (sum(1 for r in rows if r[0] == s), len(by[s])) for s in by}
    log(f"in-vocab filter {PATTERN}: kept/total {counts}")
    train = by["train"]
    synth = [k for k, (_, _, m) in enumerate(train) if m["source"] == "synthetic"]
    n = round(PCT / 100 * len(train))
    assert n <= len(synth)

    banned = {l.lower() for _, l, _, _ in rows}
    bank, rng, words, i = render.load_bank(), np.random.default_rng(SEED), [], 0
    IMG.mkdir(parents=True, exist_ok=True)
    while len(words) < n:
        if rng.random() < 0.6:
            label = "".join(rng.choice(list(LET), int(rng.integers(3, 9))))
        else:
            label = "".join(rng.choice(list(ALN), int(rng.integers(3, 7))))
        if label in banned or not in_vocab_chars(label):
            continue
        png = cv2.imencode(".png", render.render_label(label, SEED, i, bank))[1].tobytes()
        i += 1
        (IMG / f"iv_{len(words):05d}.png").write_bytes(png)
        words.append((label, png))
    pd.DataFrame({"text": [w[0] for w in words]}).to_csv(IMG / "strings.csv", index=False)
    assert not ({w[0] for w in words} & banned)

    drop = set(np.random.default_rng(0).permutation(synth)[:n].tolist())
    replay = [(l, p) for k, (l, p, _) in enumerate(train) if k not in drop] + words
    base = [(l, p) for l, p, _ in train]
    assert len(replay) == len(base)
    for name, items in (("iv_base", base), ("iv_replay30", replay)):
        d = OUT / "data" / name
        write_lmdb(d / "train" / "handwriting", items)
        for s in ("val", "test"):
            write_lmdb(d / s / "handwriting", [(l, p) for l, p, _ in by[s]])
        real = sum(1 for _, _, m in train if m["source"] != "synthetic")
        info = {"vocab_pattern": PATTERN, "rows_total": len(items), "real_rows": real, "string_rows": n if name == "iv_replay30" else 0,
                "synthetic_rows": len(items) - real - (n if name == "iv_replay30" else 0), "steps_per_epoch": -(-len(items) // 32),
                "val": len(by["val"]), "test": len(by["test"]), "dropped_out_of_vocab": {s: counts[s][0] - counts[s][1] for s in counts}}
        (d / "info.json").write_text(json.dumps(info, indent=1))
        log(f"{name}: {info}")
        print(name, info)


def in_vocab_chars(label):
    return all(c in ALN for c in label)


if __name__ == "__main__":
    main()
