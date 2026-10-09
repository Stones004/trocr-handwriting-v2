"""Pilot: sample ~3,000 real handwritten crops of the project's formats from SSL-HWD-Words (Labeled), download only those images, build a training set.

    venv_parseq/Scripts/python src/exp_oct/ssl_hwd_sample.py            # sample + download + contact sheet + LMDB

Input: outputs/exp_oct/ssl_hwd/labels.csv (from ssl_hwd_scan.py). Only labels of the form  [ ( | [ ]  CORE  [ ) | ] | . ]  are used, where CORE is 1 digit, 2 digits,
1 letter, a Roman pair (ii iv vi ix, either case) or digit+letter. Fractions, decimals, percentages and longer words are excluded. The plain words 'a', 'A', 'I' are
excluded (English words, not question labels); their marked forms ((a) a) ...) are kept. Training label = CORE (marks stripped), as in the project labels.
Caps (diversity): per-label caps below, at most MAX_PER_WRITER crops per writer, about MARKED_SHARE of each class marked when available. Seed 20261109.
Images are fetched with HTTP Range requests (one per image) in 8 threads; nothing else of the 14.5 GB zip is downloaded.
Output: outputs/exp_oct/ssl_hwd/crops/*.png, sample.csv, contact_sheet.png and the LMDB outputs/exp_oct/data/replay30_ssl3k/{train,val,test}/handwriting
(replay-30% training rows + the new rows; val/test copied unchanged).
"""
import csv
import hashlib
import random
import re
import shutil
import struct
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "exp_oct"))
from progress import log  # noqa: E402
from ssl_hwd_scan import URL, HttpFile, csv_entries  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct"
SD = OUT / "ssl_hwd"
MAX_PER_WRITER, MARKED_SHARE, SEED = 15, 0.25, 20261109
# (class, regex of CORE, cap per distinct core label)
CLASSES = [("1 digit", r"\d", 90), ("2 digits", r"\d\d", 10), ("1 letter", r"[A-Za-z]", 20), ("2 letters (Roman)", r"[ivxIVX]{2}", 25), ("Na", r"\d[A-Za-z]", 4)]
PATTERN = re.compile(r"^[(\[]?([0-9A-Za-z]+)[)\].]?$")


def classify(t):
    m = PATTERN.match(t.strip())
    if not m:
        return None
    core = m.group(1)
    for name, rx, cap in CLASSES:
        if re.fullmatch(rx, core):
            if t.strip() == core and core in ("a", "A", "I"):
                return None
            return name, core, t.strip() != core, cap
    return None


def sample(df):
    rng = random.Random(SEED)
    info = df.transcription.map(classify)
    df = df[info.notna()].copy()
    df["cls"], df["core"], df["marked"], df["cap"] = zip(*info[info.notna()])
    picked, per_writer = [], {}
    for (cls, core), g in df.groupby(["cls", "core"]):
        cap = int(g.cap.iloc[0])
        rows = g.sample(frac=1, random_state=rng.randrange(1 << 30)).to_dict("records")
        n_mark = max(1, round(cap * MARKED_SHARE))
        mk = [r for r in rows if r["marked"]]
        pl = [r for r in rows if not r["marked"]]
        order = mk[:n_mark] + pl + mk[n_mark:]  # marked first up to its share, then plain, then the rest
        k = 0
        for r in order:
            if k >= cap:
                break
            if per_writer.get(r["writer"], 0) >= MAX_PER_WRITER:
                continue
            per_writer[r["writer"]] = per_writer.get(r["writer"], 0) + 1
            picked.append(r)
            k += 1
    return pd.DataFrame(picked)


_tl = threading.local()


def fetch_image(args):
    final, name, method, csize, loff = args
    s = getattr(_tl, "s", None) or requests.Session()
    _tl.s = s
    for attempt in range(5):
        try:
            r = s.get(final, headers={"Range": f"bytes={loff}-{loff + 30 + 400 + csize}"}, timeout=120)
            r.raise_for_status()
            b = r.content
            fn, ex = struct.unpack("<HH", b[26:30])
            data = b[30 + fn + ex:30 + fn + ex + csize]
            return name, (data if method == 0 else zlib.decompress(data, -15))
        except Exception:  # noqa: BLE001
            if attempt == 4:
                return name, None
    return name, None


def main():
    labels = pd.read_csv(SD / "labels.csv", keep_default_na=False, dtype=str)
    s = sample(labels)
    print("sampled", len(s), "crops;", s.groupby("cls").size().to_dict(), "| marked", int(s.marked.sum()), "| writers", s.writer.nunique(), flush=True)
    f = HttpFile(URL)
    ents = {n: (m, c, o) for n, m, c, o in csv_entries(f, want=set(s.zip_path))}
    print("found", len(ents), "of", len(s), "entries in the zip", flush=True)
    jobs = [(f.final, n, *ents[n]) for n in s.zip_path if n in ents]
    (SD / "crops").mkdir(exist_ok=True)
    got = {}
    with ThreadPoolExecutor(8) as ex:
        for i, (n, data) in enumerate(ex.map(fetch_image, jobs), 1):
            got[n] = data
            if i % 500 == 0 or i == len(jobs):
                print(f"  downloaded {i}/{len(jobs)}", flush=True)
    keep, seen = [], set()
    for r in s.to_dict("records"):
        data = got.get(r["zip_path"])
        if not data:
            continue
        h = hashlib.md5(data).hexdigest()
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None or h in seen or im.shape[0] < 12 or im.shape[1] < 8 or im.shape[1] / im.shape[0] > 8:
            continue
        seen.add(h)
        fn = f"ssl_{len(keep):05d}.png"
        cv2.imwrite(str(SD / "crops" / fn), im)
        r.update({"filename": fn, "image_path": str(SD / "crops" / fn), "text": r["core"], "h": im.shape[0], "w": im.shape[1]})
        keep.append(r)
    k = pd.DataFrame(keep)
    k.to_csv(SD / "sample.csv", index=False)
    print(f"kept {len(k)} crops after dedupe/size filter; by class {k.groupby('cls').size().to_dict()}; marked {int(k.marked.sum())}; median size {int(k.h.median())}x{int(k.w.median())}", flush=True)
    # contact sheet: 6 random crops per class
    tiles = []
    for cls, g in k.groupby("cls"):
        for r in g.sample(min(8, len(g)), random_state=1).itertuples():
            im = cv2.resize(cv2.imread(r.image_path), (256, 64))
            cv2.putText(im, f"{r.transcription} -> {r.text}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            tiles.append(im)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles) - 3, 4)]
    cv2.imwrite(str(SD / "contact_sheet.png"), np.vstack(rows))
    # training set = replay30 + new rows
    src, dst = OUT / "data" / "replay_p30", OUT / "data" / "replay30_ssl3k"
    for split in ("val", "test"):
        if not (dst / split).exists():
            shutil.copytree(src / split, dst / split)
    t = dst / "train" / "handwriting"
    if t.exists():
        shutil.rmtree(t)
    t.mkdir(parents=True)
    senv = lmdb.open(str(src / "train" / "handwriting"), readonly=True, lock=False)
    env = lmdb.open(str(t), map_size=1024 * 1024 * 1024)
    with senv.begin() as stx, env.begin(write=True) as tx:
        n0 = int(stx.get(b"num-samples"))
        for i in range(1, n0 + 1):
            tx.put(f"image-{i:09d}".encode(), stx.get(f"image-{i:09d}".encode()))
            tx.put(f"label-{i:09d}".encode(), stx.get(f"label-{i:09d}".encode()))
        for j, r in enumerate(k.itertuples(), n0 + 1):
            tx.put(f"image-{j:09d}".encode(), Path(r.image_path).read_bytes())
            tx.put(f"label-{j:09d}".encode(), r.text.encode())
        tx.put(b"num-samples", str(n0 + len(k)).encode())
    env.close()
    total = n0 + len(k)
    steps = -(-total // 32)
    log(f"replay30_ssl3k: {n0} replay-30 rows + {len(k)} real SSL-HWD crops = {total} rows, {steps} steps/epoch, {steps * 15} steps per 15-epoch run")
    print(f"LMDB built: {total} rows, {steps} steps/epoch")


if __name__ == "__main__":
    main()
