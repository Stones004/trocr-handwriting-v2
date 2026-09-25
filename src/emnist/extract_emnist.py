"""Extracts the glyphs this project needs from the EMNIST `byclass` split (default: digits 0-9 and lowercase a-j; use
--vocab for other characters, e.g. the uppercase/roman characters that occur as single-character labels in our data).

`byclass` is the only EMNIST split that keeps upper/lower case separate (`balanced`/`letters` merge them), and we need
lowercase a-j only. Class ids in byclass: 0-9 = '0'-'9', 10-35 = 'A'-'Z', 36-61 = 'a'-'z'.

EMNIST images are stored transposed relative to normal image orientation (a quirk of the original MATLAB export), so
they are transposed back here. Each 28x28 image is white ink on black with the glyph tight-cropped to its bounding box
and centered in a square, so aspect ratio is preserved but absolute height is not (an 'a' and a 'b' both fill ~20px);
the sequence synthesizer compensates for that per letter.

Input : data/emnist/gzip.zip  (https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip, 562MB, downloaded separately)
Output: data/emnist/glyphs.npz with train_images/train_labels/test_images/test_labels (uint8; labels are the characters)

Run with venv_parseq (needs only numpy):
    venv_parseq/Scripts/python src/emnist/extract_emnist.py
"""
import argparse
import gzip
import io
import zipfile
from pathlib import Path

import numpy as np

# byclass class id -> character
BYCLASS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="data/emnist/gzip.zip")
ap.add_argument("--out", default="data/emnist/glyphs.npz")
ap.add_argument("--vocab", default="0123456789abcdefghij", help="characters to keep (any of the 62 byclass characters)")
args = ap.parse_args()
VOCAB = args.vocab


def read_idx(raw):
    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


out = {}
with zipfile.ZipFile(args.zip) as z:
    for split in ("train", "test"):
        def load(kind):
            with z.open(f"gzip/emnist-byclass-{split}-{kind}-idx{'3' if kind == 'images' else '1'}-ubyte.gz") as f:
                return read_idx(gzip.GzipFile(fileobj=io.BytesIO(f.read())).read())
        images, labels = load("images"), load("labels")
        chars = np.array(list(BYCLASS))[labels]
        keep = np.isin(chars, list(VOCAB))
        images = images[keep].transpose(0, 2, 1)  # undo EMNIST's transposed storage
        chars = chars[keep]
        out[f"{split}_images"] = np.ascontiguousarray(images)
        out[f"{split}_labels"] = chars
        counts = {c: int((chars == c).sum()) for c in VOCAB}
        print(f"[{split}] kept {keep.sum()} of {len(labels)} glyphs; per class: {counts}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, **out)
print("wrote", args.out)
