"""Read ONLY the label CSVs of SSL-HWD-Words (Labeled) without downloading the 14.5 GB zip, and count the labels per project format.

    venv_parseq/Scripts/python src/exp_oct/ssl_hwd_scan.py

How: the zip's central directory (file list) and the small per-writer CSV files are fetched with HTTP Range requests from
https://huggingface.co/datasets/Shree10/SSL-HWD-Words (public, Apache-2.0 per the dataset card). No image is downloaded.
Output: outputs/exp_oct/ssl_hwd/labels.csv (writer, image_filename, transcription, zip_path) and a printed summary
(per format: label count, writers, label length stats, top labels). Read-only on every existing project file.
"""
import csv
import io
import re
import sys
import struct
import time
import zlib
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "exp_oct" / "ssl_hwd"
URL = "https://huggingface.co/datasets/Shree10/SSL-HWD-Words/resolve/main/SSL-HWD-Words(Labeled).zip"
BLOCK = 4 * 1024 * 1024


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a one-block read-ahead cache."""

    def __init__(self, url):
        self.url, self.pos, self.s = url, 0, requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.final = r.url
        r.close()
        self.cache_start, self.cache = -1, b""
        self.requests = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, start, end):
        for attempt in range(5):
            try:
                r = self.s.get(self.final, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (403, 404):  # signed link expired: resolve again
                    self.final = self.s.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60).url
                    continue
                r.raise_for_status()
                self.requests += 1
                return r.content
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"range {start}-{end} failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > BLOCK:  # a big read (the central directory): one request, no cache
            data = self._fetch(self.pos, self.pos + n - 1)
        else:
            if not (self.cache_start <= self.pos and self.pos + n <= self.cache_start + len(self.cache)):
                end = min(self.pos + BLOCK, self.size) - 1
                self.cache, self.cache_start = self._fetch(self.pos, end), self.pos
            o = self.pos - self.cache_start
            data = self.cache[o:o + n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def fmt(t):
    if re.fullmatch(r"\d", t): return "1 digit"
    if re.fullmatch(r"\d\d", t): return "2 digits"
    if re.fullmatch(r"\d\d[A-Za-z]", t): return "NNa (2 digits + letter)"
    if re.fullmatch(r"\d[A-Za-z]", t): return "Na (digit + letter)"
    if re.fullmatch(r"[A-Za-z]", t): return "1 letter"
    if re.fullmatch(r"[ivxIVX]{2}", t): return "2 letters (Roman)"
    if re.fullmatch(r"\d+", t): return "3+ digits"
    return "other"


def csv_entries(f, want=None):
    """Stream the zip central directory in chunks and keep only the .csv entries (or the names in `want`) -> [(name, method, csize, local_header_offset)]. Handles zip64."""
    tail = f._fetch(f.size - 70000, f.size - 1)
    i = tail.rfind(b"PK")
    n_total, cd_size, cd_off = struct.unpack("<10xHII", tail[i:i + 20])[0], *struct.unpack("<II", tail[i + 12:i + 20])
    loc = tail.rfind(b"PK", 0, i)
    if loc >= 0:  # zip64: end-of-central-directory record holds the real 64-bit values
        eocd64 = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        rec = f._fetch(eocd64, eocd64 + 55)
        n_total, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
    print(f"central directory: {n_total:,} entries, {cd_size / 1e6:.0f} MB at offset {cd_off}", flush=True)
    out, buf, pos, got = [], b"", cd_off, 0
    end = cd_off + cd_size
    while pos < end or buf:
        if pos < end and len(buf) < 4_000_000:
            chunk = f._fetch(pos, min(pos + 32 * 1024 * 1024, end) - 1)
            pos += len(chunk)
            buf += chunk
        o = 0
        while o + 46 <= len(buf):
            if buf[o:o + 4] != b"PK":
                raise RuntimeError("bad central directory signature")
            method, = struct.unpack("<H", buf[o + 10:o + 12])
            csize, usize = struct.unpack("<II", buf[o + 20:o + 28])
            fn, ex, cm = struct.unpack("<HHH", buf[o + 28:o + 34])
            loff, = struct.unpack("<I", buf[o + 42:o + 46])
            rec_len = 46 + fn + ex + cm
            if o + rec_len > len(buf):
                break
            name = buf[o + 46:o + 46 + fn].decode("utf-8", errors="replace")
            if (name in want) if want is not None else name.lower().endswith(".csv"):
                extra = buf[o + 46 + fn:o + 46 + fn + ex]
                k = 0
                while k + 4 <= len(extra):
                    tag, sz = struct.unpack("<HH", extra[k:k + 4])
                    if tag == 1:
                        v = extra[k + 4:k + 4 + sz]
                        q = 0
                        if usize == 0xFFFFFFFF: q += 8
                        if csize == 0xFFFFFFFF: csize, = struct.unpack("<Q", v[q:q + 8]); q += 8
                        if loff == 0xFFFFFFFF: loff, = struct.unpack("<Q", v[q:q + 8])
                    k += 4 + sz
                out.append((name, method, csize, loff))
            got += 1
            o += rec_len
        buf = buf[o:]
        if pos >= end and (not buf or len(buf) < 46):
            break
    print(f"scanned {got:,} entries", flush=True)
    return out


def read_entry(f, method, csize, loff):
    h = f._fetch(loff, loff + 29)
    fn, ex = struct.unpack("<HH", h[26:30])
    data = f._fetch(loff + 30 + fn + ex, loff + 30 + fn + ex + csize - 1)
    return data if method == 0 else zlib.decompress(data, -15)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = HttpFile(URL)
    print(f"zip size {f.size / 1e9:.2f} GB; reading the file list (central directory) with range requests...", flush=True)
    t0 = time.time()
    ents = csv_entries(f)
    print(f"{len(ents)} csv files (writers) found in {time.time() - t0:.0f}s, {f.requests} requests, e.g. {[e[0] for e in ents[:3]]}", flush=True)
    rows = []
    for i, (n, method, csize, loff) in enumerate(ents, 1):
        txt = read_entry(f, method, csize, loff).decode("utf-8", errors="replace")
        writer = Path(n).parent.name
        for r in csv.DictReader(io.StringIO(txt)):
            rows.append((writer, r.get("image_filename", ""), r.get("transcription", ""), str(Path(n).parent).replace("\\", "/") + "/" + r.get("image_filename", "")))
        if i % 100 == 0 or i == len(ents):
            print(f"  {i}/{len(ents)} csvs, {len(rows):,} labels, {f.requests} requests", flush=True)
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["writer", "image_filename", "transcription", "zip_path"])
        w.writerows(rows)
    cnt, writers, tops = Counter(), {}, {}
    for wr, _, t, _ in rows:
        k = fmt(t.strip())
        cnt[k] += 1
        writers.setdefault(k, set()).add(wr)
        tops.setdefault(k, Counter())[t.strip()] += 1
    print(f"\nTOTAL labels {len(rows):,}, writers {len({r[0] for r in rows})}")
    for k, c in cnt.most_common():
        if k == "other":
            print(f"{k:28s} {c:>9,}")
        else:
            print(f"{k:28s} {c:>9,}  writers {len(writers[k]):>4}  top: {tops[k].most_common(8)}")


if __name__ == "__main__":
    main()
