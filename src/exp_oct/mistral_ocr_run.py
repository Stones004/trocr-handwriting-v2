"""Send the 226 reviewed ROI crops to Mistral's dedicated OCR model (mistral-ocr-latest, the model the dataset builder
in D:/OCR_codes uses) and cache the raw markdown answers.

    venv_parseq/Scripts/python src/exp_oct/mistral_ocr_run.py [--limit N]

* Same crops/truth as mistral_compare.py. Key: MISTRAL_API_KEY in <repo>/.env (only ever sent to api.mistral.ai).
* The image is sent inline as a base64 data URL to POST /v1/ocr (nothing is uploaded to or stored in the Mistral account;
  D:/OCR_codes' client uploads a file and uses a signed URL, which gives the same OCR call).
* The OCR endpoint takes no prompt, so the answer is whatever it transcribes (marks, LaTeX etc. included); scoring strips
  non-alphanumerics exactly as for Chandra (see compare_board.py).
* Resumable: answers are cached in outputs/exp_oct/roi_check/mistral_ocr_cache.json.
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "exp_oct" / "roi_check" / "mistral_ocr_cache.json"
CROPS = Path("D:/OCR_codes/data/roi_watch/input")
QUEUE = Path("D:/OCR_codes/data/roi_watch/review_queue.csv")
URL = "https://api.mistral.ai/v1/ocr"
MODEL = "mistral-ocr-latest"


def ocr(key, path, retries=6):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": MODEL, "document": {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}}
    for k in range(retries):
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
        if r.status_code == 429:
            time.sleep(5 * (k + 1))
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Mistral rejected the key (HTTP {r.status_code}).")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return "\n".join(p["markdown"] for p in r.json()["pages"]).strip()
    raise RuntimeError(f"rate limited too long for {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wait", type=float, default=1.2, help="seconds between requests")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_KEY_HERE":
        sys.exit("No key: put MISTRAL_API_KEY in .env")
    q = pd.read_csv(QUEUE, keep_default_na=False, dtype=str)
    if a.limit:
        q = q.head(a.limit)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [f for f in q.filename if f not in cache]
    print(f"{len(q)} crops, {len(todo)} still to send to {MODEL}")
    for i, f in enumerate(todo, 1):
        cache[f] = ocr(key, CROPS / f)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        if i <= 3 or i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {f!r} -> {cache[f]!r}", flush=True)
        time.sleep(a.wait)
    print("done:", len(cache), "answers in", CACHE)


if __name__ == "__main__":
    main()
