"""Send the user's 226 reviewed ROI crops to Mistral (vision chat API, free tier) and to PARSeq, then list where they disagree.

    venv_parseq/Scripts/python src/exp_oct/mistral_compare.py [--run replay_p30_s44] [--model mistral-small-latest] [--limit N]

* Key: MISTRAL_API_KEY in <repo>/.env (git-ignored). The key is only ever sent to api.mistral.ai.
* Crops: D:/OCR_codes/data/roi_watch/input, truth = corrected_text of D:/OCR_codes/data/roi_watch/review_queue.csv.
* PARSeq: outputs/exp_oct/<run>/checkpoints/parseq_<run>.ckpt (default: replay_p30_s44).
* Resumable: Mistral answers are cached in outputs/exp_oct/roi_check/mistral_cache.json, so a rerun only sends the crops still missing.
* Free tier is rate limited (about 1 request/second): the script waits between requests and backs off on HTTP 429.
* Comparison key: characters only (marks stripped, as in the labels), case-sensitive = strict, case-insensitive = lenient.
Output: outputs/exp_oct/roi_check/mistral_vs_parseq.csv and a summary on screen.
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "parseq"))
OUT = ROOT / "outputs" / "exp_oct" / "roi_check"
CACHE = OUT / "mistral_cache.json"
CROPS = Path("D:/OCR_codes/data/roi_watch/input")
QUEUE = Path("D:/OCR_codes/data/roi_watch/review_queue.csv")
URL = "https://api.mistral.ai/v1/chat/completions"
PROMPT = ("This image is a small crop of a handwritten question number from an exam paper (examples: 7, 12, 3b, 14a, iv). "
          "Reply with ONLY the characters you read (digits and letters), nothing else. Ignore brackets, dots, 'Q' prefixes and any stray strokes. "
          "Keep letter case. If you cannot read it, reply with ?")


def norm(s):
    return re.sub(r"[^0-9A-Za-z]", "", str(s))


def drop_prefix(raw):
    """Mistral often reads the drawn 'Q.' prefix as a letter ('a.2', 'c i'): drop one leading letter that is followed by a separator."""
    m = re.match(r"^\s*[A-Za-z][ .:\-]+\s*(.+)$", str(raw))
    return norm(m.group(1)) if m else norm(raw)


def ask(model, key, path, retries=6):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    body = {"model": model, "temperature": 0, "max_tokens": 20,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                      {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}]}]}
    for k in range(retries):
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
        if r.status_code == 429:
            time.sleep(5 * (k + 1))
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Mistral rejected the key (HTTP {r.status_code}). Check MISTRAL_API_KEY in .env.")
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"rate limited too long for {path}")


def parseq_preds(run, paths):
    from PIL import Image
    from model_io import decode, get_device, load_model, make_transform
    dev = get_device()
    m = load_model(str(ROOT / "outputs/exp_oct" / run / "checkpoints" / f"parseq_{run}.ckpt"), dev)
    tf = make_transform(m)
    out = [decode(m, tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))[0] for p in paths]
    return [o[0] for o in out], [float(o[1]) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="replay_p30_s44")
    ap.add_argument("--model", default="pixtral-12b-latest", help="mistral-small-latest has a 0 requests/minute limit on the free key")
    ap.add_argument("--limit", type=int, default=0, help="only the first N crops (a quick test)")
    ap.add_argument("--wait", type=float, default=2.2, help="seconds between requests")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_KEY_HERE":
        sys.exit("No key: put your key in .env as MISTRAL_API_KEY=...")
    q = pd.read_csv(QUEUE, keep_default_na=False, dtype=str)
    if a.limit:
        q = q.head(a.limit)
    paths = [str(CROPS / f) for f in q.filename]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [f for f in q.filename if f"{a.model}|{f}" not in cache]
    print(f"{len(q)} crops, {len(todo)} still to send to {a.model}")
    for i, f in enumerate(todo, 1):
        cache[f"{a.model}|{f}"] = ask(a.model, key, CROPS / f)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        if i % 20 == 0 or i == len(todo):
            print(f"  sent {i}/{len(todo)}")
        time.sleep(a.wait)
    pq, conf = parseq_preds(a.run, paths)
    d = pd.DataFrame({"filename": q.filename, "truth": q.corrected_text.str.strip(), "parseq": pq, "parseq_conf": conf,
                      "mistral_raw": [cache[f"{a.model}|{f}"] for f in q.filename]})
    d["mistral"] = d.mistral_raw.map(norm)
    d["subset"] = ["documents" if f.startswith("Doc") else "screenshots" for f in d.filename]
    d["parseq_ok"] = d.parseq == d.truth
    d["mistral_ok"] = d.mistral == d.truth
    d["mistral_ok_lenient"] = d.mistral.str.lower() == d.truth.str.lower()
    d["mistral_ok_prefix"] = d.mistral_raw.map(drop_prefix).str.lower() == d.truth.str.lower()
    d["agree"] = d.parseq == d.mistral

    def cat(r):
        if r.parseq_ok and r.mistral_ok: return "both right"
        if r.parseq_ok: return "only PARSeq right"
        if r.mistral_ok: return "only Mistral right"
        return "both wrong, same answer" if r.agree else "both wrong, different answers"
    d["outcome"] = d.apply(cat, axis=1)
    d.to_csv(OUT / "mistral_vs_parseq.csv", index=False)
    n = len(d)
    print(f"\nPARSeq ({a.run}): {d.parseq_ok.sum()}/{n} = {d.parseq_ok.mean() * 100:.1f}%")
    print(f"Mistral ({a.model}): {d.mistral_ok.sum()}/{n} = {d.mistral_ok.mean() * 100:.1f}% strict, {d.mistral_ok_lenient.mean() * 100:.1f}% ignoring case, {d.mistral_ok_prefix.mean() * 100:.1f}% also ignoring a leading letter read from the Q prefix")
    print(f"They give the same answer on {d.agree.sum()}/{n} crops; when they agree the answer is right {d[d.agree].parseq_ok.mean() * 100:.1f}% of the time")
    print("\n" + d.outcome.value_counts().to_string())
    print("\nBy subset (PARSeq / Mistral accuracy %):")
    print((d.groupby("subset")[["parseq_ok", "mistral_ok"]].mean() * 100).round(1).to_string())
    dis = d[~d.agree][["filename", "truth", "parseq", "parseq_conf", "mistral_raw", "outcome"]]
    print(f"\nDISAGREEMENTS ({len(dis)}):")
    print(dis.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
