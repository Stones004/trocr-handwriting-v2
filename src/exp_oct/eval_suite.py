"""Common evaluation for the October experiments: one function, reused for every checkpoint.

For a model, on the real test (and val) crops with charset_test = the full 94 characters (raw, case-sensitive string match):
  1 exact match + 95% Wilson interval + CER          5 tier table (conf >0.95 / 0.70-0.95 / <0.70)
  2 word-test exact match (300 synthetic words)      6 confident errors (conf > 0.9), crops copied to a folder
  3 per-crop table (filename,label,pred,conf,correct) 7 ECE (15 bins) and NLL
  4 coverage at 99% precision: threshold T chosen on VAL only, applied to TEST
conf = minimum per-character softmax probability, end token included (same as model_io.decode).
NLL = mean per-token negative log-likelihood of the true label (chars + EOS) under the model's own autoregressive decode logits.
ECE = crop-level: confidence (min char prob) vs whether the whole crop is exactly right.

CLI (venv_parseq):  python src/exp_oct/eval_suite.py --name baseline --checkpoint models/parseq_master_v3/parseq_master_v3.ckpt
                    (--checkpoint pretrained = original PARSeq)
"""
import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
from model_io import get_device, load_model, make_transform  # noqa: E402
from progress import log  # noqa: E402
from validation_metrics import lev  # noqa: E402

TEST_CSV = ROOT / "data" / "docsplit" / "seed42" / "test.csv"
VAL_CSV = ROOT / "data" / "docsplit" / "seed42" / "val.csv"
WORD_CSV = ROOT / "data" / "open_read" / "word_eval.csv"
CONF_ERR = 0.9
safe = lambda s: re.sub(r"[^A-Za-z0-9_-]", "_", str(s)) or "empty"  # noqa: E731  (file-name safe)
TIERS = [("conf > 0.95", 0.95, 2.0), ("0.70 - 0.95", 0.70, 0.95), ("conf < 0.70", -1.0, 0.70)]


# ---------------------------------------------------------------- pure statistics (unit-tested) ----------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def coverage_threshold(conf, correct, target=0.99):
    """Lowest threshold T such that the crops with conf >= T have precision >= target. inf if no threshold qualifies."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    best = math.inf
    for t in sorted(set(conf.tolist()), reverse=True):
        acc = conf >= t
        if correct[acc].mean() >= target:
            best = t
    return best


def accepted_stats(conf, correct, T):
    """(coverage, precision, n_accepted) when accepting conf >= T. Precision is nan if nothing is accepted."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    acc = conf >= T
    n = int(acc.sum())
    return (n / max(len(conf), 1), float(correct[acc].mean()) if n else float("nan"), n)


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def tier_table(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    rows = []
    for name, lo, hi in TIERS:
        m = (conf > lo) & (conf <= hi) if name != "conf < 0.70" else (conf < hi)
        n, err = int(m.sum()), int((~correct[m]).sum())
        rows.append({"tier": name, "crops": n, "errors": err, "precision": (n - err) / n if n else float("nan")})
    return rows


# ---------------------------------------------------------------- model passes -------------------------------------------
@torch.no_grad()
def run_model(model, df, tf, device, bs=64):
    """Returns preds, conf (min char prob), logits (N, L, C) on cpu, for every crop of df."""
    preds, confs, logits = [], [], []
    paths = df["image_path"].tolist()
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device)
        lg = model(x).float()
        pr, cp = model.tokenizer.decode(lg.softmax(-1))
        preds += [s.strip() for s in pr]
        confs += [c.min().item() if len(c) else 0.0 for c in cp]
        logits.append(lg.cpu())
    return preds, np.array(confs), torch.cat(logits)


def token_targets(model, labels, length):
    """(N, L) target ids (chars + EOS, pad elsewhere) aligned with the decode logits."""
    enc = model.tokenizer.encode(list(labels))[:, 1:]            # drop BOS
    out = torch.full((len(labels), length), model.tokenizer.pad_id, dtype=torch.long)
    k = min(length, enc.shape[1])
    out[:, :k] = enc[:, :k]
    return out


def nll(logits, targets, pad_id, T=1.0):
    """Mean per-token negative log-likelihood of the true label (chars + EOS) under the model's decode logits / T."""
    lp = torch.log_softmax(logits.float() / T, -1)
    m = targets != pad_id
    tok = -lp.gather(-1, targets.clamp(max=lp.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    return float(tok[m].mean())


def conf_from_logits(model, logits, T=1.0):
    _, cp = model.tokenizer.decode((logits.float() / T).softmax(-1))
    return np.array([c.min().item() if len(c) else 0.0 for c in cp])


def fit_temperature(model, logits, targets):
    """One scalar T minimising token NLL (fitted on whatever logits it is given: call it with VAL only)."""
    pad = model.tokenizer.pad_id
    grid = np.exp(np.linspace(math.log(0.3), math.log(5.0), 200))
    best = min(grid, key=lambda t: nll(logits, targets, pad, t))
    fine = np.exp(np.linspace(math.log(best / 1.05), math.log(best * 1.05), 100))
    return float(min(fine, key=lambda t: nll(logits, targets, pad, t)))


# ---------------------------------------------------------------- the suite ---------------------------------------------
def _crop_table(df, preds, conf):
    labels = df["text"].astype(str).str.strip().tolist()
    return pd.DataFrame({"filename": df["filename"], "label": labels, "pred": preds, "conf": conf,
                         "correct": [p == t for p, t in zip(preds, labels)]})


def _acc_block(tab):
    n, k = len(tab), int(tab["correct"].sum())
    lo, hi = wilson(k, n)
    edits = sum(lev(p, t) for p, t in zip(tab["pred"], tab["label"]))
    norm = lambda x: re.sub(r"^(?:Ans|ans|Q|q)", "", re.sub(r"[^A-Za-z0-9]", "", str(x)))  # noqa: E731  (open_read metric: ignore marks)
    k2 = sum(norm(p) == norm(t) for p, t in zip(tab["pred"], tab["label"]))
    return {"n": n, "correct": k, "exact": k / n, "wilson_lo": lo, "wilson_hi": hi, "exact_alnum": k2 / n,
            "cer": edits / max(sum(max(len(t), 1) for t in tab["label"]), 1)}


def evaluate_model(model, name, out_dir, device=None, with_val=True, with_words=True, temperature=None, copy_crops=True):
    """Full suite on an already-loaded model. If `temperature` is given, test/val confidences are recomputed from logits/T
    (T must have been fitted on val). Writes per-crop CSVs, logits and metrics.json into out_dir; returns the metrics dict."""
    device = device or get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().to(device)
    tf = make_transform(model)
    sets = {"test": pd.read_csv(TEST_CSV, keep_default_na=False, dtype=str)}
    if with_val:
        sets["val"] = pd.read_csv(VAL_CSV, keep_default_na=False, dtype=str)
    if with_words:
        sets["words"] = pd.read_csv(WORD_CSV, keep_default_na=False, dtype=str)
    for k, d in sets.items():
        assert not d["text"].eq("").any() and not d["image_path"].eq("").any(), f"{k}: empty label or path"
    res, tabs = {"name": name}, {}
    for k, d in sets.items():
        preds, conf, lg = run_model(model, d, tf, device)
        if temperature is not None and k in ("test", "val"):
            conf = conf_from_logits(model, lg, temperature)
        tabs[k] = _crop_table(d, preds, conf)
        tabs[k].to_csv(out_dir / f"crops_{k}.csv", index=False)
        res[k] = _acc_block(tabs[k])
        if k in ("test", "val"):
            tg = token_targets(model, tabs[k]["label"], lg.shape[1])
            res[k]["nll"] = nll(lg, tg, model.tokenizer.pad_id, temperature or 1.0)
            res[k]["ece"] = ece(conf, tabs[k]["correct"])
            torch.save({"logits": lg.half(), "targets": tg}, out_dir / f"logits_{k}.pt")
    t = tabs["test"]
    res["tiers_test"] = tier_table(t["conf"], t["correct"])
    ce = t[(~t["correct"]) & (t["conf"] > CONF_ERR)]
    res["confident_errors"] = {"count": len(ce), "detail": ce[["filename", "label", "pred", "conf"]].round(4).to_dict("records")}
    if copy_crops and len(ce):
        d = out_dir / "confident_errors"
        d.mkdir(exist_ok=True)
        for _, r in ce.iterrows():
            src = sets["test"].loc[sets["test"]["filename"] == r["filename"], "image_path"].iloc[0]
            shutil.copy(src, d / f"{Path(r['filename']).stem}__label-{safe(r['label'])}__pred-{safe(r['pred'])}.png")
    if with_val:
        T = coverage_threshold(tabs["val"]["conf"], tabs["val"]["correct"], 0.99)
        cov, prec, n = accepted_stats(t["conf"], t["correct"], T)
        vcov, vprec, _ = accepted_stats(tabs["val"]["conf"], tabs["val"]["correct"], T)
        res["coverage99"] = {"threshold_from_val": None if math.isinf(T) else T, "test_coverage": cov, "test_precision": prec,
                             "test_accepted": n, "val_coverage": vcov, "val_precision": vprec}
    res["temperature"] = temperature
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    c99 = res.get("coverage99", {})
    log(f"EVAL {name}: test {res['test']['correct']}/{res['test']['n']} = {res['test']['exact']*100:.1f}% | "
        f"words {res.get('words', {}).get('exact', float('nan'))*100:.1f}% | "
        f"cov@99 {c99.get('test_coverage', float('nan'))*100:.1f}% (prec {c99.get('test_precision', float('nan'))*100:.1f}%) | "
        f"conf-errors {res['confident_errors']['count']} | ECE {res['test'].get('ece', float('nan')):.3f} NLL {res['test'].get('nll', float('nan')):.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", required=True, help="path to .ckpt, or 'pretrained'")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = get_device()
    model = load_model("pretrained=parseq" if a.checkpoint == "pretrained" else a.checkpoint, dev)
    evaluate_model(model, a.name, a.out or ROOT / "outputs" / "exp_oct" / "eval" / a.name, dev)


if __name__ == "__main__":
    main()
