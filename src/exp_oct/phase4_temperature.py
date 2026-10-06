"""Phase 4: temperature scaling. One scalar T per model, fitted on VAL logits only (minimise token NLL), applied to TEST. No training.

    venv_parseq/Scripts/python src/exp_oct/phase4_temperature.py baseline_s42 baseline_s43 ...   (run names under outputs/exp_oct/, or master_v3_existing)

Reads outputs/exp_oct/<run>/eval/logits_{val,test}.pt + crops_{val,test}.csv written by eval_suite (the decode logits of the real val/test crops).
For each model reports before / after: ECE (15 bins), NLL, coverage at 99% precision (threshold chosen on val, applied to test), confident errors (conf > 0.9),
test accuracy (must not change: temperature never changes the argmax). Writes <run>/eval/temperature.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
from eval_suite import accepted_stats, conf_from_logits, coverage_threshold, ece, fit_temperature, nll  # noqa: E402
from model_io import load_model  # noqa: E402
from progress import log  # noqa: E402

OUT = ROOT / "outputs" / "exp_oct"
_tok = {}


def tokenizer_model():
    if "m" not in _tok:
        _tok["m"] = load_model(str(OUT / "baseline_s42" / "checkpoints" / "parseq_baseline_s42.ckpt"), torch.device("cpu"))
    return _tok["m"]


def fit_temperature_crop(model, lv, cv):
    """T minimising the crop-level binary NLL of 'is the whole crop right' under conf = min char prob (VAL only). This matches how confidence is used
    (accept / reject a crop) and does not depend on the true labels of positions after a wrong prefix."""
    y = cv["correct"].to_numpy(float)

    def bce(T):
        c = np.clip(conf_from_logits(model, lv, T), 1e-6, 1 - 1e-6)
        return float(-(y * np.log(c) + (1 - y) * np.log(1 - c)).mean())
    grid = np.exp(np.linspace(np.log(0.5), np.log(4.0), 60))
    best = min(grid, key=bce)
    fine = np.exp(np.linspace(np.log(best / 1.1), np.log(best * 1.1), 40))
    return float(min(fine, key=bce))


def block(model, lv, tv, ct, cv, lt, tt, T):
    pad = model.tokenizer.pad_id
    cvc, ctc = conf_from_logits(model, lv, T), conf_from_logits(model, lt, T)
    thr = coverage_threshold(cvc, cv["correct"], 0.99)
    cov, prec, n = accepted_stats(ctc, ct["correct"], thr)
    return {"T": T, "test_ece": ece(ctc, ct["correct"]), "test_nll": nll(lt, tt, pad, T), "val_nll": nll(lv, tv, pad, T),
            "val_threshold": None if np.isinf(thr) else float(thr), "cov99": cov, "cov99_precision": prec, "cov99_accepted": n,
            "confident_errors": int(((~ct["correct"].astype(bool)) & (ctc > 0.9)).sum()), "mean_conf": float(ctc.mean())}


def run(name):
    d = OUT / ("eval/master_v3_existing" if name == "master_v3_existing" else f"{name}/eval")
    lv, lt = torch.load(d / "logits_val.pt"), torch.load(d / "logits_test.pt")
    tv, tt = lv["targets"], lt["targets"]
    lv, lt = lv["logits"].float(), lt["logits"].float()
    cv, ct = pd.read_csv(d / "crops_val.csv", keep_default_na=False), pd.read_csv(d / "crops_test.csv", keep_default_na=False)
    for c in (cv, ct):
        c["correct"] = c["correct"].astype(str) == "True"
    model = tokenizer_model()
    T = fit_temperature(model, lv, tv)                       # A: token NLL, fitted on val logits ONLY (the literal spec)
    Tc = fit_temperature_crop(model, lv, cv)                 # B: crop-level NLL, fitted on val ONLY
    res = {"name": name, "test_acc": float(ct["correct"].mean()), "before": block(model, lv, tv, ct, cv, lt, tt, 1.0),
           "after": block(model, lv, tv, ct, cv, lt, tt, T), "after_crop": block(model, lv, tv, ct, cv, lt, tt, Tc)}
    (d / "temperature.json").write_text(json.dumps(res, indent=1))
    b, a, c = res["before"], res["after"], res["after_crop"]
    log(f"TEMP-B {name}: T={Tc:.2f} | ECE {b['test_ece']:.3f}->{c['test_ece']:.3f} | NLL {b['test_nll']:.3f}->{c['test_nll']:.3f} | "
        f"cov@99 {b['cov99']*100:.1f}->{c['cov99']*100:.1f}% | confident errors {b['confident_errors']}->{c['confident_errors']}")
    log(f"TEMP-A {name}: T={T:.2f} | ECE {b['test_ece']:.3f}->{a['test_ece']:.3f} | NLL {b['test_nll']:.3f}->{a['test_nll']:.3f} | "
        f"cov@99 {b['cov99']*100:.1f}->{a['cov99']*100:.1f}% | confident errors {b['confident_errors']}->{a['confident_errors']}")
    return res


if __name__ == "__main__":
    for r in sys.argv[1:]:
        run(r)
