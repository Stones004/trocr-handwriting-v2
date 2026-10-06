"""Phase 1: WiSE-FT (weight-space interpolation between the pretrained PARSeq and a fine-tuned checkpoint) and weight soups. No training.

    venv_parseq/Scripts/python src/exp_oct/phase1_wise_ft.py wise --ft models/parseq_master_v3/parseq_master_v3.ckpt --tag master_v3
    venv_parseq/Scripts/python src/exp_oct/phase1_wise_ft.py soup --ckpts a.ckpt b.ckpt c.ckpt --tag baseline_soup

theta = (1-a) * theta_pre + a * theta_ft for float tensors; non-float buffers are copied from ft. State-dict keys and shapes must match exactly.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/common", "src/parseq", "src/exp_oct"):
    sys.path.insert(0, str(ROOT / p))
os.environ.setdefault("TORCH_HOME", "D:/cache/torch")
import torch  # noqa: E402
from eval_suite import evaluate_model  # noqa: E402
from model_io import get_device, load_model  # noqa: E402
from progress import log  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
OUT = ROOT / "outputs" / "exp_oct" / "phase1"


def check_match(sd_a, sd_b):
    ka, kb = set(sd_a), set(sd_b)
    if ka != kb:
        sys.exit(f"STOP: state_dict keys differ. only in pretrained: {sorted(ka - kb)[:5]}, only in ft: {sorted(kb - ka)[:5]}")
    bad = [k for k in ka if sd_a[k].shape != sd_b[k].shape]
    if bad:
        sys.exit(f"STOP: shape mismatch in {bad[:5]}")
    log(f"state_dict check OK: {len(ka)} keys, all shapes identical")


def interpolate(sd_pre, sd_ft, a):
    out = {}
    for k, v in sd_ft.items():
        out[k] = ((1 - a) * sd_pre[k].float() + a * v.float()).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


def wise(ft_path, tag, dev):
    pre = load_model("pretrained=parseq", dev)
    ft = load_model(ft_path, dev)
    sd_pre, sd_ft = {k: v.cpu() for k, v in pre.state_dict().items()}, {k: v.cpu() for k, v in ft.state_dict().items()}
    check_match(sd_pre, sd_ft)
    rows = []
    for a in ALPHAS:
        m = copy.deepcopy(ft)
        m.load_state_dict(interpolate(sd_pre, sd_ft, a))
        r = evaluate_model(m, f"wise_{tag}_a{a}", OUT / f"wise_{tag}_a{a}", dev)
        rows.append(r)
    # sanity: a=1.0 identical to the checkpoint's own outputs; a=0 identical to the pretrained model's outputs
    ref_ft = evaluate_model(ft, f"ref_{tag}", OUT / f"ref_{tag}", dev, copy_crops=False)
    ref_pre = evaluate_model(pre, "ref_pretrained", OUT / "ref_pretrained", dev, copy_crops=False)
    for a, ref in ((1.0, f"ref_{tag}"), (0.0, "ref_pretrained")):
        x = pd.read_csv(OUT / f"wise_{tag}_a{a}" / "crops_test.csv")
        y = pd.read_csv(OUT / ref / "crops_test.csv")
        same = (x["pred"].fillna("") == y["pred"].fillna("")).all() and (x["conf"] - y["conf"]).abs().max() < 1e-4
        log(f"SANITY a={a} vs {ref}: predictions and confidences identical = {bool(same)}")
        assert same, f"a={a} does not reproduce {ref}"
    log(f"pretrained real test {ref_pre['test']['exact']*100:.1f}% (report: 60.7), words {ref_pre['words']['exact']*100:.1f}% (report: 63.3)")
    return rows


def soup(ckpts, tag, dev):
    models = [load_model(c, dev) for c in ckpts]
    sds = [{k: v.cpu() for k, v in m.state_dict().items()} for m in models]
    for s in sds[1:]:
        check_match(sds[0], s)
    avg = {k: (torch.stack([s[k].float() for s in sds]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()) for k, v in sds[0].items()}
    m = copy.deepcopy(models[0])
    m.load_state_dict(avg)
    return evaluate_model(m, tag, OUT / tag, dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["wise", "soup"])
    ap.add_argument("--ft")
    ap.add_argument("--ckpts", nargs="*")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = get_device()
    wise(a.ft, a.tag, dev) if a.cmd == "wise" else soup(a.ckpts, a.tag, dev)
