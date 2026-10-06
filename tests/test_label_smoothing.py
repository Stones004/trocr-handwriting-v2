"""Checks for patches/exp_oct.patch label smoothing: eps=0 reproduces the original loss exactly, pad targets stay ignored, EOS handling unchanged.
Run: venv_parseq/Scripts/python tests/test_label_smoothing.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "parseq"))
from model_io import load_model, make_transform  # noqa: E402

# 1. pad targets are ignored by smoothing (changing the logits at pad positions changes nothing); eps=0 equals the plain call
torch.manual_seed(0)
V, pad = 97, 96
lg = torch.randn(20, V)
tg = torch.randint(0, 90, (20,))
tg[[3, 4, 11, 19]] = pad
a = F.cross_entropy(lg, tg, ignore_index=pad, label_smoothing=0.1)
lg2 = lg.clone()
lg2[[3, 4, 11, 19]] = torch.randn(4, V) * 5
b = F.cross_entropy(lg2, tg, ignore_index=pad, label_smoothing=0.1)
assert torch.allclose(a, b), (a, b)
assert torch.equal(F.cross_entropy(lg, tg, ignore_index=pad, label_smoothing=0.0), F.cross_entropy(lg, tg, ignore_index=pad))
print("pad positions ignored under smoothing: OK; eps=0 identical to plain cross entropy: OK")

# 2. whole training_step on a real batch: patched code at eps=0 == the original loop; eps>0 gives a larger loss
ck = ROOT / "outputs/exp_oct/baseline_s42/checkpoints/parseq_baseline_s42.ckpt"
m = load_model(str(ck), torch.device("cpu")).train()
tf = make_transform(m)
df = pd.read_csv(ROOT / "data/docsplit/seed42/test.csv", keep_default_na=False, dtype=str).head(16)
images = torch.stack([tf(Image.open(p).convert("RGB")) for p in df.image_path])
labels = df.text.tolist()


def original_loss(model, images, labels):
    """Verbatim copy of PARSeq.training_step before the patch (cross_entropy without label_smoothing)."""
    tgt = model.tokenizer.encode(labels, model._device)
    memory = model.model.encode(images)
    tgt_perms = model.gen_tgt_perms(tgt)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)
    loss, loss_numel, n = 0, 0, (tgt_out != model.pad_id).sum().item()
    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = model.generate_attn_masks(perm)
        out = model.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = model.model.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=model.pad_id)
        loss_numel += n
        if i == 1:
            tgt_out = torch.where(tgt_out == model.eos_id, model.pad_id, tgt_out)
            n = (tgt_out != model.pad_id).sum().item()
    return loss / loss_numel


def run(fn):
    torch.manual_seed(123)
    m.rng = np.random.default_rng(7)
    return fn()


m.log = lambda *a, **k: None
ref = run(lambda: original_loss(m, images, labels))
m.label_smoothing = 0.0
new0 = run(lambda: m.training_step((images, labels), 0))
m.label_smoothing = 0.1
new1 = run(lambda: m.training_step((images, labels), 0))
print(f"original loss {ref.item():.6f} | patched eps=0 {new0.item():.6f} | patched eps=0.1 {new1.item():.6f}")
assert torch.allclose(ref, new0, atol=1e-6), "eps=0 does not match the original loss"
assert new1 > new0, "smoothing should raise the loss on a trained model"
print("label smoothing patch tests: all passed")
