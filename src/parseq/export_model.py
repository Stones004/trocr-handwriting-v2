"""Export a trained PARSeq Lightning checkpoint as a weights-only deployment checkpoint (drops optimizer/scheduler
state: ~4x smaller) and verify it loads and predicts identically.
    python src/parseq/export_model.py --checkpoint outputs/parseq_final/run/checkpoints/last.ckpt --out models/parseq_final
The output path must contain 'parseq' (PARSeq's loader infers the model class from it); load with:
    strhub.models.utils.load_from_checkpoint("models/parseq_final/parseq_final.ckpt")
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "parseq"))
from strhub.models.utils import load_from_checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
slim = {k: ck[k] for k in ("state_dict", "hyper_parameters", "pytorch-lightning_version") if k in ck}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dst = out / f"{out.name}.ckpt"
torch.save(slim, dst)

a = load_from_checkpoint(args.checkpoint).eval().cpu()
b = load_from_checkpoint(str(dst)).eval().cpu()
x = torch.randn(4, 3, *a.hparams.img_size)
with torch.no_grad():
    same = torch.allclose(a(x), b(x), atol=1e-6)
print(f"exported {dst} ({dst.stat().st_size / 1e6:.0f} MB, was {Path(args.checkpoint).stat().st_size / 1e6:.0f} MB); outputs identical: {same}")
