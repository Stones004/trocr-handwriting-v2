"""Shared PARSeq model loading, image preprocessing and decoding, used by eval_parseq.py, infer_parseq.py, predict.py
and validate_models.py so the preprocessing that must match training exists in exactly one place.

Run under venv_parseq (needs torch and third_party/parseq's `strhub`).

    model = load_model("models/parseq_final/parseq_final.ckpt")   # or load_model(None) for the pretrained zero-shot model
    tf = make_transform(model)
    (pred, min_conf, mean_conf), = decode(model, tf(image).unsqueeze(0).to(device))

`min_conf` is the smallest per-character probability in the predicted string (EOS included): the project's
review/confidence score. `mean_conf` is the average.
"""
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "parseq"))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint=None, device=None):
    """Fine-tuned Lightning checkpoint (or an `strhub` spec such as 'pretrained=parseq'); None loads the pretrained
    scene-text model from torch.hub (zero-shot, not fine-tuned on this dataset)."""
    device = device or get_device()
    if checkpoint:
        from strhub.models.utils import load_from_checkpoint
        model = load_from_checkpoint(str(checkpoint))
    else:
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    return model.eval().to(device)


def make_transform(model, resize_mode="stretch"):
    """Resize to the model's own input size (read from the checkpoint), then normalize to [-1, 1]. Must match training:
    `resize_mode` is the `data.resize_mode` the model was trained with ("stretch" = PARSeq default, "pad" = aspect-preserving + padding)."""
    from strhub.data.module import make_resize
    return T.Compose([
        make_resize(tuple(model.hparams.img_size), resize_mode),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])


def decode(model, x):
    """Run a batch (N,3,H,W) through the model. Returns one (prediction, min_conf, mean_conf) per image."""
    with torch.no_grad():
        probs = model(x).softmax(-1)
    preds, char_probs = model.tokenizer.decode(probs)
    out = []
    for pred, pr in zip(preds, char_probs):
        out.append((pred.strip(), pr.min().item() if len(pr) else 0.0, pr.mean().item() if len(pr) else 0.0))
    return out
