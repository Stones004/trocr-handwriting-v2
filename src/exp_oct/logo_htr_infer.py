"""LoGo-HTR (WACV 2026) inference on single handwritten WORD crops.

    venv_logohtr/Scripts/python src/exp_oct/logo_htr_infer.py --ckpt third_party/LoGoHTR_weights/Downstream_Full_Model_Best.ckpt img1.png img2.png
    venv_logohtr/Scripts/python src/exp_oct/logo_htr_infer.py --ckpt <ckpt> --dir some/folder [--beam 10] [--device cpu|cuda] [--dict path] [--info]

Prints  filename<TAB>prediction  per image (stdout). Everything else goes to stderr.

* Repo: third_party/LoGoHTR (https://github.com/shreemitra/LoGoHTR), patched by patches/logohtr.patch (optional backbone load, dictionary path, utf-8).
* Weights: Downstream_Full_Model_Best.ckpt from https://huggingface.co/Shree10/LoGo-HTR.
* Preprocessing (as in training): RGB, ScaleToLimitRange(w 16-1024, h 16-256), ToTensor ([0,1], no mean/std); batches are zero-padded to the max H/W and
  the mask is a bool tensor, True = padding. The model is rebuilt from ckpt["hyper_parameters"]; beam_search is called with **model.hparams (beam_size overridable).
* The checkpoint is a pickle: it is loaded with torch.load(weights_only=True). If that fails the script stops and says so; use --trust-ckpt only for a file
  whose source you trust (its sha256 is printed).
* The dictionary file is picked by size: the checkpoint's output layer has V rows -> the dictionary must have V-3 lines (pad, sos, eos are added).
  Several candidates with the right size -> the default SSL_HWD_Dict.txt (the dataset the weights were trained on), or --dict to force one.
"""
import argparse
import hashlib
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "LoGoHTR"
DEFAULT_CKPT = ROOT / "third_party" / "LoGoHTR_weights" / "Downstream_Full_Model_Best.ckpt"
EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def log(*a):
    print(*a, file=sys.stderr)


def load_checkpoint(path, trust=False):
    import torch
    log(f"checkpoint {path} sha256 {hashlib.sha256(Path(path).read_bytes()).hexdigest()}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        # The pickle references exactly one non-tensor class, Lightning's ModelCheckpoint (callback state), besides OrderedDict and the tensor rebuild
        # helpers (checked with pickletools, nothing is executed). Allow-listing that single class keeps the safe loader on.
        from pytorch_lightning.callbacks import ModelCheckpoint
        with torch.serialization.safe_globals([ModelCheckpoint]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        if not trust:
            sys.exit(f"torch.load(weights_only=True) failed: {type(e).__name__}: {str(e)[:300]}\n"
                     "The checkpoint contains objects that are not plain tensors. Re-run with --trust-ckpt ONLY if you trust the file.")
        log(f"weights_only load failed ({type(e).__name__}); --trust-ckpt given, loading the full pickle")
        return torch.load(path, map_location="cpu", weights_only=False)


def pick_dictionary(vocab_rows, forced=None):
    dd = REPO / "comer" / "datamodule"
    if forced:
        return Path(forced)
    cands = {p.name: sum(1 for _ in open(p, encoding="utf-8")) for p in sorted(dd.glob("*.txt"))}
    ok = [n for n, k in cands.items() if k == vocab_rows - 3]
    log(f"vocab rows in checkpoint: {vocab_rows} -> dictionary with {vocab_rows - 3} lines. Candidates by size: {ok} (all: {cands})")
    if not ok:
        sys.exit("no dictionary file in the repo has the size the checkpoint expects")
    if len(ok) > 1:
        log(f"ambiguous by size; using SSL_HWD_Dict.txt (the weights' own training dataset). Force another with --dict. Pick by testing both on known images.")
    return dd / ("SSL_HWD_Dict.txt" if "SSL_HWD_Dict.txt" in ok else ok[0])


def build_model(ckpt_path, dictionary=None, trust=False):
    """-> (model, vocab, ckpt). Sets LOGO_DICT before the repo's vocab is imported (the vocab is built at import time)."""
    ck = load_checkpoint(ckpt_path, trust)
    rows = ck["state_dict"]["comer_model.decoder.word_embed.0.weight"].shape[0]
    dpath = pick_dictionary(rows, dictionary)
    os.environ["LOGO_DICT"] = str(dpath)
    log(f"dictionary: {dpath.name}")
    sys.path.insert(0, str(REPO))
    from comer.datamodule import vocab
    from comer.lit_comer import LitCoMER
    assert len(vocab) == rows, (len(vocab), rows)
    keys = set(inspect.signature(LitCoMER.__init__).parameters) - {"self"}
    hp = {k: v for k, v in ck["hyper_parameters"].items() if k in keys}
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):  # the repo's Encoder prints one line per parameter; keep stdout for predictions only
        model = LitCoMER(**hp)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True), None
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n / 1e6:.2f}M | load_state_dict: {missing}")
    return model, vocab, ck


def preprocess(path):
    import numpy as np
    import torchvision.transforms as tr
    from PIL import Image
    from comer.datamodule.transforms import ScaleToLimitRange
    img = np.array(Image.open(path).convert("RGB"))
    return tr.Compose([ScaleToLimitRange(w_lo=16, w_hi=1024, h_lo=16, h_hi=256), tr.ToTensor()])(img)


def collate(tensors):
    import torch
    h = [t.size(1) for t in tensors]
    w = [t.size(2) for t in tensors]
    x = torch.zeros(len(tensors), 3, max(h), max(w))
    mask = torch.ones(len(tensors), max(h), max(w), dtype=torch.bool)  # True = padding
    for i, t in enumerate(tensors):
        x[i, :, :h[i], :w[i]] = t
        mask[i, :h[i], :w[i]] = False
    return x, mask


def to_text(vocab, seq):
    words = [vocab.idx2word[i] for i in seq if i not in (vocab.PAD_IDX, vocab.SOS_IDX, vocab.EOS_IDX)]
    return "".join(" " if w == "" else w for w in words)  # first dictionary line is a space, stripped to "" by the vocab reader


def predict(model, vocab, paths, beam=10, device="cpu", batch=8):
    import torch
    out = []
    hp = dict(model.hparams)
    hp["beam_size"] = beam
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x, mask = collate([preprocess(p) for p in chunk])
        with torch.no_grad():
            hyps = model.comer_model.beam_search(x.to(device), mask.to(device), **hp)
        out += [to_text(vocab, h.seq) for h in hyps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--dir")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--dict", help="force a dictionary file")
    ap.add_argument("--trust-ckpt", action="store_true")
    ap.add_argument("--info", action="store_true", help="only load the model and print parameter count and dictionary")
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()
    paths = [str(p) for p in a.images]
    if a.dir:
        paths += [str(p) for p in sorted(Path(a.dir).iterdir()) if p.suffix.lower() in EXT]
    model, vocab, _ = build_model(a.ckpt, a.dict, a.trust_ckpt)
    model.to(a.device)
    if a.info or not paths:
        return
    for p, t in zip(paths, predict(model, vocab, paths, a.beam, a.device)):
        print(f"{Path(p).name}\t{t}")


if __name__ == "__main__":
    main()
