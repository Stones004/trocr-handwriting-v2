"""Minimal single-image PARSeq inference: give it an image, get back the predicted text and its confidence.
Nothing else -- no batching, no CSV, no checkpoint argument to remember. Uses the production checkpoint
(inference.checkpoint in config.yaml, default models/parseq_final/parseq_final.ckpt) unless you pass another.

    venv_parseq/Scripts/python src/parseq/predict.py "D:/path/to/roi.png"

Also importable directly, e.g. from a pipeline that already has the image in memory or wants to avoid
reloading the model per call:

    from predict import predict
    text, confidence, grammar_ok = predict("roi.png")   # loads the model once, reused on later calls

`confidence` is the minimum per-character probability across the predicted sequence -- the same metric this
project's flagging/eval logic uses elsewhere (a lower value means some character in the prediction was
uncertain, not that the whole prediction is necessarily wrong). Predictions below inference.flag_threshold are
marked for review.

Also checks the prediction against the project's closed label grammar (config.yaml labels.clean_pattern -- the same
pattern used to filter training data). PARSeq's decoding isn't constrained to only produce grammar-conforming
strings, so this catches a different failure mode than confidence does: an out-of-distribution crop can produce a
fluent-looking but nonsense prediction ("Q11", "13 b") with high confidence, which a confidence threshold alone
would miss.
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_grammar import matches_grammar  # noqa: E402
from model_io import decode, load_model, make_transform  # noqa: E402
from project_config import cfg, resolve  # noqa: E402

DEFAULT_CHECKPOINT = resolve(cfg["inference"]["checkpoint"])
FLAG_THRESHOLD = cfg["inference"]["flag_threshold"]

_model = None
_transform = None


def _load(checkpoint):
    global _model, _transform
    if _model is None:
        _model = load_model(checkpoint)
        _transform = make_transform(_model)
    return _model, _transform


def predict(image_path, checkpoint=DEFAULT_CHECKPOINT):
    """Returns (prediction: str, confidence: float, matches_grammar: bool) for one image."""
    model, transform = _load(checkpoint)
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    (prediction, confidence, _), = decode(model, x)
    return prediction, confidence, matches_grammar(prediction)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: predict.py <image_path>")
    prediction, confidence, ok = predict(sys.argv[1])
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    if confidence < FLAG_THRESHOLD:
        print(f"REVIEW: confidence is below the review threshold ({FLAG_THRESHOLD})")
    if not ok:
        print("WARNING: prediction does not match the expected label grammar (digit run, Roman numeral, or subpart letter) -- treat with suspicion regardless of confidence")
