"""
PARSeq baseline (pretrained zero-shot, or fine-tuned) on the same test split
used for evaluating this project's fine-tuned model.

With no --checkpoint, uses the pretrained scene-text checkpoint from
https://github.com/baudm/parseq via torch.hub -- NOT fine-tuned on this
dataset, a quick sanity check on whether the architecture looks promising.

With --checkpoint pointing at a Lightning .ckpt produced by
third_party/parseq/train.py (see docs/CODEBASE_GUIDE.md), evaluates that
fine-tuned model instead. Must be run with the venv_parseq interpreter
(needs strhub importable):

    venv_parseq/Scripts/python src/parseq/eval_parseq.py --checkpoint "outputs/parseq/<run>/checkpoints/epoch=20-...ckpt"

Confidence is the per-character softmax probability from PARSeq's own decode
step (min/mean across the predicted sequence, EOS included) -- the PARSeq
analog of a per-token confidence score.

Defaults come from config.yaml: the flag threshold is inference.flag_threshold, and MLflow logs go to paths.mlflow_db.

Usage:
    python src/parseq/eval_parseq.py --csv some_labelled.csv --checkpoint models/parseq_master_v3/parseq_master_v3.ckpt --output reports/results.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_io import decode, get_device, load_model, make_transform  # noqa: E402
from project_config import cfg, path  # noqa: E402
from validation_metrics import lev  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="labelled CSV with columns filename,image_path,text")
ap.add_argument("--checkpoint", default=None, help="path to a fine-tuned .ckpt from third_party/parseq/train.py; omit for the pretrained zero-shot model")
ap.add_argument("--flag-threshold", type=float, default=cfg["inference"]["flag_threshold"], help="flag for review if min char confidence falls below this")
ap.add_argument("--resize-mode", default="stretch", choices=["stretch", "pad"], help="must match the data.resize_mode the checkpoint was trained with")
ap.add_argument("--output", default=None, help="optional per-sample results CSV")
ap.add_argument("--mlflow-run", default=None, help="log test metrics into this MLflow run (run name, e.g. 2026-09-21_11-07-04, in --mlflow-experiment)")
ap.add_argument("--mlflow-experiment", default="parseq")
args = ap.parse_args()

device = get_device()
model = load_model(args.checkpoint, device)
img_transform = make_transform(model, args.resize_mode)

df = pd.read_csv(args.csv)
exact = edits = chars = flagged = 0
rows = []
for _, r in df.iterrows():
    image = Image.open(r["image_path"]).convert("RGB")
    x = img_transform(image).unsqueeze(0).to(device)
    (pred, min_conf, mean_conf), = decode(model, x)
    truth = str(r["text"]).strip()
    is_flagged = min_conf < args.flag_threshold
    exact += pred == truth
    edits += lev(pred, truth)
    chars += max(len(truth), 1)
    flagged += is_flagged
    rows.append({
        "filename": r["filename"], "target": truth, "prediction": pred,
        "exact_match": pred == truth, "seq_min_prob": min_conf, "seq_mean_prob": mean_conf,
    })

print(f"Model: PARSeq (fine-tuned: {args.checkpoint})" if args.checkpoint else "Model: PARSeq (pretrained, zero-shot)")
print("Samples:", len(df))
print("Exact-match accuracy:", exact / max(len(df), 1))
print("CER:", edits / max(chars, 1))
print(f"Flagged for review (min conf < {args.flag_threshold}):", flagged, f"({flagged / max(len(df), 1) * 100:.1f}%)")

if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("Per-sample results written to", args.output)

if args.mlflow_run:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("sqlite:///" + path("mlflow_db").as_posix())
    client = MlflowClient()
    exp = client.get_experiment_by_name(args.mlflow_experiment)
    runs = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{args.mlflow_run}'") if exp else []
    if not runs:
        sys.exit(f"MLflow run '{args.mlflow_run}' not found in experiment '{args.mlflow_experiment}'")
    run_id = runs[0].info.run_id
    for k, v in {"test_exact_match": exact / max(len(df), 1), "test_cer": edits / max(chars, 1),
                 "test_flagged_frac": flagged / max(len(df), 1), "test_samples": len(df)}.items():
        client.log_metric(run_id, k, v)
    if args.checkpoint:
        client.set_tag(run_id, "test_checkpoint", Path(args.checkpoint).name)
    if args.output:
        client.log_artifact(run_id, args.output, artifact_path="test_results")
    print("Logged test metrics to MLflow run", args.mlflow_run)
