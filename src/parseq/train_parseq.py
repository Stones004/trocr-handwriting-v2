"""Launch one PARSeq fine-tuning run with the project's settings (replaces the old Windows-only train_parseq.bat).

    venv_parseq/Scripts/python src/parseq/train_parseq.py [--experiment parseq_handwriting_fixed] [hydra overrides ...]

It runs third_party/parseq/train.py with the chosen recipe from configs/parseq/experiment/<name>.yaml and fills in the
project defaults from config.yaml: the LMDB folder (paths.lmdb_dir), its dataset folder name (lmdb.dataset_name), the
MLflow store (paths.mlflow_db), and the temp/cache locations (env.tmp_dir, env.torch_home). Anything after the
options is passed straight to Hydra, so any value can still be overridden for one run, e.g.

    python src/parseq/train_parseq.py --experiment parseq_handwriting_fixed model.lr=1e-3 seed=7 data.root_dir=data/all_lmdb hydra.run.dir=outputs/parseq_final/run
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import ROOT, subprocess_env, train_command  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--experiment", default="parseq_handwriting", help="recipe name under configs/parseq/experiment/")
ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model.lr=1e-3 seed=7")
args = ap.parse_args()

sys.exit(subprocess.run(train_command(args.experiment, args.overrides), cwd=ROOT, env=subprocess_env()).returncode)
