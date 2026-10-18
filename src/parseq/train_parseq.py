"""Launch one PARSeq fine-tuning run with the project's settings (replaces the old Windows-only train_parseq.bat).

    venv_parseq/Scripts/python src/parseq/train_parseq.py [hydra overrides ...]

It runs third_party/parseq/train.py with the recipe in config.yaml (`train:` section: the parseq_master_v3 recipe), the LMDB folder
(paths.lmdb_dir), its dataset folder name (lmdb.dataset_name), the MLflow store (paths.mlflow_db) and the temp/cache locations (env.tmp_dir, env.torch_home). Anything after the
options is passed straight to Hydra, so any value can still be overridden for one run, e.g.

    python src/parseq/train_parseq.py model.lr=1e-3 +seed=7 hydra.run.dir=outputs/my_run
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from project_config import ROOT, subprocess_env, train_command  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model.lr=1e-3 seed=7")
args = ap.parse_args()

sys.exit(subprocess.run(train_command(args.overrides), cwd=ROOT, env=subprocess_env()).returncode)
