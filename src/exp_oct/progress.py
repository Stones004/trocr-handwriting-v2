"""Live progress log for the October experiment suite. Every script calls log(); watch it with
    Get-Content outputs\\exp_oct\\progress.log -Wait -Tail 30
"""
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "outputs" / "exp_oct" / "progress.log"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
