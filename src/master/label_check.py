"""Find real crops whose label is probably wrong, without reviewing all of them.

Idea: split the real crops by DOCUMENT into 5 folds, train 5 PARSeq models (each on 4 folds, real data only, the recipe in config.yaml), and let
each model read the fold it never saw. A model that did not learn a crop's label can only repeat it if the image supports it, so a disagreement
(or a low-confidence reading) marks a suspicious label. The original pretrained PARSeq, which never saw our labels either, gives a second opinion.

    venv_parseq/Scripts/python src/master/label_check.py train     # 5 folds: train + out-of-fold predictions -> data/label_check/oof_predictions.csv
    venv_parseq/Scripts/python src/master/label_check.py sheets    # review sheets (most suspicious first) + a random audit sample -> data/label_check/

Nothing in the dataset is changed; this only reads it and writes files under data/label_check/ (temporary LMDBs and checkpoints are deleted afterwards).
Labels checked = the normalized labels of the dataset (characters only; brackets, dots, Q prefix stripped).
"""
import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.path.insert(0, str(ROOT / "src" / "parseq"))
from project_config import cfg, path, resolve, subprocess_env, train_command  # noqa: E402

OUT = ROOT / "data" / "label_check"
TMP = ROOT / "data" / "_label_check_tmp"
K, SEED = 5, 42


def real_crops():
    d = pd.read_csv(ROOT / "data" / "master_dataset.csv", keep_default_na=False, dtype=str)
    d = d[d.source == "real"].drop_duplicates("filename")
    img = resolve(cfg["image_dir"])
    return pd.DataFrame({"filename": d.filename, "image_path": [str(img / f) for f in d.filename], "text": d.text, "doc": d.doc})


def make_folds(df):
    """Greedy, balanced by crop count; whole documents stay together."""
    docs = df.doc.value_counts()
    order = list(docs.index)
    random.Random(SEED).shuffle(order)
    order.sort(key=lambda x: -docs[x])
    load, fold_of = [0] * K, {}
    for doc in order:
        k = load.index(min(load))
        fold_of[doc] = k
        load[k] += docs[doc]
    return df.assign(fold=df.doc.map(fold_of))


def train():
    from model_io import decode, get_device, load_model, make_transform
    from PIL import Image
    df = make_folds(real_crops())
    print("real crops:", len(df), "| per fold:", df.fold.value_counts().sort_index().tolist(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env, dev = subprocess_env(), get_device()
    parts = []
    for k in range(K):
        pred_csv = OUT / f"oof_fold{k}.csv"
        if pred_csv.exists():
            parts.append(pd.read_csv(pred_csv, keep_default_na=False, dtype=str)); continue
        run = ROOT / "outputs" / "parseq_label_check" / f"fold{k}"
        final = run / "checkpoints" / "final.ckpt"
        if not final.exists():
            data = TMP / f"fold{k}"
            data.mkdir(parents=True, exist_ok=True)
            tr, ho = df[df.fold != k], df[df.fold == k]
            cols = ["filename", "image_path", "text"]
            tr[cols].to_csv(data / "train.csv", index=False)
            ho[cols].to_csv(data / "val.csv", index=False)       # PARSeq needs a val set; fixed epochs, so it never influences the weights
            ho[cols].to_csv(data / "test.csv", index=False)
            subprocess.run([sys.executable, str(ROOT / "src" / "parseq" / "prepare_parseq_dataset.py"), "--processed-dir", str(data),
                            "--output-dir", str(data / "lmdb")], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
            print(f"[fold {k}] training on {len(tr)} crops, reading {len(ho)} unseen", flush=True)
            ov = [f"data.root_dir={(data / 'lmdb').as_posix()}", f"hydra.run.dir={run.as_posix()}", f"+mlflow_db={(TMP / 'mlflow.db').as_posix()}"]
            r = subprocess.run(train_command(ov), cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not final.exists():
                sys.exit(f"[fold {k}] training failed (exit {r.returncode})")
        model = load_model(str(final), dev)
        tf = make_transform(model)
        ho = df[df.fold == k]
        rows = []
        for r in ho.itertuples():
            pred, mn, mean = decode(model, tf(Image.open(r.image_path).convert("RGB")).unsqueeze(0).to(dev))[0]
            rows.append({"filename": r.filename, "doc": r.doc, "fold": k, "label": r.text, "pred": pred, "min_conf": f"{mn:.4f}", "mean_conf": f"{mean:.4f}",
                         "image_path": r.image_path})
        out = pd.DataFrame(rows)
        out.to_csv(pred_csv, index=False)
        parts.append(out.astype(str))
        print(f"[fold {k}] out-of-fold exact match {(out.pred == out.label).mean() * 100:.1f}%", flush=True)
        del model
    res = pd.concat(parts, ignore_index=True)
    pre = load_model("pretrained=parseq", dev)
    tf = make_transform(pre)
    res["pretrained"] = [decode(pre, tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev))[0][0] for p in res.image_path]
    res.to_csv(OUT / "oof_predictions.csv", index=False)
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(ROOT / "outputs" / "parseq_label_check", ignore_errors=True)
    print("wrote", OUT / "oof_predictions.csv", "| overall out-of-fold exact match", f"{(res.pred == res.label).mean() * 100:.1f}%")


def sheets(per_sheet=30, audit_n=150):
    from PIL import Image, ImageDraw
    res = pd.read_csv(OUT / "oof_predictions.csv", keep_default_na=False, dtype=str)
    res["min_conf"] = res.min_conf.astype(float)
    norm = lambda s: "".join(c for c in str(s) if c.isalnum())  # noqa: E731
    res["agree"] = res.pred == res.label
    res["pre_agrees"] = [norm(p) == l for p, l in zip(res.pretrained, res.label)]
    res["unsure"] = res.agree & (res.min_conf < 0.5)
    # priority: 1 = the CV model AND the pretrained model both disagree with the label; 2 = the CV model disagrees; 3 = agrees but unsure
    res["priority"] = 0
    res.loc[res.unsure, "priority"] = 3
    res.loc[~res.agree, "priority"] = 2
    res.loc[~res.agree & ~res.pre_agrees, "priority"] = 1
    flagged = res[res.priority > 0].sort_values(["priority", "label", "min_conf"]).reset_index(drop=True)
    audit = res[res.priority == 0].sample(min(audit_n, int((res.priority == 0).sum())), random_state=SEED).sort_values("label").reset_index(drop=True)
    flagged.assign(decision="", corrected_label="").to_csv(OUT / "review_flagged.csv", index=False)
    audit.assign(decision="", corrected_label="").to_csv(OUT / "review_audit.csv", index=False)
    print(f"real crops {len(res)} | trusted (model agrees, confident) {int((res.priority == 0).sum())} | to review: {len(flagged)} "
          f"[priority 1: {int((flagged.priority == 1).sum())}, 2: {int((flagged.priority == 2).sum())}, 3: {int((flagged.priority == 3).sum())}] | audit sample {len(audit)}")
    cols, cw, ch = 5, 230, 110
    for name, d in (("flagged", flagged), ("audit", audit)):
        for s in range(0, len(d), per_sheet):
            chunk = d.iloc[s:s + per_sheet]
            sh = Image.new("RGB", (cols * cw, ((len(chunk) + cols - 1) // cols) * (ch + 34)), "white")
            dr = ImageDraw.Draw(sh)
            for i, r in enumerate(chunk.itertuples()):
                im = Image.open(r.image_path).convert("RGB")
                im.thumbnail((cw - 8, ch))
                x, y = (i % cols) * cw, (i // cols) * (ch + 34)
                sh.paste(im, (x + 4, y + 34))
                dr.text((x + 4, y + 2), f"#{s + i}  label={r.label!r}", fill=(0, 0, 0))
                dr.text((x + 4, y + 16), f"model={r.pred!r} (p{r.priority})" if name == "flagged" else f"model={r.pred!r}", fill=(200, 0, 0))
            sh.save(OUT / f"{name}_{s // per_sheet:02d}.png")
    print("sheets written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sheets"])
    {"train": train, "sheets": sheets}[ap.parse_args().cmd]()
