from pathlib import Path
import argparse
import random
import re

import pandas as pd
import yaml


AMBIGUOUS_LABELS = {"ambigous", "ambiguous"}

# Excel/XML embeds a literal escape like _x000D_ (carriage return),
# _x000A_ (line feed), etc. when a cell has an internal line break and
# the reader doesn't decode it. Strip these out rather than dropping
# the whole row -- the remaining text is often still valid ground truth
# (e.g. "(iv)_x000D_" -> "(iv)").
EXCEL_ARTIFACT_PATTERN = re.compile(r"_x000[0-9A-Fa-f]_")

# Matches the "clean" symbol-grammar labels this project is actually
# built around: a digit run (optionally with a single trailing subpart
# letter, e.g. "6b"), a Roman numeral, or a bare subpart letter --
# optionally wrapped in parens and/or with trailing "." / ")" / ".)".
# Anything that doesn't match (annotator notes like "Ans 4 (a)",
# multi-token combos with internal spaces, stray artifacts like "|")
# is real content but not part of the closed symbol vocabulary this
# pipeline constrains decoding to -- it gets routed to freetext.csv
# instead of train/val/test, same treatment as ambiguous rows.
CLEAN_PATTERN = re.compile(
    r"^\(?(?:[0-9]+[a-jA-J]?|[IVXivx]+|[a-jA-J])(?:\.\)|[.)])?\)?\Z"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    annotation_file = Path(cfg["annotation_file"])
    image_dir = Path(cfg["image_dir"])
    filename_col = cfg["filename_column"]
    label_col = cfg["label_column"]

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            "Edit image_dir in config.yaml."
        )

    df = pd.read_excel(annotation_file)

    for col in (filename_col, label_col):
        if col not in df.columns:
            raise ValueError(
                f"Missing column {col!r}. "
                f"Available columns: {list(df.columns)}"
            )

    # IMPORTANT:
    # We intentionally preserve duplicate rows.
    work = df[[filename_col, label_col]].copy()

    work[filename_col] = work[filename_col].astype("string").str.strip()
    work[label_col] = work[label_col].astype("string").str.strip()

    # Strip Excel line-break artifacts (_x000D_ etc.) that leak through
    # when a cell has an internal line break. Log how many rows were
    # affected so it's visible, not silent.
    artifact_mask = work[label_col].str.contains(
        EXCEL_ARTIFACT_PATTERN, na=False
    )
    excel_artifact_count = int(artifact_mask.sum())

    work[label_col] = (
        work[label_col]
        .str.replace(EXCEL_ARTIFACT_PATTERN, "", regex=True)
        .str.strip()
    )

    # Remove only rows with no OCR target.
    valid_label = (
        work[label_col].notna()
        & (work[label_col] != "")
        & (work[label_col].str.lower() != "nan")
    )

    missing_label_count = int((~valid_label).sum())
    work = work.loc[valid_label].copy()

    # "ambigous" is a reviewer/annotation label, not handwritten
    # ground truth. Keep it separately and do not train on it.
    normalized_labels = work[label_col].str.lower()
    ambiguous_mask = normalized_labels.isin(AMBIGUOUS_LABELS)

    ambiguous = work.loc[ambiguous_mask].copy()
    work = work.loc[~ambiguous_mask].copy()

    # Split out free-text labels (real content, but outside the closed
    # symbol grammar this pipeline is built around) from the clean
    # symbol-pattern labels that go into train/val/test.
    clean_mask = work[label_col].apply(
        lambda text: bool(CLEAN_PATTERN.match(str(text)))
    )

    freetext = work.loc[~clean_mask].copy()
    work = work.loc[clean_mask].copy()

    # Resolve image paths.
    work["image_path"] = work[filename_col].map(
        lambda filename: str(image_dir / str(filename))
    )

    exists_mask = work["image_path"].map(
        lambda path: Path(path).is_file()
    )

    missing_images = work.loc[
        ~exists_mask, filename_col
    ].tolist()

    work = work.loc[exists_mask].copy()

    # Rename to the schema expected by dataset.py.
    work = work.rename(
        columns={
            filename_col: "filename",
            label_col: "text",
        }
    )

    work = work[
        ["filename", "image_path", "text"]
    ]

    # Preserve duplicates.
    # DO NOT use drop_duplicates().
    indices = list(range(len(work)))

    random.Random(
        cfg["split"]["seed"]
    ).shuffle(indices)

    n = len(indices)

    train_end = int(
        n * float(cfg["split"]["train"])
    )

    val_end = train_end + int(
        n * float(cfg["split"]["val"])
    )

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    out = Path("data/processed")
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    work.iloc[train_idx].to_csv(
        out / "train.csv",
        index=False
    )

    work.iloc[val_idx].to_csv(
        out / "val.csv",
        index=False
    )

    work.iloc[test_idx].to_csv(
        out / "test.csv",
        index=False
    )

    # Save ambiguous samples separately.
    if len(ambiguous):

        ambiguous = ambiguous.rename(
            columns={
                filename_col: "filename",
                label_col: "text",
            }
        )

        ambiguous["image_path"] = ambiguous[
            "filename"
        ].map(
            lambda filename:
            str(image_dir / str(filename))
        )

        ambiguous = ambiguous[
            ["filename", "image_path", "text"]
        ]

        ambiguous.to_csv(
            out / "ambiguous.csv",
            index=False
        )

    else:

        pd.DataFrame(
            columns=[
                "filename",
                "image_path",
                "text"
            ]
        ).to_csv(
            out / "ambiguous.csv",
            index=False
        )

    # Save free-text samples separately -- real content, outside the
    # closed symbol grammar. Kept out of train/val/test so they don't
    # dilute the vocab-constrained model; route these to unconstrained
    # decoding + mandatory human review at inference time instead.
    if len(freetext):

        freetext = freetext.rename(
            columns={
                filename_col: "filename",
                label_col: "text",
            }
        )

        freetext["image_path"] = freetext[
            "filename"
        ].map(
            lambda filename:
            str(image_dir / str(filename))
        )

        freetext = freetext[
            ["filename", "image_path", "text"]
        ]

        freetext.to_csv(
            out / "freetext.csv",
            index=False
        )

    else:

        pd.DataFrame(
            columns=[
                "filename",
                "image_path",
                "text"
            ]
        ).to_csv(
            out / "freetext.csv",
            index=False
        )

    # Save missing image references.
    if missing_images:

        pd.DataFrame(
            {"filename": missing_images}
        ).to_csv(
            out / "missing_images.csv",
            index=False
        )

    else:

        pd.DataFrame(
            {"filename": []}
        ).to_csv(
            out / "missing_images.csv",
            index=False
        )

    print(
        "=== Dataset preparation ==="
    )

    print(
        f"Original rows:              {len(df)}"
    )

    print(
        f"Excel artifacts cleaned:    "
        f"{excel_artifact_count}"
    )

    print(
        f"Missing/empty labels:       "
        f"{missing_label_count}"
    )

    print(
        f"Ambiguous rows excluded:    "
        f"{len(ambiguous)}"
    )

    print(
        f"Free-text rows excluded:    "
        f"{len(freetext)}"
    )

    print(
        f"Missing images:              "
        f"{len(missing_images)}"
    )

    print(
        f"Usable OCR rows:             "
        f"{len(work)}"
    )

    print(
        f"Train:                       "
        f"{len(train_idx)}"
    )

    print(
        f"Validation:                  "
        f"{len(val_idx)}"
    )

    print(
        f"Test:                        "
        f"{len(test_idx)}"
    )

    print(
        "Duplicates were NOT removed."
    )

    print(
        f"Output directory:            {out}"
    )

    print(
        f"Ambiguous file:              "
        f"{out / 'ambiguous.csv'}"
    )

    print(
        f"Free-text file:              "
        f"{out / 'freetext.csv'}"
    )

    print(
        f"Missing-image report:        "
        f"{out / 'missing_images.csv'}"
    )


if __name__ == "__main__":
    main()