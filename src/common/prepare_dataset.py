from pathlib import Path
import argparse
import random

import pandas as pd
import yaml


AMBIGUOUS_LABELS = {"ambigous", "ambiguous"}


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
        f"Missing/empty labels:       "
        f"{missing_label_count}"
    )

    print(
        f"Ambiguous rows excluded:    "
        f"{len(ambiguous)}"
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
        f"Missing-image report:        "
        f"{out / 'missing_images.csv'}"
    )


if __name__ == "__main__":
    main()