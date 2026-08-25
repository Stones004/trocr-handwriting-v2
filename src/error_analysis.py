import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-small-handwritten"


def normalize_for_comparison(text):
    return str(text).strip()


def levenshtein_ops(reference, prediction):
    """
    Return edit distance plus substitution/deletion/insertion counts.
    """
    reference = list(reference)
    prediction = list(prediction)

    n = len(reference)
    m = len(prediction)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if reference[i - 1] == prediction[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    # Backtrack to classify operations.
    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and reference[i - 1] == prediction[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1

        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            substitutions += 1
            i -= 1
            j -= 1

        elif (
            i > 0
            and dp[i][j] == dp[i - 1][j] + 1
        ):
            deletions += 1
            i -= 1

        else:
            insertions += 1
            j -= 1

    return (
        dp[n][m],
        substitutions,
        deletions,
        insertions,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="checkpoints/best",
    )

    parser.add_argument(
        "--csv",
        default="data/processed/test.csv",
    )

    parser.add_argument(
        "--output",
        default="error_analysis.csv",
    )

    parser.add_argument(
        "--beams",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TrOCR ERROR ANALYSIS")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print("\nLoading processor...")

    processor = TrOCRProcessor.from_pretrained(
        args.model
    )

    print("Loading model...")

    model = VisionEncoderDecoderModel.from_pretrained(
        args.model
    )

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Load test set
    # ---------------------------------------------------------

    df = pd.read_csv(args.csv)

    print(
        "\nTest samples:",
        len(df),
    )

    results = []

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    for index, row in df.iterrows():

        image_path = row["image_path"]

        target = normalize_for_comparison(
            row["text"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        pixel_values = processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(device)

        with torch.no_grad():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=16,
                num_beams=args.beams,
                early_stopping=True,
            )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        prediction = normalize_for_comparison(
            prediction
        )

        (
            distance,
            substitutions,
            deletions,
            insertions,
        ) = levenshtein_ops(
            target,
            prediction,
        )

        exact = (
            target == prediction
        )

        results.append(
            {
                "filename": row["filename"],
                "image_path": image_path,
                "target": target,
                "prediction": prediction,
                "exact_match": exact,
                "edit_distance": distance,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
            }
        )

        print(
            f"[{index + 1:03d}/{len(df):03d}] "
            f"GT={target!r:<10} "
            f"PRED={prediction!r:<10} "
            f"{'OK' if exact else 'ERROR'}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        args.output,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    exact_accuracy = (
        results_df["exact_match"].mean()
    )

    total_chars = sum(
        len(x)
        for x in results_df["target"]
    )

    total_distance = (
        results_df["edit_distance"].sum()
    )

    cer = (
        total_distance / total_chars
        if total_chars
        else 0.0
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Exact match: "
        f"{exact_accuracy:.4f} "
        f"({exact_accuracy * 100:.2f}%)"
    )

    print(
        f"CER: "
        f"{cer:.4f} "
        f"({cer * 100:.2f}%)"
    )

    print(
        "\nTotal substitutions:",
        results_df["substitutions"].sum(),
    )

    print(
        "Total deletions:",
        results_df["deletions"].sum(),
    )

    print(
        "Total insertions:",
        results_df["insertions"].sum(),
    )

    # ---------------------------------------------------------
    # Failed predictions
    # ---------------------------------------------------------

    errors = results_df[
        ~results_df["exact_match"]
    ].copy()

    print(
        "\nFailed samples:",
        len(errors),
    )

    print("\n" + "-" * 70)
    print("FAILED PREDICTIONS")
    print("-" * 70)

    for _, row in errors.iterrows():

        print(
            f"\n{row['filename']}"
        )

        print(
            f"  Target:     {row['target']!r}"
        )

        print(
            f"  Prediction: {row['prediction']!r}"
        )

        print(
            f"  Distance:   {row['edit_distance']}"
        )

        print(
            f"  Sub:        {row['substitutions']}"
        )

        print(
            f"  Delete:     {row['deletions']}"
        )

        print(
            f"  Insert:     {row['insertions']}"
        )

    print(
        "\nFull results saved to:",
        Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()