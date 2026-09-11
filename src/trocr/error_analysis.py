import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from constraints import (
    get_closed_vocab_processor,
    get_tracking_closed_vocab_processor,
    get_logits_processor_from_csvs,
    transition_confidence,
)
from dataset import resize_with_padding, get_target_size


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

    parser.add_argument(
        "--no-constrain",
        action="store_true",
        help="disable constrained decoding entirely",
    )

    parser.add_argument(
        "--charmask",
        action="store_true",
        help="use the old flat label-token mask instead of the closed-vocabulary trie (weaker, kept for comparison)",
    )

    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.5,
        help="flag a sample for review if its min post-mask token confidence falls below this",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="calibration temperature from calibrate_temperature.py (1.0 = unscaled)",
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
    # Vocab-constrained decoding
    # ---------------------------------------------------------

    logits_processor = None
    tracking_processor = None

    if not args.no_constrain:
        csv_paths = ["data/processed/train.csv", "data/processed/val.csv"]

        if args.charmask:
            logits_processor = get_logits_processor_from_csvs(processor, csv_paths)
        else:
            logits_processor = get_closed_vocab_processor(processor, model, csv_paths)

        # The raw pre-mask top1/top2 margin (useful for flagging genuinely
        # ambiguous glyphs, e.g. 6 vs b, or catching confidently-wrong
        # predictions the post-mask confidence misses) is only reliably
        # attributable in greedy decoding -- see constraints.py docstring.
        # Only build the tracking variant when --beams 1 is passed.
        if args.beams == 1 and not args.charmask:
            tracking_processor = get_tracking_closed_vocab_processor(
                processor, model, csv_paths
            )
            from transformers import LogitsProcessorList
            logits_processor = LogitsProcessorList([tracking_processor])

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

        image = resize_with_padding(
            image, target_size=get_target_size(processor)
        )

        pixel_values = processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(device)

        if tracking_processor is not None:
            tracking_processor.reset()

        with torch.no_grad():

            outputs = model.generate(
                pixel_values,
                max_new_tokens=16,
                num_beams=args.beams,
                early_stopping=True,
                logits_processor=logits_processor,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prediction = processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )[0]

        prediction = normalize_for_comparison(
            prediction
        )

        conf = transition_confidence(model, outputs, temperature=args.temperature)[0]
        min_conf = conf["seq_min_prob"]
        flagged = min_conf < args.flag_threshold

        margin_info = {"min_margin": None, "any_overridden": None}
        if tracking_processor is not None:
            margin_info = tracking_processor.raw_margin_confidence()
            if margin_info["min_margin"] < 0.1:
                flagged = True

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
                "min_confidence": min_conf,
                "mean_confidence": conf["seq_mean_prob"],
                "raw_min_margin": margin_info["min_margin"],
                "grammar_overrode_top_pick": margin_info["any_overridden"],
                "flagged_for_review": flagged,
            }
        )

        print(
            f"[{index + 1:03d}/{len(df):03d}] "
            f"GT={target!r:<10} "
            f"PRED={prediction!r:<10} "
            f"conf={min_conf:.2f} "
            f"{'FLAGGED ' if flagged else ''}"
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

    flagged_count = results_df["flagged_for_review"].sum()

    print(
        f"Flagged for review (min_confidence < {args.flag_threshold}"
        + (" or raw margin < 0.1" if args.beams == 1 else "")
        + "):",
        flagged_count,
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