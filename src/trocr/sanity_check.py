import torch
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-small-handwritten"
CSV_PATH = "data/processed/train.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("=" * 60)
    print("TrOCR GPU SANITY CHECK")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Device
    # ---------------------------------------------------------

    print("\n[1] Device")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

    print("Using:", DEVICE)

    # ---------------------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------------------

    print("\n[2] Loading dataset")

    df = pd.read_csv(CSV_PATH)

    print("Training samples:", len(df))

    # Take only a tiny batch
    batch_df = df.head(20)

    print("\nSamples:")

    for _, row in batch_df.iterrows():

        print(
            f"  {row['filename']} -> "
            f"{row['text']!r}"
        )

    # ---------------------------------------------------------
    # 3. Load processor
    # ---------------------------------------------------------

    print("\n[3] Loading TrOCR processor")

    processor = TrOCRProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Processor loaded successfully.")

    # ---------------------------------------------------------
    # 4. Load model
    # ---------------------------------------------------------

    print("\n[4] Loading TrOCR model")

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME
    )

    # TrOCR generation/training configuration
    model.config.decoder_start_token_id = (
        processor.tokenizer.cls_token_id
    )

    model.config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        processor.tokenizer.sep_token_id
    )

    model.to(DEVICE)

    model.train()

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # 5. Prepare images
    # ---------------------------------------------------------

    print("\n[5] Preparing images")

    pixel_values = []
    labels = []

    for _, row in batch_df.iterrows():

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        print(
            f"  {row['filename']}: "
            f"{image.size} -> {row['text']!r}"
        )

        pixels = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        tokenized = processor.tokenizer(
            str(row["text"]),
            padding="max_length",
            max_length=16,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        tokenized[
            tokenized == processor.tokenizer.pad_token_id
        ] = -100

        pixel_values.append(pixels)
        labels.append(tokenized)

    pixel_values = torch.stack(
        pixel_values
    ).to(DEVICE)

    labels = torch.stack(
        labels
    ).to(DEVICE)

    print(
        "\nPixel tensor:",
        tuple(pixel_values.shape)
    )

    print(
        "Label tensor:",
        tuple(labels.shape)
    )

    # ---------------------------------------------------------
    # 6. Forward pass
    # ---------------------------------------------------------

    print("\n[6] Forward pass")

    outputs = model(
        pixel_values=pixel_values,
        labels=labels
    )

    print(
        "Loss:",
        outputs.loss.item()
    )

    # ---------------------------------------------------------
    # 7. Backward pass
    # ---------------------------------------------------------

    print("\n[7] Backward pass")

    outputs.loss.backward()

    print(
        "Backward pass completed successfully."
    )

    # ---------------------------------------------------------
    # 8. Test generation
    # ---------------------------------------------------------

    print("\n[8] Generation test")

    model.eval()

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=16,
            num_beams=2
        )

    predictions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("\nPredictions:")

    for i, prediction in enumerate(predictions):

        print(
            f"  Target: {batch_df.iloc[i]['text']!r}"
        )

        print(
            f"  Output: {prediction!r}"
        )

    # ---------------------------------------------------------
    # Done
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()