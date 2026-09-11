import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# Basic enhancement functions
# ============================================================

def unsharp(gray, sigma=1.0, amount=0.8):
    """Apply unsharp masking while preserving grayscale information."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)

    sharpened = cv2.addWeighted(
        gray,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# Aspect-ratio preserving resize
# ============================================================

def resize_with_padding(
    image,
    target_size=384,
    pad_value=255,
):
    """
    Resize while preserving aspect ratio, then place the
    result in the centre of a square white canvas.

    Example:
        101 x 173
            ↓
        proportional resize
            ↓
        fit inside 384 x 384
            ↓
        white padding
    """

    height, width = image.shape[:2]

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = max(
        1,
        round(width * scale),
    )

    new_height = max(
        1,
        round(height * scale),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    if image.ndim == 2:

        canvas = np.full(
            (
                target_size,
                target_size,
            ),
            pad_value,
            dtype=np.uint8,
        )

    else:

        canvas = np.full(
            (
                target_size,
                target_size,
                image.shape[2],
            ),
            pad_value,
            dtype=np.uint8,
        )

    x = (
        target_size - new_width
    ) // 2

    y = (
        target_size - new_height
    ) // 2

    canvas[
        y:y + new_height,
        x:x + new_width,
    ] = resized

    return canvas


# ============================================================
# Generate preprocessing variants
# ============================================================

def make_variants(image):
    """
    Generate both the original preprocessing experiments and
    the new aspect-ratio-preserving experiments.

    IMPORTANT:
    The aspect-ratio variants are explicitly resized to
    384x384 BEFORE being passed to the TrOCR processor.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    bilateral = cv2.bilateralFilter(
        gray,
        d=7,
        sigmaColor=35,
        sigmaSpace=35,
    )

    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )

    # --------------------------------------------------------
    # Aspect-ratio-preserving base
    # --------------------------------------------------------

    aspect_padding = resize_with_padding(
        image,
        target_size=384,
        pad_value=255,
    )

    aspect_gray = cv2.cvtColor(
        aspect_padding,
        cv2.COLOR_BGR2GRAY,
    )

    aspect_padding_sharpen = cv2.cvtColor(
        unsharp(
            aspect_gray,
            sigma=1.0,
            amount=0.8,
        ),
        cv2.COLOR_GRAY2BGR,
    )

    aspect_padding_clahe = cv2.cvtColor(
        clahe.apply(aspect_gray),
        cv2.COLOR_GRAY2BGR,
    )

    aspect_otsu = cv2.threshold(
        aspect_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    aspect_padding_otsu = cv2.cvtColor(
        aspect_otsu,
        cv2.COLOR_GRAY2BGR,
    )

    # --------------------------------------------------------
    # Return ALL variants
    # --------------------------------------------------------

    return {

        # Original baseline
        "original": image,

        # Original preprocessing experiments
        "grayscale": cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR,
        ),

        "mild_sharpen": cv2.cvtColor(
            unsharp(
                gray,
                sigma=1.0,
                amount=0.8,
            ),
            cv2.COLOR_GRAY2BGR,
        ),

        "strong_sharpen": cv2.cvtColor(
            unsharp(
                gray,
                sigma=1.0,
                amount=1.5,
            ),
            cv2.COLOR_GRAY2BGR,
        ),

        "clahe": cv2.cvtColor(
            clahe.apply(gray),
            cv2.COLOR_GRAY2BGR,
        ),

        "clahe_sharpen": cv2.cvtColor(
            unsharp(
                clahe.apply(gray),
                sigma=1.0,
                amount=0.8,
            ),
            cv2.COLOR_GRAY2BGR,
        ),

        "bilateral": cv2.cvtColor(
            bilateral,
            cv2.COLOR_GRAY2BGR,
        ),

        "bilateral_sharpen": cv2.cvtColor(
            unsharp(
                bilateral,
                sigma=1.0,
                amount=0.8,
            ),
            cv2.COLOR_GRAY2BGR,
        ),

        "otsu": cv2.cvtColor(
            otsu,
            cv2.COLOR_GRAY2BGR,
        ),

        "adaptive_threshold": cv2.cvtColor(
            adaptive,
            cv2.COLOR_GRAY2BGR,
        ),

        # NEW: geometry-preserving experiments
        "aspect_ratio_padding": aspect_padding,

        "aspect_padding_sharpen":
            aspect_padding_sharpen,

        "aspect_padding_clahe":
            aspect_padding_clahe,

        "aspect_padding_otsu":
            aspect_padding_otsu,
    }


# ============================================================
# TrOCR inference
# ============================================================

def predict(
    model,
    processor,
    image_bgr,
    device,
    num_beams=4,
):
    """Run TrOCR inference on one BGR image."""

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    pixel_values = processor(
        images=pil_image,
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=16,
            num_beams=num_beams,
            early_stopping=True,
        )

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return prediction


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare TrOCR preprocessing methods, "
            "including aspect-ratio-preserving padding."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image.",
    )

    parser.add_argument(
        "--model",
        default="checkpoints/best",
        help="TrOCR checkpoint.",
    )

    parser.add_argument(
        "--target",
        default=None,
        help="Known ground-truth label.",
    )

    parser.add_argument(
        "--beams",
        type=int,
        default=4,
        help="Number of beams.",
    )

    parser.add_argument(
        "--save-dir",
        default="preprocess_test_results",
        help="Directory for processed images.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("PREPROCESSING + ASPECT RATIO TEST")
    print("=" * 70)

    print(
        "Image:",
        args.image,
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        args.image
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {args.image}"
        )

    height, width = image.shape[:2]

    print(
        "Original size:",
        f"{width} x {height}",
    )

    print(
        "Original aspect ratio:",
        f"{width / height:.3f}",
    )

    # --------------------------------------------------------
    # Generate variants
    # --------------------------------------------------------

    variants = make_variants(
        image
    )

    print(
        f"\nGenerated {len(variants)} variants."
    )

    # --------------------------------------------------------
    # Load processor and model
    # --------------------------------------------------------

    print("\nLoading processor...")

    processor = (
        TrOCRProcessor.from_pretrained(
            args.model
        )
    )

    print("Loading model...")

    model = (
        VisionEncoderDecoderModel.from_pretrained(
            args.model
        )
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        args.save_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # Run all variants
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "PREDICTIONS"
    )

    print(
        "-" * 70
    )

    for name, variant in variants.items():

        output_path = (
            output_dir
            / f"{name}.png"
        )

        cv2.imwrite(
            str(output_path),
            variant,
        )

        prediction = predict(
            model=model,
            processor=processor,
            image_bgr=variant,
            device=device,
            num_beams=args.beams,
        )

        exact = False

        if args.target is not None:

            exact = (
                prediction
                == args.target.strip()
            )

        results.append(
            {
                "variant": name,
                "prediction": prediction,
                "exact": exact,
            }
        )

        marker = ""

        if args.target is not None:

            marker = (
                "  <-- EXACT MATCH"
                if exact
                else ""
            )

        print(
            f"{name:<28}"
            f" -> {prediction!r}"
            f"{marker}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    if args.target is not None:

        print(
            "Ground truth:",
            repr(args.target),
        )

        matches = [
            result
            for result in results
            if result["exact"]
        ]

        if matches:

            print(
                "\nExact-match variants:"
            )

            for result in matches:

                print(
                    "  -",
                    result["variant"],
                )

        else:

            print(
                "\nNo variant produced "
                "an exact match."
            )

    print(
        "\nProcessed images saved to:"
    )

    print(
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()