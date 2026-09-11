"""
Checks how much upscaling TrOCR's processor is doing across your dataset.
TrOCR resizes every crop to a fixed internal size (384x384 for
trocr-small-handwritten) before encoding -- if your source crops are much
smaller than that, every single one gets blurred by interpolation before
the model ever sees it. This tells you how widespread that is.

Usage: python check_resolution.py
"""
import pandas as pd
from PIL import Image

TARGET_SIZE = 384  # confirm against your actual processor if unsure:
# from transformers import TrOCRProcessor
# p = TrOCRProcessor.from_pretrained("checkpoints/best")
# print(p.image_processor.size)

widths, heights = [], []
for split in ("train", "val", "test"):
    df = pd.read_csv(f"data/processed/{split}.csv")
    for path in df["image_path"]:
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

widths = pd.Series(widths)
heights = pd.Series(heights)

print(f"Total crops checked: {len(widths)}")
print(f"\nWidth  -- min: {widths.min()}, median: {widths.median():.0f}, max: {widths.max()}")
print(f"Height -- min: {heights.min()}, median: {heights.median():.0f}, max: {heights.max()}")

below_target = ((widths < TARGET_SIZE) | (heights < TARGET_SIZE)).sum()
print(f"\nCrops smaller than {TARGET_SIZE}x{TARGET_SIZE} in either dimension: "
      f"{below_target}/{len(widths)} ({below_target/len(widths)*100:.1f}%)")

severely_small = ((widths < TARGET_SIZE // 2) | (heights < TARGET_SIZE // 2)).sum()
print(f"Crops smaller than {TARGET_SIZE//2}x{TARGET_SIZE//2} (2x+ upscale needed): "
      f"{severely_small}/{len(widths)} ({severely_small/len(widths)*100:.1f}%)")