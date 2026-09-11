from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Light augmentation for handwritten single-token crops. Deliberately
# excludes flips (horizontal/vertical) -- those change character identity
# for this vocab (a flipped '6' can read as '9', mirrored letters become
# different letters), which corrupts the label rather than diversifying
# it. Kept small in magnitude since a rare-class row may get resampled
# many times per epoch by the WeightedRandomSampler in train.py -- these
# transforms exist to make those repeats visually distinct, not to
# simulate wildly different handwriting.
TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomApply(
        [transforms.RandomRotation(degrees=4)], p=0.5
    ),
    transforms.RandomApply(
        [transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))], p=0.3
    ),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
    ),
])


def resize_with_padding(image, target_size=384, pad_value=255):
    """
    Aspect-ratio-preserving resize onto a square canvas of side target_size,
    instead of TrOCRProcessor's default behavior of stretching directly to
    a square (non-uniform width/height scale factors on any non-square
    crop). Your dataset's crops range from 29-292px wide and 44-649px
    tall -- some far from square -- so a naive stretch meaningfully warps
    character shape on top of the separate blur-from-upscaling issue.
    This function doesn't remove the upscale (content still gets
    magnified to roughly fill target_size, since aspect-preserving means
    scale = min(target/w, target/h)) -- it only removes the UNEVEN
    stretch distortion, which is a real but different problem.

    PIL adaptation of the cv2-based version in test_preprocessing.py --
    same logic, ported so it can run directly on the PIL images this
    pipeline already loads, without adding an opencv dependency here.
    """
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new(image.mode, (target_size, target_size), pad_value)
    offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def get_target_size(processor):
    """
    Reads the processor's actual configured input size rather than
    hardcoding 384 -- stays correct if the model/processor ever changes.
    Falls back to 384 (trocr-small-handwritten's default) if the size
    isn't reported in the expected shape.
    """
    size = getattr(processor.image_processor, "size", None)
    if isinstance(size, dict):
        return size.get("height") or size.get("shortest_edge") or 384
    if isinstance(size, int):
        return size
    return 384


class HandwritingDataset(Dataset):
    def __init__(self, csv_file, processor, max_target_length=32, augment=False):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.target_size = get_target_size(processor)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        if self.augment:
            image = TRAIN_AUGMENT(image)
        image = resize_with_padding(image, target_size=self.target_size)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            str(row["text"]), padding="max_length",
            max_length=self.max_target_length, truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }