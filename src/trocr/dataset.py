from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class HandwritingDataset(Dataset):
    def __init__(self, csv_file, processor, max_target_length=32):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
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
