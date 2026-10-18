"""PyTorch Dataset for the noise/clean crop classifier. Reads a split CSV written by prepare_dataset.py
(columns: filename, image_path, label, doc) and yields (image_tensor, label) pairs.
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class NoiseDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), torch.tensor(int(row["label"]), dtype=torch.long)

    @property
    def labels(self):
        return self.df["label"].to_numpy()


def class_weights(csv_path):
    """Inverse-frequency weight per class [clean, noise], for a class-balanced loss."""
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    n = counts.sum()
    return torch.tensor([n / (2 * max(c, 1)) for c in counts], dtype=torch.float32)
