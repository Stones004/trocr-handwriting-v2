"""Model construction, transforms and checkpoint I/O for the noise classifier, shared by train.py and predict.py.

Backbone is a torchvision ResNet-18 pretrained on ImageNet, fine-tuned with a 2-way head (clean=0, noise=1). Crops
are small (commonly well under 100px) and grayscale; they're upsampled to img_size and replicated to 3 channels so
the pretrained stem still applies -- transfer learning from ImageNet consistently beats training a CNN from scratch
on a dataset this size (~7k images).
"""
import torch
from torch import nn
from torchvision import models, transforms

IMG_SIZE = 128
CLASSES = ["clean", "noise"]


def make_transform(img_size=IMG_SIZE, train=False):
    ops = [transforms.Grayscale(num_output_channels=3), transforms.Resize((img_size, img_size))]
    if train:
        ops += [transforms.RandomRotation(5), transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def save_checkpoint(model, path, img_size=IMG_SIZE):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "img_size": img_size, "classes": CLASSES}, path)


def load_model(checkpoint, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("img_size", IMG_SIZE)
