"""PyTorch Dataset + transforms for HAM10000."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def _load_resized(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


class HAMDataset(Dataset):
    """Caches images (resized to cache_size) in RAM for fast epochs."""

    def __init__(self, df, transform, cache_size: int = 256):
        self.labels = df["label"].tolist()
        self.paths = df["path"].tolist()
        self.transform = transform
        with ThreadPoolExecutor(max_workers=8) as ex:
            self.images = list(ex.map(lambda p: _load_resized(p, cache_size), self.paths))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return self.transform(img), self.labels[i]


def denormalize(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) uint8 image."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    x = (t.cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
