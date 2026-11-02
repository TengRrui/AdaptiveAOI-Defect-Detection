from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | None, image_size: int | None = None) -> torch.Tensor | None:
    if not path:
        return None
    mask = Image.open(path).convert("L")
    if image_size is not None:
        mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    arr = (np.array(mask) > 0).astype("float32")
    return torch.from_numpy(arr).unsqueeze(0)
