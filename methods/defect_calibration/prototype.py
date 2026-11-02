from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def l2_normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(vec.float(), p=2, dim=-1, eps=eps)


def prototype(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 1:
        return l2_normalize(features)
    if features.numel() == 0:
        raise ValueError("Cannot build prototype from empty features.")
    return l2_normalize(features.float().mean(dim=0))


def mask_to_patch_selector(mask_path: str | Path, patch_shape: tuple[int, int], min_value: float = 0.25) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    arr = (np.array(mask) > 0).astype(np.float32)
    resized = cv2.resize(arr, (patch_shape[1], patch_shape[0]), interpolation=cv2.INTER_AREA)
    return resized.reshape(-1) >= float(min_value)


def topk_selector(score_map: np.ndarray, top_ratio: float = 0.05, min_patches: int = 4) -> np.ndarray:
    flat = score_map.reshape(-1)
    k = max(int(flat.size * float(top_ratio)), int(min_patches))
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    selector = np.zeros(flat.size, dtype=bool)
    selector[idx] = True
    return selector


def projection_scores(features: torch.Tensor, normal_proto: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
    centered = features.float().cpu() - normal_proto.float().cpu().view(1, -1)
    direction = l2_normalize(direction.float().cpu()).view(1, -1)
    score = torch.matmul(centered, direction.t()).squeeze(1)
    score = torch.relu(score)
    return score.numpy().astype(np.float32)
