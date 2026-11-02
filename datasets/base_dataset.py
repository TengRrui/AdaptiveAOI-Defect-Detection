from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from datasets.transforms import image_transform, load_mask, load_rgb


class AOIManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        image_size: int = 512,
        dataset: str | None = None,
        category: str | None = None,
        split: str | None = None,
        label: int | None = None,
        return_mask: bool = True,
    ):
        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            df = pd.read_csv(manifest)
        if dataset is not None:
            df = df[df["dataset"].astype(str).str.lower() == dataset.lower()]
        if category is not None:
            df = df[df["category"].astype(str) == category]
        if split is not None:
            df = df[df["split"].astype(str).str.lower() == split.lower()]
        if label is not None:
            df = df[df["label"].astype(int) == int(label)]
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.transform = image_transform(self.image_size)
        self.return_mask = return_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index].to_dict()
        image = load_rgb(item["image_path"])
        tensor = self.transform(image)
        mask_path = item.get("mask_path", "")
        mask = load_mask(mask_path, self.image_size) if self.return_mask and isinstance(mask_path, str) and mask_path else None
        item["label"] = int(item["label"])
        item["height"] = int(item["height"])
        item["width"] = int(item["width"])
        return {"image": tensor, "mask": mask, "meta": item}

    @property
    def categories(self) -> list[str]:
        return sorted(self.df["category"].astype(str).unique().tolist())


def collate_batch(batch):
    images = [b["image"] for b in batch]
    masks = [b["mask"] for b in batch]
    metas = [b["meta"] for b in batch]
    import torch

    return {"image": torch.stack(images, dim=0), "mask": masks, "meta": metas}
