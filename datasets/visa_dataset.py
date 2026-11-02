from __future__ import annotations

from pathlib import Path

from datasets.base_dataset import AOIManifestDataset
from datasets.manifest import build_visa_manifest


class VisADataset(AOIManifestDataset):
    def __init__(self, root: str | Path, data_root: str | Path | None = None, image_size: int = 512, category: str | None = None, split: str | None = None, label: int | None = None):
        manifest = build_visa_manifest(Path(root), Path(data_root) if data_root else None)
        super().__init__(manifest, image_size=image_size, dataset="visa", category=category, split=split, label=label)
