from __future__ import annotations

from pathlib import Path

from datasets.base_dataset import AOIManifestDataset
from datasets.manifest import build_dagm_manifest


class DAGMDataset(AOIManifestDataset):
    def __init__(self, root: str | Path, image_size: int = 512, category: str | None = None, split: str | None = None, label: int | None = None):
        manifest, _ = build_dagm_manifest(Path(root))
        super().__init__(manifest, image_size=image_size, dataset="dagm", category=category, split=split, label=label)
