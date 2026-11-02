from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.anomaly_scorer import image_score, percentile_threshold
from core.coreset import select_coreset
from core.domain_build_contract import default_feature_layers
from core.memory_bank import MemoryBank, sample_patches
from datasets.base_dataset import AOIManifestDataset, collate_batch
from methods.patchcore.model import FastPatchCoreAOI
from tools.path_utils import memory_dir


def manifest_path_for(config: dict, dataset: str) -> Path:
    return Path(config["output_root"]) / "results" / f"manifest_{dataset}.csv"


def build_memory(config: dict, category: str | None = None) -> dict:
    dataset = str(config["dataset"]).lower()
    category = category or str(config["category"])
    manifest_path = manifest_path_for(config, dataset)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/build_manifest.py first.")

    train_set = AOIManifestDataset(
        manifest_path,
        image_size=int(config["image_size"]),
        dataset=dataset,
        category=category,
        split="train",
        label=0,
        return_mask=False,
    )
    if len(train_set) == 0:
        raise RuntimeError(f"No train normal samples found for {dataset}/{category}.")

    loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_batch,
    )
    model = FastPatchCoreAOI(config)
    start = time.perf_counter()
    sampled_chunks = []
    original_memory_size = 0
    ratio = float(config.get("coreset_ratio", 0.1))
    max_total = int(config.get("max_memory_patches", 50000))
    per_image_max = max(1, max_total // max(1, len(train_set)))
    seed = int(config.get("random_seed", 42))

    strategy = str(config.get("coreset_strategy", "random")).lower()
    selector_metadata = {"coreset_strategy": "random", "coreset_selector_version": "legacy-random"}
    # Retain the original code path for the established random baseline.
    if strategy == "random":
        for batch_idx, batch in enumerate(loader):
            patches, _ = model.extract(batch["image"])
            flat = patches.reshape(-1, patches.shape[-1]).detach().cpu()
            original_memory_size += int(flat.shape[0])
            sampled = sample_patches(flat, ratio=ratio, max_count=per_image_max * int(batch["image"].shape[0]), seed=seed + batch_idx)
            sampled_chunks.append(sampled)
        memory_features = torch.cat(sampled_chunks, dim=0)
        if memory_features.shape[0] > max_total:
            memory_features = sample_patches(memory_features, ratio=1.0, max_count=max_total, seed=seed)
    else:
        # Keep a bounded, balanced normal candidate pool before applying the more
        # expensive selector.  IDs make the spatial selector reproducible.
        per_image_candidate = int(config.get("coreset_pre_sample_per_image", 64))
        candidate_limit = int(config.get("coreset_candidate_limit", max_total))
        ids, spatial_ids = [], []
        for batch_idx, batch in enumerate(loader):
            patches, patch_shape = model.extract(batch["image"])
            for image_offset in range(patches.shape[0]):
                flat = patches[image_offset].reshape(-1, patches.shape[-1]).detach().cpu()
                original_memory_size += int(flat.shape[0])
                keep = min(len(flat), max(1, per_image_candidate))
                index = torch.randperm(len(flat), generator=torch.Generator().manual_seed(seed + batch_idx * 1009 + image_offset))[:keep]
                sampled_chunks.append(flat[index])
                row = torch.arange(patch_shape[0]).repeat_interleave(patch_shape[1])
                col = torch.arange(patch_shape[1]).repeat(patch_shape[0])
                bins = (row // max(1, patch_shape[0] // 4)) * 4 + (col // max(1, patch_shape[1] // 4))
                ids.append(np.full(keep, batch_idx * int(patches.shape[0]) + image_offset, dtype=np.int64))
                spatial_ids.append(bins[index].numpy().astype(np.int64))
        candidate_features = torch.cat(sampled_chunks, dim=0)
        candidate_ids = np.concatenate(ids)
        candidate_spatial = np.concatenate(spatial_ids)
        if len(candidate_features) > candidate_limit:
            pre_indices = np.random.default_rng(seed).choice(len(candidate_features), candidate_limit, replace=False)
            candidate_features = candidate_features[torch.from_numpy(pre_indices)]
            candidate_ids, candidate_spatial = candidate_ids[pre_indices], candidate_spatial[pre_indices]
        memory_features, selector_metadata = select_coreset(
            candidate_features,
            target_size=max_total,
            strategy=strategy,
            seed=seed,
            image_ids=candidate_ids,
            spatial_ids=candidate_spatial,
            kmeans_batch_size=int(config.get("coreset_kmeans_batch_size", 1024)),
            kmeans_max_iter=int(config.get("coreset_kmeans_max_iter", 50)),
        )

    bank = MemoryBank(knn_k=int(config.get("knn_k", 1))).fit(memory_features)
    model.set_memory(bank, {})

    threshold_scores = []
    limit = int(config.get("threshold_sample_limit", 80))
    for idx in range(min(len(train_set), limit)):
        sample = train_set[idx]
        result = model.score_tensor(sample["image"])
        threshold_scores.append(image_score(result["score_map"]))
    threshold = percentile_threshold(threshold_scores, float(config.get("threshold_percentile", 99.0)))

    build_time = time.perf_counter() - start
    metadata = {
        "baseline": "FastPatchCore-AOI",
        "dataset": dataset,
        "category": category,
        "backbone": model.extractor.backbone_name,
        "requested_backbone": config.get("backbone", "wide_resnet50_2"),
        "feature_layers": list(config.get("layers", default_feature_layers())),
        "image_size": int(config["image_size"]),
        "original_memory_size": int(original_memory_size),
        "coreset_memory_size": int(memory_features.shape[0]),
        "feature_dim": int(memory_features.shape[1]),
        "build_time": float(build_time),
        "sample_count": int(len(train_set)),
        "knn_k": int(config.get("knn_k", 1)),
        "coreset_ratio": ratio,
        "threshold_method": config.get("threshold_method", "normal_percentile"),
        "threshold_percentile": float(config.get("threshold_percentile", 99.0)),
        "threshold": float(threshold),
        "threshold_sample_count": int(len(threshold_scores)),
        **selector_metadata,
    }
    out_dir = memory_dir(config, dataset, category)
    memory_path = out_dir / "memory_bank.pt"
    bank.save(memory_path, metadata)
    pd.DataFrame([metadata]).to_csv(out_dir / "memory_metadata.csv", index=False, encoding="utf-8-sig")
    metadata["memory_path"] = str(memory_path)
    return metadata
