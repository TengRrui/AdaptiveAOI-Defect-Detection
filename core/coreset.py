"""Reproducible feature subset selectors for PatchCore memory banks.

The original random selector remains the baseline.  The additional selectors are
intended for memory *coverage*, not for threshold or support-set calibration.
"""

from __future__ import annotations

import time

import numpy as np
import torch


SUPPORTED_CORESET_STRATEGIES = {"random", "stratified_spatial", "minibatch_kmeans"}


def _as_cpu_features(features: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = features.detach().cpu() if isinstance(features, torch.Tensor) else torch.as_tensor(features)
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N, D] features, got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32).contiguous()


def _random_indices(total: int, target: int, seed: int) -> np.ndarray:
    if target >= total:
        return np.arange(total, dtype=np.int64)
    return np.random.default_rng(seed).choice(total, size=target, replace=False).astype(np.int64)


def _stratified_spatial_indices(image_ids: np.ndarray, spatial_ids: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Balance samples across source images and patch-position bins.

    It deliberately never looks at labels or test data.  A deterministic shuffled
    round-robin prevents a large homogeneous image from consuming the full bank.
    """
    rng = np.random.default_rng(seed)
    total = len(image_ids)
    if target >= total:
        return np.arange(total, dtype=np.int64)
    groups: list[np.ndarray] = []
    for image_id in np.unique(image_ids):
        image_indices = np.flatnonzero(image_ids == image_id)
        # Spatial bins are shuffled within image, then interleaved.  This keeps
        # representation from different image regions when enough samples exist.
        per_bin = [image_indices[spatial_ids[image_indices] == bin_id] for bin_id in np.unique(spatial_ids[image_indices])]
        per_bin = [rng.permutation(items) for items in per_bin if len(items)]
        ordered: list[int] = []
        while any(len(items) for items in per_bin):
            for pos, items in enumerate(per_bin):
                if len(items):
                    ordered.append(int(items[0]))
                    per_bin[pos] = items[1:]
        groups.append(np.asarray(ordered, dtype=np.int64))
    groups = [groups[idx] for idx in rng.permutation(len(groups))]
    selected: list[int] = []
    cursor = 0
    while len(selected) < target and any(cursor < len(group) for group in groups):
        for group in groups:
            if cursor < len(group):
                selected.append(int(group[cursor]))
                if len(selected) == target:
                    break
        cursor += 1
    return np.asarray(selected, dtype=np.int64)


def select_coreset(
    features: torch.Tensor | np.ndarray,
    target_size: int,
    strategy: str = "random",
    seed: int = 42,
    image_ids: np.ndarray | None = None,
    spatial_ids: np.ndarray | None = None,
    kmeans_batch_size: int = 1024,
    kmeans_max_iter: int = 50,
) -> tuple[torch.Tensor, dict]:
    """Select a compact normal-feature bank and return reproducibility metadata.

    ``minibatch_kmeans`` returns centers rather than observed patches.  This is a
    legitimate compact normal prototype bank and avoids materialising expensive
    greedy pairwise distance matrices for large feature pools.
    """
    matrix = _as_cpu_features(features)
    strategy = str(strategy).lower()
    if strategy not in SUPPORTED_CORESET_STRATEGIES:
        raise ValueError(f"Unsupported coreset strategy '{strategy}'.")
    total = int(matrix.shape[0])
    target = min(max(1, int(target_size)), total)
    started = time.perf_counter()
    metadata = {
        "coreset_strategy": strategy,
        "coreset_seed": int(seed),
        "coreset_source_size": total,
        "coreset_target_size": target,
        "coreset_feature_dim": int(matrix.shape[1]),
        "coreset_selector_version": "balanced-v1",
    }
    if target == total:
        selected = matrix
        metadata["coreset_selection_kind"] = "all_features"
    elif strategy == "random":
        indices = _random_indices(total, target, seed)
        selected = matrix[torch.from_numpy(indices)]
        metadata.update({"coreset_selection_kind": "sampled_patches", "coreset_index_count": int(len(indices))})
    elif strategy == "stratified_spatial":
        if image_ids is None:
            image_ids = np.zeros(total, dtype=np.int64)
        if spatial_ids is None:
            spatial_ids = np.zeros(total, dtype=np.int64)
        image_ids = np.asarray(image_ids).reshape(-1)
        spatial_ids = np.asarray(spatial_ids).reshape(-1)
        if len(image_ids) != total or len(spatial_ids) != total:
            raise ValueError("image_ids and spatial_ids must align with features")
        indices = _stratified_spatial_indices(image_ids, spatial_ids, target, seed)
        selected = matrix[torch.from_numpy(indices)]
        metadata.update({
            "coreset_selection_kind": "stratified_observed_patches",
            "coreset_index_count": int(len(indices)),
            "coreset_source_images": int(len(np.unique(image_ids))),
            "coreset_spatial_bins": int(len(np.unique(spatial_ids))),
        })
    else:
        from sklearn.cluster import MiniBatchKMeans

        # sklearn consumes a NumPy view; the candidate pool is deliberately bounded
        # by the caller before reaching this selector.
        estimator = MiniBatchKMeans(
            n_clusters=target,
            batch_size=max(target, int(kmeans_batch_size)),
            max_iter=max(1, int(kmeans_max_iter)),
            n_init=1,
            # k-means++ becomes disproportionately expensive for thousands of
            # 1536-D patch centers. A fixed random initialisation keeps this
            # candidate practical and reproducible for the comparison protocol.
            init="random",
            random_state=int(seed),
            reassignment_ratio=0.0,
        )
        estimator.fit(matrix.numpy())
        selected = torch.from_numpy(estimator.cluster_centers_.astype(np.float32, copy=False))
        metadata.update({
            "coreset_selection_kind": "cluster_centers",
            "coreset_kmeans_batch_size": int(estimator.batch_size),
            "coreset_kmeans_max_iter": int(kmeans_max_iter),
            "coreset_kmeans_iterations": int(estimator.n_iter_),
            "coreset_kmeans_inertia": float(estimator.inertia_),
        })
    metadata["coreset_build_ms"] = float((time.perf_counter() - started) * 1000.0)
    return selected.contiguous(), metadata
