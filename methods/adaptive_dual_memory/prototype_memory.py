from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class DefectPrototypeMemory:
    """A compact memory that is physically separate from the normal memory bank."""

    prototypes: torch.Tensor
    reliability: torch.Tensor
    radius: torch.Tensor
    normal_floor: float
    normal_scale: float
    metadata: dict

    def __post_init__(self) -> None:
        count = int(self.prototypes.shape[0])
        if self.prototypes.ndim != 2 or count == 0:
            raise ValueError("Defect prototypes must have shape [K, D], K > 0.")
        if self.reliability.numel() != count or self.radius.numel() != count:
            raise ValueError("Prototype reliability/radius count mismatch.")

    @property
    def count(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.prototypes.shape[1])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "prototypes": self.prototypes.detach().cpu().float(),
                "reliability": self.reliability.detach().cpu().float(),
                "radius": self.radius.detach().cpu().float(),
                "normal_floor": float(self.normal_floor),
                "normal_scale": float(self.normal_scale),
                "metadata": dict(self.metadata),
            },
            tmp,
        )
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DefectPrototypeMemory":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(
            prototypes=payload["prototypes"].float(),
            reliability=payload["reliability"].float(),
            radius=payload["radius"].float(),
            normal_floor=float(payload["normal_floor"]),
            normal_scale=float(payload["normal_scale"]),
            metadata=dict(payload.get("metadata", {})),
        )


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _euclidean_to_rows(features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(features * features, axis=1, keepdims=True)
        + np.sum(rows * rows, axis=1)[None, :]
        - 2.0 * features @ rows.T
    )
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32, copy=False)


def mine_defect_prototypes(
    defect_feature_sets: Iterable[torch.Tensor | np.ndarray],
    defect_normal_distances: Iterable[torch.Tensor | np.ndarray],
    normal_patch_distances: torch.Tensor | np.ndarray,
    *,
    max_prototypes: int = 32,
    top_ratio: float = 0.05,
    min_patches_per_image: int = 4,
    floor_percentile: float = 99.0,
    min_image_support: int = 2,
    seed: int = 0,
) -> DefectPrototypeMemory:
    """Mine weakly supervised defect patches and compress them into reliable medoids.

    Only image-level defect labels are assumed. Candidate patches must be among the
    most abnormal patches in their source image and exceed a normal-derived floor.
    The resulting memory is never merged into the normal memory bank.
    """

    feature_sets = [_as_numpy(item) for item in defect_feature_sets]
    distance_sets = [_as_numpy(item).reshape(-1) for item in defect_normal_distances]
    if len(feature_sets) != len(distance_sets) or not feature_sets:
        raise ValueError("Defect features and normal-distance sets must be non-empty and aligned.")

    normal_distances = _as_numpy(normal_patch_distances).reshape(-1)
    if normal_distances.size == 0:
        raise ValueError("Normal patch-distance reference is empty.")
    normal_floor = float(np.percentile(normal_distances, float(floor_percentile)))
    normal_hi = float(np.percentile(normal_distances, min(99.9, float(floor_percentile) + 0.8)))
    normal_scale = max(normal_hi - normal_floor, float(np.std(normal_distances)), 1e-4)

    candidate_features: list[np.ndarray] = []
    candidate_sources: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    for source_index, (features, distances) in enumerate(zip(feature_sets, distance_sets)):
        if features.ndim != 2 or len(features) != len(distances):
            raise ValueError("Each defect feature set must have shape [P, D] and match its distances.")
        keep = min(len(distances), max(int(min_patches_per_image), int(np.ceil(len(distances) * float(top_ratio)))))
        top_indices = np.argpartition(distances, -keep)[-keep:]
        selected = top_indices[distances[top_indices] >= normal_floor]
        if selected.size == 0:
            selected = top_indices[np.argsort(distances[top_indices])[-max(1, min(min_patches_per_image, keep)):]]
        candidate_features.append(features[selected])
        candidate_sources.append(np.full(len(selected), source_index, dtype=np.int32))
        candidate_scores.append(distances[selected].astype(np.float32, copy=False))

    candidates = np.concatenate(candidate_features, axis=0)
    sources = np.concatenate(candidate_sources, axis=0)
    candidate_normal_scores = np.concatenate(candidate_scores, axis=0)
    if len(candidates) == 0:
        raise RuntimeError("No defect patch candidates survived mining.")
    candidates = _as_numpy(F.normalize(torch.from_numpy(candidates), p=2, dim=1))

    cluster_count = min(int(max_prototypes), max(1, len(candidates) // max(4, int(min_patches_per_image))))
    clusterer = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        batch_size=min(1024, max(32, len(candidates))),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.0,
    )
    assignments = clusterer.fit_predict(candidates)
    centers = _as_numpy(F.normalize(torch.from_numpy(clusterer.cluster_centers_.astype(np.float32)), p=2, dim=1))

    medoids: list[np.ndarray] = []
    reliabilities: list[float] = []
    radii: list[float] = []
    support_rows: list[dict] = []
    for cluster_index in range(cluster_count):
        member_indices = np.flatnonzero(assignments == cluster_index)
        if member_indices.size == 0:
            continue
        distinct_sources = int(np.unique(sources[member_indices]).size)
        member_features = candidates[member_indices]
        center = centers[cluster_index : cluster_index + 1]
        center_distances = _euclidean_to_rows(member_features, center).reshape(-1)
        medoid = member_features[int(np.argmin(center_distances))]
        global_index = int(member_indices[int(np.argmin(center_distances))])
        # The medoid retains its measured distance from the normal memory. This
        # evidence is used only for reliability and is never written to that bank.
        separation = float(candidate_normal_scores[global_index])
        separation_score = float(np.clip((separation - normal_floor) / normal_scale, 0.0, 1.0))
        support_score = float(np.sqrt(distinct_sources / max(1, len(feature_sets))))
        reliability = float(np.clip(0.65 * separation_score + 0.35 * support_score, 0.0, 1.0))
        radius = float(max(np.percentile(center_distances, 90.0), 1e-4))
        medoids.append(medoid)
        reliabilities.append(reliability)
        radii.append(radius)
        support_rows.append(
            {
                "cluster": int(cluster_index),
                "patch_count": int(member_indices.size),
                "image_support": distinct_sources,
                "reliability": reliability,
                "radius": radius,
            }
        )

    if not medoids:
        raise RuntimeError("Prototype clustering produced no usable medoids.")
    order = np.argsort(-np.asarray(reliabilities, dtype=np.float32))
    supported = [int(index) for index in order if support_rows[int(index)]["image_support"] >= int(min_image_support)]
    if not supported:
        supported = [int(index) for index in order]
    supported = supported[: int(max_prototypes)]
    prototype_tensor = F.normalize(torch.from_numpy(np.stack([medoids[index] for index in supported])).float(), p=2, dim=1)
    reliability_tensor = torch.tensor([reliabilities[index] for index in supported], dtype=torch.float32)
    radius_tensor = torch.tensor([radii[index] for index in supported], dtype=torch.float32)
    metadata = {
        "method": "weakly_supervised_defect_patch_mining",
        "candidate_patch_count": int(len(candidates)),
        "prototype_count": int(len(supported)),
        "defect_image_count": int(len(feature_sets)),
        "top_ratio": float(top_ratio),
        "floor_percentile": float(floor_percentile),
        "normal_floor": normal_floor,
        "normal_scale": normal_scale,
        "min_image_support": int(min_image_support),
        "seed": int(seed),
        "clusters": [support_rows[index] for index in supported],
        "normal_memory_contaminated": False,
    }
    return DefectPrototypeMemory(
        prototypes=prototype_tensor,
        reliability=reliability_tensor,
        radius=radius_tensor,
        normal_floor=normal_floor,
        normal_scale=normal_scale,
        metadata=metadata,
    )
