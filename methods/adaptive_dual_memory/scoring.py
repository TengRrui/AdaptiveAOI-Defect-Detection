from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .prototype_memory import DefectPrototypeMemory


def _as_tensor(value: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


class DualMemoryScorer:
    """Relative dual-memory score with an optional reliability gate."""

    def __init__(self, memory: DefectPrototypeMemory, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.prototypes = F.normalize(memory.prototypes.to(self.device).float(), p=2, dim=1)
        self.reliability = memory.reliability.to(self.device).float().clamp(0.0, 1.0)
        self.radius = memory.radius.to(self.device).float().clamp_min(1e-4)
        self.normal_floor = float(memory.normal_floor)
        self.normal_scale = max(float(memory.normal_scale), 1e-4)

    @torch.inference_mode()
    def components(
        self,
        patch_features: torch.Tensor | np.ndarray,
        normal_distances: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor]:
        features = F.normalize(_as_tensor(patch_features, self.device), p=2, dim=1)
        normal = _as_tensor(normal_distances, self.device).reshape(-1)
        # Both operands are unit-normalized, so Euclidean distance can be
        # recovered from one compact matrix product. This avoids the generic
        # cdist kernel overhead for a tiny prototype bank.
        cosine = features @ self.prototypes.T
        distances = torch.sqrt(torch.clamp(2.0 - 2.0 * cosine, min=0.0))
        defect, nearest = torch.min(distances, dim=1)
        reliability = self.reliability[nearest]
        radius = self.radius[nearest]
        contrast = torch.relu(normal - defect)
        normal_gate = torch.sigmoid((normal - self.normal_floor) / self.normal_scale)
        radius_gate = torch.sigmoid((radius - defect) / (0.15 * radius + 1e-4))
        gate = reliability * normal_gate * radius_gate
        return {
            "normal": normal,
            "defect": defect,
            "contrast": contrast,
            "gate": gate,
            "nearest_prototype": nearest,
        }

    @torch.inference_mode()
    def score(
        self,
        patch_features: torch.Tensor | np.ndarray,
        normal_distances: torch.Tensor | np.ndarray,
        *,
        alpha: float,
        gated: bool = True,
    ) -> torch.Tensor:
        parts = self.components(patch_features, normal_distances)
        modifier = parts["gate"] if gated else torch.ones_like(parts["gate"])
        return parts["normal"] + float(alpha) * modifier * parts["contrast"]


@torch.inference_mode()
def score_defect_direction(
    patch_features: torch.Tensor | np.ndarray,
    normal_distances: torch.Tensor | np.ndarray,
    normal_prototype: torch.Tensor,
    defect_prototype: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    device = patch_features.device if isinstance(patch_features, torch.Tensor) else torch.device("cpu")
    features = _as_tensor(patch_features, device)
    normal = _as_tensor(normal_distances, device).reshape(-1)
    normal_proto = F.normalize(normal_prototype.to(device).float(), p=2, dim=0)
    defect_proto = F.normalize(defect_prototype.to(device).float(), p=2, dim=0)
    direction = F.normalize(defect_proto - normal_proto, p=2, dim=0)
    projection = torch.relu((features - normal_proto) @ direction)
    return normal + float(alpha) * projection
