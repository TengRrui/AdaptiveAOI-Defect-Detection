from __future__ import annotations

import numpy as np
import torch

from methods.adaptive_dual_memory import DefectPrototypeMemory, DualMemoryScorer, mine_defect_prototypes


def test_defect_prototypes_are_compact_separate_and_roundtrip(tmp_path):
    generator = torch.Generator().manual_seed(7)
    defect_sets = [
        torch.nn.functional.normalize(torch.randn(64, 16, generator=generator) + 0.35, p=2, dim=1)
        for _ in range(6)
    ]
    defect_distances = [np.linspace(0.2, 1.0, 64, dtype=np.float32) for _ in defect_sets]
    normal_distances = np.linspace(0.05, 0.5, 512, dtype=np.float32)
    memory = mine_defect_prototypes(
        defect_sets,
        defect_distances,
        normal_distances,
        max_prototypes=8,
        min_image_support=1,
        seed=7,
    )
    assert 1 <= memory.count <= 8
    assert memory.metadata["normal_memory_contaminated"] is False
    target = memory.save(tmp_path / "defect_prototypes.pt")
    restored = DefectPrototypeMemory.load(target)
    assert torch.allclose(memory.prototypes, restored.prototypes)
    assert torch.allclose(memory.reliability, restored.reliability)


def test_reliability_gate_never_reduces_normal_memory_patch_score():
    prototypes = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
    memory = DefectPrototypeMemory(
        prototypes=prototypes,
        reliability=torch.tensor([0.9, 0.8]),
        radius=torch.tensor([0.5, 0.5]),
        normal_floor=0.3,
        normal_scale=0.1,
        metadata={"normal_memory_contaminated": False},
    )
    features = torch.nn.functional.normalize(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), p=2, dim=1)
    normal_scores = torch.tensor([0.8, 0.6])
    scored = DualMemoryScorer(memory).score(features, normal_scores, alpha=0.5, gated=True)
    assert torch.all(scored >= normal_scores)
    assert memory.metadata["normal_memory_contaminated"] is False
