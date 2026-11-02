"""Model artifacts must reject executable pickle while retaining native round trips."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from core.memory_bank import MemoryBank
from methods.adaptive_dual_memory.prototype_memory import DefectPrototypeMemory
from methods.defect_calibration.defect_direction import DefectDirection


LOADERS = [MemoryBank.load, DefectDirection.load, DefectPrototypeMemory.load]


class ExecutableArtifact:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return open, (str(self.marker), "w")


@pytest.mark.parametrize("loader", LOADERS)
def test_loader_rejects_pickle_side_effect(loader, tmp_path: Path) -> None:
    artifact = tmp_path / "untrusted.pt"
    marker = tmp_path / "unexpected_execution.txt"
    torch.save(ExecutableArtifact(marker), artifact)
    with pytest.raises(pickle.UnpicklingError):
        loader(artifact)
    assert not marker.exists()


def test_native_memory_bank_round_trip(tmp_path: Path) -> None:
    bank = MemoryBank(device="cpu", backend="sklearn")
    features = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    bank.fit(features)
    path = tmp_path / "normal.pt"
    bank.save(path, {"knn_k": 1, "name": "synthetic"})
    restored, metadata = MemoryBank.load(path, device="cpu", backend="sklearn")
    np.testing.assert_array_equal(restored.features, features)
    assert metadata["name"] == "synthetic"


def test_native_defect_direction_round_trip(tmp_path: Path) -> None:
    original = DefectDirection(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), 0.25, {"count": 2})
    path = original.save(tmp_path / "direction.pt")
    restored = DefectDirection.load(path)
    torch.testing.assert_close(restored.direction, original.direction)
    assert restored.alpha == 0.25
    assert restored.metadata == {"count": 2}


def test_native_defect_prototypes_round_trip(tmp_path: Path) -> None:
    original = DefectPrototypeMemory(torch.tensor([[1.0, 0.0]]), torch.tensor([0.75]), torch.tensor([0.5]), 0.1, 0.2, {"count": 1})
    path = original.save(tmp_path / "prototypes.pt")
    restored = DefectPrototypeMemory.load(path)
    torch.testing.assert_close(restored.prototypes, original.prototypes)
    torch.testing.assert_close(restored.reliability, original.reliability)
    assert restored.metadata == {"count": 1}
