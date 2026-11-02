"""Chunked tensor search must handle a neighbour count larger than early chunks."""
import pytest
import torch

from core.memory_bank import MemoryBank


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_neighbours_accumulate_across_small_chunks(device: str) -> None:
    features = torch.arange(8, dtype=torch.float32, device=device).reshape(-1, 1)
    bank = MemoryBank(knn_k=6, device=device, torch_gpu_chunk_size=2)
    # The tensor kernel can be checked on CPU without requiring a CUDA index.
    bank.features = features.cpu().numpy()
    bank._gpu_features = features
    bank._gpu_feature_norms = (features * features).sum(dim=1)
    distances, indices = bank._search_torch_gpu_flat(torch.tensor([[0.0]], device=device))
    torch.testing.assert_close(distances.cpu(), torch.tensor([[0.0, 1.0, 4.0, 9.0, 16.0, 25.0]]))
    assert indices.cpu().tolist() == [[0, 1, 2, 3, 4, 5]]
