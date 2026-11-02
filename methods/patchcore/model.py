from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.anomaly_scorer import image_score, patch_scores_to_map
from core.domain_build_contract import default_feature_layers
from core.feature_extractor import PatchFeatureExtractor
from core.memory_bank import MemoryBank


class FastPatchCoreAOI:
    def __init__(self, config: dict):
        self.config = config
        self.extractor = PatchFeatureExtractor(
            backbone=config.get("backbone", "wide_resnet50_2"),
            layers=config.get("layers", default_feature_layers()),
            device=config.get("device", "cuda"),
            output_root=config.get("output_root", "outputs"),
            use_fp16=bool(config.get("use_fp16", False)),
            non_blocking_transfer=bool(config.get("non_blocking_transfer", True)),
        )
        self.memory_bank: MemoryBank | None = None
        self.metadata = {}

    @property
    def device(self):
        return self.extractor.device

    @torch.inference_mode()
    def extract(self, image_tensor: torch.Tensor, timer=None) -> tuple[torch.Tensor, tuple[int, int]]:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return self.extractor(image_tensor, timer=timer)

    def load_memory(self, memory_path: str | Path):
        self.memory_bank, self.metadata = MemoryBank.load(
            memory_path,
            backend=self.config.get("memory_backend", "auto"),
            device=self.device,
            fallback_backend=self.config.get("memory_backend_fallback", "faiss_cpu_flat"),
            ivf_nlist=int(self.config.get("memory_ivf_nlist", 256)),
            ivf_nprobe=int(self.config.get("memory_ivf_nprobe", 32)),
            torch_gpu_chunk_size=int(self.config.get("torch_gpu_chunk_size", 0)),
        )
        self.metadata = {**self.metadata, **self.memory_bank.runtime_metadata()}
        return self

    def set_memory(self, bank: MemoryBank, metadata: dict):
        self.memory_bank = bank
        self.metadata = metadata
        return self

    def score_tensor(self, image_tensor: torch.Tensor) -> dict:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is not loaded.")
        patches, patch_shape = self.extract(image_tensor)
        scores = self.memory_bank.score(patches[0])
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        score_map = patch_scores_to_map(scores, patch_shape)
        return {
            "patch_scores": scores,
            "score_map": score_map,
            "image_score": image_score(score_map),
            "patch_shape": patch_shape,
        }

    def threshold(self) -> float:
        return float(self.metadata.get("threshold", self.config.get("threshold", 0.5)))

    def predict_label(self, score: float) -> int:
        return int(float(score) >= self.threshold())

    def close_build_runtime(self) -> None:
        """Release a temporary build extractor before its worker thread exits.

        CUDA module/tensor destruction during Python thread teardown can wait
        inside the driver while still holding the interpreter lock on Windows.
        Target-domain builds use a short-lived model, so move that model back
        to CPU explicitly after all feature work has completed.  The persisted
        Memory Bank and its dtype are unaffected.
        """

        self.memory_bank = None
        device = self.extractor.device
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            self.extractor.to("cpu")
            self.extractor.device = torch.device("cpu")
            torch.cuda.empty_cache()
