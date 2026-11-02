from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


SUPPORTED_BACKENDS = {
    "auto",
    "faiss_gpu_flat",
    "faiss_gpu_ivf",
    "faiss_cpu_flat",
    "sklearn",
    "torch_gpu_flat",
}


def sample_patches(features: torch.Tensor, ratio: float, max_count: int | None = None, seed: int = 42) -> torch.Tensor:
    if features.ndim != 2:
        features = features.reshape(-1, features.shape[-1])
    total = features.shape[0]
    keep = max(1, int(total * float(ratio))) if ratio < 1 else total
    if max_count is not None:
        keep = min(keep, int(max_count))
    if keep >= total:
        return features
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:keep]
    return features.cpu()[indices]


class MemoryBank:
    """Patch memory with CPU FAISS/sklearn and optional GPU search backends.

    Serialized banks remain CPU feature tensors. Backend choice is runtime-only so an
    existing ``memory_bank.pt`` can be reused without mutation.
    """

    def __init__(
        self,
        knn_k: int = 1,
        backend: str = "auto",
        device: str | torch.device = "cuda",
        fallback_backend: str = "faiss_cpu_flat",
        ivf_nlist: int = 256,
        ivf_nprobe: int = 32,
        torch_gpu_chunk_size: int = 0,
    ):
        self.knn_k = int(knn_k)
        self.features: np.ndarray | None = None
        self.backend_requested = str(backend or "auto").lower()
        self.fallback_backend = str(fallback_backend or "faiss_cpu_flat").lower()
        self.device = torch.device(device)
        self.ivf_nlist = int(ivf_nlist)
        self.ivf_nprobe = int(ivf_nprobe)
        self.torch_gpu_chunk_size = int(torch_gpu_chunk_size)
        self.backend = "none"
        self.index = None
        self._gpu_features: torch.Tensor | None = None
        self._gpu_feature_norms: torch.Tensor | None = None
        self._faiss_resources = None
        self.backend_info: dict[str, object] = {}

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features is not None else 0

    @property
    def uses_gpu_queries(self) -> bool:
        return self.backend in {"faiss_gpu_flat", "faiss_gpu_ivf", "torch_gpu_flat"}

    @staticmethod
    def _import_faiss():
        import faiss

        return faiss

    @classmethod
    def faiss_gpu_available(cls) -> tuple[bool, str]:
        try:
            faiss = cls._import_faiss()
        except Exception as exc:
            return False, f"faiss import failed: {type(exc).__name__}: {exc}"
        required = ("StandardGpuResources", "index_cpu_to_gpu")
        missing = [name for name in required if not hasattr(faiss, name)]
        if missing:
            return False, f"faiss-gpu symbols unavailable: {', '.join(missing)}"
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is false"
        return True, "available"

    def _clear_runtime_index(self) -> None:
        self.index = None
        self._gpu_features = None
        self._gpu_feature_norms = None
        self._faiss_resources = None

    def _set_backend_info(self, started: float, **extra) -> None:
        self.backend_info = {
            **self.backend_info,
            "memory_backend_requested": self.backend_requested,
            "memory_backend": self.backend,
            "memory_feature_dim": self.feature_dim,
            "memory_feature_count": int(len(self.features)) if self.features is not None else 0,
            "memory_index_build_ms": float((time.perf_counter() - started) * 1000.0),
            "memory_ivf_nlist": int(self.ivf_nlist) if self.backend == "faiss_gpu_ivf" else 0,
            "memory_ivf_nprobe": int(self.ivf_nprobe) if self.backend == "faiss_gpu_ivf" else 0,
            **extra,
        }

    def _fit_faiss_cpu(self) -> None:
        faiss = self._import_faiss()
        index = faiss.IndexFlatL2(self.feature_dim)
        index.add(self.features)
        self.index = index
        self.backend = "faiss_cpu_flat"

    def _fit_sklearn(self) -> None:
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(n_neighbors=min(self.knn_k, len(self.features)), metric="euclidean")
        index.fit(self.features)
        self.index = index
        self.backend = "sklearn"

    def _fit_torch_gpu(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("torch_gpu_flat requires an available CUDA device")
        before = int(torch.cuda.memory_allocated(self.device))
        self._gpu_features = torch.as_tensor(self.features, dtype=torch.float32, device=self.device).contiguous()
        self._gpu_feature_norms = (self._gpu_features * self._gpu_features).sum(dim=1)
        torch.cuda.synchronize(self.device)
        after = int(torch.cuda.memory_allocated(self.device))
        self.backend = "torch_gpu_flat"
        self.backend_info["memory_index_gpu_bytes"] = max(0, after - before)

    def _fit_faiss_gpu_flat(self) -> None:
        faiss = self._import_faiss()
        available, reason = self.faiss_gpu_available()
        if not available:
            raise RuntimeError(reason)
        self._faiss_resources = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatL2(self.feature_dim)
        cpu_index.add(self.features)
        self.index = faiss.index_cpu_to_gpu(self._faiss_resources, self.device.index or 0, cpu_index)
        self.backend = "faiss_gpu_flat"

    def _fit_faiss_gpu_ivf(self) -> None:
        faiss = self._import_faiss()
        available, reason = self.faiss_gpu_available()
        if not available:
            raise RuntimeError(reason)
        nlist = min(max(1, self.ivf_nlist), len(self.features))
        if len(self.features) < 2:
            raise RuntimeError("faiss_gpu_ivf requires at least two memory features")
        quantizer = faiss.IndexFlatL2(self.feature_dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist)
        cpu_index.train(self.features)
        cpu_index.add(self.features)
        cpu_index.nprobe = min(max(1, self.ivf_nprobe), nlist)
        self._faiss_resources = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self._faiss_resources, self.device.index or 0, cpu_index)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = cpu_index.nprobe
        self.backend = "faiss_gpu_ivf"
        self.ivf_nlist = nlist
        self.ivf_nprobe = int(cpu_index.nprobe)

    def _fit_requested_backend(self, requested: str) -> None:
        if requested == "faiss_cpu_flat":
            self._fit_faiss_cpu()
        elif requested == "sklearn":
            self._fit_sklearn()
        elif requested == "torch_gpu_flat":
            self._fit_torch_gpu()
        elif requested == "faiss_gpu_flat":
            self._fit_faiss_gpu_flat()
        elif requested == "faiss_gpu_ivf":
            self._fit_faiss_gpu_ivf()
        else:
            raise ValueError(f"Unsupported memory backend: {requested}")

    def fit(self, features: torch.Tensor | np.ndarray):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        self.features = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
        if self.features.ndim != 2 or self.features.shape[0] == 0:
            raise ValueError(f"Memory features must have shape [N, D] with N > 0; got {self.features.shape}.")
        if self.backend_requested not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported memory backend '{self.backend_requested}'. Supported: {sorted(SUPPORTED_BACKENDS)}")
        if self.fallback_backend not in SUPPORTED_BACKENDS - {"auto"}:
            raise ValueError(f"Unsupported fallback backend '{self.fallback_backend}'.")

        self._clear_runtime_index()
        started = time.perf_counter()
        fallback_reason = None
        requested = "faiss_cpu_flat" if self.backend_requested == "auto" else self.backend_requested
        try:
            self._fit_requested_backend(requested)
        except Exception as exc:
            fallback_reason = f"{requested} failed: {type(exc).__name__}: {exc}"
            fallback = self.fallback_backend
            if requested.startswith("faiss_gpu") and fallback == "faiss_cpu_flat" and self.device.type == "cuda" and torch.cuda.is_available():
                # A CUDA exact search remains available on Windows environments with faiss-cpu only.
                fallback = "torch_gpu_flat"
            try:
                self._fit_requested_backend(fallback)
            except Exception as fallback_exc:
                if fallback != "faiss_cpu_flat":
                    self._fit_faiss_cpu()
                    fallback_reason += f"; {fallback} failed: {type(fallback_exc).__name__}: {fallback_exc}; used faiss_cpu_flat"
                else:
                    self._fit_sklearn()
                    fallback_reason += f"; faiss_cpu_flat failed: {type(fallback_exc).__name__}: {fallback_exc}; used sklearn"
        self._set_backend_info(started, memory_backend_fallback_reason=fallback_reason or "")
        self.backend_info.setdefault("memory_index_gpu_bytes", 0)
        return self

    def _as_numpy_query(self, patch_features: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(patch_features, torch.Tensor):
            query = patch_features.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            query = np.asarray(patch_features, dtype=np.float32)
        if query.ndim == 3:
            query = query.reshape(-1, query.shape[-1])
        if query.ndim != 2 or query.shape[1] != self.feature_dim:
            raise ValueError(f"Query feature dimension {query.shape[-1] if query.ndim else 'unknown'} does not match memory dimension {self.feature_dim}.")
        return np.ascontiguousarray(query)

    def _as_gpu_query(self, patch_features: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(patch_features, torch.Tensor):
            query = patch_features
            if query.ndim == 3:
                query = query.reshape(-1, query.shape[-1])
            query = query.to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            query = torch.as_tensor(np.asarray(patch_features, dtype=np.float32), device=self.device)
            if query.ndim == 3:
                query = query.reshape(-1, query.shape[-1])
        if query.ndim != 2 or query.shape[1] != self.feature_dim:
            raise ValueError(f"Query feature dimension {query.shape[-1] if query.ndim else 'unknown'} does not match memory dimension {self.feature_dim}.")
        return query.contiguous()

    def _search_torch_gpu_flat(self, patch_features: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._as_gpu_query(patch_features)
        if self._gpu_features is None or self._gpu_feature_norms is None:
            raise RuntimeError("torch_gpu_flat index is not initialized")
        k = min(self.knn_k, self._gpu_features.shape[0])
        chunk_size = self.torch_gpu_chunk_size or self._gpu_features.shape[0]
        best_distances = None
        best_indices = None
        for start in range(0, self._gpu_features.shape[0], chunk_size):
            bank_chunk = self._gpu_features[start:start + chunk_size]
            norm_chunk = self._gpu_feature_norms[start:start + chunk_size]
            distances = (query * query).sum(dim=1, keepdim=True) + norm_chunk.unsqueeze(0) - 2.0 * (query @ bank_chunk.T)
            chunk_values, chunk_indices = torch.topk(distances, k=min(k, bank_chunk.shape[0]), dim=1, largest=False)
            chunk_indices = chunk_indices + start
            if best_distances is None:
                best_distances, best_indices = chunk_values, chunk_indices
            else:
                combined_values = torch.cat([best_distances, chunk_values], dim=1)
                combined_indices = torch.cat([best_indices, chunk_indices], dim=1)
                best_distances, positions = torch.topk(combined_values, k=min(k, combined_values.shape[1]), dim=1, largest=False)
                best_indices = torch.gather(combined_indices, 1, positions)
        return best_distances, best_indices

    def _score_torch_gpu_flat(self, patch_features: torch.Tensor | np.ndarray) -> torch.Tensor:
        distances, _ = self._search_torch_gpu_flat(patch_features)
        return torch.sqrt(distances[:, 0].clamp_min(0.0))

    def _score_faiss_gpu(self, patch_features: torch.Tensor | np.ndarray):
        query = self._as_gpu_query(patch_features)
        try:
            # faiss.contrib.torch_utils enables zero-copy CUDA tensor search when faiss-gpu is installed.
            import faiss.contrib.torch_utils  # noqa: F401

            distances, _ = self.index.search(query, min(self.knn_k, len(self.features)))
            if isinstance(distances, torch.Tensor):
                return torch.sqrt(distances[:, 0].clamp_min(0.0))
        except Exception as exc:
            self.backend_info["memory_gpu_query_adapter_error"] = f"{type(exc).__name__}: {exc}"
        raise RuntimeError("FAISS GPU tensor search is unavailable; select torch_gpu_flat or a CPU backend.")

    def score(self, patch_features: torch.Tensor | np.ndarray) -> np.ndarray | torch.Tensor:
        if self.features is None:
            raise RuntimeError("Memory bank is not fitted.")
        if self.backend == "torch_gpu_flat":
            return self._score_torch_gpu_flat(patch_features)
        if self.backend in {"faiss_gpu_flat", "faiss_gpu_ivf"}:
            return self._score_faiss_gpu(patch_features)

        query = self._as_numpy_query(patch_features)
        if self.backend == "faiss_cpu_flat":
            distances, _ = self.index.search(query, min(self.knn_k, len(self.features)))
            scores = np.sqrt(np.maximum(distances[:, 0], 0.0))
        elif self.backend == "sklearn":
            distances, _ = self.index.kneighbors(query)
            scores = distances[:, 0]
        else:
            raise RuntimeError(f"Memory backend is not initialized: {self.backend}")
        return scores.astype(np.float32)

    def nearest_indices(self, patch_features: torch.Tensor | np.ndarray) -> np.ndarray | torch.Tensor:
        """Return top-1 memory indices for CPU/GPU numerical consistency checks."""
        if self.features is None:
            raise RuntimeError("Memory bank is not fitted.")
        if self.backend == "torch_gpu_flat":
            _, indices = self._search_torch_gpu_flat(patch_features)
            return indices[:, 0]
        if self.backend in {"faiss_gpu_flat", "faiss_gpu_ivf"}:
            query = self._as_gpu_query(patch_features)
            try:
                import faiss.contrib.torch_utils  # noqa: F401

                _, indices = self.index.search(query, 1)
                if isinstance(indices, torch.Tensor):
                    return indices[:, 0]
            except Exception as exc:
                raise RuntimeError(f"FAISS GPU tensor index query failed: {type(exc).__name__}: {exc}") from exc
            raise RuntimeError("FAISS GPU tensor index query is unavailable")
        query = self._as_numpy_query(patch_features)
        if self.backend == "faiss_cpu_flat":
            _, indices = self.index.search(query, 1)
        elif self.backend == "sklearn":
            _, indices = self.index.kneighbors(query, n_neighbors=1)
        else:
            raise RuntimeError(f"Memory backend is not initialized: {self.backend}")
        return indices[:, 0].astype(np.int64)

    def runtime_metadata(self) -> dict[str, object]:
        return dict(self.backend_info)

    def save(self, path: str | Path, metadata: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": torch.from_numpy(self.features), "metadata": metadata}, path)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **runtime_options):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        bank = cls(knn_k=int(payload.get("metadata", {}).get("knn_k", 1)), **runtime_options)
        bank.fit(payload["features"])
        return bank, payload.get("metadata", {})
