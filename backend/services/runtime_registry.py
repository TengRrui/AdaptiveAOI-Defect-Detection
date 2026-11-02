from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from backend.config import IntegrationSettings
from backend.errors import ApiError


def _stamp(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


@dataclass
class RuntimeEntry:
    key: str
    model_id: str
    dataset: str
    category: str
    device: str
    inferencer: Any
    created_at: str
    use_count: int
    active_count: int
    lock: threading.RLock


class RuntimeRegistry:
    """Thread-safe LRU cache for real PatchCore inferencers and memory banks."""

    def __init__(self, settings: IntegrationSettings):
        self.settings = settings
        self._entries: OrderedDict[str, RuntimeEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._execution_lock = threading.RLock()

    def resolve_device(self, requested: str | None) -> tuple[str, list[str]]:
        # Keep the API import path light.  PyTorch and the algorithm runtime are
        # loaded by the background warm-up thread (or the first real job), not
        # while Uvicorn is still trying to expose the liveness endpoint.
        import torch
        from core.runtime import select_device

        request = str(requested or self.settings.dynamic_settings().get("device") or self.settings.defaults.get("device", "auto")).lower()
        warnings: list[str] = []
        if request not in {"auto", "cpu", "cuda"}:
            raise ApiError("INVALID_DEVICE", "device must be one of auto, cpu, or cuda.", 422)
        if request == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif request == "cuda":
            selected = select_device("cuda")
            resolved = selected.type
            if resolved != "cuda":
                if bool(self.settings.runtime.get("allow_cpu_fallback", True)):
                    warnings.append("CUDA is unavailable in the algorithm environment; inference uses CPU.")
                else:
                    raise ApiError("CUDA_UNAVAILABLE", "CUDA was requested but is unavailable.", 409)
        else:
            resolved = "cpu"
        return resolved, warnings

    @staticmethod
    def _key(spec: dict[str, Any]) -> str:
        payload = {
            "model_id": spec["model_id"],
            "dataset": spec["dataset"],
            "category": spec["category"],
            "device": spec["device"],
            "use_fp16": bool(spec["config"].get("use_fp16", False)),
            "backbone": spec["config"].get("backbone"),
            "layers": spec["config"].get("layers"),
            "memory": str(spec["memory_path"]),
            "memory_mtime": _stamp(Path(spec["memory_path"])),
            "calibration": str(spec.get("calibration_path") or ""),
            "calibration_mtime": _stamp(Path(spec["calibration_path"])) if spec.get("calibration_path") else None,
            "defect_direction": str(spec.get("defect_direction_path") or ""),
            "defect_direction_mtime": _stamp(Path(spec["defect_direction_path"])) if spec.get("defect_direction_path") else None,
            "defect_prototypes": str(spec.get("defect_prototype_path") or ""),
            "defect_prototypes_mtime": _stamp(Path(spec["defect_prototype_path"])) if spec.get("defect_prototype_path") else None,
            "lightweight_calibrator": str(spec.get("lightweight_calibrator_path") or ""),
            "lightweight_calibrator_mtime": _stamp(Path(spec["lightweight_calibrator_path"])) if spec.get("lightweight_calibrator_path") else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def has_model(self, model_id: str, dataset: str, category: str) -> bool:
        with self._lock:
            return any(entry.model_id == model_id and entry.dataset == dataset and entry.category == category for entry in self._entries.values())

    def get_or_create(
        self,
        spec: dict[str, Any],
        *,
        progress_callback: Callable[[str, str, int], None] | None = None,
    ) -> tuple[RuntimeEntry, bool]:
        key = self._key(spec)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry, True
            memory_path = Path(spec["memory_path"])
            if not memory_path.exists():
                raise ApiError("MEMORY_BANK_MISSING", "所选数据集和类别的记忆库不存在。请先构建 memory_bank.pt，或在界面中选择已有的类别。", 409, {"expected_path": str(memory_path)})
            config = dict(spec["config"])
            config["output_root"] = str(self.settings.runtime_root / "cache_artifacts" / spec["dataset"] / spec["category"])
            config["device"] = spec["device"]
            if progress_callback is not None:
                progress_callback("initializing_python_runtime", "正在加载检测算法运行时组件", 30)
            from methods.patchcore.infer import PatchCoreInferencer

            if progress_callback is not None:
                progress_callback("loading_model", "正在初始化 GPU 计算环境并加载基础模型", 45)

            inferencer = PatchCoreInferencer(
                config,
                category=spec["category"],
                memory_path=memory_path,
                progress_callback=progress_callback,
            )
            entry = RuntimeEntry(
                key=key,
                model_id=spec["model_id"],
                dataset=spec["dataset"],
                category=spec["category"],
                device=spec["device"],
                inferencer=inferencer,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                use_count=0,
                active_count=0,
                lock=threading.RLock(),
            )
            self._entries[key] = entry
            self._evict_if_needed(protected_key=key)
            return entry, False

    def _evict_if_needed(self, protected_key: str | None = None) -> None:
        limit = max(1, int(self.settings.runtime.get("max_cached_runtimes", 1)))
        while len(self._entries) > limit:
            candidate_key = next(
                (
                    key
                    for key, value in self._entries.items()
                    if key != protected_key and value.active_count == 0
                ),
                None,
            )
            if candidate_key is None:
                # Active/leased runtimes may temporarily exceed the cache
                # target.  They are reconsidered when the last lease exits.
                break
            entry = self._entries.pop(candidate_key)
            entry.inferencer.visual_save_queue.close()
        if not self._entries:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @contextmanager
    def execution(self, entry: RuntimeEntry, output_root: Path, threshold: float | None) -> Iterator[Any]:
        execution_lock = self._execution_lock if bool(self.settings.runtime.get("serialize_inference", True)) else entry.lock
        with self._lock:
            if self._entries.get(entry.key) is not entry:
                raise ApiError("RUNTIME_EVICTED", "推理资源刚刚发生切换，请重新提交检测任务。", 409)
            entry.active_count += 1
            self._entries.move_to_end(entry.key)
        with execution_lock, entry.lock:
            inferencer = entry.inferencer
            original_root = inferencer.config.get("output_root")
            original_threshold = inferencer.threshold_value
            inferencer.config["output_root"] = str(output_root)
            if threshold is not None:
                inferencer.threshold_value = float(threshold)
            try:
                entry.use_count += 1
                yield inferencer
            finally:
                inferencer.config["output_root"] = original_root
                inferencer.threshold_value = original_threshold
                with self._lock:
                    entry.active_count = max(0, entry.active_count - 1)
                    self._evict_if_needed()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Serialize adaptation against image, batch, and video inference."""
        with self._execution_lock:
            yield

    def memory_path_in_use(self, memory_path: str | Path) -> bool:
        target = Path(memory_path).resolve()
        with self._lock:
            return any(
                entry.active_count > 0 and Path(entry.inferencer.memory_path).resolve() == target
                for entry in self._entries.values()
            )

    def invalidate(self, dataset: str | None = None, category: str | None = None, *, fail_on_active: bool = True) -> int:
        with self._lock:
            keys = [key for key, entry in self._entries.items() if (dataset is None or entry.dataset == dataset) and (category is None or entry.category == category)]
            active = [key for key in keys if self._entries[key].active_count > 0]
            if active and fail_on_active:
                raise ApiError(
                    "RUNTIME_IN_USE",
                    "当前推理资源正在被检测任务使用，任务结束前不能释放或替换。",
                    409,
                    {"active_runtime_count": len(active)},
                )
            keys = [key for key in keys if key not in active]
            for key in keys:
                entry = self._entries.pop(key)
                entry.inferencer.visual_save_queue.close()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return len(keys)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cached_runtime_count": len(self._entries),
                "runtimes": [
                    {
                        "model_id": entry.model_id,
                        "dataset": entry.dataset,
                        "category": entry.category,
                        "device": entry.device,
                        "created_at": entry.created_at,
                        "use_count": entry.use_count,
                        "active_count": entry.active_count,
                    }
                    for entry in self._entries.values()
                ],
            }

    def shutdown(self) -> None:
        # Process teardown must not close queues underneath a worker that is
        # still unwinding.  The operating system reclaims those resources.
        self.invalidate(fail_on_active=False)
