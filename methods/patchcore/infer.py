from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from tools.path_utils import ensure_dir, memory_dir


LIGHTWEIGHT_MODE = "normal_memory_defect_guided_lite"


class _LazyVisualSaveQueue:
    """Defer OpenCV/postprocess imports until the first real output is saved."""

    def __init__(self, *, enabled: bool, queue_size: int):
        self.enabled = bool(enabled)
        self.queue_size = int(queue_size)
        self._delegate = None

    def _ensure(self):
        if self._delegate is None:
            from core.postprocess import VisualSaveQueue

            self._delegate = VisualSaveQueue(enabled=self.enabled, queue_size=self.queue_size)
        return self._delegate

    def submit(self, payloads):
        return self._ensure().submit(payloads)

    def resolve(self, ticket):
        return self._ensure().resolve(ticket)

    def close(self):
        if self._delegate is not None:
            self._delegate.close()


def default_memory_path(config: dict, dataset: str | None = None, category: str | None = None) -> Path:
    return memory_dir(config, dataset or config["dataset"], category or config["category"]) / "memory_bank.pt"


def default_defect_direction_path(config: dict, dataset: str, category: str) -> Path:
    return Path(config["output_root"]) / "calibrations" / dataset / category / "defect_direction.pt"


def default_defect_prototype_path(config: dict, dataset: str, category: str) -> Path:
    return Path(config["output_root"]) / "calibrations" / dataset / category / "defect_prototypes.pt"


def default_lightweight_calibrator_path(config: dict, dataset: str, category: str) -> Path:
    return Path(config["output_root"]) / "calibrations" / dataset / category / "lightweight_calibrator.json"


def _patch_image_score(values: torch.Tensor | np.ndarray) -> float:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return image_score(np.asarray(values, dtype=np.float32).reshape(1, -1))


class PatchCoreInferencer:
    def __init__(
        self,
        config: dict,
        category: str | None = None,
        memory_path: str | Path | None = None,
        progress_callback: Callable[[str, str, int], None] | None = None,
    ):
        # Keep importing this module itself cheap.  The backend can expose
        # liveness and finish profile resolution without importing OpenCV,
        # torchvision transforms, tile inference, or visualization helpers.
        # These bindings are installed only when the one RuntimeRegistry owner
        # actually constructs the formal inferencer.
        global FastPatchCoreAOI, Image, RUNTIME_FIELDS, StageTimer, image_score, np, patch_scores_to_map, sync_if_cuda, timing_totals, torch
        import numpy as np
        import torch
        from PIL import Image
        from core.anomaly_scorer import image_score, patch_scores_to_map
        from core.runtime import RUNTIME_FIELDS, StageTimer, sync_if_cuda, timing_totals
        from methods.patchcore.model import FastPatchCoreAOI

        self.config = dict(config)
        if category is not None:
            self.config["category"] = category
        self.dataset = str(self.config["dataset"]).lower()
        self.category = str(self.config["category"])
        if progress_callback is not None:
            progress_callback("loading_model", "正在初始化 GPU 计算环境并加载基础模型", 45)
        started = time.perf_counter()
        self.model = FastPatchCoreAOI(self.config)
        sync_if_cuda(self.model.device)
        self.model_load_ms = float((time.perf_counter() - started) * 1000.0)
        self.memory_path = Path(memory_path) if memory_path else default_memory_path(self.config, self.dataset, self.category)
        sync_if_cuda(self.model.device)
        if progress_callback is not None:
            progress_callback("loading_memory_bank", "正在加载当前正式 Memory Bank", 75)
        started = time.perf_counter()
        self.model.load_memory(self.memory_path)
        sync_if_cuda(self.model.device)
        self.memory_bank_load_ms = float((time.perf_counter() - started) * 1000.0)
        if progress_callback is not None:
            # MemoryBank.load deserializes the immutable features and builds
            # the selected runtime index as one atomic operation.  Publish the
            # index boundary only after that operation has actually completed.
            progress_callback("loading_index", "正在加载并校验近邻检索索引", 90)
        self._apply_scoring_memory_limit()
        self.pin_memory = bool(self.config.get("pin_memory", False))
        self.visual_save_queue = _LazyVisualSaveQueue(
            enabled=bool(self.config.get("async_save_outputs", False)),
            queue_size=int(self.config.get("save_queue_size", 2)),
        )
        self.calibration = self._load_calibration()
        self.threshold_value = self._resolve_threshold()
        self.pixel_threshold_value = self._resolve_pixel_threshold()
        self.defect_direction = self._load_defect_direction()
        self.dual_memory = self._load_dual_memory()
        self.lightweight_profile = self._load_lightweight_profile()
        if progress_callback is not None:
            progress_callback("warming_up", "正在完成检测资源预热", 95)

    def _apply_scoring_memory_limit(self) -> None:
        limit = int(self.config.get("score_memory_limit", 0) or 0)
        if limit <= 0 or self.model.memory_bank is None or self.model.memory_bank.features is None:
            return
        features = self.model.memory_bank.features
        total = int(features.shape[0])
        if total <= limit:
            self.model.metadata["scoring_memory_size"] = total
            self.model.metadata["original_memory_size"] = total
            return
        seed = int(self.config.get("random_seed", 42))
        strategy = str(self.config.get("score_memory_strategy", "random")).lower()
        # Legacy random compression remains bit-for-bit compatible as the default.
        if strategy == "random":
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(total, size=limit, replace=False))
            selected = features[indices]
            selection_metadata = {"score_memory_strategy": "random", "score_memory_selection_kind": "legacy-random"}
        elif strategy == "minibatch_kmeans":
            from core.coreset import select_coreset

            selected_tensor, selection_metadata = select_coreset(
                features,
                target_size=limit,
                strategy="minibatch_kmeans",
                seed=seed,
                kmeans_batch_size=int(self.config.get("score_memory_kmeans_batch_size", 1024)),
                kmeans_max_iter=int(self.config.get("score_memory_kmeans_max_iter", 20)),
            )
            selected = selected_tensor.numpy()
            selection_metadata["score_memory_strategy"] = "minibatch_kmeans"
        else:
            # Existing serialized banks do not retain source-image/spatial IDs.
            # Do not present a random subset as spatially stratified; such a bank
            # must be rebuilt using build_memory with coreset_strategy instead.
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(total, size=limit, replace=False))
            selected = features[indices]
            selection_metadata = {
                "score_memory_strategy": "random",
                "score_memory_requested_strategy": strategy,
                "score_memory_fallback_reason": "serialized memory lacks image_ids/spatial_ids; rebuild with coreset_strategy for spatial stratification",
            }
        self.model.memory_bank.fit(selected)
        self.model.metadata["original_memory_size"] = total
        self.model.metadata["scoring_memory_size"] = int(limit)
        self.model.metadata.update(selection_metadata)

    def _load_calibration(self) -> dict | None:
        if not bool(self.config.get("use_calibrated_threshold", False)):
            return None
        from core.calibration import default_calibration_path, load_calibration

        path = Path(self.config.get("calibration_path") or default_calibration_path(self.config, self.dataset, self.category))
        if path.exists():
            return load_calibration(path)
        return None

    def _resolve_threshold(self) -> float:
        if self.calibration and "image_threshold" in self.calibration:
            return float(self.calibration["image_threshold"])
        if "threshold" in self.config:
            return float(self.config["threshold"])
        return self.model.threshold()

    def _resolve_pixel_threshold(self) -> float:
        if self.calibration and "pixel_threshold" in self.calibration:
            return float(self.calibration["pixel_threshold"])
        return float(self.threshold_value)

    def _load_defect_direction(self) -> object | None:
        if not bool(self.config.get("use_defect_direction", False)):
            return None
        path = Path(self.config.get("defect_direction_path") or default_defect_direction_path(self.config, self.dataset, self.category))
        if not path.exists():
            return None
        from methods.defect_calibration.defect_direction import DefectDirection

        direction = DefectDirection.load(path)
        if "defect_alpha" in self.config:
            direction.alpha = float(self.config["defect_alpha"])
        return direction

    def _load_dual_memory(self) -> object | None:
        if not bool(self.config.get("use_dual_memory", False)):
            return None
        path = Path(self.config.get("defect_prototype_path") or default_defect_prototype_path(self.config, self.dataset, self.category))
        if not path.is_file():
            return None
        from methods.adaptive_dual_memory import DefectPrototypeMemory, DualMemoryScorer

        return DualMemoryScorer(DefectPrototypeMemory.load(path), device=self.model.device)

    def _load_lightweight_profile(self) -> dict | None:
        requested_mode = str(self.config.get("method_mode_resolved") or self.config.get("method_mode") or "")
        path_value = self.config.get("lightweight_calibrator_path") or default_lightweight_calibrator_path(self.config, self.dataset, self.category)
        # Read-only inference needs only the frozen JSON loader.  Import the
        # leaf module directly so package-level training exports (including
        # scikit-learn CV metrics) cannot lengthen cold start.
        from methods.lightweight_calibration.lightweight_profile import load_lightweight_profile

        profile = load_lightweight_profile(path_value)
        if profile is None or requested_mode not in {"", LIGHTWEIGHT_MODE}:
            return None
        return profile

    def output_paths(self, image_path: str | Path):
        image_path = Path(image_path)
        digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8]
        stem = f"{image_path.stem}_{digest}"
        base = Path(self.config["output_root"])
        heatmap_path = base / "heatmaps" / self.dataset / self.category / f"{stem}_heatmap.png"
        mask_path = base / "masks" / self.dataset / self.category / f"{stem}_mask.png"
        overlay_path = base / "overlays" / self.dataset / self.category / f"{stem}_overlay.png"
        marked_image_path = base / "marked_images" / self.dataset / self.category / f"{stem}_marked.png"
        return heatmap_path, mask_path, overlay_path, marked_image_path

    def _heatmap_target_size(self, height: int, width: int) -> tuple[int, int]:
        max_side = int(self.config.get("heatmap_output_size", 0) or 0)
        if max_side <= 0 or max(height, width) <= max_side:
            return int(height), int(width)
        scale = float(max_side) / float(max(height, width))
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    def _score_patch_features(self, patch_features: torch.Tensor, patch_shape: tuple[int, int], timer: StageTimer) -> tuple[np.ndarray, dict]:
        if self.model.memory_bank.uses_gpu_queries:
            with timer.track("memory_score_ms"):
                memory_scores = self.model.memory_bank.score(patch_features)
            with timer.track("device_to_host_ms"):
                if isinstance(memory_scores, torch.Tensor):
                    memory_scores = memory_scores.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    memory_scores = np.asarray(memory_scores, dtype=np.float32)
        else:
            with timer.track("device_to_host_ms"):
                memory_query = patch_features.detach().cpu().numpy().astype(np.float32, copy=False)
            with timer.track("memory_score_ms"):
                memory_scores = self.model.memory_bank.score(memory_query)
        memory_map = patch_scores_to_map(memory_scores, patch_shape)
        defect_map = None
        defect_img_score = 0.0
        baseline_score = image_score(memory_map)
        decision_score = baseline_score
        lite_result = None
        dual_alpha = float(self.config.get("dual_memory_alpha", 0.0) or 0.0)
        with timer.track("defect_score_ms"):
            if self.lightweight_profile is not None:
                from methods.lightweight_calibration import apply_lightweight_profile

                lite_result = apply_lightweight_profile(memory_scores, baseline_score, self.lightweight_profile)
                decision_score = float(lite_result["final_score"])
            elif self.dual_memory is not None and dual_alpha > 0.0:
                dual_scores = self.dual_memory.score(
                    patch_features,
                    memory_scores,
                    alpha=dual_alpha,
                    gated=True,
                )
                decision_score = _patch_image_score(dual_scores)
                defect_img_score = float(decision_score - image_score(memory_map))
            elif self.defect_direction is not None:
                defect_map = self.defect_direction.score_map(patch_features, patch_shape)
                defect_img_score = image_score(defect_map)
        with timer.track("fusion_ms"):
            if defect_map is not None:
                score_map = memory_map + float(self.defect_direction.alpha) * defect_map
                decision_score = image_score(score_map)
            else:
                score_map = memory_map
        if lite_result is None:
            lite_result = {
                "baseline_score": float(baseline_score),
                "calibrated_score": float(decision_score),
                "final_score": float(decision_score),
                "calibration_applied": False,
                "gate_status": "not_applicable",
                "fallback_used": False,
                "calibration_version": "",
            }
        return score_map, {
            "memory_score": image_score(memory_map),
            "defect_score": float(defect_img_score),
            "defect_alpha": dual_alpha if self.dual_memory is not None else (float(self.defect_direction.alpha) if self.defect_direction is not None else 0.0),
            "decision_score": float(decision_score),
            "baseline_score": float(lite_result["baseline_score"]),
            "calibrated_score": float(lite_result["calibrated_score"]),
            "final_score": float(lite_result["final_score"]),
            "calibration_applied": bool(lite_result["calibration_applied"]),
            "gate_status": str(lite_result["gate_status"]),
            "fallback_used": bool(lite_result["fallback_used"]),
            "calibration_version": str(lite_result.get("calibration_version") or ""),
            "calibration_signal": float(lite_result.get("calibration_signal", 0.0)),
            "method_mode": LIGHTWEIGHT_MODE if self.lightweight_profile is not None else ("normal_defect_dual_memory" if self.dual_memory is not None else ("defect_direction" if self.defect_direction is not None else "normal_memory")),
            "memory_backend": str(self.model.memory_bank.backend),
        }

    def _preprocess(self, image: Image.Image, timer: StageTimer, image_size: int | None = None) -> torch.Tensor:
        from core.image_io import preprocess_rgb

        target_size = int(image_size if image_size is not None else self.config["image_size"])
        tensor, timing = preprocess_rgb(image, target_size, pin_memory=self.pin_memory)
        timer.add("resize_ms", float(timing["resize_ms"]))
        timer.add("tensor_convert_ms", float(timing["tensor_convert_ms"]))
        timer.add("preprocess_ms", float(timing["resize_ms"]) + float(timing["tensor_convert_ms"]))
        timer.pinned_memory = bool(getattr(timer, "pinned_memory", False) or timing["pinned_memory"])
        return tensor

    def _infer_full_image(
        self,
        image: Image.Image,
        timer: StageTimer,
        image_size: int | None = None,
        heatmap_size: tuple[int, int] | None = None,
    ):
        from core.postprocess import resize_heatmap

        width, height = image.size
        with timer.track("tile_schedule_ms"):
            pass
        tensor = self._preprocess(image, timer, image_size=image_size)
        patches, patch_shape = self.model.extract(tensor, timer=timer)
        score_map, extra = self._score_patch_features(patches[0], patch_shape, timer)
        if heatmap_size is None:
            target_h, target_w = self._heatmap_target_size(height, width)
        else:
            target_h, target_w = (int(heatmap_size[0]), int(heatmap_size[1]))
        with timer.track("heatmap_resize_ms"):
            heatmap = resize_heatmap(score_map, height=target_h, width=target_w)
        return heatmap, extra, 1

    def _infer_tiled(self, image, timer: StageTimer):
        from core.tile_inference import crop_tile, finalize_fusion, fuse_tile, init_fusion, schedule_tiles

        width, height = image.size
        tile_size = int(self.config.get("tile_size", 768))
        overlap = int(self.config.get("tile_overlap", 128))
        batch_size = int(self.config.get("tile_batch_size", 4))
        fusion = str(self.config.get("tile_fusion", "max"))
        with timer.track("tile_schedule_ms"):
            boxes = schedule_tiles(width, height, tile_size=tile_size, overlap=overlap)
            acc, weight = init_fusion(height, width, fusion)

        memory_scores = []
        defect_scores = []
        decision_scores = []
        baseline_scores = []
        calibrated_scores = []
        calibration_applied = []
        fallback_used = []
        gate_statuses = []
        calibration_versions = []
        for start in range(0, len(boxes), batch_size):
            batch_boxes = boxes[start:start + batch_size]
            tensors = [self._preprocess(crop_tile(image, box), timer) for box in batch_boxes]
            batch_tensor = torch.stack(tensors, dim=0)
            patches, patch_shape = self.model.extract(batch_tensor, timer=timer)
            for i, box in enumerate(batch_boxes):
                score_map, extra = self._score_patch_features(patches[i], patch_shape, timer)
                memory_scores.append(extra["memory_score"])
                defect_scores.append(extra["defect_score"])
                decision_scores.append(extra["decision_score"])
                baseline_scores.append(extra.get("baseline_score", extra["decision_score"]))
                calibrated_scores.append(extra.get("calibrated_score", extra["decision_score"]))
                calibration_applied.append(bool(extra.get("calibration_applied", False)))
                fallback_used.append(bool(extra.get("fallback_used", False)))
                gate_statuses.append(str(extra.get("gate_status", "not_applicable")))
                calibration_versions.append(str(extra.get("calibration_version", "")))
                with timer.track("fusion_ms"):
                    fuse_tile(acc, weight, score_map, box, fusion)
        with timer.track("fusion_ms"):
            heatmap = finalize_fusion(acc, weight, fusion)
        return heatmap, {
            "memory_score": float(max(memory_scores)) if memory_scores else 0.0,
            "defect_score": float(max(defect_scores)) if defect_scores else 0.0,
            "defect_alpha": float(self.config.get("dual_memory_alpha", 0.0)) if self.dual_memory is not None else (float(self.defect_direction.alpha) if self.defect_direction is not None else 0.0),
            "decision_score": float(max(decision_scores)) if decision_scores else 0.0,
            "baseline_score": float(max(baseline_scores)) if baseline_scores else 0.0,
            "calibrated_score": float(max(calibrated_scores)) if calibrated_scores else 0.0,
            "final_score": float(max(decision_scores)) if decision_scores else 0.0,
            "calibration_applied": bool(any(calibration_applied)),
            "gate_status": "calibration_applied" if any(calibration_applied) else (gate_statuses[0] if gate_statuses else "not_applicable"),
            "fallback_used": bool(any(fallback_used)),
            "calibration_version": next((value for value in calibration_versions if value), ""),
            "method_mode": LIGHTWEIGHT_MODE if self.lightweight_profile is not None else ("normal_defect_dual_memory" if self.dual_memory is not None else ("defect_direction" if self.defect_direction is not None else "normal_memory")),
            "memory_backend": str(self.model.memory_bank.backend),
        }, len(boxes)

    def _infer_coarse_to_fine(self, image: Image.Image, timer: StageTimer):
        """Run a low-resolution full-image pass then refine only proposed ROIs.

        The coarse heatmap always remains the base map. Fine scores are fused only
        inside selected ROIs, so a coarse response outside a fine ROI is never
        silently discarded.
        """
        from core.tile_inference import crop_tile, finalize_fusion, fuse_tile, init_fusion, propose_rois

        width, height = image.size
        coarse_image_size = int(self.config.get("coarse_image_size", self.config["image_size"]))
        fine_image_size = int(self.config.get("fine_image_size", self.config["image_size"]))
        fine_tile_size = int(self.config.get("fine_tile_size", self.config.get("tile_size", 768)))
        fusion = str(self.config.get("coarse_fine_fusion", "max")).lower()
        threshold_value = self.config.get("coarse_threshold")
        coarse_threshold = self.threshold_value if threshold_value is None else float(threshold_value)

        coarse_heatmap, coarse_extra, _ = self._infer_full_image(
            image,
            timer,
            image_size=coarse_image_size,
            heatmap_size=(height, width),
        )
        with timer.track("tile_schedule_ms"):
            selection = propose_rois(
                coarse_heatmap=coarse_heatmap,
                coarse_threshold=coarse_threshold,
                roi_topk=int(self.config.get("roi_topk", 1)),
                roi_min_area=int(self.config.get("roi_min_area", 1)),
                roi_expand_ratio=float(self.config.get("roi_expand_ratio", 0.0)),
                roi_merge_iou=float(self.config.get("roi_merge_iou", 0.3)),
                fine_tile_size=fine_tile_size,
                max_fine_rois=int(self.config.get("max_fine_rois", 1)),
                min_fine_rois=int(self.config.get("min_fine_rois", 1)),
            )

        roi_metadata = {
            "coarse_to_fine": True,
            "inference_mode": "coarse_to_fine",
            "coarse_image_size": coarse_image_size,
            "fine_image_size": fine_image_size,
            "fine_tile_size": fine_tile_size,
            "coarse_threshold": float(coarse_threshold),
            "coarse_score": float(image_score(coarse_heatmap)),
            "coarse_component_count": int(selection.component_count),
            "coarse_retained_component_count": int(selection.retained_component_count),
            "coarse_fallback_count": int(selection.fallback_count),
            "fine_roi_count": int(len(selection.proposals)),
            "fine_roi_boxes": json.dumps(
                [
                    {
                        "x0": proposal.box.x0,
                        "y0": proposal.box.y0,
                        "x1": proposal.box.x1,
                        "y1": proposal.box.y1,
                        "score": round(float(proposal.score), 8),
                        "area": int(proposal.area),
                        "source": proposal.source,
                    }
                    for proposal in selection.proposals
                ],
                ensure_ascii=False,
            ),
            "coarse_no_candidate_reason": "" if selection.proposals else selection.reason,
            "coarse_candidate_reason": selection.reason,
            "coarse_fine_fusion": fusion,
        }
        if not selection.proposals:
            roi_metadata["fine_roi_coverage"] = 0.0
            return coarse_heatmap, coarse_extra, 1, roi_metadata

        fine_acc, fine_weight = init_fusion(height, width, "max" if fusion in {"max", "replace", "fine"} else "average")
        fine_memory_scores = []
        fine_defect_scores = []
        fine_decision_scores = []
        fine_baseline_scores = []
        fine_calibrated_scores = []
        fine_calibration_applied = []
        fine_fallback_used = []
        fine_gate_statuses = []
        fine_calibration_versions = []
        for proposal in selection.proposals:
            roi_image = crop_tile(image, proposal.box)
            fine_map, fine_extra, _ = self._infer_full_image(
                roi_image,
                timer,
                image_size=fine_image_size,
                heatmap_size=(proposal.box.height, proposal.box.width),
            )
            fine_memory_scores.append(float(fine_extra["memory_score"]))
            fine_defect_scores.append(float(fine_extra["defect_score"]))
            fine_decision_scores.append(float(fine_extra["decision_score"]))
            fine_baseline_scores.append(float(fine_extra.get("baseline_score", fine_extra["decision_score"])))
            fine_calibrated_scores.append(float(fine_extra.get("calibrated_score", fine_extra["decision_score"])))
            fine_calibration_applied.append(bool(fine_extra.get("calibration_applied", False)))
            fine_fallback_used.append(bool(fine_extra.get("fallback_used", False)))
            fine_gate_statuses.append(str(fine_extra.get("gate_status", "not_applicable")))
            fine_calibration_versions.append(str(fine_extra.get("calibration_version", "")))
            with timer.track("fusion_ms"):
                fuse_tile(fine_acc, fine_weight, fine_map, proposal.box, "max" if fusion in {"max", "replace", "fine"} else "average")

        with timer.track("fusion_ms"):
            fine_coverage = np.isfinite(fine_acc) if fusion in {"max", "replace", "fine"} else fine_weight > 0
            fine_heatmap = finalize_fusion(fine_acc, fine_weight, "max" if fusion in {"max", "replace", "fine"} else "average")
            heatmap = np.asarray(coarse_heatmap, dtype=np.float32).copy()
            if fusion == "max":
                heatmap[fine_coverage] = np.maximum(heatmap[fine_coverage], fine_heatmap[fine_coverage])
            elif fusion in {"replace", "fine"}:
                heatmap[fine_coverage] = fine_heatmap[fine_coverage]
            elif fusion in {"average", "avg", "mean"}:
                heatmap[fine_coverage] = (heatmap[fine_coverage] + fine_heatmap[fine_coverage]) / 2.0
            else:
                raise ValueError(f"Unsupported coarse_fine_fusion mode: {fusion}")
        roi_metadata["fine_roi_coverage"] = float(fine_coverage.mean())
        extra = {
            "memory_score": float(max([coarse_extra["memory_score"], *fine_memory_scores])),
            "defect_score": float(max([coarse_extra["defect_score"], *fine_defect_scores])),
            "defect_alpha": float(self.config.get("dual_memory_alpha", 0.0)) if self.dual_memory is not None else (float(self.defect_direction.alpha) if self.defect_direction is not None else 0.0),
            "decision_score": float(max([coarse_extra["decision_score"], *fine_decision_scores])),
            "baseline_score": float(max([coarse_extra.get("baseline_score", coarse_extra["decision_score"]), *fine_baseline_scores])),
            "calibrated_score": float(max([coarse_extra.get("calibrated_score", coarse_extra["decision_score"]), *fine_calibrated_scores])),
            "final_score": float(max([coarse_extra["decision_score"], *fine_decision_scores])),
            "calibration_applied": bool(coarse_extra.get("calibration_applied", False) or any(fine_calibration_applied)),
            "gate_status": "calibration_applied" if (coarse_extra.get("calibration_applied", False) or any(fine_calibration_applied)) else str(coarse_extra.get("gate_status", fine_gate_statuses[0] if fine_gate_statuses else "not_applicable")),
            "fallback_used": bool(coarse_extra.get("fallback_used", False) or any(fine_fallback_used)),
            "calibration_version": str(coarse_extra.get("calibration_version") or next((value for value in fine_calibration_versions if value), "")),
            "method_mode": LIGHTWEIGHT_MODE if self.lightweight_profile is not None else ("normal_defect_dual_memory" if self.dual_memory is not None else ("defect_direction" if self.defect_direction is not None else "normal_memory")),
            "memory_backend": str(self.model.memory_bank.backend),
        }
        return heatmap, extra, 1 + len(selection.proposals), roi_metadata

    @staticmethod
    def _refresh_timing(result: dict) -> None:
        times = {field: float(result.get(field, 0.0) or 0.0) for field in RUNTIME_FIELDS if field != "total_ms"}
        timing = timing_totals(times, result.get("timing_mode", "strict_total"))
        result.update({key: value for key, value in timing.items() if key != "timing_mode"})
        result["inference_time_ms"] = float(result["total_ms"])
        result["scoring_ms"] = float(result.get("memory_score_ms", 0.0) + result.get("defect_score_ms", 0.0) + result.get("fusion_ms", 0.0))

    def finalize_save_results(self, results: list[dict], close: bool = False) -> None:
        """Resolve background saves before a caller materialises result.csv."""
        errors: list[str] = []
        try:
            for result in results:
                ticket = result.pop("_save_ticket", None)
                if ticket is None:
                    # Synchronous infer_image already resolved this result.
                    continue
                outcome = self.visual_save_queue.resolve(ticket)
                result["encode_ms"] = float(outcome.encode_ms)
                result["disk_write_ms"] = float(outcome.disk_write_ms)
                result["save_queue_wait_ms"] = float(outcome.queue_wait_ms)
                result["save_output_ms"] = float(outcome.disk_write_ms)
                for key in ("heatmap_path", "mask_path", "overlay_path", "marked_image_path"):
                    # A path becomes externally valid only after its write succeeds.
                    result[key] = str(outcome.paths.get(key, ""))
                self._refresh_timing(result)
                errors.extend(outcome.errors)
        finally:
            if close:
                self.visual_save_queue.close()
        if errors:
            error_path = Path(self.config["output_root"]) / "logs" / "async_save_errors.md"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(f"- {message}" for message in errors) + "\n")
            raise RuntimeError(f"Visual output save failed for {len(errors)} file(s). See {error_path}")

    def infer_image(
        self,
        image_path: str | Path,
        meta: dict | None = None,
        save_outputs: bool = True,
        generate_visuals: bool | None = None,
    ) -> dict:
        from core.image_io import prepare_image

        prepared = prepare_image(image_path)
        return self.infer_prepared(prepared, meta=meta, save_outputs=save_outputs, generate_visuals=generate_visuals)

    def infer_prepared(
        self,
        prepared: PreparedImage,
        meta: dict | None = None,
        save_outputs: bool = True,
        generate_visuals: bool | None = None,
    ) -> dict:
        from core.postprocess import (
            array_profile,
            binary_mask,
            make_marked_image,
            make_overlay,
            normalize_uint8,
        )

        image_path = prepared.image_path
        meta = meta or {}
        timer = StageTimer(self.model.device)
        timer.add("file_read_ms", prepared.file_read_ms)
        timer.add("image_read_ms", prepared.file_read_ms)  # Backward-compatible alias.
        timer.add("decode_ms", prepared.decode_ms)
        timer.add("color_convert_ms", prepared.color_convert_ms)
        timer.add("queue_wait_ms", prepared.queue_wait_ms)
        image = prepared.image
        width, height = prepared.width, prepared.height

        use_coarse_to_fine = bool(self.config.get("use_coarse_to_fine", False))
        use_tiling = bool(self.config.get("use_tiling", False)) and not use_coarse_to_fine
        if use_coarse_to_fine:
            heatmap, extra, tile_count, coarse_to_fine_metadata = self._infer_coarse_to_fine(image, timer)
        elif use_tiling:
            heatmap, extra, tile_count = self._infer_tiled(image, timer)
            coarse_to_fine_metadata = {"coarse_to_fine": False, "inference_mode": "full_tile"}
        else:
            heatmap, extra, tile_count = self._infer_full_image(image, timer)
            coarse_to_fine_metadata = {
                "coarse_to_fine": False,
                "inference_mode": "coarse_only" if bool(self.config.get("coarse_only", False)) else "resize_full_image",
            }

        write_visuals = bool(save_outputs) and not bool(self.config.get("no_save_visuals", False))
        if generate_visuals is not None:
            write_visuals = write_visuals and bool(generate_visuals)
        save_heatmap = write_visuals and bool(self.config.get("save_heatmap", True))
        save_mask = write_visuals and bool(self.config.get("save_mask", True))
        save_overlay = write_visuals and bool(self.config.get("save_overlay", True))

        threshold = self.threshold_value
        with timer.track("image_score_ms"):
            score = float(extra.get("decision_score", image_score(heatmap)))
            pred = int(score >= threshold)
        postprocess_arrays = [array_profile("score_heatmap", heatmap, "_infer_full_image/_infer_tiled", reusable=True)]
        mask = None
        if save_mask or bool(self.config.get("return_mask_array", False)):
            with timer.track("mask_ms"):
                mask = binary_mask(heatmap, self.pixel_threshold_value)
            postprocess_arrays.append(array_profile("binary_mask", mask, "infer_prepared", reusable=save_mask))
        heatmap_path, mask_path, overlay_path, marked_image_path = self.output_paths(image_path)
        heatmap_u8 = None
        if save_heatmap or save_overlay:
            with timer.track("heatmap_normalize_ms"):
                heatmap_u8 = normalize_uint8(heatmap)
            postprocess_arrays.append(array_profile("heatmap_uint8", heatmap_u8, "infer_prepared", reusable=save_heatmap and save_overlay))
        overlay = None
        if save_overlay:
            with timer.track("overlay_ms"):
                overlay = make_overlay(image, heatmap_u8=heatmap_u8)
            postprocess_arrays.append(array_profile("overlay_rgb", overlay, "infer_prepared", reusable=False))
        marked_image = None
        marked_regions: list[tuple[int, int, int, int]] = []
        if save_overlay and mask is not None:
            # Keep the display-only red-box rendering within the existing
            # visualization timing bucket; no new inference stage is added.
            with timer.track("overlay_ms"):
                marked_image, marked_regions = make_marked_image(image, mask, enabled=bool(pred == 1))
            postprocess_arrays.append(array_profile("marked_image_rgb", marked_image, "infer_prepared", reusable=False))
        payloads = {}
        if save_heatmap:
            payloads["heatmap_path"] = (heatmap_path, heatmap_u8)
        if save_mask and mask is not None:
            payloads["mask_path"] = (mask_path, mask)
        if save_overlay and overlay is not None:
            payloads["overlay_path"] = (overlay_path, overlay)
        # A normal sample intentionally has no red boxes.  Avoid creating a
        # duplicate artifact in that case; the UI falls back to the original.
        if save_overlay and marked_image is not None and marked_regions:
            payloads["marked_image_path"] = (marked_image_path, marked_image)
        ticket = self.visual_save_queue.submit(payloads)

        timer.add("visualization_ms", timer.get("heatmap_normalize_ms") + timer.get("mask_ms") + timer.get("overlay_ms"))
        timer.add("postprocess_ms", timer.get("heatmap_resize_ms") + timer.get("visualization_ms"))
        times = {field: timer.get(field) for field in RUNTIME_FIELDS if field != "total_ms"}
        timing = timing_totals(times, self.config.get("timing_mode", "strict_total"))
        result = {
            "dataset": self.dataset,
            "category": self.category,
            "image_path": str(image_path),
            "label_gt": int(meta.get("label", -1)),
            "label_name": str(meta.get("label_name", "")),
            "defect_type": str(meta.get("defect_type", "")),
            "gt_mask_path": str(meta.get("mask_path", "")) if meta.get("mask_path", "") == meta.get("mask_path", "") else "",
            "score": float(score),
            "image_score": float(score),
            "baseline_score": float(extra.get("baseline_score", score)),
            "calibrated_score": float(extra.get("calibrated_score", score)),
            "final_score": float(extra.get("final_score", score)),
            "calibration_applied": bool(extra.get("calibration_applied", False)),
            "gate_status": str(extra.get("gate_status", "not_applicable")),
            "fallback_used": bool(extra.get("fallback_used", False)),
            "calibration_version": str(extra.get("calibration_version", "")),
            "memory_score": float(extra["memory_score"]),
            "memory_backend": str(extra["memory_backend"]),
            "memory_backend_requested": str(self.model.memory_bank.backend_requested),
            "memory_index_build_ms": float(self.model.memory_bank.backend_info.get("memory_index_build_ms", 0.0)),
            "memory_index_gpu_bytes": int(self.model.memory_bank.backend_info.get("memory_index_gpu_bytes", 0)),
            "defect_score": float(extra["defect_score"]),
            "defect_alpha": float(extra["defect_alpha"]),
            "method_mode": str(extra.get("method_mode") or "normal_memory"),
            "threshold": float(threshold),
            "pixel_threshold": float(self.pixel_threshold_value),
            "pred_label": int(pred),
            "heatmap_path": "",
            "mask_path": "",
            "overlay_path": "",
            "marked_image_path": "",
            "marked_region_count": int(len(marked_regions)),
            "tile_count": int(tile_count),
            "use_tiling": bool(use_tiling),
            **coarse_to_fine_metadata,
            "precision_mode": "fp16" if bool(self.config.get("use_fp16", False)) else "fp32",
            "pinned_memory": bool(getattr(timer, "pinned_memory", False)),
            "non_blocking_transfer": bool(self.model.extractor.non_blocking_transfer),
            "timing_mode": str(timing["timing_mode"]),
            "generate_visualizations": bool(write_visuals),
            "visualization_status": "generated" if write_visuals else "not_generated",
            "inference_time_ms": float(timing["total_ms"]),
            "speed_mode": bool(self.config.get("speed_mode", False)),
            "extreme_speed_mode": bool(self.config.get("extreme_speed_mode", False)),
            "scoring_memory_size": int(self.model.metadata.get("scoring_memory_size", self.model.memory_bank.features.shape[0] if self.model.memory_bank and self.model.memory_bank.features is not None else 0)),
            "original_memory_size": int(self.model.metadata.get("original_memory_size", self.model.memory_bank.features.shape[0] if self.model.memory_bank and self.model.memory_bank.features is not None else 0)),
            "coreset_strategy": str(self.model.metadata.get("coreset_strategy", "legacy-random")),
            "score_memory_strategy": str(self.model.metadata.get("score_memory_strategy", "none")),
            **times,
            **{k: v for k, v in timing.items() if k != "timing_mode"},
        }
        result["scoring_ms"] = float(result["memory_score_ms"] + result["defect_score_ms"] + result["fusion_ms"])
        if bool(self.config.get("retain_postprocess_arrays", False)):
            result["_heatmap_array"] = heatmap
            result["_mask_array"] = mask
        result["_postprocess_array_profile"] = postprocess_arrays
        result["_save_ticket"] = ticket
        if not self.visual_save_queue.enabled:
            # Preserve direct synchronous infer_image behaviour: output paths and
            # save timings are ready before the method returns.
            self.finalize_save_results([result], close=False)
        return result

    def save_single_result(self, result: dict, output_path: str | Path | None = None) -> Path:
        self.finalize_save_results([result], close=True)
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        if output_path is None:
            image_stem = Path(clean["image_path"]).stem
            output_path = Path(self.config["output_root"]) / "results" / f"{self.dataset}_{self.category}_{image_stem}_result.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path


def run_batch_inference(
    config: dict,
    category: str | None = None,
    split: str = "test",
    limit: int | None = None,
    result_name: str = "result.csv",
) -> tuple[object, Path]:
    # Dataset-manifest inference is the only code path that needs Pandas.
    # Deferring it prevents Pandas/pyarrow from delaying API liveness and does
    # not change any scoring, preprocessing or model behaviour.
    import pandas as pd
    from core.image_io import iter_prepared_images

    dataset = str(config["dataset"]).lower()
    category = category or str(config["category"])
    manifest_path = Path(config["output_root"]) / "results" / f"manifest_{dataset}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/build_manifest.py first.")
    df = pd.read_csv(manifest_path)
    df = df[(df["dataset"].astype(str).str.lower() == dataset) & (df["category"].astype(str) == category) & (df["split"].astype(str).str.lower() == split)]
    if limit:
        df = df.head(int(limit))
    inferencer = PatchCoreInferencer(config, category=category)
    raw_results = []
    items = df.to_dict("records")
    paths = [item["image_path"] for item in items]
    prefetch = iter_prepared_images(
        paths,
        enabled=bool(config.get("async_prefetch", False)),
        queue_size=int(config.get("prefetch_queue_size", 2)),
    )
    try:
        for item, prepared in zip(items, prefetch):
            result = inferencer.infer_prepared(prepared, meta=item, save_outputs=True)
            raw_results.append(result)
        inferencer.finalize_save_results(raw_results, close=True)
    except Exception:
        inferencer.visual_save_queue.close()
        raise
    records = [{k: v for k, v in result.items() if not k.startswith("_")} for result in raw_results]
    result_dir = ensure_dir(Path(config["output_root"]) / "results" / dataset / category)
    result_csv = result_dir / result_name
    out_df = pd.DataFrame(records)
    out_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    return out_df, result_csv
