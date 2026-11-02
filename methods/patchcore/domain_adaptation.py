"""Target-domain adaptation with a pure normal bank and optional defect memory.

This module is deliberately separate from the API/UI layer.  It receives two
already-separated path lists and never concatenates defect samples into the
normal feature stream. Defect images may calibrate a threshold or build a
physically separate compact prototype resource; neither operation mutates the
normal memory.
"""
from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from core.anomaly_scorer import image_score, patch_scores_to_map
from core.calibration import choose_image_threshold
from core.domain_build_contract import (
    CREATE_ALLOWED_MODES,
    DEFAULT_CREATE_MODE,
    PRECISION_MODES,
    default_feature_layers,
    dtype_name,
)
from core.image_io import prepare_image, preprocess_rgb
from core.memory_bank import MemoryBank
from methods.adaptive_dual_memory import DualMemoryScorer, mine_defect_prototypes
from methods.lightweight_calibration import MODE as LIGHTWEIGHT_MODE
from methods.lightweight_calibration import fit_lightweight_profile, write_lightweight_profile
from methods.patchcore.model import FastPatchCoreAOI


ProgressCallback = Callable[[str, int, int, str], None]
CancelCallback = Callable[[], bool]
SUPPORTED_DOMAIN_THRESHOLD_METHODS = {
    "normal_percentile",
    "mean_std",
    "best_f1",
    "recall_priority",
}
# Compatibility name retained for callers that inspect builder capabilities.
# Unlike historical profile readers, this function is a creation path and must
# expose only the current creation policy.
SUPPORTED_DOMAIN_METHOD_MODES = CREATE_ALLOWED_MODES


def _patch_image_score(values: torch.Tensor | np.ndarray) -> float:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return image_score(np.asarray(values, dtype=np.float32).reshape(1, -1))


def _calibration_rank(labels: list[int], scores: list[float], threshold: float) -> tuple[float, float, float]:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float32)
    metrics = _confusion(labels, scores, threshold)
    if len(np.unique(y)) < 2:
        return 0.0, 0.0, float(metrics["f1"])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), float(metrics["f1"])


class DomainAdaptationCancelled(RuntimeError):
    pass


def _check_cancel(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise DomainAdaptationCancelled("目标域适配任务已取消。")


def _notify(callback: ProgressCallback | None, stage: str, current: int, total: int, message: str) -> None:
    if callback is not None:
        callback(stage, int(current), int(total), message)


def _batches(items: list[Path], size: int) -> Iterable[tuple[int, list[Path]]]:
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def _confusion(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int32)
    pred = (np.asarray(scores, dtype=np.float32) >= float(threshold)).astype(np.int32)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def build_and_calibrate_domain(
    normal_paths: Iterable[str | Path],
    defect_paths: Iterable[str | Path],
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> dict[str, Any]:
    """Build one normal-only bank, calibrate it, and persist the core artifacts."""
    requested_mode = str(config.get("method_mode") or DEFAULT_CREATE_MODE).lower()
    if requested_mode not in SUPPORTED_DOMAIN_METHOD_MODES:
        raise ValueError("该目标域构建模式已停止用于新建资源，请选择当前支持的构建方式。")
    normals = [Path(path).resolve() for path in normal_paths]
    supplied_defects = [Path(path).resolve() for path in defect_paths]
    # A true normal-only run must not consume defect samples at all.  The only
    # creation mode that consumes defects is the lightweight guided calibrator.
    defects = supplied_defects if requested_mode == LIGHTWEIGHT_MODE else []
    if not normals:
        raise ValueError("正常样本为空，无法生成记忆库。")
    if not defects and requested_mode == LIGHTWEIGHT_MODE:
        raise ValueError("缺陷校准样本为空，无法执行有监督阈值校准。")
    if set(normals) & set(defects):
        raise ValueError("正常样本与缺陷校准样本存在重复文件。")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile_identity = str(config.get("normal_bank_id") or out.name.lstrip(".").removesuffix(".partial") or "normal_memory_bank")
    profile_created_at = str(config.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    hashable_config = {key: value for key, value in config.items() if key not in {"output_root", "config_hash"}}
    profile_config_hash = str(config.get("config_hash") or hashlib.sha256(json.dumps(hashable_config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest())
    config = dict(config)
    config.update(normal_bank_id=profile_identity, created_at=profile_created_at, config_hash=profile_config_hash)
    batch_size = max(1, int(config.get("batch_size", 1)))
    seed = int(config.get("random_seed", 42))
    ratio = float(config.get("coreset_ratio", 0.1))
    max_memory = max(1, int(config.get("max_memory_patches", 50000)))
    image_size = int(config.get("image_size", 256))
    method = str(config.get("threshold_method", "best_f1")).lower()
    if method not in SUPPORTED_DOMAIN_THRESHOLD_METHODS:
        raise ValueError(f"当前代码不支持阈值方法：{method}")

    _check_cancel(cancelled)
    _notify(progress, "loading_model", 0, len(normals) + len(defects), "正在准备目标域特征提取资源")
    started = time.perf_counter()
    model = FastPatchCoreAOI(config)
    requested_precision = str(config.get("requested_precision") or config.get("precision") or ("fp16" if config.get("use_fp16") else "fp32")).lower()
    actual_precision = str(config.get("actual_precision") or ("fp16" if model.extractor.use_fp16 else "fp32")).lower()
    if requested_precision not in PRECISION_MODES or actual_precision not in PRECISION_MODES:
        raise ValueError("目标域构建精度契约无效，只支持 fp16 或 fp32。")

    selected_chunks: list[torch.Tensor] = []
    selected_rows: list[torch.Tensor] = []
    original_patch_count = 0
    per_image_cap = max(1, max_memory // len(normals))
    processed = 0
    for start, paths in _batches(normals, batch_size):
        _check_cancel(cancelled)
        tensors = []
        for path in paths:
            prepared = prepare_image(path)
            tensor, _ = preprocess_rgb(prepared.image, image_size, pin_memory=bool(config.get("pin_memory", False)))
            tensors.append(tensor)
        patches, _ = model.extract(torch.stack(tensors, dim=0))
        for offset, path in enumerate(paths):
            flat = patches[offset].reshape(-1, patches.shape[-1]).detach().cpu()
            patch_count = int(flat.shape[0])
            original_patch_count += patch_count
            keep = min(patch_count, per_image_cap, max(1, int(round(patch_count * ratio))))
            generator = torch.Generator(device="cpu").manual_seed(seed + start + offset)
            indices = torch.randperm(patch_count, generator=generator)[:keep]
            selected_chunks.append(flat[indices])
            selected_rows.append(
                torch.stack(
                    [
                        torch.full((keep,), start + offset, dtype=torch.int64),
                        indices.to(dtype=torch.int64),
                    ],
                    dim=1,
                )
            )
            processed += 1
            _notify(progress, "building_memory", processed, len(normals) + len(defects), f"正在提取正常样本特征：{path.name}")

    memory_features = torch.cat(selected_chunks, dim=0).contiguous()
    coreset_indices = torch.cat(selected_rows, dim=0).contiguous()
    if len(memory_features) > max_memory:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        final_indices = torch.randperm(len(memory_features), generator=generator)[:max_memory]
        memory_features = memory_features[final_indices]
        coreset_indices = coreset_indices[final_indices]

    bank = MemoryBank(knn_k=int(config.get("knn_k", 1)), backend="faiss_cpu_flat", device="cpu").fit(memory_features)
    precision_metadata = {
        "requested_precision": requested_precision,
        "actual_precision": actual_precision,
        "compute_precision": actual_precision,
        "device": model.device.type,
        "amp_enabled": bool(model.extractor.use_fp16),
        **model.extractor.precision_metadata(),
        # Patch embeddings are normalized in FP32 before nearest-neighbour
        # indexing; MemoryBank/FAISS intentionally persist/query FP32.
        "feature_dtype": dtype_name(memory_features.dtype),
        "memory_bank_dtype": "float32",
        "index_dtype": "float32",
        "fallback_reason": str(config.get("precision_fallback_reason") or ""),
    }
    memory_metadata = {
        "baseline": "FastPatchCore-AOI",
        "dataset": str(config.get("dataset", "custom")),
        "category": str(config.get("category", "custom")),
        "backbone": model.extractor.backbone_name,
        "requested_backbone": str(config.get("backbone", "wide_resnet50_2")),
        "feature_layers": list(config.get("layers", default_feature_layers())),
        "image_size": image_size,
        "original_memory_size": original_patch_count,
        "coreset_memory_size": int(memory_features.shape[0]),
        "feature_dim": int(memory_features.shape[1]),
        "sample_count": len(normals),
        "knn_k": int(config.get("knn_k", 1)),
        "coreset_ratio": ratio,
        "coreset_strategy": "deterministic_random_per_normal_image",
        "coreset_seed": seed,
        "memory_source": "normal_samples_only",
        "normal_memory_enabled": True,
        "defect_samples_written_to_memory_bank": False,
        **precision_metadata,
    }
    memory_path = out / "memory_bank.pt"
    bank.save(memory_path, memory_metadata)
    coreset_payload = {
        "columns": ["normal_sample_index", "source_patch_index"],
        "indices": coreset_indices,
        "normal_sample_count": len(normals),
        "seed": seed,
    }
    torch.save(coreset_payload, out / "coreset_indices.pt")
    # Keep the legacy filename above for compatibility while also persisting
    # the explicit resource name used by the target-domain profile contract.
    torch.save(coreset_payload, out / "normal_coreset_index.pt")
    model.set_memory(bank, memory_metadata)

    labels: list[int] = []
    base_scores: list[float] = []
    base_patch_rows: list[np.ndarray] = []
    normal_patch_reference: list[np.ndarray] = []
    defect_feature_sets: list[torch.Tensor] = []
    defect_distance_sets: list[np.ndarray] = []
    calibration_paths = [(path, 0, "normal") for path in normals] + [(path, 1, "defect") for path in defects]
    for index, (path, label, role) in enumerate(calibration_paths):
        _check_cancel(cancelled)
        prepared = prepare_image(path)
        tensor, _ = preprocess_rgb(prepared.image, image_size, pin_memory=bool(config.get("pin_memory", False)))
        patches, _ = model.extract(tensor)
        distances = bank.score(patches[0])
        if isinstance(distances, torch.Tensor):
            distances_np = distances.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            distances_np = np.asarray(distances, dtype=np.float32)
        labels.append(label)
        base_patch_rows.append(distances_np)
        base_scores.append(_patch_image_score(distances_np))
        if label == 0:
            normal_patch_reference.append(distances_np)
        elif label == 1 and requested_mode in {"normal_defect_dual_memory", "auto"}:
            defect_feature_sets.append(patches[0].detach().cpu().half())
            defect_distance_sets.append(distances_np)
        processed = len(normals) + index + 1
        _notify(progress, "calibrating_threshold", processed, 3 * len(normals) + 2 * len(defects), f"正在评分{('正常' if label == 0 else '缺陷')}校准样本：{path.name}")

    effective_normal_only = requested_mode == "normal_only" or not defects
    baseline_strategy = "normal_percentile" if effective_normal_only else method
    baseline_threshold_detail = choose_image_threshold(
        labels,
        base_scores,
        strategy=baseline_strategy,
        percentile=float(config.get("threshold_percentile", 99.0)),
        mean_std_k=float(config.get("threshold_mean_std_k", 3.0)),
        target_recall=float(config.get("threshold_target_recall", 0.95)),
    )
    baseline_threshold = float(baseline_threshold_detail["threshold"])
    baseline_rank = _calibration_rank(labels, base_scores, baseline_threshold) if defects else None

    resolved_mode = "normal_only" if effective_normal_only else "normal_defect_calibration"
    selected_alpha = 0.0
    selected_scores = list(base_scores)
    selected_threshold_detail = dict(baseline_threshold_detail)
    prototype_memory = None
    candidate_rank = None
    candidate_threshold_detail = None
    alpha_search: list[dict[str, Any]] = []
    lightweight_profile = None
    lightweight_candidate_profile = None
    lightweight_rank = None
    lightweight_threshold_detail = None
    lightweight_scores: list[float] = []
    lightweight_component_comparison: list[dict[str, Any]] = []
    auto_recommendation: dict[str, Any] | None = None
    if defects and requested_mode in {LIGHTWEIGHT_MODE, "auto"}:
        _notify(progress, "calibrating_lightweight", 0, len(normals) + len(defects), "正在拟合轻量单调校准与安全门控")
        lightweight_candidate_profile = fit_lightweight_profile(
            normal_patch_reference,
            [row for row, label in zip(base_patch_rows, labels) if label == 1],
            [score for score, label in zip(base_scores, labels) if label == 0],
            [score for score, label in zip(base_scores, labels) if label == 1],
            threshold=baseline_threshold,
            threshold_strategy=method,
            config={
                **config,
                "baseline_threshold": baseline_threshold,
                "lightweight_mode": LIGHTWEIGHT_MODE,
            },
            seed=seed,
        )
        lightweight_component_comparison = list(lightweight_candidate_profile.get("component_comparison") or [])
        lightweight_scores = [float(value) for value in (lightweight_candidate_profile.get("calibration_metrics") or {}).get("final_scores", [])]
        lightweight_metric_payload = dict(lightweight_candidate_profile.get("calibration_metrics") or {})
        lightweight_metric_payload.pop("final_scores", None)
        lightweight_metric_payload.pop("signals", None)
        lightweight_metric_payload.pop("pooled_features", None)
        lightweight_threshold_detail = choose_image_threshold(
            labels,
            lightweight_scores,
            strategy=method,
            percentile=float(config.get("threshold_percentile", 99.0)),
            mean_std_k=float(config.get("threshold_mean_std_k", 3.0)),
            target_recall=float(config.get("threshold_target_recall", 0.95)),
        )
        lightweight_threshold = float(lightweight_threshold_detail["threshold"])
        lightweight_rank = _calibration_rank(labels, lightweight_scores, lightweight_threshold)
        lightweight_candidate_profile["image_threshold"] = lightweight_threshold
        lightweight_candidate_profile["baseline_threshold"] = baseline_threshold
        lightweight_candidate_profile["threshold_strategy"] = str(lightweight_threshold_detail.get("strategy") or method)
        lightweight_metric_payload["threshold_detail"] = lightweight_threshold_detail
        lightweight_candidate_profile["calibration_metrics"] = lightweight_metric_payload
        if requested_mode == LIGHTWEIGHT_MODE:
            lightweight_profile = lightweight_candidate_profile
            resolved_mode = LIGHTWEIGHT_MODE
            selected_scores = lightweight_scores
            selected_threshold_detail = dict(lightweight_threshold_detail)
            selected_alpha = 0.0
        _notify(progress, "calibrating_lightweight", len(normals) + len(defects), len(normals) + len(defects), "轻量单调校准与安全门控拟合完成")
    if defects and requested_mode in {"normal_defect_dual_memory", "auto"}:
        _notify(progress, "mining_defect_patches", 0, len(defects), "正在挖掘缺陷补丁并生成独立紧凑原型")
        prototype_memory = mine_defect_prototypes(
            defect_feature_sets,
            defect_distance_sets,
            np.concatenate(normal_patch_reference),
            max_prototypes=int(config.get("max_defect_prototypes", 32)),
            top_ratio=float(config.get("defect_top_ratio", 0.05)),
            min_patches_per_image=int(config.get("min_defect_patches_per_image", 4)),
            floor_percentile=float(config.get("normal_floor_percentile", 99.0)),
            min_image_support=int(config.get("min_defect_image_support", 2)),
            seed=seed,
        )
        dual = DualMemoryScorer(prototype_memory, device=model.device)
        addition_rows: list[np.ndarray] = []
        for normal_index, path in enumerate(normals):
            _check_cancel(cancelled)
            prepared = prepare_image(path)
            tensor, _ = preprocess_rgb(prepared.image, image_size, pin_memory=bool(config.get("pin_memory", False)))
            patches, _ = model.extract(tensor)
            normal_distance = bank.score(patches[0])
            parts = dual.components(patches[0], normal_distance)
            addition_rows.append((parts["gate"] * parts["contrast"]).detach().cpu().numpy())
            _notify(progress, "validating_dual_memory", normal_index + 1, len(normals) + len(defects), f"正在验证正常样本门控评分：{path.name}")
        for defect_index, (features_cpu, distances_np) in enumerate(zip(defect_feature_sets, defect_distance_sets)):
            parts = dual.components(features_cpu.float(), distances_np)
            addition_rows.append((parts["gate"] * parts["contrast"]).detach().cpu().numpy())
            _notify(progress, "validating_dual_memory", len(normals) + defect_index + 1, len(normals) + len(defects), "正在验证缺陷样本门控评分")

        candidates = [float(value) for value in config.get("dual_alpha_candidates", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0])]
        best_choice = None
        for alpha in candidates:
            candidate_scores = [
                _patch_image_score(base + alpha * addition)
                for base, addition in zip(base_patch_rows, addition_rows)
            ]
            detail = choose_image_threshold(
                labels,
                candidate_scores,
                strategy=method,
                percentile=float(config.get("threshold_percentile", 99.0)),
                mean_std_k=float(config.get("threshold_mean_std_k", 3.0)),
                target_recall=float(config.get("threshold_target_recall", 0.95)),
            )
            threshold_value = float(detail["threshold"])
            rank = _calibration_rank(labels, candidate_scores, threshold_value)
            alpha_search.append({"alpha": alpha, "auroc": rank[0], "average_precision": rank[1], "f1": rank[2], "threshold": threshold_value})
            ordering = (*rank, -alpha)
            if best_choice is None or ordering > best_choice[0]:
                best_choice = (ordering, alpha, candidate_scores, detail, rank)
        assert best_choice is not None
        _, candidate_alpha, candidate_scores, candidate_threshold_detail, candidate_rank = best_choice
        assert baseline_rank is not None
        candidate_non_decreasing = all(current >= baseline - 1e-12 for current, baseline in zip(candidate_rank, baseline_rank))
        if requested_mode == "normal_defect_dual_memory" or candidate_non_decreasing:
            resolved_mode = "normal_defect_dual_memory"
            selected_alpha = float(candidate_alpha)
            selected_scores = list(candidate_scores)
            selected_threshold_detail = dict(candidate_threshold_detail)
            prototype_memory.metadata.update(
                {
                    "alpha": selected_alpha,
                    "scoring": "reliability_gated_relative_distance",
                    "localization_map": "normal_memory_only",
                    "frozen": True,
                    "test_time_update_allowed": False,
                }
            )
            prototype_memory.save(out / "defect_prototypes.pt")
            torch.save(
                {
                    "prototype_index": torch.arange(prototype_memory.count, dtype=torch.int64),
                    "reliability": prototype_memory.reliability.detach().cpu().float(),
                    "radius": prototype_memory.radius.detach().cpu().float(),
                    "metadata": dict(prototype_memory.metadata),
                },
                out / "defect_prototype_index.pt",
            )
        else:
            prototype_memory = None

    if requested_mode == "auto" and defects and baseline_rank is not None:
        candidates_for_recommendation: list[tuple[str, tuple[float, float, float], float]] = [
            ("normal_defect_calibration", baseline_rank, 0.0),
        ]
        if lightweight_rank is not None and lightweight_threshold_detail is not None:
            candidates_for_recommendation.append((LIGHTWEIGHT_MODE, lightweight_rank, 1.0))
        if candidate_rank is not None and candidate_threshold_detail is not None:
            candidates_for_recommendation.append(("normal_defect_dual_memory", candidate_rank, 2.0))
        # Prefer the lower-complexity candidate when validation metrics tie.
        recommended_mode, recommended_rank, _ = max(candidates_for_recommendation, key=lambda item: (item[1][0], item[1][1], item[1][2], -item[2]))
        auto_recommendation = {
            "candidates": [
                {"method_mode": name, "auroc": rank[0], "average_precision": rank[1], "f1": rank[2]}
                for name, rank, _ in candidates_for_recommendation
            ],
            "recommended_mode": recommended_mode,
            "selection_rule": "calibration AUROC/AP/F1 with lower-complexity tie break; final test excluded",
        }
        if recommended_mode == LIGHTWEIGHT_MODE and lightweight_rank is not None and lightweight_threshold_detail is not None:
            resolved_mode = LIGHTWEIGHT_MODE
            lightweight_profile = lightweight_candidate_profile
            selected_scores = list(lightweight_scores)
            selected_threshold_detail = dict(lightweight_threshold_detail)
            selected_alpha = 0.0
            prototype_memory = None
            for stale in (out / "defect_prototypes.pt", out / "defect_prototype_index.pt"):
                stale.unlink(missing_ok=True)
        elif recommended_mode == "normal_defect_calibration":
            resolved_mode = "normal_defect_calibration"
            selected_scores = list(base_scores)
            selected_threshold_detail = dict(baseline_threshold_detail)
            selected_alpha = 0.0
            prototype_memory = None
            for stale in (out / "defect_prototypes.pt", out / "defect_prototype_index.pt"):
                stale.unlink(missing_ok=True)

    threshold = float(selected_threshold_detail["threshold"])
    metrics = _confusion(labels, selected_scores, threshold)
    normal_scores = [score for label, score in zip(labels, selected_scores) if label == 0]
    defect_scores = [score for label, score in zip(labels, selected_scores) if label == 1]
    threshold_payload = {
        "dataset": str(config.get("dataset", "custom")),
        "category": str(config.get("category", "custom")),
        "threshold_strategy": str(selected_threshold_detail.get("strategy") or method),
        "image_threshold": threshold,
        "pixel_threshold": baseline_threshold,
        "image_threshold_detail": selected_threshold_detail,
        "normal_shot_used": len(normals),
        "defect_shot_used": len(defects),
        "seed": seed,
        "frozen": True,
        "method_mode_requested": requested_mode,
        "method_mode_resolved": resolved_mode,
        "algorithm_mode": resolved_mode,
        "normal_memory_enabled": True,
        "lightweight_calibration_enabled": resolved_mode == LIGHTWEIGHT_MODE,
        "defect_guidance_enabled": resolved_mode == LIGHTWEIGHT_MODE,
        "dual_memory_alpha": selected_alpha,
        "lightweight_calibration": bool(resolved_mode == LIGHTWEIGHT_MODE),
        "lightweight_calibration_path": "lightweight_calibrator.json" if resolved_mode == LIGHTWEIGHT_MODE else None,
        "defect_samples_written_to_memory_bank": False,
        "test_time_update_allowed": False,
        "precision": precision_metadata,
    }
    (out / "threshold.json").write_text(json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    score_rows = [
        {
            "sample_role": "normal" if label == 0 else "defect",
            "sample_index": index if label == 0 else index - len(normals),
            "baseline_score": float(base),
            "selected_score": float(selected),
        }
        for index, (label, base, selected) in enumerate(zip(labels, base_scores, selected_scores))
    ]
    summary = {
        "threshold": threshold,
        "pixel_threshold": baseline_threshold,
        "threshold_method": threshold_payload["threshold_strategy"],
        "method_mode_requested": requested_mode,
        "method_mode_resolved": resolved_mode,
        "dual_memory_alpha": selected_alpha,
        "baseline_calibration_rank": None if baseline_rank is None else {"auroc": baseline_rank[0], "average_precision": baseline_rank[1], "f1": baseline_rank[2]},
        "candidate_calibration_rank": None if candidate_rank is None else {"auroc": candidate_rank[0], "average_precision": candidate_rank[1], "f1": candidate_rank[2]},
        "lightweight_calibration_rank": None if lightweight_rank is None else {"auroc": lightweight_rank[0], "average_precision": lightweight_rank[1], "f1": lightweight_rank[2]},
        "lightweight_component_comparison": lightweight_component_comparison,
        "alpha_search": alpha_search,
        "prototype_metadata": prototype_memory.metadata if prototype_memory is not None else None,
        "metrics": metrics,
        "normal_score_summary": {
            "minimum": float(min(normal_scores)),
            "maximum": float(max(normal_scores)),
            "mean": float(np.mean(normal_scores)),
        },
        "defect_score_summary": None if not defect_scores else {
            "minimum": float(min(defect_scores)),
            "maximum": float(max(defect_scores)),
            "mean": float(np.mean(defect_scores)),
        },
        "scores": score_rows,
        "elapsed_seconds": float(time.perf_counter() - started),
        "normal_memory_features_only": True,
        "defect_resource_separate": resolved_mode == "normal_defect_dual_memory",
        "lightweight_profile": lightweight_profile,
        "lightweight_finetuning_used": False,
        "base_method": "normal patch memory nearest-neighbour anomaly detection",
        "references": ["PatchCore (CVPR 2022)", "M3DM (CVPR 2023)", "SegAD (CVPR 2024)"],
        "precision": precision_metadata,
    }
    (out / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    scoring_config = {
        "method_version": "normal-memory-v1" if resolved_mode == "normal_only" else "normal-memory-defect-guided-lite-v1",
        "method_mode_requested": requested_mode,
        "method_mode_resolved": resolved_mode,
        "algorithm_mode": resolved_mode,
        "normal_memory_enabled": True,
        "lightweight_calibration_enabled": resolved_mode == LIGHTWEIGHT_MODE,
        "defect_guidance_enabled": resolved_mode == LIGHTWEIGHT_MODE,
        "base_method": "normal patch memory nearest-neighbour anomaly detection",
        "distance_metric": "euclidean",
        "normal_memory_source": "normal_samples_only",
        "defect_resource_separate": resolved_mode == "normal_defect_dual_memory",
        "lightweight_calibration": lightweight_profile if resolved_mode == LIGHTWEIGHT_MODE else None,
        "image_scoring": "reliability_gated_relative_distance" if resolved_mode == "normal_defect_dual_memory" else ("normal_distance_plus_monotonic_defect_guided_calibration" if resolved_mode == LIGHTWEIGHT_MODE else "normal_memory_distance"),
        "localization_map": "normal_memory_only",
        "image_threshold": threshold,
        "pixel_threshold": baseline_threshold,
        "inference_mode": "inference_only",
        "frozen": True,
        "test_time_update_allowed": False,
        "precision": precision_metadata,
    }
    (out / "scoring_config.json").write_text(json.dumps(scoring_config, ensure_ascii=False, indent=2), encoding="utf-8")
    fusion_parameters = {
        "enabled": resolved_mode == "normal_defect_dual_memory",
        "formula_version": "normal_plus_gated_positive_margin_v1",
        "alpha": selected_alpha,
        "normal_floor": float(prototype_memory.normal_floor) if prototype_memory is not None else None,
        "normal_scale": float(prototype_memory.normal_scale) if prototype_memory is not None else None,
        "prototype_count": int(prototype_memory.count) if prototype_memory is not None else 0,
        "reliability_gate": "prototype_support_x_normal_floor_x_prototype_radius" if prototype_memory is not None else None,
        "frozen": True,
    }
    (out / "fusion_parameters.json").write_text(json.dumps(fusion_parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    lightweight_path = None
    if resolved_mode == LIGHTWEIGHT_MODE and lightweight_profile is not None:
        lightweight_path = out / "lightweight_calibrator.json"
        write_lightweight_profile(lightweight_path, lightweight_profile)
    defect_resource_path = out / "defect_prototypes.pt"
    complexity_summary = {
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.extractor.parameters())),
        "additional_trainable_parameters": 0,
        "backbone_passes_per_image": 1,
        "normal_memory_feature_count": int(memory_features.shape[0]),
        "normal_memory_bytes": int(memory_path.stat().st_size),
        "defect_prototype_count": int(prototype_memory.count) if prototype_memory is not None else 0,
        "defect_resource_bytes": int(defect_resource_path.stat().st_size) if defect_resource_path.is_file() else 0,
        "lightweight_calibrator_parameters": int((lightweight_profile or {}).get("parameter_count", 0)),
        "lightweight_calibrator_bytes": int(lightweight_path.stat().st_size) if lightweight_path is not None and lightweight_path.is_file() else 0,
        "lightweight_additional_ann_count": 0,
        "lightweight_backbone_passes_per_image": 1,
        "runtime_latency_measured_during_build": False,
        "runtime_latency_note": "Use the fixed fair benchmark; build-time elapsed time is not inference latency.",
        "frozen": True,
    }
    (out / "complexity_summary.json").write_text(json.dumps(complexity_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _notify(progress, "calibration_complete", 3 * len(normals) + 2 * len(defects), 3 * len(normals) + 2 * len(defects), "目标域方法构建与校准完成")
    result = {
        "memory_path": memory_path,
        "memory_metadata": memory_metadata,
        "precision_metadata": precision_metadata,
        "threshold": threshold_payload,
        "calibration_summary": summary,
        "defect_prototype_path": out / "defect_prototypes.pt" if resolved_mode == "normal_defect_dual_memory" else None,
        "lightweight_calibrator_path": lightweight_path,
    }
    # Do not leave CUDA tensors to Python's background-thread frame teardown.
    # On Windows that teardown can block the API interpreter after the durable
    # job has already been marked complete.  Release the final batch tensor and
    # explicitly close the temporary extractor while the build stage is still
    # controlled and observable.
    patches = None
    tensor = None
    tensors = []
    model.close_build_runtime()
    return result
