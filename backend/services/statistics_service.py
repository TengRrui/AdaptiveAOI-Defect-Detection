from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

from backend.errors import ApiError


LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
MODULES = {"single", "batch", "video"}
STATUS_MAP = {"completed": "success", "failed": "failed", "cancelled": "cancelled"}
STAGE_KEYS = (
    "input_read_ms",
    "decode_ms",
    "preprocess_ms",
    "feature_extraction_ms",
    "ann_search_ms",
    "anomaly_scoring_ms",
    "inference_ms",
    "postprocess_ms",
    "visualization_ms",
    "encode_ms",
    "save_ms",
    "end_to_end_ms",
    "frame_total_ms",
)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _parse_datetime(value: str | datetime | None, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ApiError("INVALID_STATISTICS_TIME", f"{field} 不是有效的 ISO 时间。", 422) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def _local_iso(value: datetime | None) -> str | None:
    return value.astimezone(LOCAL_TIMEZONE).isoformat(timespec="seconds") if value else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "total": None, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "total": float(sum(clean)),
        "mean": float(fmean(clean)),
        "p50": _percentile(clean, 50.0),
        "p95": _percentile(clean, 95.0),
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return Path(text).name if text else fallback


def _safe_parent_name(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return Path(text).parent.name or fallback


def _manifest_time(manifest: dict[str, Any]) -> datetime | None:
    for key in ("created_at", "updated_at", "completed_at"):
        try:
            parsed = _parse_datetime(manifest.get(key), key)
        except ApiError:
            continue
        if parsed is not None:
            return parsed
    return None


def _normal_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return STATUS_MAP.get(raw, raw if raw in {"success", "failed", "cancelled", "running"} else "running")


def _single_timing(row: dict[str, Any] | None) -> dict[str, float | None]:
    if not row:
        return {}
    metadata = row.get("metadata") or {}
    timing = dict(metadata.get("timing") or {})
    if timing:
        return {key: _number(value) for key, value in timing.items() if isinstance(key, str)}
    core = metadata.get("core_result") or {}
    if not core:
        return {}
    mapped: dict[str, float | None] = {}
    input_values = [_number(core.get(key)) for key in ("file_read_ms", "decode_ms", "color_convert_ms")]
    if any(value is not None for value in input_values):
        mapped["input_read_ms"] = float(sum(value or 0.0 for value in input_values))
    aliases = {
        "preprocess_ms": "preprocess_ms",
        "feature_extraction_ms": "feature_ms",
        "ann_search_ms": "memory_score_ms",
        "postprocess_ms": "postprocess_ms",
        "visualization_ms": "visualization_ms",
        "encode_ms": "encode_ms",
        "save_ms": "disk_write_ms",
        "inference_ms": "detection_only_ms",
        "end_to_end_ms": "strict_total_ms",
    }
    for target, source in aliases.items():
        value = _number(core.get(source))
        if value is not None:
            mapped[target] = value
    anomaly_parts = [_number(core.get(key)) for key in ("defect_score_ms", "fusion_ms", "image_score_ms")]
    if any(value is not None for value in anomaly_parts):
        mapped["anomaly_scoring_ms"] = float(sum(value or 0.0 for value in anomaly_parts))
    return mapped


def _stage_statistics(records: list[dict[str, Any]], *, task_stage_totals: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in STAGE_KEYS:
        values: list[float] = []
        total_values: list[float] = []
        for record in records:
            if task_stage_totals:
                stage_payload = (record.get("stage_statistics") or {}).get(stage) or {}
                mean_value = _number(stage_payload.get("mean_ms"))
                total_value = _number(stage_payload.get("total_ms"))
                if mean_value is not None:
                    values.append(mean_value)
                if total_value is not None:
                    total_values.append(total_value)
            else:
                value = _number((record.get("timing") or {}).get(stage))
                if value is not None:
                    values.append(value)
                    total_values.append(value)
        statistics = _stats(values)
        if not values and not total_values:
            continue
        rows.append(
            {
                "stage": stage,
                "count": statistics["count"],
                "total_ms": float(sum(total_values)) if total_values else None,
                "mean_ms": statistics["mean"],
                "p50_ms": statistics["p50"],
                "p95_ms": statistics["p95"],
                "min_ms": statistics["min"],
                "max_ms": statistics["max"],
            }
        )
    return rows


def _bucket_mode(start: datetime | None, end: datetime | None, records: list[dict[str, Any]]) -> str:
    if start is not None and end is not None:
        span = end - start
    else:
        timestamps = [record["created_at_utc"] for record in records if record.get("created_at_utc")]
        span = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else timedelta(days=1)
    if span <= timedelta(days=2):
        return "hour"
    if span <= timedelta(days=120):
        return "day"
    if span <= timedelta(days=730):
        return "week"
    return "month"


def _bucket_start(value: datetime, mode: str) -> datetime:
    local = value.astimezone(LOCAL_TIMEZONE)
    if mode == "hour":
        return local.replace(minute=0, second=0, microsecond=0)
    if mode == "week":
        day = (local - timedelta(days=local.weekday())).date()
        return datetime.combine(day, datetime.min.time(), LOCAL_TIMEZONE)
    if mode == "month":
        return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def _time_series(records: list[dict[str, Any]], module: str, mode: str) -> list[dict[str, Any]]:
    buckets: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        created_at = record.get("created_at_utc")
        if created_at is not None:
            buckets[_bucket_start(created_at, mode)].append(record)
    output: list[dict[str, Any]] = []
    for bucket, items in sorted(buckets.items()):
        row: dict[str, Any] = {"bucket": bucket.isoformat(timespec="seconds"), "bucket_mode": mode, "task_count": len(items)}
        if module == "single":
            row.update(
                normal=sum(item.get("prediction") == "normal" for item in items),
                anomaly=sum(item.get("prediction") == "anomaly" for item in items),
                failed=sum(item.get("status") == "failed" for item in items),
            )
            inference = [_number(item.get("pure_inference_ms")) for item in items]
            end_to_end = [_number(item.get("end_to_end_ms")) for item in items]
            row["average_inference_ms"] = _stats([value for value in inference if value is not None])["mean"]
            end_stats = _stats([value for value in end_to_end if value is not None])
            row["average_end_to_end_ms"] = end_stats["mean"]
            row["p95_end_to_end_ms"] = end_stats["p95"]
        elif module == "batch":
            row.update(
                image_count=sum(item.get("total_images", 0) for item in items),
                normal=sum(item.get("normal_count", 0) for item in items),
                anomaly=sum(item.get("anomaly_count", 0) for item in items),
                failed_items=sum(item.get("failed_items", 0) for item in items),
                success_tasks=sum(item.get("status") == "success" for item in items),
                failed_tasks=sum(item.get("status") == "failed" for item in items),
            )
            throughput = [value for value in (_number(item.get("throughput")) for item in items) if value is not None]
            throughput_stats = _stats(throughput)
            row.update(
                average_throughput=throughput_stats["mean"],
                p50_throughput=throughput_stats["p50"],
                p95_throughput=throughput_stats["p95"],
                min_throughput=throughput_stats["min"],
                max_throughput=throughput_stats["max"],
            )
        else:
            row.update(
                frame_count=sum(item.get("total_frames", 0) for item in items),
                normal=sum(item.get("normal_count", 0) for item in items),
                anomaly=sum(item.get("anomaly_count", 0) for item in items),
                failed_items=sum(item.get("failed_items", 0) for item in items),
                duration_seconds=sum(item.get("duration_seconds", 0.0) for item in items),
                success_tasks=sum(item.get("status") == "success" for item in items),
                failed_tasks=sum(item.get("status") == "failed" for item in items),
            )
            frame_values = [value for value in (_number(item.get("average_frame_ms")) for item in items) if value is not None]
            pipeline_fps = [value for value in (_number(item.get("full_pipeline_fps")) for item in items) if value is not None]
            row["average_frame_ms"] = _stats(frame_values)["mean"]
            row["p50_frame_ms"] = _stats([value for value in (_number(item.get("p50_frame_ms")) for item in items) if value is not None])["mean"]
            row["p95_frame_ms"] = _stats([value for value in (_number(item.get("p95_frame_ms")) for item in items) if value is not None])["mean"]
            row["average_full_pipeline_fps"] = _stats(pipeline_fps)["mean"]
        output.append(row)
    return output


def _distribution(records: list[dict[str, Any]], module: str) -> list[dict[str, Any]]:
    if module == "single":
        counts = {
            "normal": sum(item.get("prediction") == "normal" for item in records),
            "anomaly": sum(item.get("prediction") == "anomaly" for item in records),
            "failed": sum(item.get("status") == "failed" for item in records),
        }
    else:
        counts = {
            "normal": sum(item.get("normal_count", 0) for item in records),
            "anomaly": sum(item.get("anomaly_count", 0) for item in records),
            "failed": sum(item.get("failed_items", 0) for item in records),
        }
    total = sum(counts.values())
    return [
        {"result": key, "count": int(value), "percentage": float(value / total * 100.0) if total else None}
        for key, value in counts.items()
    ]


def _score_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [value for value in (_number(item.get("score")) for item in records) if value is not None]
    thresholds = [value for value in (_number(item.get("threshold")) for item in records) if value is not None]
    statistics = _stats(values)
    if not values:
        return {"bins": [], "statistics": statistics, "average_threshold": None}
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        bins = [{"lower": minimum, "upper": maximum, "label": f"{minimum:.4f}", "count": len(values)}]
    else:
        count = min(12, max(5, int(round(math.sqrt(len(values))))))
        width = (maximum - minimum) / count
        bins = []
        for index in range(count):
            lower = minimum + index * width
            upper = maximum if index == count - 1 else minimum + (index + 1) * width
            contained = sum((lower <= value <= upper) if index == count - 1 else (lower <= value < upper) for value in values)
            bins.append({"lower": lower, "upper": upper, "label": f"{lower:.3f}–{upper:.3f}", "count": contained})
    return {"bins": bins, "statistics": statistics, "average_threshold": _stats(thresholds)["mean"]}


def _category_distribution(records: list[dict[str, Any]], module: str) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "anomaly": 0})
    for item in records:
        category = str(item.get("category") or "未记录类别")
        if module == "single":
            if item.get("status") != "success":
                continue
            groups[category]["total"] += 1
            groups[category]["anomaly"] += int(item.get("prediction") == "anomaly")
        else:
            groups[category]["total"] += int(item.get("normal_count", 0)) + int(item.get("anomaly_count", 0))
            groups[category]["anomaly"] += int(item.get("anomaly_count", 0))
    return [
        {
            "category": category,
            "total": values["total"],
            "anomaly": values["anomaly"],
            "anomaly_rate": float(values["anomaly"] / values["total"] * 100.0) if values["total"] else None,
        }
        for category, values in sorted(groups.items(), key=lambda item: item[1]["total"], reverse=True)
    ]


def _scale_distribution(values: list[int]) -> list[dict[str, Any]]:
    if not values:
        return []
    maximum = max(values)
    if maximum <= 10:
        edges = [1, 3, 6, 11]
    elif maximum <= 50:
        edges = [1, 11, 21, 51]
    elif maximum <= 100:
        edges = [1, 11, 51, 101]
    elif maximum <= 500:
        edges = [1, 11, 51, 101, 501]
    else:
        edges = [1, 11, 51, 101, 501, maximum + 1]
    output = []
    for lower, exclusive_upper in zip(edges, edges[1:]):
        upper = exclusive_upper - 1
        label = f"{lower}–{upper}" if exclusive_upper <= maximum + 1 else f"{lower}+"
        output.append({"label": label, "count": sum(lower <= value < exclusive_upper for value in values)})
    return [row for row in output if row["count"] > 0]


def _ratio_distribution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = [(0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.000001)]
    values = [value for value in (_number(item.get("anomaly_rate")) for item in records) if value is not None]
    return [
        {"label": f"{lower:g}%–{min(upper, 100):g}%", "count": sum(lower <= value < upper for value in values)}
        for lower, upper in edges
    ] if values else []


def _visualization_comparison(records: list[dict[str, Any]], module: str) -> list[dict[str, Any]]:
    output = []
    for enabled in (True, False):
        group = [item for item in records if item.get("generate_visualizations") is enabled and item.get("status") == "success"]
        if not group:
            continue
        end_values = [value for value in (_number(item.get("end_to_end_ms")) for item in group) if value is not None]
        row: dict[str, Any] = {
            "generate_visualizations": enabled,
            "task_count": len(group),
            "average_end_to_end_ms": _stats(end_values)["mean"],
        }
        if module == "batch":
            row["average_throughput"] = _stats([value for value in (_number(item.get("throughput")) for item in group) if value is not None])["mean"]
            row["average_item_ms"] = _stats([value for value in (_number(item.get("average_item_ms")) for item in group) if value is not None])["mean"]
        elif module == "video":
            row["average_full_pipeline_fps"] = _stats([value for value in (_number(item.get("full_pipeline_fps")) for item in group) if value is not None])["mean"]
            row["average_frame_ms"] = _stats([value for value in (_number(item.get("average_frame_ms")) for item in group) if value is not None])["mean"]
            encode_totals = []
            for item in group:
                encode = (item.get("stage_statistics") or {}).get("encode_ms") or {}
                value = _number(encode.get("total_ms"))
                if value is not None:
                    encode_totals.append(value)
            row["average_encode_ms"] = _stats(encode_totals)["mean"]
        output.append(row)
    return output


class StatisticsService:
    """Aggregate real SQLite results and controlled job manifests for the dashboard."""

    def __init__(self, settings: Any, storage: Any):
        self.settings = settings
        self.storage = storage
        self._source_cache_lock = threading.RLock()
        self._source_cache_revision = -1
        self._database_rows_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._manifest_cache: dict[str, list[dict[str, Any]]] | None = None

    def _reset_source_cache(self, revision: int) -> None:
        if self._source_cache_revision == revision:
            return
        self._source_cache_revision = revision
        self._database_rows_cache = {}
        self._manifest_cache = None

    def _manifests(self, module: str, start: datetime | None, end: datetime | None, revision: int) -> list[dict[str, Any]]:
        with self._source_cache_lock:
            self._reset_source_cache(revision)
            if self._manifest_cache is None:
                grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in MODULES}
                deleted_record_keys = self.storage.deleted_history_keys()
                root = self.settings.jobs_root
                if root.exists():
                    for directory in root.iterdir():
                        if not directory.is_dir():
                            continue
                        path = directory / "job_manifest.json"
                        if not path.is_file():
                            continue
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            continue
                        kind = str(payload.get("kind") or "") if isinstance(payload, dict) else ""
                        if kind not in MODULES or str(payload.get("job_id") or "") in deleted_record_keys:
                            continue
                        created = _manifest_time(payload)
                        if created is None:
                            continue
                        payload["_created_at_utc"] = created
                        grouped[kind].append(payload)
                self._manifest_cache = grouped
            manifests = list(self._manifest_cache.get(module) or [])
        return [
            payload
            for payload in manifests
            if (start is None or payload["_created_at_utc"] >= start)
            and (end is None or payload["_created_at_utc"] <= end)
        ]

    def _database_rows(self, start: datetime | None, end: datetime | None, revision: int) -> list[dict[str, Any]]:
        key = (start.isoformat() if start else "", end.isoformat() if end else "")
        with self._source_cache_lock:
            self._reset_source_cache(revision)
            cached = self._database_rows_cache.get(key)
            if cached is None:
                cached = self.storage.statistics_rows(start, end)
                self._database_rows_cache[key] = cached
            return cached

    def _single_records(
        self,
        manifests: list[dict[str, Any]],
        database_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {str(row.get("result_id")): row for row in database_rows if row.get("result_id")}
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in database_rows:
            metadata = row.get("metadata") or {}
            task_id = str(metadata.get("task_id") or "")
            if task_id:
                by_task[task_id].append(row)
        records = []
        used_result_ids: set[str] = set()
        for manifest in manifests:
            task_id = str(manifest.get("job_id") or "")
            result_id = str(manifest.get("result_id") or "")
            row = by_id.get(result_id) or next(iter(by_task.get(task_id) or []), None)
            if row and row.get("result_id"):
                used_result_ids.add(str(row["result_id"]))
            records.append(self._single_record(manifest, row))
        for row in database_rows:
            if row.get("batch_id") or str(row.get("result_id")) in used_result_ids:
                continue
            records.append(self._single_record({}, row))
        return records

    @staticmethod
    def _single_record(manifest: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
        metadata = (row or {}).get("metadata") or {}
        selection = manifest.get("selection") or {}
        created = manifest.get("_created_at_utc")
        if created is None and row:
            created = _parse_datetime(row.get("created_at"), "created_at")
        timing = _single_timing(row)
        status = _normal_status(manifest.get("status") or ("completed" if row else "failed"))
        prediction = str((row or {}).get("prediction") or "").lower()
        if prediction not in {"normal", "anomaly"}:
            prediction = "failed" if status == "failed" else "unknown"
        profile = selection.get("memory_bank_profile") or {}
        generate_value = metadata.get("generate_visualizations")
        if generate_value is None and row:
            generate_value = bool(row.get("result_image_url") or row.get("heatmap_url") or row.get("mask_url"))
        config_value = metadata.get("config_version") or profile.get("config", {}).get("final_profile") or ""
        return {
            "record_key": str(manifest.get("job_id") or (row or {}).get("result_id") or ""),
            "created_at_utc": created,
            "created_at": _local_iso(created),
            "filename": _safe_name((row or {}).get("filename") or metadata.get("filename"), "未记录文件名"),
            "dataset": str(metadata.get("dataset") or selection.get("dataset") or ""),
            "category": str(metadata.get("category") or selection.get("category") or ""),
            "config": _safe_name(config_value, ""),
            "device": str((row or {}).get("device") or selection.get("device") or ""),
            "model_version": str((row or {}).get("model_id") or selection.get("model_id") or ""),
            "memory_bank_id": str(metadata.get("memory_bank_id") or selection.get("memory_bank_id") or ""),
            "prediction": prediction,
            "status": status,
            "score": _number((row or {}).get("anomaly_score")),
            "threshold": _number((row or {}).get("threshold")),
            "pure_inference_ms": _number(timing.get("inference_ms")),
            "end_to_end_ms": _number(timing.get("end_to_end_ms")),
            "generate_visualizations": generate_value if isinstance(generate_value, bool) else None,
            "timing": timing,
        }

    def _batch_records(self, manifests: list[dict[str, Any]], database_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in database_rows:
            batch_id = str(row.get("batch_id") or "")
            if batch_id:
                by_batch[batch_id].append(row)
        records = []
        used: set[str] = set()
        for manifest in manifests:
            job_id = str(manifest.get("job_id") or "")
            used.add(job_id)
            records.append(self._batch_record(manifest, by_batch.get(job_id, [])))
        for batch_id, rows in by_batch.items():
            if batch_id not in used:
                records.append(self._batch_record({"job_id": batch_id}, rows))
        return records

    @staticmethod
    def _batch_record(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        selection = manifest.get("selection") or {}
        created = manifest.get("_created_at_utc")
        if created is None and rows:
            created = _parse_datetime(rows[0].get("created_at"), "created_at")
        timing = manifest.get("timing") or {}
        stage_statistics = dict(timing.get("stages") or {})
        if not stage_statistics and rows:
            for stage in STAGE_KEYS:
                values = [value for value in (_number(_single_timing(row).get(stage)) for row in rows) if value is not None]
                if values:
                    stats = _stats(values)
                    stage_statistics[stage] = {
                        "count": stats["count"], "total_ms": stats["total"], "mean_ms": stats["mean"],
                        "p50_ms": stats["p50"], "p95_ms": stats["p95"], "min_ms": stats["min"], "max_ms": stats["max"],
                    }
        normal = sum(not bool(row.get("is_anomaly")) for row in rows)
        anomaly = sum(bool(row.get("is_anomaly")) for row in rows)
        success = _integer(manifest.get("success_count")) or len(rows)
        failed = _integer(manifest.get("failure_count"))
        total = _integer(manifest.get("input_count")) or success + failed
        source_names = [_safe_parent_name((row.get("metadata") or {}).get("source_path"), "") for row in rows]
        input_name = next((name for name in source_names if name), "批量输入目录")
        metadata = (rows[0].get("metadata") or {}) if rows else {}
        generate_value = manifest.get("generate_visualizations")
        if generate_value is None:
            generate_value = selection.get("save_visuals")
        if generate_value is None and rows:
            generate_value = any(bool(row.get("result_image_url")) for row in rows)
        task_end = _number(timing.get("task_end_to_end_ms"))
        end_stats = stage_statistics.get("end_to_end_ms") or {}
        inference_stats = stage_statistics.get("inference_ms") or {}
        throughput = _number(timing.get("throughput_images_per_second"))
        if throughput is None and task_end and success:
            throughput = float(success / (task_end / 1000.0))
        status = _normal_status(manifest.get("status") or ("completed" if rows else "failed"))
        return {
            "record_key": str(manifest.get("job_id") or (rows[0].get("batch_id") if rows else "")),
            "created_at_utc": created,
            "created_at": _local_iso(created),
            "input_name": input_name,
            "dataset": str(metadata.get("dataset") or selection.get("dataset") or ""),
            "category": str(metadata.get("category") or selection.get("category") or ""),
            "config": _safe_name(metadata.get("config_version"), ""),
            "device": str(rows[0].get("device") if rows else selection.get("device") or ""),
            "model_version": str(rows[0].get("model_id") if rows else selection.get("model_id") or ""),
            "memory_bank_id": str(metadata.get("memory_bank_id") or selection.get("memory_bank_id") or ""),
            "prediction": "anomaly" if anomaly else ("normal" if success else "failed"),
            "status": status,
            "total_images": total,
            "successful_items": success,
            "failed_items": failed,
            "normal_count": normal,
            "anomaly_count": anomaly,
            "anomaly_rate": float(anomaly / success * 100.0) if success else None,
            "end_to_end_ms": task_end,
            "average_item_ms": _number(end_stats.get("mean_ms")),
            "average_inference_ms": _number(inference_stats.get("mean_ms")),
            "throughput": throughput,
            "generate_visualizations": generate_value if isinstance(generate_value, bool) else None,
            "stage_statistics": stage_statistics,
        }

    @staticmethod
    def _video_records(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for manifest in manifests:
            summary = manifest.get("summary") or {}
            selection = manifest.get("selection") or {}
            metadata = manifest.get("input_video_metadata") or summary.get("input_video_metadata") or {}
            timing = summary.get("timing") or {}
            stages = dict(timing.get("stages") or {})
            success_frames = _integer(summary.get("successful_processed_frames") or (manifest.get("progress") or {}).get("successful_frames"))
            failed_frames = _integer(summary.get("failed_frames") or (manifest.get("progress") or {}).get("failed_frames"))
            anomaly_frames = _integer(summary.get("anomaly_frames") or (manifest.get("progress") or {}).get("anomaly_frames"))
            normal_frames = _integer(summary.get("normal_frames")) or max(0, success_frames - anomaly_frames)
            generate_value = summary.get("generate_visualizations")
            if generate_value is None:
                generate_value = selection.get("generate_visualizations")
            if generate_value is None:
                generate_value = bool(summary.get("output_video_path"))
            frame_stats = stages.get("frame_total_ms") or {}
            average_frame = _number(frame_stats.get("mean_ms")) or _number(summary.get("average_frame_pipeline_ms"))
            p50_frame = _number(frame_stats.get("p50_ms"))
            p95_frame = _number(frame_stats.get("p95_ms"))
            if not stages:
                inference_average = _number(summary.get("average_inference_ms"))
                if inference_average is not None and success_frames:
                    stages["inference_ms"] = {"count": success_frames, "total_ms": inference_average * success_frames, "mean_ms": inference_average}
                total_processing = _number(summary.get("total_processing_ms"))
                if total_processing is not None:
                    stages["end_to_end_ms"] = {"count": 1, "total_ms": total_processing, "mean_ms": total_processing}
                if average_frame is not None:
                    stages["frame_total_ms"] = {"count": success_frames, "total_ms": average_frame * success_frames, "mean_ms": average_frame}
            full_fps = _number(summary.get("full_pipeline_fps")) or _number(summary.get("processing_throughput_fps"))
            pure_fps = _number(summary.get("pure_inference_fps"))
            average_inference = _number(summary.get("average_inference_ms"))
            if pure_fps is None and average_inference and average_inference > 0:
                pure_fps = 1000.0 / average_inference
            status = _normal_status(manifest.get("status"))
            created = manifest.get("_created_at_utc")
            records.append(
                {
                    "record_key": str(manifest.get("job_id") or ""),
                    "created_at_utc": created,
                    "created_at": _local_iso(created),
                    "filename": _safe_name(manifest.get("original_filename") or summary.get("input_video_path"), "未记录视频名"),
                    "dataset": str(summary.get("dataset") or selection.get("dataset") or ""),
                    "category": str(summary.get("category") or selection.get("category") or ""),
                    "config": _safe_name(summary.get("config_path"), ""),
                    "device": str(summary.get("device") or selection.get("device") or ""),
                    "model_version": str(summary.get("model") or selection.get("model_id") or ""),
                    "memory_bank_id": str(summary.get("memory_bank_id") or selection.get("memory_bank_id") or ""),
                    "prediction": "anomaly" if anomaly_frames else ("normal" if success_frames else "failed"),
                    "status": status,
                    "duration_seconds": _number(metadata.get("duration_seconds")) or 0.0,
                    "total_frames": _integer(summary.get("total_decoded_records") or metadata.get("reported_frame_count")),
                    "successful_items": success_frames,
                    "failed_items": failed_frames,
                    "normal_count": normal_frames,
                    "anomaly_count": anomaly_frames,
                    "anomaly_rate": float(anomaly_frames / success_frames * 100.0) if success_frames else None,
                    "input_fps": _number(summary.get("source_fps") or metadata.get("source_fps")),
                    "pure_inference_fps": pure_fps,
                    "full_pipeline_fps": full_fps,
                    "reached_input_fps": bool(summary.get("reached_input_fps")) if summary.get("reached_input_fps") is not None else None,
                    "average_frame_ms": average_frame,
                    "p50_frame_ms": p50_frame,
                    "p95_frame_ms": p95_frame,
                    "end_to_end_ms": _number(timing.get("end_to_end_ms")) or _number(summary.get("total_processing_ms")),
                    "generate_visualizations": generate_value if isinstance(generate_value, bool) else None,
                    "stage_statistics": stages,
                }
            )
        return records

    @staticmethod
    def _filter_options(records: list[dict[str, Any]]) -> dict[str, list[str]]:
        return {
            key: sorted({str(item.get(key)) for item in records if item.get(key) not in (None, "")})
            for key in ("dataset", "category", "config", "device", "model_version", "memory_bank_id", "prediction", "status")
        }

    @staticmethod
    def _apply_filters(records: list[dict[str, Any]], filters: dict[str, str | None]) -> list[dict[str, Any]]:
        output = records
        for key, expected in filters.items():
            if expected not in (None, "", "all"):
                output = [item for item in output if str(item.get(key) or "") == str(expected)]
        return output

    @staticmethod
    def _summary(records: list[dict[str, Any]], module: str) -> dict[str, Any]:
        success_tasks = sum(item.get("status") == "success" for item in records)
        failed_tasks = sum(item.get("status") == "failed" for item in records)
        cancelled_tasks = sum(item.get("status") == "cancelled" for item in records)
        if module == "single":
            anomalies = sum(item.get("prediction") == "anomaly" for item in records)
            normals = sum(item.get("prediction") == "normal" for item in records)
            scores = [value for value in (_number(item.get("score")) for item in records) if value is not None]
            end_values = [value for value in (_number(item.get("end_to_end_ms")) for item in records) if value is not None]
            inference = [value for value in (_number(item.get("pure_inference_ms")) for item in records) if value is not None]
            return {
                "task_count": len(records), "success_tasks": success_tasks, "failed_tasks": failed_tasks,
                "cancelled_tasks": cancelled_tasks, "normal_count": normals, "anomaly_count": anomalies,
                "anomaly_rate": float(anomalies / (normals + anomalies) * 100.0) if normals + anomalies else None,
                "average_score": _stats(scores)["mean"], "average_end_to_end_ms": _stats(end_values)["mean"],
                "average_inference_ms": _stats(inference)["mean"],
                "visualized_tasks": sum(item.get("generate_visualizations") is True for item in records),
            }
        total_items = sum(item.get("total_images" if module == "batch" else "total_frames", 0) for item in records)
        anomaly_items = sum(item.get("anomaly_count", 0) for item in records)
        normal_items = sum(item.get("normal_count", 0) for item in records)
        end_values = [value for value in (_number(item.get("end_to_end_ms")) for item in records) if value is not None]
        if module == "batch":
            item_times = [value for value in (_number(item.get("average_item_ms")) for item in records) if value is not None]
            throughput = [value for value in (_number(item.get("throughput")) for item in records) if value is not None]
            return {
                "task_count": len(records), "success_tasks": success_tasks, "failed_tasks": failed_tasks,
                "cancelled_tasks": cancelled_tasks, "total_images": total_items, "normal_count": normal_items,
                "anomaly_count": anomaly_items, "average_batch_size": float(total_items / len(records)) if records else None,
                "average_task_ms": _stats(end_values)["mean"], "average_item_ms": _stats(item_times)["mean"],
                "average_throughput": _stats(throughput)["mean"],
            }
        durations = [float(item.get("duration_seconds", 0.0)) for item in records]
        anomaly_rates = [value for value in (_number(item.get("anomaly_rate")) for item in records) if value is not None]
        full_fps = [value for value in (_number(item.get("full_pipeline_fps")) for item in records) if value is not None]
        pure_fps = [value for value in (_number(item.get("pure_inference_fps")) for item in records) if value is not None]
        reached = [item.get("reached_input_fps") for item in records if item.get("reached_input_fps") is not None]
        return {
            "task_count": len(records), "success_tasks": success_tasks, "failed_tasks": failed_tasks,
            "cancelled_tasks": cancelled_tasks, "total_duration_seconds": float(sum(durations)),
            "total_frames": total_items, "anomaly_frames": anomaly_items,
            "average_anomaly_rate": _stats(anomaly_rates)["mean"], "average_full_pipeline_fps": _stats(full_fps)["mean"],
            "average_pure_inference_fps": _stats(pure_fps)["mean"],
            "average_task_ms": _stats(end_values)["mean"],
            "reached_input_fps_rate": float(sum(bool(value) for value in reached) / len(reached) * 100.0) if reached else None,
        }

    @staticmethod
    def _detail_rows(records: list[dict[str, Any]], module: str) -> list[dict[str, Any]]:
        rows = []
        for item in records:
            common = {
                "time": item.get("created_at"), "dataset": item.get("dataset"),
                "category": item.get("category"), "status": item.get("status"),
                "generate_visualizations": item.get("generate_visualizations"),
            }
            if module == "single":
                common.update(
                    filename=item.get("filename"), prediction=item.get("prediction"), score=item.get("score"),
                    threshold=item.get("threshold"), pure_inference_ms=item.get("pure_inference_ms"),
                    end_to_end_ms=item.get("end_to_end_ms"),
                )
            elif module == "batch":
                common.update(
                    input_name=item.get("input_name"), total_images=item.get("total_images"),
                    successful_items=item.get("successful_items"), failed_items=item.get("failed_items"),
                    anomaly_count=item.get("anomaly_count"), anomaly_rate=item.get("anomaly_rate"),
                    end_to_end_ms=item.get("end_to_end_ms"), average_item_ms=item.get("average_item_ms"),
                    throughput=item.get("throughput"),
                )
            else:
                common.update(
                    filename=item.get("filename"), duration_seconds=item.get("duration_seconds"),
                    total_frames=item.get("total_frames"), anomaly_count=item.get("anomaly_count"),
                    anomaly_rate=item.get("anomaly_rate"), input_fps=item.get("input_fps"),
                    pure_inference_fps=item.get("pure_inference_fps"), full_pipeline_fps=item.get("full_pipeline_fps"),
                    end_to_end_ms=item.get("end_to_end_ms"),
                )
            rows.append(common)
        return rows

    def query(
        self,
        *,
        module: str,
        start_time: str | None = None,
        end_time: str | None = None,
        dataset: str | None = None,
        category: str | None = None,
        config: str | None = None,
        device: str | None = None,
        model_version: str | None = None,
        memory_bank_id: str | None = None,
        prediction: str | None = None,
        task_status: str | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        if module not in MODULES:
            raise ApiError("INVALID_STATISTICS_MODULE", "统计模块必须为 single、batch 或 video。", 422)
        start = _parse_datetime(start_time, "开始时间")
        end = _parse_datetime(end_time, "结束时间")
        now = datetime.now(timezone.utc)
        if start is not None and end is not None and start > end:
            raise ApiError("INVALID_STATISTICS_RANGE", "开始时间不能晚于结束时间。", 422)
        if end is not None and end > now + timedelta(seconds=5):
            raise ApiError("INVALID_STATISTICS_RANGE", "结束时间不能晚于当前时间。", 422)
        safe_page = max(1, int(page))
        safe_page_size = min(100, max(5, int(page_size)))
        revision = self.storage.statistics_revision()
        manifests = self._manifests(module, start, end, revision)
        database_rows = self._database_rows(start, end, revision)
        if module == "single":
            records = self._single_records(manifests, database_rows)
        elif module == "batch":
            records = self._batch_records(manifests, database_rows)
        else:
            records = self._video_records(manifests)
        records = [
            record for record in records
            if record.get("created_at_utc") is not None
            and (start is None or record["created_at_utc"] >= start)
            and (end is None or record["created_at_utc"] <= end)
        ]
        options = self._filter_options(records)
        filtered = self._apply_filters(
            records,
            {
                "dataset": dataset, "category": category, "config": config, "device": device,
                "model_version": model_version, "memory_bank_id": memory_bank_id,
                "prediction": prediction, "status": task_status,
            },
        )
        mode = _bucket_mode(start, end, filtered)
        sort_aliases = {
            "created_at": "created_at_utc", "score": "score", "end_to_end_ms": "end_to_end_ms",
            "throughput": "throughput", "item_count": "total_images" if module == "batch" else "total_frames",
            "anomaly_rate": "anomaly_rate", "full_pipeline_fps": "full_pipeline_fps",
        }
        sort_key = sort_aliases.get(sort_by, "created_at_utc")
        reverse = str(sort_order).lower() != "asc"
        filtered.sort(
            key=lambda item: (item.get(sort_key) is not None, item.get(sort_key)),
            reverse=reverse,
        )
        total = len(filtered)
        offset = (safe_page - 1) * safe_page_size
        page_records = filtered[offset : offset + safe_page_size]
        details = self._detail_rows(page_records, module)
        timing_rows = _stage_statistics(filtered, task_stage_totals=module in {"batch", "video"})
        response = {
            "module": module,
            "timezone": "Asia/Shanghai",
            "range_boundary": "开始时间和结束时间均包含在统计区间内",
            "statistics_revision": revision,
            "current_time": _local_iso(now),
            "summary": self._summary(filtered, module),
            "time_series": _time_series(filtered, module, mode),
            "result_distribution": _distribution(filtered, module),
            "category_distribution": _category_distribution(filtered, module),
            "timing_statistics": {"stages": timing_rows},
            "visualization_comparison": _visualization_comparison(filtered, module),
            "task_details": details,
            "pagination": {
                "page": safe_page, "page_size": safe_page_size, "total": total,
                "total_pages": max(1, math.ceil(total / safe_page_size)) if total else 1,
            },
            "filter_options": options,
            "active_filters": {
                "start_time": _local_iso(start), "end_time": _local_iso(end), "dataset": dataset,
                "category": category, "config": config, "device": device, "model_version": model_version,
                "memory_bank_id": memory_bank_id, "prediction": prediction, "task_status": task_status,
                "sort_by": sort_by, "sort_order": sort_order,
            },
        }
        if module == "single":
            response["score_distribution"] = _score_distribution(filtered)
        elif module == "batch":
            response["scale_distribution"] = _scale_distribution([int(item.get("total_images", 0)) for item in filtered if item.get("total_images", 0) > 0])
        else:
            response["anomaly_ratio_distribution"] = _ratio_distribution(filtered)
            response["fps_comparison"] = [
                {
                    "label": f"视频 {index + 1}", "input_fps": item.get("input_fps"),
                    "pure_inference_fps": item.get("pure_inference_fps"), "full_pipeline_fps": item.get("full_pipeline_fps"),
                }
                for index, item in enumerate(filtered[:30])
            ]
        return response


__all__ = ["StatisticsService"]
