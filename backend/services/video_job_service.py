"""Asynchronous, single-task orchestration for offline video-file detection.

The Streamlit UI only submits and polls these jobs.  Model loading, decoding,
per-frame inference and MP4 encoding stay in this algorithm-system process and
reuse :class:`RuntimeRegistry` entries instead of loading a model per frame.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.adapters.result_serializer import artifact_url
from backend.errors import ApiError
from backend.services.detection_context import DetectionContext
from backend.services.job_service import JobPaths, JobService


class _LazyModule:
    """Resolve optional video-native dependencies only when a video job runs."""

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)


cv2 = _LazyModule("cv2")


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
MAX_VIDEO_UPLOAD_BYTES = 500 * 1024 * 1024
HEARTBEAT_TIMEOUT_SECONDS = 20.0
PROGRESS_PERSIST_SECONDS = 0.25
PREVIEW_FRAME_INTERVAL = 5
_BROWSER_NATIVE_FOURCC = {"H264", "AVC1", "X264", "VP80", "VP90", "AV01"}
_ACTIVE_VIDEO_STATES = {
    "queued",
    "initializing",
    "running",
    "pausing",
    "paused",
    "resuming",
    "cancelling",
}
_TERMINAL_VIDEO_STATES = {"completed", "cancelled", "failed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_filename(filename: str, fallback: str = "upload.mp4") -> str:
    name = Path(filename or fallback).name.strip()
    return name or fallback


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_write_preview(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("当前预览帧无法编码为 JPEG。")
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def _phash_frame(frame: Any) -> str:
    """Return a small deterministic perceptual hash for representative-frame dedup."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(getattr(frame, "shape", ())) == 3 else frame
    gray = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    median = float(gray.mean())
    return "".join("1" if float(value) >= median else "0" for value in gray.reshape(-1))


def _hamming_distance(left: str, right: str) -> int:
    if not left or not right or len(left) != len(right):
        return 64
    return sum(a != b for a, b in zip(left, right))


def _metadata_for(path: Path) -> dict[str, Any]:
    """Validate metadata and decode one real frame before enabling Start."""
    # Loading the video inference package also imports PatchCore.  A probe is
    # user-triggered, so defer it and let the API expose liveness first.
    from video.video_reader import VideoInputError, VideoReader

    try:
        with VideoReader(path) as reader:
            metadata = reader.metadata
            assert metadata is not None
            first = next(reader.iter_frames(max_frames=1), None)
            if first is None:
                raise VideoInputError(f"视频无法解码出任何有效帧：{path}")
            payload = metadata.as_dict()
    except (VideoInputError, OSError, ValueError) as exc:
        raise ApiError("VIDEO_NOT_DECODABLE", f"视频无法打开或解码：{exc}", 422) from exc
    payload["file_name"] = path.name
    payload["file_size_bytes"] = path.stat().st_size
    payload["decodable"] = True
    return payload


def _browser_preview_for(path: Path, metadata: dict[str, Any]) -> Path | None:
    """Create a browser-only VP8 preview when the source codec is not web-safe.

    The original staged file remains the sole inference input.  The preview is
    contained in the same controlled upload directory and is never written
    back to the user's source file.
    """
    fourcc = str(metadata.get("fourcc") or "").upper()
    if path.suffix.lower() in {".mp4", ".mov"} and fourcc in _BROWSER_NATIVE_FOURCC:
        return None
    target = path.parent / "browser_preview.webm"
    temporary = path.parent / "browser_preview.tmp.webm"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or metadata.get("source_fps") or 20.0)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if width <= 0 or height <= 0 or not writer.isOpened():
        capture.release()
        writer.release()
        temporary.unlink(missing_ok=True)
        return None
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written <= 0 or not temporary.exists() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        return None
    os.replace(temporary, target)
    return target


@dataclass
class VideoTask:
    job: JobPaths
    source_path: Path
    original_filename: str
    metadata: dict[str, Any]
    selection: dict[str, Any]
    detection_context: DetectionContext | None = None
    resume_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    state: str = "queued"
    current_frame: int = -1
    current_time_ms: float = 0.0
    anomaly_score: float | None = None
    prediction: str = ""
    anomaly_frames: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    preview_path: Path | None = None
    last_heartbeat: float = field(default_factory=time.monotonic)
    last_persist: float = 0.0
    error: dict[str, str] | None = None
    summary: dict[str, Any] | None = None
    model_cache_hit: bool | None = None
    runtime_device: str = ""
    runtime_model_id: str = ""
    runtime_dataset: str = ""
    runtime_category: str = ""
    runtime_memory_bank_id: str = ""
    runtime_threshold: float | None = None
    runtime_config_version: str = ""
    frame_contexts: dict[int, dict[str, Any]] = field(default_factory=dict)
    keyframe_feedback_targets: list[dict[str, Any]] = field(default_factory=list)
    started_monotonic: float | None = None
    active_segment_started_monotonic: float | None = None
    active_processing_seconds: float = 0.0
    paused_seconds: float = 0.0
    pause_started_monotonic: float | None = None
    paused_at: str | None = None
    last_progress_at: str | None = None
    progress_samples: list[tuple[float, int]] = field(default_factory=list)
    current_fps: float | None = None
    average_processing_fps: float | None = None
    complete_fps: float | None = None
    eta_seconds: float | None = None
    highest_anomaly_score: float | None = None
    inference_total_ms: float = 0.0
    inference_count: int = 0

    def __post_init__(self) -> None:
        self.resume_event.set()


class VideoJobService:
    """Owns one mutually-exclusive local video-file detection task at a time."""

    def __init__(self, settings: Any, jobs: JobService, adapter: Any, registry: Any, logger: Any):
        self.settings = settings
        self.jobs = jobs
        self.adapter = adapter
        self.registry = registry
        self.logger = logger
        self._lock = threading.RLock()
        self._tasks: dict[str, VideoTask] = {}
        self._active_job_id: str | None = None

    @property
    def upload_root(self) -> Path:
        root = self.settings.upload_root / "video_files"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _active_task(self) -> VideoTask | None:
        if not self._active_job_id:
            return None
        task = self._tasks.get(self._active_job_id)
        if task and task.state in _ACTIVE_VIDEO_STATES:
            return task
        return None

    @staticmethod
    def _active_seconds(task: VideoTask, now: float | None = None) -> float:
        seconds = float(task.active_processing_seconds)
        if task.active_segment_started_monotonic is not None:
            seconds += max(0.0, (now if now is not None else time.monotonic()) - task.active_segment_started_monotonic)
        return seconds

    @staticmethod
    def _paused_seconds(task: VideoTask, now: float | None = None) -> float:
        seconds = float(task.paused_seconds)
        if task.pause_started_monotonic is not None:
            seconds += max(0.0, (now if now is not None else time.monotonic()) - task.pause_started_monotonic)
        return seconds

    @staticmethod
    def _start_active_segment(task: VideoTask, now: float | None = None) -> None:
        if task.active_segment_started_monotonic is None:
            task.active_segment_started_monotonic = now if now is not None else time.monotonic()

    @staticmethod
    def _close_active_segment(task: VideoTask, now: float | None = None) -> None:
        if task.active_segment_started_monotonic is None:
            return
        timestamp = now if now is not None else time.monotonic()
        task.active_processing_seconds += max(0.0, timestamp - task.active_segment_started_monotonic)
        task.active_segment_started_monotonic = None

    def _refresh_performance(self, task: VideoTask, *, record_progress: bool = False) -> None:
        """Refresh rate facts only at worker progress/state boundaries.

        Status polling intentionally never calls this method.  Therefore a
        paused or terminal task cannot acquire a different FPS merely because
        wall-clock time advanced in the browser or API process.
        """
        active_seconds = self._active_seconds(task)
        completed_items = max(0, task.current_frame + 1)
        if record_progress and (not task.progress_samples or task.progress_samples[-1][1] != completed_items):
            task.progress_samples.append((active_seconds, completed_items))
            task.progress_samples = task.progress_samples[-32:]
            task.last_progress_at = _utc_now()
        task.average_processing_fps = completed_items / active_seconds if completed_items and active_seconds > 0 else None
        task.complete_fps = task.average_processing_fps
        if len(task.progress_samples) >= 2:
            latest_active, latest_count = task.progress_samples[-1]
            earliest_index = 0
            for index in range(len(task.progress_samples) - 2, -1, -1):
                if latest_active - task.progress_samples[index][0] >= 5.0:
                    earliest_index = index
                    break
            earliest_active, earliest_count = task.progress_samples[earliest_index]
            active_delta = latest_active - earliest_active
            count_delta = latest_count - earliest_count
            task.current_fps = count_delta / active_delta if count_delta > 0 and active_delta > 0 else task.average_processing_fps
        else:
            task.current_fps = task.average_processing_fps
        total = int(task.metadata.get("effective_total_frames") or task.metadata.get("reported_frame_count", 0))
        remaining = max(0, total - completed_items)
        if task.state in {"pausing", "paused", "resuming", "cancelling", "cancelled", "failed"}:
            task.eta_seconds = None
        elif task.state == "completed":
            task.eta_seconds = 0.0
        else:
            rate = task.current_fps or task.average_processing_fps
            task.eta_seconds = remaining / rate if rate and rate > 0 else None

    def _freeze_timing(self, task: VideoTask) -> None:
        now = time.monotonic()
        self._close_active_segment(task, now)
        if task.pause_started_monotonic is not None:
            task.paused_seconds += max(0.0, now - task.pause_started_monotonic)
            task.pause_started_monotonic = None
            task.paused_at = None
        self._refresh_performance(task)

    def _require_task(self, job_id: str) -> VideoTask:
        with self._lock:
            task = self._tasks.get(job_id)
        if task is None:
            durable = self.jobs.durable_state(job_id)
            if durable and durable.get("task_kind") == "video":
                raise ApiError("VIDEO_JOB_NOT_ACTIVE", "视频任务已持久化，但当前不在运行进程中；中断任务可重新运行。", 409)
            raise ApiError("VIDEO_JOB_NOT_FOUND", "视频检测任务不存在。", 404)
        return task

    def _historical_manifest(self, job_id: str) -> tuple[dict[str, Any], Path]:
        safe_id = str(job_id or "").strip()
        if not safe_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in safe_id):
            raise ApiError("INVALID_VIDEO_JOB_ID", "视频任务标识无效。", 422)
        job_root = (self.settings.jobs_root / safe_id).resolve()
        try:
            job_root.relative_to(self.settings.jobs_root.resolve())
        except ValueError as exc:
            raise ApiError("INVALID_VIDEO_JOB_ID", "视频任务标识无效。", 422) from exc
        manifest_path = job_root / "job_manifest.json"
        if not manifest_path.exists():
            raise ApiError("VIDEO_JOB_NOT_FOUND", "视频检测任务不存在。", 404)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError("VIDEO_JOB_INVALID", "视频任务记录无法读取。", 500) from exc
        if not isinstance(manifest, dict):
            raise ApiError("VIDEO_JOB_INVALID", "视频任务记录格式无效。", 500)
        return manifest, job_root

    def _historical_frame_record(self, job_id: str, frame_index: int) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
        manifest, job_root = self._historical_manifest(job_id)
        summary = dict(manifest.get("summary") or {})
        jsonl_value = str(summary.get("jsonl_path") or "")
        jsonl_path = Path(jsonl_value).resolve() if jsonl_value else (job_root / "results" / "video" / "frame_results.jsonl").resolve()
        try:
            jsonl_path.relative_to(job_root)
        except ValueError as exc:
            raise ApiError("VIDEO_FRAME_INDEX_INVALID", "视频帧明细路径无效。", 500) from exc
        if not jsonl_path.exists():
            raise ApiError("VIDEO_FRAME_NOT_FOUND", "该视频任务没有保存帧级检测明细。", 404)
        target_index = int(frame_index)
        record: dict[str, Any] = {}
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and int(item.get("frame_index", -1)) == target_index:
                        record = item
                        break
        except OSError as exc:
            raise ApiError("VIDEO_FRAME_INDEX_UNREADABLE", "视频帧级明细无法读取。", 500) from exc
        if not record:
            raise ApiError("VIDEO_FRAME_NOT_FOUND", "指定视频帧不在已保存检测明细中。", 404)
        source_value = str(
            summary.get("input_video_path")
            or manifest.get("source_path")
            or manifest.get("staged_input_path")
            or ""
        )
        source = Path(source_value).resolve() if source_value else Path()
        allowed_roots = [self.settings.runtime_root.resolve(), self.settings.upload_root.resolve()]
        if not source_value or not source.exists() or not any(
            _is_relative_to(source, root) for root in allowed_roots
        ):
            raise ApiError("VIDEO_SOURCE_UNAVAILABLE", "原始视频文件已不可用，无法读取该帧。", 404)
        return manifest, record, source, job_root

    def historical_frame_preview(self, job_id: str, frame_index: int) -> Path:
        """Lazily extract one persisted frame for history browsing."""
        _manifest, _record, source, job_root = self._historical_frame_record(job_id, frame_index)
        target = job_root / "results" / "video" / "frame_previews" / f"frame_{int(frame_index):08d}.jpg"
        if target.exists() and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ApiError("VIDEO_NOT_DECODABLE", "原始视频当前无法解码。", 422)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ApiError("VIDEO_FRAME_NOT_FOUND", "指定视频帧无法读取。", 404)
            temporary = target.with_name(f"{target.stem}.part{target.suffix}")
            if not cv2.imwrite(str(temporary), frame):
                raise ApiError("VIDEO_FRAME_WRITE_FAILED", "视频帧预览生成失败。", 500)
            os.replace(temporary, target)
            return target
        finally:
            capture.release()

    def _historical_feedback_context(self, job_id: str, frame_index: int) -> dict[str, Any]:
        manifest, record, source, _job_root = self._historical_frame_record(job_id, frame_index)
        summary = dict(manifest.get("summary") or {})
        selection = dict(manifest.get("selection") or {})
        memory_profile = dict(selection.get("memory_bank_profile") or {})
        input_metadata = dict(summary.get("input_video_metadata") or manifest.get("input_video_metadata") or {})
        method_mode = str(
            selection.get("method_mode")
            or memory_profile.get("method_mode_resolved")
            or record.get("method_mode")
            or "normal_only"
        )
        index = int(frame_index)
        memory_bank_id = str(selection.get("memory_bank_id") or summary.get("memory_bank_id") or "")
        config_profile_id = str(selection.get("config_profile_id") or summary.get("config_profile_id") or "")
        config_profile_version = int(selection.get("config_profile_version") or 0)
        return {
            "mode": "video",
            "task_id": str(job_id),
            "sample_id": f"frame:{index}",
            "result_id": None,
            "frame_index": index,
            "timestamp_ms": float(record.get("timestamp_ms") or 0.0),
            "source_label": f"{manifest.get('original_filename') or source.name} · 帧 {index}",
            "detection_time": str(record.get("detection_time") or manifest.get("created_at") or ""),
            "dataset": str(selection.get("dataset") or summary.get("dataset") or ""),
            "category": str(selection.get("category") or summary.get("category") or ""),
            "model_version": str(selection.get("model_id") or summary.get("model") or ""),
            "config_version": str(summary.get("calibration_version") or config_profile_version or ""),
            "memory_bank_id": memory_bank_id,
            "source_memory_bank_id": memory_bank_id,
            "source_profile_id": str(selection.get("profile_id") or memory_bank_id),
            "source_target_domain_id": str(selection.get("target_domain_id") or memory_bank_id),
            "source_config_snapshot_id": str(
                selection.get("config_snapshot_id")
                or (f"{config_profile_id}:v{config_profile_version}" if config_profile_id and config_profile_version else config_profile_id)
                or ""
            ),
            "threshold": record.get("threshold", selection.get("threshold")),
            "prediction": str(record.get("prediction") or ""),
            "score": record.get("score", record.get("anomaly_score")),
            "configuration_frozen": True,
            "config_profile_id": config_profile_id,
            "config_profile_version": config_profile_version,
            "method_mode": method_mode,
            "profile_method_mode": method_mode,
            "scoring_method_mode": str(record.get("method_mode") or method_mode),
            "image_size": memory_profile.get("image_size") or selection.get("image_size"),
            "feature_dimension": memory_profile.get("feature_dimension") or selection.get("feature_dimension"),
            "feature_layers": memory_profile.get("feature_layers") or selection.get("feature_layers") or [],
            "backbone": memory_profile.get("backbone") or selection.get("backbone"),
            "inference_mode": "inference_only",
            "official_test_mode": bool(selection.get("official_test_mode", False)),
            "test_data_update_allowed": False,
            "data_role": "official_test_set" if selection.get("official_test_mode", False) else "inference_input",
            "evaluation_mode": "official_test" if selection.get("official_test_mode", False) else "inference_only",
            "source_artifact_ref": f"video_frame:{job_id}:{index}",
            "source_sha256": str(manifest.get("source_sha256") or ""),
            "source_width": int(input_metadata.get("width") or 0),
            "source_height": int(input_metadata.get("height") or 0),
            "perceptual_hash": "",
            "_source_path": str(source),
        }

    def _durable_status(self, job_id: str) -> dict[str, Any] | None:
        durable = self.jobs.durable_state(job_id)
        if not durable or durable.get("task_kind") != "video":
            return None
        manifest = dict(durable.get("payload") or {})
        progress = dict(manifest.get("progress") or {})
        runtime_status = str(durable.get("runtime_status") or durable.get("status") or "interrupted")
        persisted_context = dict(manifest.get("detection_context") or (manifest.get("selection") or {}).get("detection_context") or {})
        persisted_revision = manifest.get("runtime_revision")
        if persisted_revision is None:
            persisted_revision = persisted_context.get("runtime_revision")
        return {
            "job_id": job_id,
            "status": runtime_status,
            "complete": runtime_status == "completed",
            "cancelled": runtime_status == "cancelled",
            "interrupted": runtime_status == "interrupted",
            "rerunnable": bool(durable.get("rerunnable")),
            "completed_items": int(manifest.get("completed_items") or progress.get("current_frame", -1) + 1 or 0),
            "total_items": int(manifest.get("total_items") or progress.get("total_frames") or 0),
            "active_processing_seconds": manifest.get("active_processing_seconds"),
            "paused_seconds": manifest.get("paused_seconds"),
            "current_fps": manifest.get("current_fps"),
            "inference_fps": manifest.get("inference_fps"),
            "complete_fps": manifest.get("complete_fps"),
            "eta_seconds": None if runtime_status == "interrupted" else manifest.get("eta_seconds"),
            "paused_at": manifest.get("paused_at"),
            "last_progress_at": manifest.get("last_progress_at"),
            "error": manifest.get("error"),
            "input": {
                "original_filename": manifest.get("original_filename"),
                "metadata": manifest.get("input_video_metadata") or {},
            },
            "progress": progress,
            "runtime_revision": int(persisted_revision) if persisted_revision is not None else None,
            "detection_context": persisted_context,
            "summary": manifest.get("summary") or {},
            "artifacts": manifest.get("artifacts") or {},
            "manifest": manifest,
        }

    def upload_and_inspect(self, filename: str, content: bytes) -> dict[str, Any]:
        reservation = self.reserve_upload(filename)
        source = Path(reservation["source_path"])
        try:
            _atomic_write_bytes(source, content)
            return self.inspect_staged_upload(
                reservation["upload_id"],
                source,
                size_bytes=len(content),
                source_sha256=hashlib.sha256(content).hexdigest(),
            )
        except Exception:
            self.discard_upload(reservation["upload_id"])
            raise

    def reserve_upload(self, filename: str) -> dict[str, str]:
        safe_name = _safe_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            raise ApiError(
                "UNSUPPORTED_VIDEO_TYPE",
                f"不支持的视频扩展名：{suffix or '无扩展名'}。允许 .mp4、.avi、.mov、.mkv。",
                415,
                {"supported_extensions": sorted(SUPPORTED_VIDEO_SUFFIXES)},
            )
        upload_id = f"video_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
        upload_dir = self.upload_root / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        source = upload_dir / safe_name
        return {"upload_id": upload_id, "source_path": str(source)}

    def discard_upload(self, upload_id: str) -> None:
        directory = (self.upload_root / str(upload_id)).resolve()
        try:
            directory.relative_to(self.upload_root.resolve())
        except ValueError:
            return
        shutil.rmtree(directory, ignore_errors=True)

    def inspect_staged_upload(
        self,
        upload_id: str,
        source: Path,
        *,
        size_bytes: int,
        source_sha256: str,
    ) -> dict[str, Any]:
        upload_dir = (self.upload_root / str(upload_id)).resolve()
        source = Path(source).resolve()
        try:
            source.relative_to(upload_dir)
        except ValueError as exc:
            raise ApiError("VIDEO_UPLOAD_INVALID", "视频上传路径不在受控目录内。", 403) from exc
        if size_bytes <= 0 or not source.exists():
            raise ApiError("EMPTY_VIDEO_UPLOAD", "所选视频为空，无法开始检测。", 422)
        if size_bytes > MAX_VIDEO_UPLOAD_BYTES:
            raise ApiError("VIDEO_TOO_LARGE", "视频文件超过当前软件 500 MB 上传上限。", 413, {"max_bytes": MAX_VIDEO_UPLOAD_BYTES})
        safe_name = source.name
        try:
            metadata = _metadata_for(source)
        except ApiError as exc:
            # Decoder exceptions often embed the controlled staging path.
            # Keep filesystem details in server logs, never in the API/UI.
            self.logger.warning("video_upload_decode_failed upload_id=%s code=%s", upload_id, exc.code)
            raise ApiError(
                "VIDEO_NOT_DECODABLE",
                "视频无法打开或解码，请检查文件完整性和编码格式。",
                422,
            ) from exc
        native_preview = source.suffix.lower() in {".mp4", ".mov"} and str(metadata.get("fourcc") or "").upper() in _BROWSER_NATIVE_FOURCC
        metadata["browser_preview_generated"] = False
        metadata["browser_preview_pending"] = not native_preview
        metadata["browser_preview_format"] = "原始视频" if native_preview else "预览转换中"
        manifest = {
            "upload_id": upload_id,
            "created_at": _utc_now(),
            "original_filename": safe_name,
            "staged_input_path": str(source),
            "browser_preview_path": "",
            "metadata": metadata,
            "source_sha256": source_sha256,
            "note": "上传视频仅用于本地离线视频文件检测；未写入项目根目录。",
        }
        manifest_path = upload_dir / "upload_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not native_preview:
            threading.Thread(
                target=self._build_browser_preview,
                args=(upload_id, source),
                name=f"AdaptiveAOI-VideoPreview-{upload_id[-8:]}",
                daemon=True,
            ).start()
        return self.public_upload(upload_id)

    def _build_browser_preview(self, upload_id: str, source: Path) -> None:
        manifest_path = self.upload_root / str(upload_id) / "upload_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = dict(manifest.get("metadata") or {})
            preview = _browser_preview_for(source, metadata)
            metadata["browser_preview_generated"] = bool(preview)
            metadata["browser_preview_pending"] = False
            metadata["browser_preview_format"] = "VP8 / WebM" if preview else "原始视频"
            manifest["metadata"] = metadata
            manifest["browser_preview_path"] = str(preview) if preview else ""
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, manifest_path)
        except Exception as exc:
            self.logger.warning("video_preview_generation_failed upload_id=%s error=%s", upload_id, exc)

    def public_upload(self, upload_id: str) -> dict[str, Any]:
        manifest = self._upload_manifest(upload_id)
        metadata = dict(manifest.get("metadata") or {})
        metadata.pop("path", None)
        return {
            "upload_id": manifest.get("upload_id"),
            "created_at": manifest.get("created_at"),
            "original_filename": manifest.get("original_filename"),
            "metadata": metadata,
            "source_sha256": manifest.get("source_sha256"),
            "preview_url": f"/api/video/uploads/{upload_id}/content",
            "note": manifest.get("note"),
        }

    def upload_content(self, upload_id: str) -> tuple[Path, str]:
        manifest = self._upload_manifest(upload_id)
        source = Path(str(manifest.get("browser_preview_path") or manifest["staged_input_path"])).resolve()
        upload_dir = (self.upload_root / upload_id).resolve()
        try:
            source.relative_to(upload_dir)
        except ValueError as exc:
            raise ApiError("VIDEO_UPLOAD_INVALID", "视频预览路径不在受控目录内。", 403) from exc
        if not source.exists():
            source = Path(str(manifest["staged_input_path"])).resolve()
        media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        return source, media_type

    def _upload_manifest(self, upload_id: str) -> dict[str, Any]:
        if not upload_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in upload_id):
            raise ApiError("INVALID_VIDEO_UPLOAD_ID", "视频上传标识非法。", 422)
        manifest_path = self.upload_root / upload_id / "upload_manifest.json"
        if not manifest_path.exists():
            raise ApiError("VIDEO_UPLOAD_NOT_FOUND", "视频上传记录不存在，请重新选择视频。", 404)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError("VIDEO_UPLOAD_INVALID", "视频上传记录损坏，请重新选择视频。", 500) from exc
        source = Path(str(payload.get("staged_input_path", ""))).resolve()
        upload_root = self.upload_root.resolve()
        try:
            source.relative_to(upload_root)
        except ValueError as exc:
            raise ApiError("VIDEO_UPLOAD_INVALID", "视频上传路径不在受控目录内。", 403) from exc
        if not source.exists():
            raise ApiError("VIDEO_UPLOAD_MISSING", "已选择的视频暂存文件不存在，请重新选择视频。", 404)
        return payload

    def start(self, upload_id: str, selection: dict[str, Any], request_id: str) -> dict[str, Any]:
        with self._lock:
            current = self._active_task()
            if current is not None:
                raise ApiError(
                    "VIDEO_JOB_ALREADY_ACTIVE",
                    "已有视频检测任务正在运行；请等待完成、停止或恢复该任务，不能重复启动冲突任务。",
                    409,
                    {"active_job_id": current.job.job_id, "status": current.state},
                )

            uploaded = self._upload_manifest(upload_id)
            source = Path(str(uploaded["staged_input_path"])).resolve()
            metadata = _metadata_for(source)
            metadata["source_sha256"] = str(uploaded.get("source_sha256") or "")
            # Official-test isolation is a server-owned flag.  The browser
            # cannot opt a test video into the feedback optimizer by sending a
            # forged selection value.
            selection = dict(selection)
            context_value = selection.get("detection_context")
            frozen_context = DetectionContext.from_value(context_value) if context_value is not None else None
            if frozen_context is not None:
                selection["runtime_revision"] = frozen_context.runtime_revision
                selection["detection_context"] = frozen_context.as_dict()
            dynamic = self.settings.dynamic_settings()
            selection["official_test_mode"] = bool(selection.get("official_test_mode") or dynamic.get("official_test_mode"))
            selection["feedback_optimization_locked"] = bool(selection.get("feedback_optimization_locked") or dynamic.get("feedback_optimization_locked"))
            job = self.jobs.create("video")
            task = VideoTask(
                job=job,
                source_path=source,
                original_filename=str(uploaded.get("original_filename") or source.name),
                metadata=metadata,
                selection=dict(selection),
                detection_context=frozen_context,
            )
            self.adapter.domain_profiles.reserve(selection.get("memory_bank_id"), job.job_id)
            self._tasks[job.job_id] = task
            self._active_job_id = job.job_id
            self.jobs.update(
                job,
                request_id=request_id,
                status="queued",
                processing_mode="offline_video_file_frame_by_frame_detection",
                source_upload_id=upload_id,
                source_path=str(source),
                original_filename=task.original_filename,
                input_video_metadata=metadata,
                selection=selection,
                runtime_revision=frozen_context.runtime_revision if frozen_context else selection.get("runtime_revision"),
                detection_context=frozen_context.as_dict() if frozen_context else selection.get("detection_context"),
                note="当前模式对本地视频文件执行逐帧检测，实时性取决于硬件、模型和视频分辨率。",
            )
            worker = threading.Thread(target=self._run, args=(task, request_id), name=f"AdaptiveAOI-Video-{job.job_id}", daemon=False)
            task.thread = worker
            worker.start()
            return self.status(job.job_id, heartbeat=True)

    def _persist_live_state(self, task: VideoTask, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - task.last_persist < PROGRESS_PERSIST_SECONDS:
            return
        task.last_persist = now
        total_frames = int(task.metadata.get("effective_total_frames") or task.metadata.get("reported_frame_count", 0))
        progress = min(1.0, max(0.0, (task.current_frame + 1) / total_frames)) if total_frames else None
        active_seconds = self._active_seconds(task)
        paused_seconds = self._paused_seconds(task)
        self.jobs.update(
            task.job,
            status=task.state,
            completed_items=max(0, task.current_frame + 1),
            total_items=total_frames,
            active_processing_seconds=active_seconds,
            paused_seconds=paused_seconds,
            current_fps=task.current_fps,
            inference_fps=(
                task.inference_count / (task.inference_total_ms / 1000.0)
                if task.inference_count and task.inference_total_ms > 0
                else None
            ),
            complete_fps=task.complete_fps,
            eta_seconds=task.eta_seconds,
            paused_at=task.paused_at,
            last_progress_at=task.last_progress_at,
            progress={
                "current_frame": task.current_frame,
                "total_frames": total_frames,
                "current_time_ms": task.current_time_ms,
                "progress": progress,
                "anomaly_score": task.anomaly_score,
                "prediction": task.prediction,
                "successful_frames": task.successful_frames,
                "failed_frames": task.failed_frames,
                "anomaly_frames": task.anomaly_frames,
                "active_processing_seconds": active_seconds,
                "paused_seconds": paused_seconds,
                "current_fps": task.current_fps,
                "average_processing_fps": task.average_processing_fps,
                "complete_fps": task.complete_fps,
                "estimated_remaining_seconds": task.eta_seconds,
                "paused_at": task.paused_at,
                "last_progress_at": task.last_progress_at,
            },
            preview_path=str(task.preview_path) if task.preview_path else "",
        )

    def _run(self, task: VideoTask, request_id: str) -> None:
        try:
            with self._lock:
                if task.cancel_event.is_set():
                    task.state = "cancelled"
                    self._persist_live_state(task, force=True)
                    return
                task.state = "initializing"
                self._persist_live_state(task, force=True)

            resolved, entry, cache_hit = self.adapter.resolve_video_runtime(**task.selection)
            task.model_cache_hit = cache_hit
            task.runtime_device = str(resolved["device"])
            task.runtime_model_id = str(resolved["model_id"])
            task.runtime_dataset = str(resolved["dataset"])
            task.runtime_category = str(resolved["category"])
            task.runtime_memory_bank_id = str(resolved["memory_bank_id"])
            task.runtime_threshold = float(resolved["threshold"]) if resolved.get("threshold") is not None else None
            task.runtime_config_version = self.settings.profile_path(resolved["model_id"]).name
            if task.cancel_event.is_set():
                task.state = "cancelled"
                self._persist_live_state(task, force=True)
                return

            output_dir = task.job.results_dir / "video"
            output_video = output_dir / f"{task.source_path.stem}_检测结果.mp4"
            preview_path = output_dir / "preview" / "latest_frame.jpg"
            task.preview_path = preview_path

            def progress_callback(event: dict[str, Any]) -> None:
                with self._lock:
                    task.current_frame = int(event.get("frame_index", task.current_frame))
                    task.current_time_ms = task.current_frame / float(task.metadata["source_fps"]) * 1000.0
                    self._refresh_performance(task, record_progress=True)
                    self._persist_live_state(task)

            def result_callback(result: Any) -> None:
                with self._lock:
                    task.current_frame = int(result.frame_index)
                    task.current_time_ms = float(result.timestamp_ms)
                    task.anomaly_score = float(result.anomaly_score) if result.anomaly_score is not None else None
                    if task.anomaly_score is not None:
                        task.highest_anomaly_score = max(task.highest_anomaly_score or task.anomaly_score, task.anomaly_score)
                    if result.inference_time_ms is not None:
                        task.inference_total_ms += float(result.inference_time_ms)
                        task.inference_count += 1
                    task.prediction = str(result.prediction)
                    task.frame_contexts[int(result.frame_index)] = {
                        "frame_index": int(result.frame_index),
                        "timestamp_ms": float(result.timestamp_ms),
                        "prediction": str(result.prediction),
                        "prediction_label": result.prediction_label,
                        "score": float(result.anomaly_score) if result.anomaly_score is not None else None,
                        "threshold": float(result.threshold) if result.threshold is not None else task.runtime_threshold,
                        "detection_time": _utc_now(),
                        "runtime_revision": (
                            task.detection_context.runtime_revision
                            if task.detection_context is not None
                            else task.selection.get("runtime_revision")
                        ),
                    }
                    while len(task.frame_contexts) > 512:
                        task.frame_contexts.pop(next(iter(task.frame_contexts)))
                    if result.status == "success":
                        task.successful_frames += 1
                        if result.prediction_label == 1:
                            task.anomaly_frames += 1
                    elif result.status == "failed":
                        task.failed_frames += 1
                    self._refresh_performance(task, record_progress=True)
                    self._persist_live_state(task)

            def preview_callback(frame: Any, result: Any) -> None:
                if int(result.frame_index) % PREVIEW_FRAME_INTERVAL == 0:
                    _atomic_write_preview(preview_path, frame)

            def pause_callback() -> bool:
                if task.resume_event.is_set():
                    return False
                with self._lock:
                    if task.cancel_event.is_set():
                        return False
                    now = time.monotonic()
                    self._close_active_segment(task, now)
                    if task.pause_started_monotonic is None:
                        task.pause_started_monotonic = now
                        task.paused_at = _utc_now()
                    task.state = "paused"
                    task.eta_seconds = None
                    self._refresh_performance(task)
                    self._persist_live_state(task, force=True)
                while not task.resume_event.wait(0.10):
                    if task.cancel_event.is_set():
                        break
                with self._lock:
                    now = time.monotonic()
                    if task.pause_started_monotonic is not None:
                        task.paused_seconds += max(0.0, now - task.pause_started_monotonic)
                        task.pause_started_monotonic = None
                        task.paused_at = None
                    if task.cancel_event.is_set():
                        task.state = "cancelling"
                        task.eta_seconds = None
                    else:
                        task.state = "running"
                        self._start_active_segment(task, now)
                    self._refresh_performance(task)
                    self._persist_live_state(task, force=True)
                return False

            def paused_seconds_callback() -> float:
                with self._lock:
                    return self._paused_seconds(task)

            def cancelled() -> bool:
                return task.cancel_event.is_set()

            with self._lock:
                task.state = "running"
                task.started_monotonic = time.monotonic()
                self._start_active_segment(task, task.started_monotonic)
                self._refresh_performance(task)
                self._persist_live_state(task, force=True)
            generate_visualizations = bool(task.selection.get("generate_visualizations", True))
            from video import VideoDetectionOptions, run_video_detection

            options = VideoDetectionOptions(
                input_video=task.source_path,
                output_video=output_video,
                output_dir=output_dir,
                config=resolved["config"],
                memory_path=resolved["memory_path"],
                model_name="AdaptiveAOI正常记忆检测",
                config_path=str(self.settings.profile_path(resolved["model_id"])),
                device=resolved["device"],
                threshold=resolved["threshold"],
                generate_visualizations=generate_visualizations,
                frame_stride=max(1, int(task.selection.get("frame_stride", 1))),
                save_overlay=generate_visualizations,
                save_heatmap=generate_visualizations and bool(task.selection.get("save_heatmap", False)),
                save_mask=generate_visualizations and bool(task.selection.get("save_mask", False)),
                save_jsonl=bool(task.selection.get("save_jsonl", True)),
                save_csv=True,
                save_keyframes=generate_visualizations,
                overwrite=False,
                progress_callback=progress_callback,
                preview_callback=preview_callback,
                result_callback=result_callback,
                pause_callback=pause_callback,
                paused_seconds_callback=paused_seconds_callback,
                cancel_callback=cancelled,
                model_load_ms=None if cache_hit else getattr(entry.inferencer, "model_load_ms", None),
                memory_bank_load_ms=None if cache_hit else getattr(entry.inferencer, "memory_bank_load_ms", None),
                model_reused=bool(cache_hit),
                memory_bank_reused=bool(cache_hit),
                runtime_revision=(
                    task.detection_context.runtime_revision
                    if task.detection_context is not None
                    else task.selection.get("runtime_revision")
                ),
                detection_context=(
                    task.detection_context.as_dict()
                    if task.detection_context is not None
                    else dict(task.selection.get("detection_context") or {})
                ),
            )
            with self.registry.execution(entry, task.job.root, resolved["threshold"]) as inferencer:
                summary = run_video_detection(options, inferencer=inferencer)

            with self._lock:
                task.summary = summary
                task.state = "cancelled" if bool(summary.get("cancelled")) else "completed"
                actual_total_frames = int(summary.get("total_decoded_records", 0))
                reported_total_frames = int(task.metadata.get("reported_frame_count", 0))
                task.metadata["effective_total_frames"] = max(reported_total_frames, actual_total_frames)
                task.current_frame = max(task.current_frame, actual_total_frames - 1)
                task.successful_frames = int(summary.get("successful_processed_frames", 0))
                task.failed_frames = int(summary.get("failed_frames", 0))
                task.anomaly_frames = int(summary.get("anomaly_frames", 0))
                task.anomaly_score = summary.get("highest_anomaly_score")
                task.prediction = "INCOMPLETE" if task.state == "cancelled" else "COMPLETED"
                self._freeze_timing(task)
                summary_fps = summary.get("full_pipeline_fps")
                if isinstance(summary_fps, (int, float)):
                    task.average_processing_fps = float(summary_fps)
                    task.complete_fps = float(summary_fps)
                task.eta_seconds = 0.0 if task.state == "completed" else None
                task.keyframe_feedback_targets = self._keyframe_targets(task, summary)
                self.jobs.update(
                    task.job,
                    status=task.state,
                    completed_at=_utc_now(),
                    model_cache_hit=cache_hit,
                    complete=task.state == "completed",
                    summary=summary,
                    artifacts=self._artifact_urls(summary, task),
                    warning="任务已停止，输出仅包含停止前的成功帧，未标记为完整结果。" if task.state == "cancelled" else "",
                )
                self._persist_live_state(task, force=True)
                self.logger.info("video_job_finished job_id=%s status=%s", task.job.job_id, task.state)
        except ApiError as exc:
            with self._lock:
                self._freeze_timing(task)
                task.state = "failed"
                task.eta_seconds = None
                task.error = {"code": exc.code, "message": exc.message}
                self.jobs.fail(task.job, exc.code, exc.message)
                self._persist_live_state(task, force=True)
        except Exception as exc:
            self.logger.exception("video_job_failed job_id=%s", task.job.job_id)
            with self._lock:
                self._freeze_timing(task)
                task.state = "failed"
                task.eta_seconds = None
                task.error = {"code": "VIDEO_INFERENCE_FAILED", "message": f"视频检测失败：{type(exc).__name__}: {exc}"}
                self.jobs.fail(task.job, task.error["code"], task.error["message"])
                self._persist_live_state(task, force=True)
        finally:
            self.adapter.domain_profiles.release(task.selection.get("memory_bank_id"), task.job.job_id)
            with self._lock:
                if self._active_job_id == task.job.job_id:
                    self._active_job_id = None

    def _artifact_urls(self, summary: dict[str, Any], task: VideoTask) -> dict[str, Any]:
        urls = {
            "result_video_url": artifact_url(summary.get("output_video_path"), self.settings.jobs_root),
            "jsonl_url": artifact_url(summary.get("jsonl_path"), self.settings.jobs_root),
            "csv_url": artifact_url(summary.get("csv_path"), self.settings.jobs_root),
            "summary_url": artifact_url(summary.get("summary_path"), self.settings.jobs_root),
            "preview_url": artifact_url(task.preview_path, self.settings.jobs_root) if task.preview_path and task.preview_path.exists() else None,
            "keyframes": {name: artifact_url(path, self.settings.jobs_root) for name, path in (summary.get("keyframes") or {}).items()},
            "visualization_status": summary.get("visualization_status", "generated"),
        }
        return urls

    @staticmethod
    def _read_frame_records(task: VideoTask, summary: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        jsonl_value = str(summary.get("jsonl_path") or "")
        if jsonl_value:
            path = Path(jsonl_value).resolve()
            try:
                path.relative_to(task.job.results_dir.resolve())
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict) and item.get("status") == "success":
                            records.append(item)
            except (ValueError, OSError, json.JSONDecodeError):
                records = []
        if not records and summary.get("csv_path"):
            path = Path(str(summary["csv_path"])).resolve()
            try:
                path.relative_to(task.job.results_dir.resolve())
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    records = [dict(item) for item in csv.DictReader(handle) if item.get("status") == "success"]
            except (ValueError, OSError, csv.Error):
                records = []
        return records

    def _keyframe_targets(self, task: VideoTask, summary: dict[str, Any]) -> list[dict[str, Any]]:
        records = self._read_frame_records(task, summary)
        if not records:
            return []
        by_name: dict[str, dict[str, Any] | None] = {
            "first_frame": records[0],
            "normal_representative": next((item for item in records if str(item.get("prediction_label")) == "0"), None),
            "first_anomaly": next((item for item in records if str(item.get("prediction_label")) == "1"), None),
            "highest_anomaly_score": max(records, key=lambda item: float(item.get("anomaly_score") or float("-inf"))),
            "last_frame": records[-1],
        }
        available = set((summary.get("keyframes") or {}).keys())
        targets: list[dict[str, Any]] = []
        for name, item in by_name.items():
            if name not in available or not item:
                continue
            targets.append(
                {
                    "keyframe_name": name,
                    "frame_index": int(item["frame_index"]),
                    "timestamp_ms": float(item.get("timestamp_ms") or 0.0),
                    "prediction": str(item.get("prediction") or ""),
                    "score": item.get("anomaly_score"),
                    "threshold": item.get("threshold"),
                }
            )
        return targets

    def feedback_context(self, job_id: str, frame_index: int) -> dict[str, Any]:
        """Validate a frame and return server-owned facts for feedback storage."""
        try:
            task = self._require_task(job_id)
        except ApiError as exc:
            if exc.code != "VIDEO_JOB_NOT_ACTIVE":
                raise
            return self._historical_feedback_context(job_id, frame_index)
        index = int(frame_index)
        with self._lock:
            record = dict(task.frame_contexts.get(index) or {})
            summary = dict(task.summary or {})
            if not record and summary:
                record = next(
                    (item for item in self._read_frame_records(task, summary) if int(item.get("frame_index", -1)) == index),
                    {},
                )
            if not record:
                raise ApiError("VIDEO_FRAME_NOT_FOUND", "当前帧尚未完成检测，或不属于该视频任务。", 404)
            perceptual_hash = ""
            # Recompute the frame hash from the service-owned staged video.
            # A browser-provided hash is never trusted as de-duplication
            # evidence for an optimization candidate.
            capture = cv2.VideoCapture(str(task.source_path))
            try:
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        perceptual_hash = _phash_frame(frame)
            finally:
                capture.release()
            memory_bank_id = str(task.runtime_memory_bank_id or "")
            config_profile_id = str(task.selection.get("config_profile_id") or "")
            config_profile_version = int(task.selection.get("config_profile_version") or 0)
            memory_profile = dict(task.selection.get("memory_bank_profile") or {})
            profile_method_mode = str(
                task.selection.get("method_mode")
                or memory_profile.get("method_mode_resolved")
                or "normal_only"
            )
            return {
                "mode": "video",
                "task_id": task.job.job_id,
                "sample_id": f"frame:{index}",
                "result_id": None,
                "frame_index": index,
                "timestamp_ms": float(record.get("timestamp_ms") or 0.0),
                "source_label": f"{task.original_filename} · 帧 {index}",
                "detection_time": str(record.get("detection_time") or self.jobs.read_manifest(task.job).get("created_at") or ""),
                "dataset": task.runtime_dataset,
                "category": task.runtime_category,
                "model_version": task.runtime_model_id,
                "config_version": task.runtime_config_version,
            "memory_bank_id": memory_bank_id,
            "runtime_revision": (
                int(record["runtime_revision"])
                if record.get("runtime_revision") is not None
                else (selection.get("detection_context") or {}).get("runtime_revision")
            ),
            "source_memory_bank_id": memory_bank_id,
                "source_profile_id": str(task.selection.get("profile_id") or memory_bank_id),
                "source_target_domain_id": str(task.selection.get("target_domain_id") or memory_bank_id),
                "source_config_snapshot_id": str(
                    task.selection.get("config_snapshot_id")
                    or (f"{config_profile_id}:v{config_profile_version}" if config_profile_id and config_profile_version else config_profile_id)
                    or task.runtime_config_version
                ),
                "threshold": record.get("threshold", task.runtime_threshold),
                "prediction": str(record.get("prediction") or ""),
                "score": record.get("score", record.get("anomaly_score")),
                "configuration_frozen": True,
                "config_profile_id": config_profile_id,
                "config_profile_version": config_profile_version,
                "method_mode": profile_method_mode,
                "profile_method_mode": profile_method_mode,
                "scoring_method_mode": str(record.get("method_mode") or profile_method_mode),
                "image_size": memory_profile.get("image_size") or task.selection.get("image_size"),
                "feature_dimension": memory_profile.get("feature_dimension") or task.selection.get("feature_dimension"),
                "feature_layers": memory_profile.get("feature_layers") or task.selection.get("feature_layers") or [],
                "backbone": memory_profile.get("backbone") or task.selection.get("backbone"),
                "inference_mode": "inference_only",
                "official_test_mode": bool(task.selection.get("official_test_mode", False)),
                "test_data_update_allowed": False,
                "data_role": "official_test_set" if task.selection.get("official_test_mode", False) else "inference_input",
                "evaluation_mode": "official_test" if task.selection.get("official_test_mode", False) else "inference_only",
                "source_artifact_ref": f"video_frame:{task.job.job_id}:{index}",
                "source_sha256": str(task.metadata.get("source_sha256") or ""),
                "source_width": int(task.metadata.get("width") or 0),
                "source_height": int(task.metadata.get("height") or 0),
                "perceptual_hash": perceptual_hash,
                # Internal-only value consumed by the candidate worker.  API
                # responses must strip it before reaching the browser.
                "_source_path": str(task.source_path),
            }

    def extract_feedback_frame(self, job_id: str, frame_index: int, destination: str | Path) -> Path:
        """Extract one validated source frame into a job-owned artifact path.

        The input video is always the staged file owned by this service; callers
        cannot provide an arbitrary server path.  The destination is restricted
        to the feedback job tree by the candidate service before this method is
        called.
        """
        task = self._require_task(job_id)
        index = int(frame_index)
        if index < 0:
            raise ApiError("INVALID_VIDEO_FRAME", "视频帧编号无效。", 422)
        cap = cv2.VideoCapture(str(task.source_path))
        try:
            if not cap.isOpened():
                raise ApiError("VIDEO_NOT_DECODABLE", "源视频当前无法解码。", 422)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ApiError("VIDEO_FRAME_NOT_FOUND", "指定视频帧无法读取。", 404)
            target = Path(destination)
            target = target.resolve()
            jobs_root = self.settings.jobs_root.resolve()
            try:
                target.relative_to(jobs_root)
            except ValueError as exc:
                raise ApiError("VIDEO_FRAME_DESTINATION_INVALID", "视频帧输出位置不在受控任务目录内。", 403) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            # OpenCV selects its encoder from the final suffix.  ``.png.part``
            # has no registered encoder, so keep ``.png`` as the temporary
            # file's final suffix and atomically rename it afterwards.
            temporary = target.with_name(f"{target.stem}.part{target.suffix}")
            if not cv2.imwrite(str(temporary), frame):
                raise ApiError("VIDEO_FRAME_WRITE_FAILED", "视频帧保存失败。", 500)
            os.replace(temporary, target)
            return target
        finally:
            cap.release()

    def feedback_interval_contexts(self, job_id: str, start_ms: float, end_ms: float, max_frames: int = 8) -> list[dict[str, Any]]:
        """Return representative, non-adjacent frames for a feedback interval."""
        task = self._require_task(job_id)
        start = max(0.0, float(start_ms))
        end = max(start, float(end_ms))
        limit = min(32, max(1, int(max_frames)))
        with self._lock:
            rows = [
                dict(item)
                for item in self._read_frame_records(task, dict(task.summary or {}))
                if start <= float(item.get("timestamp_ms") or 0.0) <= end
            ]
        # Highest anomaly peaks are reviewed first.  A minimum frame gap keeps
        # hundreds of adjacent nearly-identical frames out of the pool.
        rows.sort(key=lambda item: float(item.get("anomaly_score") or item.get("score") or 0.0), reverse=True)
        selected: list[dict[str, Any]] = []
        selected_hashes: list[str] = []
        min_gap = max(1, int(task.selection.get("frame_stride") or 1) * 5)
        capture = cv2.VideoCapture(str(task.source_path))
        try:
            for row in rows:
                frame_index = int(row.get("frame_index", -1))
                if frame_index < 0 or any(abs(frame_index - int(item.get("frame_index", -1))) < min_gap for item in selected):
                    continue
                perceptual_hash = ""
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        perceptual_hash = _phash_frame(frame)
                # A small Hamming distance means the candidate is visually
                # equivalent to a previously selected peak.  This is an
                # additional guard beyond frame spacing, not a label decision.
                if perceptual_hash and any(_hamming_distance(perceptual_hash, old) <= 4 for old in selected_hashes):
                    continue
                context = self.feedback_context(job_id, frame_index)
                context["selection_reason"] = "区间异常峰值代表帧（已过滤相邻相似帧）"
                context["perceptual_hash"] = perceptual_hash
                context["interval_start_ms"] = start
                context["interval_end_ms"] = end
                selected.append(context)
                if perceptual_hash:
                    selected_hashes.append(perceptual_hash)
                if len(selected) >= limit:
                    break
        finally:
            capture.release()
        return selected

    def status(self, job_id: str, heartbeat: bool = False) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(job_id)
        if task is None:
            durable = self._durable_status(job_id)
            if durable is not None:
                return durable
            raise ApiError("VIDEO_JOB_NOT_FOUND", "视频检测任务不存在。", 404)
        with self._lock:
            if heartbeat and task.state in _ACTIVE_VIDEO_STATES:
                task.last_heartbeat = time.monotonic()
                self.jobs.update(task.job, status=task.state)
            manifest = self.jobs.read_manifest(task.job)
            total = int(task.metadata.get("effective_total_frames") or task.metadata.get("reported_frame_count", 0))
            progress = min(1.0, max(0.0, (task.current_frame + 1) / total)) if total else None
            completed_items = max(0, task.current_frame + 1)
            active_seconds = self._active_seconds(task)
            paused_seconds = self._paused_seconds(task)
            inference_fps = (
                task.inference_count / (task.inference_total_ms / 1000.0)
                if task.inference_count and task.inference_total_ms > 0
                else None
            )
            summary = dict(task.summary or {})
            current_frame_context = dict(task.frame_contexts.get(task.current_frame) or {})
            return {
                "job_id": task.job.job_id,
                "status": task.state,
                "complete": task.state == "completed",
                "cancelled": task.state == "cancelled",
                "completed_items": completed_items,
                "total_items": total,
                "active_processing_seconds": active_seconds,
                "paused_seconds": paused_seconds,
                "current_fps": task.current_fps,
                "inference_fps": inference_fps,
                "complete_fps": task.complete_fps,
                "eta_seconds": task.eta_seconds,
                "paused_at": task.paused_at,
                "last_progress_at": task.last_progress_at,
                "error": task.error,
                "input": {"original_filename": task.original_filename, "metadata": task.metadata},
                "progress": {
                    "current_frame": task.current_frame,
                    "total_frames": total,
                    "current_time_ms": task.current_time_ms,
                    "duration_seconds": task.metadata.get("duration_seconds"),
                    "progress": progress,
                    "anomaly_score": task.anomaly_score,
                    "prediction": task.prediction,
                    "successful_frames": task.successful_frames,
                    "failed_frames": task.failed_frames,
                    "anomaly_frames": task.anomaly_frames,
                    "active_processing_seconds": active_seconds,
                    "paused_seconds": paused_seconds,
                    "current_fps": task.current_fps,
                    "average_processing_fps": task.average_processing_fps,
                    "pure_inference_fps": inference_fps,
                    "full_pipeline_fps": task.complete_fps,
                    "highest_anomaly_score": task.highest_anomaly_score,
                    "estimated_remaining_seconds": task.eta_seconds,
                    "paused_at": task.paused_at,
                    "last_progress_at": task.last_progress_at,
                    "stage": task.state,
                },
                "model_cache_hit": task.model_cache_hit,
                "runtime_device": task.runtime_device or task.selection.get("device", "auto"),
                "generate_visualizations": bool(task.selection.get("generate_visualizations", True)),
                "visualization_status": summary.get("visualization_status") or (
                    "generated" if bool(task.selection.get("generate_visualizations", True)) else "not_generated"
                ),
                "feedback_runtime": {
                    "model_version": task.runtime_model_id,
                    "memory_bank_id": task.runtime_memory_bank_id,
                    "threshold": task.runtime_threshold,
                    "configuration_frozen": True,
                    "runtime_revision": (
                        task.detection_context.runtime_revision
                        if task.detection_context is not None
                        else task.selection.get("runtime_revision")
                    ),
                },
                "runtime_revision": (
                    task.detection_context.runtime_revision
                    if task.detection_context is not None
                    else task.selection.get("runtime_revision")
                ),
                "detection_context": (
                    task.detection_context.as_dict()
                    if task.detection_context is not None
                    else dict(task.selection.get("detection_context") or {})
                ),
                "current_frame_context": current_frame_context,
                "keyframe_feedback_targets": list(task.keyframe_feedback_targets),
                "summary": summary,
                "artifacts": self._artifact_urls(summary, task) if summary else {"preview_url": artifact_url(task.preview_path, self.settings.jobs_root) if task.preview_path and task.preview_path.exists() else None},
                "manifest": manifest,
            }

    def rerun(self, job_id: str, request_id: str) -> dict[str, Any]:
        durable = self.jobs.durable_state(job_id)
        if not durable or durable.get("task_kind") != "video":
            raise ApiError("VIDEO_JOB_NOT_FOUND", "视频检测任务不存在。", 404)
        if durable.get("status") != "interrupted":
            raise ApiError("VIDEO_JOB_NOT_INTERRUPTED", "只有因服务重启中断的视频任务可以重新运行。", 409)
        manifest = dict(durable.get("payload") or {})
        upload_id = str(manifest.get("source_upload_id") or "")
        selection = dict(manifest.get("selection") or {})
        if not upload_id or not selection:
            raise ApiError("VIDEO_JOB_RERUN_DATA_MISSING", "中断任务缺少可重新运行的视频或运行参数。", 409)
        created = self.start(upload_id, selection, request_id)
        self.jobs.states.record_rerun(job_id, str(created["job_id"]))
        return created

    def pause(self, job_id: str) -> dict[str, Any]:
        task = self._require_task(job_id)
        with self._lock:
            if task.state in {"pausing", "paused"}:
                return self.status(job_id, heartbeat=True)
            if task.state != "running":
                raise ApiError("VIDEO_JOB_NOT_RUNNING", "只有运行中的视频任务可以暂停。", 409, {"status": task.state})
            task.resume_event.clear()
            task.state = "pausing"
            task.eta_seconds = None
            self._persist_live_state(task, force=True)
        return self.status(job_id, heartbeat=True)

    def resume(self, job_id: str) -> dict[str, Any]:
        task = self._require_task(job_id)
        with self._lock:
            if task.state == "resuming":
                return self.status(job_id, heartbeat=True)
            if task.state not in {"pausing", "paused"}:
                raise ApiError("VIDEO_JOB_NOT_PAUSED", "当前视频任务不处于暂停状态。", 409, {"status": task.state})
            task.state = "resuming"
            task.eta_seconds = None
            self._persist_live_state(task, force=True)
            task.resume_event.set()
        return self.status(job_id, heartbeat=True)

    def cancel(self, job_id: str, reason: str = "用户取消") -> dict[str, Any]:
        task = self._require_task(job_id)
        with self._lock:
            if task.state in _TERMINAL_VIDEO_STATES:
                return self.status(job_id, heartbeat=False)
            task.cancel_event.set()
            task.resume_event.set()
            task.state = "cancelling"
            task.eta_seconds = None
            self.jobs.update(task.job, status="cancelling", cancel_reason=reason, complete=False)
            self._persist_live_state(task, force=True)
        return self.status(job_id, heartbeat=False)

    def stop(self, job_id: str, reason: str = "用户停止") -> dict[str, Any]:
        """Backward-compatible alias for clients using the former stop route."""
        return self.cancel(job_id, reason=reason)

    def open_result_directory(self, job_id: str) -> dict[str, Any]:
        task = self._require_task(job_id)
        directory = task.job.results_dir / "video"
        if not directory.exists():
            raise ApiError("VIDEO_OUTPUT_NOT_READY", "结果目录尚未生成。", 409)
        if not hasattr(os, "startfile"):
            raise ApiError("OPEN_DIRECTORY_UNSUPPORTED", "当前运行环境不支持从软件打开结果目录。", 501)
        try:
            os.startfile(str(directory))  # type: ignore[attr-defined]
        except OSError as exc:
            raise ApiError("OPEN_DIRECTORY_FAILED", f"无法打开结果目录：{exc}", 500) from exc
        return {"job_id": job_id, "output_directory": str(directory)}

    def open_result_video(self, job_id: str) -> dict[str, Any]:
        """Open only this task's verified result video with the OS player."""
        task = self._require_task(job_id)
        video_path = task.job.results_dir / "video" / f"{task.source_path.stem}_检测结果.mp4"
        if not video_path.exists() or not video_path.is_file():
            raise ApiError("VIDEO_RESULT_NOT_READY", "检测结果视频尚未生成或已被移动。", 409)
        if not hasattr(os, "startfile"):
            raise ApiError("OPEN_VIDEO_UNSUPPORTED", "当前运行环境不支持从软件打开结果视频。", 501)
        try:
            os.startfile(str(video_path))  # type: ignore[attr-defined]
        except OSError as exc:
            raise ApiError("OPEN_VIDEO_FAILED", f"无法使用系统播放器打开结果视频：{exc}", 500) from exc
        return {"job_id": job_id, "result_video": str(video_path)}

    def shutdown(self) -> None:
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            if task.state in _ACTIVE_VIDEO_STATES:
                self.cancel(task.job.job_id, reason="软件或后端正在关闭")
        for task in tasks:
            if task.thread and task.thread.is_alive():
                task.thread.join(timeout=10.0)


__all__ = ["SUPPORTED_VIDEO_SUFFIXES", "VideoJobService"]
