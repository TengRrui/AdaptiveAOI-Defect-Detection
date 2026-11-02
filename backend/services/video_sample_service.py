"""Bounded video sampling utilities for target-domain adaptation.

The service keeps video decoding and frame selection in one place so image,
video and mixed target-domain builds all hand the existing PatchCore builder a
single list of ordinary image paths.  Only representative, quality-checked
frames are materialized under a task-owned temporary directory.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable



class _LazyModule:
    """Keep OpenCV/NumPy out of the Backend service-construction path."""

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)


cv2 = _LazyModule("cv2")
np = _LazyModule("numpy")


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CancelCallback = Callable[[], bool]
ProgressCallback = Callable[[str, int, int, str], None]


class VideoSampleCancelled(RuntimeError):
    pass


def _check_cancel(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise VideoSampleCancelled("视频样本处理已取消。")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(frame: np.ndarray) -> str:
    # Canonicalize to RGB so an image file and an extracted video frame with
    # identical pixels receive the same exact-content hash.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _phash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    return small >= float(small.mean())


def _phash_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    return 1.0 - float(np.count_nonzero(left != right)) / float(left.size)


def _fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class VideoSampleService:
    """Probe and sample local videos with bounded memory and deterministic selection."""

    def __init__(self, temp_root: str | Path):
        self.temp_root = Path(temp_root).expanduser().resolve()
        self.temp_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def supported_video(path: str | Path) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

    @staticmethod
    def _files_from_inputs(files: Iterable[str | Path] | None, directories: Iterable[str | Path] | None) -> list[Path]:
        found: dict[str, Path] = {}
        for raw in list(files or []):
            try:
                path = Path(str(raw)).expanduser().resolve()
            except (OSError, ValueError):
                continue
            # Explicitly supplied files are retained even when missing or
            # using an unsupported extension so the scan can return a clear
            # per-file warning instead of silently pretending the input was
            # never provided.  Directory discovery below remains restricted
            # to supported extensions.
            if not path.is_dir():
                found[str(path).casefold()] = path
        for raw in list(directories or []):
            try:
                root = Path(str(raw)).expanduser().resolve()
            except (OSError, ValueError):
                continue
            if not root.is_dir():
                continue
            try:
                candidates = root.rglob("*")
            except OSError:
                candidates = []
            for path in candidates:
                if path.is_file() and VideoSampleService.supported_video(path):
                    found[str(path).casefold()] = path
        return sorted(found.values(), key=lambda item: str(item).casefold())

    @staticmethod
    def probe_video(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
        candidate = Path(path).expanduser().resolve()
        result: dict[str, Any] = {
            "path": str(candidate),
            "name": candidate.name,
            "valid": False,
            "error": "",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration_seconds": 0.0,
            "codec": "",
            "size_bytes": int(candidate.stat().st_size) if candidate.is_file() else 0,
            "source_video_hash": "",
        }
        if not candidate.is_file():
            result["error"] = "视频文件不存在或不可访问。"
            return result
        if not VideoSampleService.supported_video(candidate):
            result["error"] = "视频格式不在当前解码器支持范围内。"
            return result
        capture = cv2.VideoCapture(str(candidate))
        try:
            if not capture.isOpened():
                result["error"] = "视频无法打开或解码。"
                return result
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
            ok, frame = capture.read()
            if fps <= 0.0 or width <= 0 or height <= 0 or not ok or frame is None:
                result["error"] = "视频元数据无效或首帧无法解码。"
                return result
            result.update(
                {
                    "valid": True,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_seconds": (frame_count / fps) if frame_count > 0 else 0.0,
                    "codec": _fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
                }
            )
        except Exception as exc:
            result["error"] = "视频读取失败，请更换文件后重试。"
            result["detail"] = type(exc).__name__
        finally:
            capture.release()
        if result["valid"] and include_hash:
            try:
                result["source_video_hash"] = _sha256_file(candidate)
                result["video_id"] = result["source_video_hash"][:16]
            except OSError:
                result["error"] = "视频文件哈希读取失败。"
                result["valid"] = False
        return result

    def scan(self, files: Iterable[str | Path] | None, directories: Iterable[str | Path] | None) -> dict[str, Any]:
        paths = self._files_from_inputs(files, directories)
        videos = [self.probe_video(path) for path in paths]
        valid = [item for item in videos if item.get("valid")]
        invalid = [item for item in videos if not item.get("valid")]
        return {
            "video_count": len(paths),
            "valid_video_count": len(valid),
            "invalid_video_count": len(paths) - len(valid),
            "total_duration_seconds": round(sum(float(item.get("duration_seconds") or 0.0) for item in valid), 3),
            "scanned_frame_count": 0,
            "candidate_frame_count": 0,
            "deduplicated_frame_count": 0,
            "rejected_frame_count": len(invalid),
            "selected_frame_count": 0,
            "videos": [self._public_probe(item) for item in videos],
            "warnings": [f"{item['name']}：{item.get('error')}" for item in videos if not item.get("valid")],
        }

    @staticmethod
    def _public_probe(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key not in {"path", "detail"}}

    @staticmethod
    def _sampling_defaults(config: dict[str, Any] | None, target_count: int) -> dict[str, Any]:
        raw = dict(config or {})
        method = str(raw.get("sampling_method") or raw.get("method") or "auto").lower()
        aliases = {"fixed_frame": "fixed_frame_interval", "fixed_time": "fixed_time_interval", "fps": "target_fps", "uniform": "uniform_count"}
        method = aliases.get(method, method)
        if method not in {"fixed_frame_interval", "fixed_time_interval", "target_fps", "uniform_count", "auto"}:
            method = "auto"
        return {
            "sampling_method": method,
            "frame_interval": max(1, int(raw.get("frame_interval", 15))),
            "time_interval_seconds": max(0.05, float(raw.get("time_interval_seconds", 1.0))),
            "target_fps": max(0.1, float(raw.get("target_fps", 1.0))),
            "max_frames_per_video": max(1, int(raw.get("max_frames_per_video", max(60, target_count)))),
            "start_time": max(0.0, float(raw.get("start_time", 0.0))),
            "end_time": _as_float(raw.get("end_time"), None),
            "deduplicate": bool(raw.get("deduplicate", True)),
            "similarity_threshold": min(0.999, max(0.5, float(raw.get("similarity_threshold", 0.95)))),
            "filter_blur": bool(raw.get("filter_blur", True)),
            "blur_threshold": max(0.0, float(raw.get("blur_threshold", 40.0))),
            "filter_brightness": bool(raw.get("filter_brightness", True)),
            "dark_threshold": min(255.0, max(0.0, float(raw.get("dark_threshold", 20.0)))),
            "bright_threshold": min(255.0, max(0.0, float(raw.get("bright_threshold", 235.0)))),
            "final_max_samples": max(1, int(raw.get("final_max_samples", target_count))),
            "seed": int(raw.get("seed", 42)),
        }

    @staticmethod
    def _interval_for_video(intervals: Any, video: dict[str, Any]) -> list[dict[str, Any]]:
        if not intervals:
            return []
        values: Any = intervals
        if isinstance(intervals, dict):
            values = intervals.get(video.get("path")) or intervals.get(video.get("name")) or intervals.get(video.get("video_id")) or []
        if not isinstance(values, list):
            values = [values]
        result: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            start = _as_float(item.get("start_time", item.get("start_seconds", 0.0)), 0.0) or 0.0
            end = _as_float(item.get("end_time", item.get("end_seconds")), None)
            frame_start = item.get("start_frame")
            frame_end = item.get("end_frame")
            duration = float(video.get("duration_seconds") or 0.0)
            if start < 0.0 or (duration > 0 and start > duration + 1e-6) or (end is not None and (end < start or (duration > 0 and end > duration + 1e-6))):
                raise ValueError(f"{video.get('name') or '缺陷视频'} 的缺陷时间区间超出视频时长。")
            if frame_start is not None and int(frame_start) < 0:
                raise ValueError(f"{video.get('name') or '缺陷视频'} 的缺陷帧区间无效。")
            if frame_end is not None and frame_start is not None and int(frame_end) < int(frame_start):
                raise ValueError(f"{video.get('name') or '缺陷视频'} 的缺陷帧区间无效。")
            result.append({"start_time": max(0.0, start), "end_time": end, "start_frame": frame_start, "end_frame": frame_end})
        return result

    @staticmethod
    def _in_intervals(frame_index: int, timestamp_ms: float, intervals: list[dict[str, Any]]) -> bool:
        if not intervals:
            return True
        seconds = timestamp_ms / 1000.0
        for item in intervals:
            start_frame = item.get("start_frame")
            end_frame = item.get("end_frame")
            if start_frame is not None and frame_index < int(start_frame):
                continue
            if end_frame is not None and frame_index > int(end_frame):
                continue
            if seconds < float(item.get("start_time", 0.0)):
                continue
            end_time = item.get("end_time")
            if end_time is not None and seconds > float(end_time):
                continue
            return True
        return False

    def _extract_one(
        self,
        video: dict[str, Any],
        role: str,
        config: dict[str, Any],
        output_dir: Path,
        intervals: Any,
        target_count: int,
        progress: ProgressCallback | None,
        cancelled: CancelCallback | None,
        excluded_sample_ids: set[str] | None = None,
    ) -> tuple[list[Path], list[dict[str, Any]], dict[str, int]]:
        sampling = self._sampling_defaults(config, target_count)
        path = Path(str(video["path"]))
        fps = float(video["fps"])
        frame_count = int(video.get("frame_count") or 0)
        start_index = max(0, int(round(sampling["start_time"] * fps)))
        end_time = sampling.get("end_time")
        if end_time is not None:
            end_index = min(frame_count - 1 if frame_count else 2**31 - 1, int(round(float(end_time) * fps)))
        else:
            end_index = frame_count - 1 if frame_count else 2**31 - 1
        if end_index < start_index:
            raise ValueError(f"{path.name} 的起止时间范围无效。")
        method = sampling["sampling_method"]
        uniform_targets: set[int] = set()
        if method == "uniform_count":
            count = min(sampling["max_frames_per_video"], max(1, target_count))
            uniform_targets = set(int(item) for item in np.linspace(start_index, end_index, count))
        interval_definitions = self._interval_for_video(intervals, video)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"{path.name} 无法打开或解码。")
        selected_paths: list[Path] = []
        manifest: list[dict[str, Any]] = []
        counters = {"scanned": 0, "candidate": 0, "deduplicated": 0, "rejected": 0, "selected": 0}
        last_candidate_time = -math.inf
        next_target_frame = start_index
        previous_hash: str | None = None
        previous_phash: np.ndarray | None = None
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
            frame_index = start_index
            while True:
                _check_cancel(cancelled)
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                counters["scanned"] += 1
                if frame_index > end_index:
                    break
                timestamp_ms = float(frame_index / fps * 1000.0)
                should_sample = False
                if method == "fixed_frame_interval":
                    should_sample = (frame_index - start_index) % sampling["frame_interval"] == 0
                elif method in {"fixed_time_interval", "auto"}:
                    interval = sampling["time_interval_seconds"] if method == "fixed_time_interval" else max(0.5, 1.0 / sampling["target_fps"])
                    should_sample = timestamp_ms / 1000.0 + 1e-6 >= last_candidate_time + interval
                elif method == "target_fps":
                    if frame_index >= next_target_frame:
                        should_sample = True
                        next_target_frame = frame_index + max(1, int(round(fps / sampling["target_fps"])))
                elif method == "uniform_count":
                    should_sample = frame_index in uniform_targets
                if not should_sample or not self._in_intervals(frame_index, timestamp_ms, interval_definitions):
                    frame_index += 1
                    continue
                last_candidate_time = timestamp_ms / 1000.0
                counters["candidate"] += 1
                height, width = frame.shape[:2]
                if width < 16 or height < 16:
                    counters["rejected"] += 1
                    frame_index += 1
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sampling["filter_blur"] and blur_score < sampling["blur_threshold"]:
                    counters["rejected"] += 1
                    frame_index += 1
                    continue
                if sampling["filter_brightness"] and (brightness < sampling["dark_threshold"] or brightness > sampling["bright_threshold"]):
                    counters["rejected"] += 1
                    frame_index += 1
                    continue
                digest = _frame_hash(frame)
                current_phash = _phash(frame)
                if previous_hash == digest or (sampling["deduplicate"] and previous_phash is not None and _phash_similarity(previous_phash, current_phash) >= sampling["similarity_threshold"]):
                    counters["deduplicated"] += 1
                    frame_index += 1
                    continue
                if len(selected_paths) >= sampling["max_frames_per_video"]:
                    break
                sample_id = f"{role}_{video['video_id']}_{frame_index:09d}"
                if excluded_sample_ids and sample_id in excluded_sample_ids:
                    counters["rejected"] += 1
                    frame_index += 1
                    continue
                frame_name = f"{video['video_id']}_{frame_index:09d}.jpg"
                frame_path = output_dir / role / frame_name
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    counters["rejected"] += 1
                    frame_index += 1
                    continue
                selected_paths.append(frame_path)
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "source_type": "video",
                        "source_video_id": video["video_id"],
                        "source_video_name": video["name"],
                        "frame_index": frame_index,
                        "timestamp_ms": round(timestamp_ms, 3),
                        "original_fps": fps,
                        "width": width,
                        "height": height,
                        "frame_hash": digest,
                        "extraction_method": method,
                        "quality_score": round(blur_score, 4),
                        "selected": True,
                        "usage": "normal_memory" if role == "normal" else "defect_calibration_only",
                        "defect_interval": interval_definitions,
                        "path": str(frame_path),
                    }
                )
                counters["selected"] += 1
                previous_hash = digest
                previous_phash = current_phash
                if progress is not None:
                    progress("video_sampling", counters["scanned"], max(1, frame_count), f"正在抽取{('正常' if role == 'normal' else '缺陷')}视频：{video['name']}")
                frame_index += 1
        finally:
            capture.release()
        return selected_paths, manifest, counters

    @staticmethod
    def _balanced_select(paths: list[Path], manifest: list[dict[str, Any]], limit: int, seed: int) -> tuple[list[Path], list[dict[str, Any]]]:
        if limit <= 0 or len(paths) <= limit:
            return paths, manifest
        by_video: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for path, row in zip(paths, manifest):
            by_video.setdefault(str(row.get("source_video_id") or "unknown"), []).append((path, row))
        keys = sorted(by_video)
        if keys:
            offset = int(seed) % len(keys)
            keys = keys[offset:] + keys[:offset]
        chosen: list[tuple[Path, dict[str, Any]]] = []
        while len(chosen) < limit and any(by_video.values()):
            for key in keys:
                bucket = by_video.get(key) or []
                if bucket and len(chosen) < limit:
                    chosen.append(bucket.pop(0))
        chosen_paths = [item[0] for item in chosen]
        chosen_ids = {id(item[1]) for item in chosen}
        for row in manifest:
            row["selected"] = id(row) in chosen_ids
        for path in paths:
            if path not in chosen_paths:
                try:
                    path.unlink()
                except OSError:
                    pass
        # Keep rejected-by-budget rows in the manifest with selected=False so
        # the frozen sample ledger remains auditable.  The returned path list
        # contains only the rows actually handed to the feature builder.
        return chosen_paths, manifest

    def extract(
        self,
        role: str,
        files: Iterable[str | Path] | None,
        directories: Iterable[str | Path] | None,
        config: dict[str, Any] | None,
        *,
        target_count: int,
        intervals: Any = None,
        task_id: str | None = None,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
        excluded_sample_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Extract task-scoped frames and remove partial output on failure.

        The caller can only clean a successful result because ``work_dir`` is
        part of that result.  Therefore extraction itself owns rollback until
        the result has been returned.
        """

        task_token = str(task_id or hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16])
        try:
            return self._extract_impl(
                role,
                files,
                directories,
                config,
                target_count=target_count,
                intervals=intervals,
                task_id=task_token,
                progress=progress,
                cancelled=cancelled,
                excluded_sample_ids=excluded_sample_ids,
            )
        except BaseException:
            shutil.rmtree(self.temp_root / task_token, ignore_errors=True)
            raise

    def _extract_impl(
        self,
        role: str,
        files: Iterable[str | Path] | None,
        directories: Iterable[str | Path] | None,
        config: dict[str, Any] | None,
        *,
        target_count: int,
        intervals: Any = None,
        task_id: str | None = None,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
        excluded_sample_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        role = "defect" if str(role).lower().startswith("defect") else "normal"
        # Probe once for valid/invalid output while preserving deterministic order.
        all_paths = self._files_from_inputs(files, directories)
        probed = [self.probe_video(path) for path in all_paths]
        videos = [item for item in probed if item.get("valid")]
        invalid = [item for item in probed if not item.get("valid")]
        work = self.temp_root / f"{task_id or hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]}" / role
        work.mkdir(parents=True, exist_ok=True)
        all_paths_out: list[Path] = []
        all_manifest: list[dict[str, Any]] = []
        counters = {"scanned": 0, "candidate": 0, "deduplicated": 0, "rejected": len(invalid), "selected": 0}
        excluded = {str(item) for item in (excluded_sample_ids or []) if str(item).strip()}
        for index, video in enumerate(videos):
            _check_cancel(cancelled)
            paths_out, manifest, stats = self._extract_one(video, role, config or {}, work, intervals, target_count, progress, cancelled, excluded)
            all_paths_out.extend(paths_out)
            all_manifest.extend(manifest)
            for key in ("scanned", "candidate", "deduplicated", "rejected", "selected"):
                counters[key] += int(stats.get(key, 0))
            if progress is not None:
                progress("video_sampling", index + 1, max(1, len(videos)), f"已完成视频：{video['name']}")
        sampling = self._sampling_defaults(config, target_count)
        all_paths_out, all_manifest = self._balanced_select(all_paths_out, all_manifest, min(sampling["final_max_samples"], max(1, target_count)), sampling["seed"])
        selected_by_path = {
            Path(str(row.get("path"))): row
            for row in all_manifest
            if row.get("selected")
        }
        selected_manifest = [selected_by_path[path] for path in all_paths_out if path in selected_by_path]
        # A frame may be encountered in two different videos (for example a
        # copied clip).  Keep the first deterministic representative and mark
        # later occurrences as deduplicated rather than inflating the budget.
        seen_hashes: set[str] = set()
        unique_paths: list[Path] = []
        unique_manifest: list[dict[str, Any]] = []
        dedup_paths: list[Path] = []
        dedup_rows: list[dict[str, Any]] = []
        for path, row in zip(all_paths_out, selected_manifest):
            digest = str(row.get("frame_hash") or "")
            if digest and digest in seen_hashes:
                counters["deduplicated"] += 1
                row["selected"] = False
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if digest:
                seen_hashes.add(digest)
            dedup_paths.append(path)
            dedup_rows.append(row)
        all_paths_out = dedup_paths
        # selected rows preserve their original manifest order, while all
        # non-selected budget candidates remain auditable.
        all_manifest = [row for row in all_manifest if not row.get("selected")]
        all_manifest.extend(dedup_rows)
        # Keep the summary count aligned with the actual files returned to the
        # feature builder.  The manifest intentionally retains unselected
        # budget candidates for auditability, so its row count is not a
        # selected-frame count.
        counters["selected"] = len(all_paths_out)
        return {
            "paths": all_paths_out,
            "manifest": all_manifest,
            "videos": probed,
            "summary": {
                "video_count": len(probed),
                "valid_video_count": len(videos),
                "invalid_video_count": len(invalid),
                "total_duration_seconds": round(sum(float(item.get("duration_seconds") or 0.0) for item in videos), 3),
                "scanned_frame_count": counters["scanned"],
                "candidate_frame_count": counters["candidate"],
                "deduplicated_frame_count": counters["deduplicated"],
                "rejected_frame_count": counters["rejected"],
                "selected_frame_count": counters["selected"],
                "warnings": [f"{item['name']}：{item.get('error')}" for item in invalid],
                "sampling_config": sampling,
                "source_video_hashes": {str(item.get("name")): str(item.get("source_video_hash")) for item in videos},
            },
            "work_dir": work.parent,
        }

    @staticmethod
    def cleanup(result: dict[str, Any]) -> None:
        work_dir = result.get("work_dir")
        if work_dir:
            shutil.rmtree(Path(str(work_dir)), ignore_errors=True)
