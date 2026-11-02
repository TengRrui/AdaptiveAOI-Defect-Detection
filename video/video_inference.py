"""Offline video-file, frame-by-frame inference using the official PatchCore core.

The module intentionally contains no UI logic and never reloads the model for
each frame.  It is an offline video-file pipeline, not a claim of live-camera
or industrial real-time operation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from core.image_io import PreparedImage
from core.postprocess import binary_mask, make_overlay, normalize_uint8
from methods.patchcore.infer import PatchCoreInferencer

from .video_reader import VideoInputError, VideoReader
from .video_types import CancelCallback, FrameResult, PauseCallback, PausedSecondsCallback, PreviewCallback, ProgressCallback
from .video_writer import VideoOutputError, VideoWriter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_artifact_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, payload = cv2.imencode(suffix, image)
    if not ok:
        raise VideoOutputError(f"无法编码帧图像：{path}")
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload.tobytes())
    os.replace(temporary, path)


def _frame_to_pil_rgb(frame: np.ndarray) -> Image.Image:
    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"不支持的视频帧通道形状：{frame.shape}")
    return Image.fromarray(rgb)


def _boxes_from_heatmap(heatmap: np.ndarray, threshold: float, min_area: int = 16) -> list[dict[str, Any]]:
    mask = binary_mask(heatmap, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict[str, Any]] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < int(min_area):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        region = heatmap[y : y + height, x : x + width]
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": area,
                "max_score": float(np.max(region)) if region.size else float(threshold),
            }
        )
    return sorted(boxes, key=lambda item: (item["max_score"], item["area"]), reverse=True)


def _annotate_frame(
    bgr_frame: np.ndarray,
    heatmap: np.ndarray,
    result: FrameResult,
    overlay_enabled: bool,
) -> np.ndarray:
    if overlay_enabled:
        overlay_rgb = make_overlay(_frame_to_pil_rgb(bgr_frame), heatmap=heatmap)
        output = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    else:
        output = bgr_frame.copy()
    color = (0, 0, 220) if result.prediction_label == 1 else (0, 170, 0)
    for box in result.boxes:
        cv2.rectangle(output, (box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"]), color, 2)
    label = f"{result.prediction} score={result.anomaly_score:.4f} threshold={result.threshold:.4f}"
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 620), 40), (18, 18, 18), -1)
    cv2.putText(output, label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, f"frame={result.frame_index} time={result.timestamp_ms / 1000.0:.2f}s", (14, output.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return output


def _error_frame(frame: np.ndarray, frame_index: int, message: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 44), (0, 0, 190), -1)
    cv2.putText(output, f"FRAME {frame_index} INFERENCE ERROR", (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


@dataclass
class VideoDetectionOptions:
    input_video: str | Path
    output_video: str | Path
    config: dict[str, Any]
    memory_path: str | Path | None = None
    model_name: str = "PatchCoreAOI"
    config_path: str = ""
    device: str = ""
    threshold: float | None = None
    generate_visualizations: bool = True
    frame_stride: int = 1
    start_time: float | None = None
    end_time: float | None = None
    max_frames: int | None = None
    save_overlay: bool = True
    save_heatmap: bool = False
    save_mask: bool = False
    save_jsonl: bool = True
    save_csv: bool = True
    save_keyframes: bool = True
    overwrite: bool = False
    progress_callback: ProgressCallback | None = None
    cancel_callback: CancelCallback | None = None
    pause_callback: PauseCallback | None = None
    paused_seconds_callback: PausedSecondsCallback | None = None
    preview_callback: PreviewCallback | None = None
    result_callback: Callable[[FrameResult], None] | None = None
    model_load_ms: float | None = None
    memory_bank_load_ms: float | None = None
    model_reused: bool = False
    memory_bank_reused: bool = False
    output_dir: str | Path | None = None
    runtime_revision: int | None = None
    detection_context: dict[str, Any] | None = None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _duration_stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "total_ms": None, "mean_ms": None, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(clean),
        "total_ms": float(sum(clean)),
        "mean_ms": float(statistics.fmean(clean)),
        "p50_ms": _percentile(clean, 50.0),
        "p95_ms": _percentile(clean, 95.0),
        "min_ms": float(min(clean)),
        "max_ms": float(max(clean)),
    }


def _write_records(records: list[FrameResult], jsonl_path: Path, csv_path: Path, save_jsonl: bool, save_csv: bool) -> tuple[str, str]:
    plain = [record.as_dict() for record in records]
    jsonl_output = ""
    csv_output = ""
    if save_jsonl:
        payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in plain)
        _atomic_write_text(jsonl_path, payload)
        jsonl_output = str(jsonl_path)
    if save_csv:
        rows = []
        for row in plain:
            serializable = dict(row)
            serializable["boxes"] = json.dumps(serializable["boxes"], ensure_ascii=False, separators=(",", ":"))
            rows.append(serializable)
        fieldnames = list(rows[0].keys()) if rows else ["frame_index", "status"]
        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        _atomic_write_text(csv_path, buffer.getvalue())
        csv_output = str(csv_path)
    return jsonl_output, csv_output


def _write_keyframes(keyframes: dict[str, np.ndarray], keyframe_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for name, frame in keyframes.items():
        path = keyframe_dir / f"{name}.png"
        _atomic_write_image(path, frame)
        outputs[name] = str(path)
    return outputs


def run_video_detection(
    options: VideoDetectionOptions,
    inferencer: Any | None = None,
    inferencer_factory: Callable[[dict[str, Any], str | Path | None], Any] | None = None,
) -> dict[str, Any]:
    """Process a local video frame-by-frame through one loaded official model.

    `inferencer` and `inferencer_factory` are injection points for automated
    tests.  Normal callers leave both unset; exactly one PatchCoreInferencer is
    created before decoding the first frame.
    """

    if int(options.frame_stride) < 1:
        raise ValueError("frame_stride 必须大于或等于 1。")
    detection_context = dict(options.detection_context or {})
    runtime_revision = options.runtime_revision
    if runtime_revision is None and detection_context.get("runtime_revision") is not None:
        runtime_revision = int(detection_context["runtime_revision"])
    if runtime_revision is not None:
        runtime_revision = int(runtime_revision)
        detection_context["runtime_revision"] = runtime_revision
    input_path = Path(options.input_video).expanduser().resolve()
    requested_output = Path(options.output_video).expanduser().resolve()
    if input_path == requested_output:
        raise ValueError("输出视频不能覆盖输入视频。")
    output_dir = Path(options.output_dir).expanduser().resolve() if options.output_dir else requested_output.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = _safe_artifact_path(output_dir / "frame_results.jsonl", options.overwrite)
    csv_path = _safe_artifact_path(output_dir / "frame_results.csv", options.overwrite)
    summary_path = _safe_artifact_path(output_dir / "video_summary.json", options.overwrite)
    keyframe_dir = output_dir / "keyframes"
    if keyframe_dir.exists() and not options.overwrite:
        keyframe_dir = _safe_artifact_path(keyframe_dir, False)

    config = dict(options.config)
    config.update(
        {
            "save_heatmap": False,
            "save_mask": False,
            "save_overlay": False,
            "no_save_visuals": True,
            "async_save_outputs": False,
            "retain_postprocess_arrays": bool(options.generate_visualizations),
            "timing_mode": "strict_total" if options.generate_visualizations else "no_save_total",
        }
    )
    if options.threshold is not None:
        config["threshold"] = float(options.threshold)
        config["use_calibrated_threshold"] = False
    factory = inferencer_factory or (lambda cfg, memory: PatchCoreInferencer(cfg, category=cfg.get("category"), memory_path=memory))
    owns_inferencer = inferencer is None
    model_initialization_ms = 0.0
    if inferencer is None:
        started = time.perf_counter()
        inferencer = factory(config, options.memory_path)
        model_initialization_ms = (time.perf_counter() - started) * 1000.0

    # A cached backend inferencer was originally created for image inference.
    # Retain its heatmap only for this call, then restore the original setting
    # before releasing RuntimeRegistry's execution lock.
    restore_inferencer_config: dict[str, tuple[bool, Any]] = {}
    if hasattr(inferencer, "config") and isinstance(inferencer.config, dict):
        for key in ("retain_postprocess_arrays", "timing_mode"):
            restore_inferencer_config[key] = (key in inferencer.config, inferencer.config.get(key))
        inferencer.config["retain_postprocess_arrays"] = bool(options.generate_visualizations)
        inferencer.config["timing_mode"] = "strict_total" if options.generate_visualizations else "no_save_total"

    records: list[FrameResult] = []
    keyframes: dict[str, np.ndarray] = {}
    highest_score = float("-inf")
    warnings: list[str] = ["本结果为离线视频文件逐帧检测，不代表工业实时视频在线检测性能。"]
    writer: VideoWriter | None = None
    writer_info = None
    cancelled = False
    processing_started = time.perf_counter()
    last_frame_index = -1
    try:
        with VideoReader(input_path) as reader:
            metadata = reader.metadata
            assert metadata is not None
            output_fps = metadata.source_fps if int(options.frame_stride) == 1 else metadata.source_fps / int(options.frame_stride)
            if options.generate_visualizations:
                writer = VideoWriter(requested_output, output_fps, metadata.width, metadata.height, overwrite=options.overwrite)
                writer_info = writer.open()
            frame_iterator = iter(reader.iter_frames(options.start_time, options.end_time, options.max_frames))
            while True:
                # Pause before requesting the next packet.  This prevents the
                # decoder from reading ahead after the worker has acknowledged
                # a safe frame boundary.  A frame already in inference is
                # allowed to finish before the pause is acknowledged.
                while options.pause_callback is not None and bool(options.pause_callback()):
                    if options.cancel_callback is not None and bool(options.cancel_callback()):
                        cancelled = True
                        warnings.append("处理被用户取消；输出仅包含取消前已经成功处理的帧。")
                        break
                    time.sleep(0.05)
                if cancelled:
                    break
                if options.cancel_callback is not None and bool(options.cancel_callback()):
                    cancelled = True
                    warnings.append("处理被用户取消；输出仅包含取消前已经成功处理的帧。")
                    break
                try:
                    packet = next(frame_iterator)
                except StopIteration:
                    break
                last_frame_index = packet.frame_index
                if packet.frame_index % int(options.frame_stride) != 0:
                    records.append(
                        FrameResult(
                            frame_index=packet.frame_index,
                            timestamp_ms=packet.timestamp_ms,
                            source_fps=metadata.source_fps,
                            original_width=metadata.width,
                            original_height=metadata.height,
                            inference_width=int(config.get("image_size", 0)),
                            inference_height=int(config.get("image_size", 0)),
                            inference_time_ms=None,
                            frame_pipeline_ms=None,
                            anomaly_score=None,
                            prediction="SKIPPED_BY_STRIDE",
                            prediction_label=None,
                            threshold=None,
                            status="skipped_by_stride",
                            decode_ms=packet.decode_ms,
                            runtime_revision=runtime_revision,
                            detection_context=dict(detection_context),
                        )
                    )
                    continue
                frame_started = time.perf_counter()
                try:
                    color_started = time.perf_counter()
                    pil_image = _frame_to_pil_rgb(packet.bgr_frame)
                    color_convert_ms = (time.perf_counter() - color_started) * 1000.0
                    prepared = PreparedImage(
                        image_path=Path(f"{input_path.stem}__frame_{packet.frame_index:06d}.png"),
                        image=pil_image,
                        width=metadata.width,
                        height=metadata.height,
                        file_read_ms=0.0,
                        decode_ms=packet.decode_ms,
                        color_convert_ms=color_convert_ms,
                    )
                    core = inferencer.infer_prepared(prepared, meta=None, save_outputs=False, generate_visuals=False)
                    threshold = float(core["threshold"])
                    heatmap = None
                    boxes: list[dict[str, Any]] = []
                    extra_postprocess_ms = 0.0
                    if options.generate_visualizations:
                        heatmap = np.asarray(core.get("_heatmap_array"), dtype=np.float32)
                        if heatmap.ndim != 2:
                            raise RuntimeError("当前推理接口未返回二维 heatmap；无法生成视频可视化。")
                        boxes_started = time.perf_counter()
                        boxes = _boxes_from_heatmap(heatmap, threshold)
                        extra_postprocess_ms = float((time.perf_counter() - boxes_started) * 1000.0)
                    result = FrameResult(
                        frame_index=packet.frame_index,
                        timestamp_ms=packet.timestamp_ms,
                        source_fps=metadata.source_fps,
                        original_width=metadata.width,
                        original_height=metadata.height,
                        inference_width=int(config.get("image_size", 0)),
                        inference_height=int(config.get("image_size", 0)),
                        inference_time_ms=float(core.get("detection_only_ms", core.get("inference_time_ms", core.get("total_ms", 0.0)))),
                        frame_pipeline_ms=None,
                        anomaly_score=float(core["score"]),
                        baseline_score=float(core.get("baseline_score", core["score"])),
                        calibrated_score=float(core.get("calibrated_score", core["score"])),
                        final_score=float(core.get("final_score", core["score"])),
                        calibration_applied=bool(core.get("calibration_applied", False)),
                        gate_status=str(core.get("gate_status", "not_applicable")),
                        fallback_used=bool(core.get("fallback_used", False)),
                        calibration_version=str(core.get("calibration_version", "")),
                        method_mode=str(core.get("method_mode") or config.get("method_mode_resolved") or "normal_memory"),
                        prediction="ANOMALY" if int(core["pred_label"]) else "NORMAL",
                        prediction_label=int(core["pred_label"]),
                        threshold=threshold,
                        heatmap_shape=[int(value) for value in heatmap.shape] if heatmap is not None else None,
                        heatmap_min=float(np.min(heatmap)) if heatmap is not None else None,
                        heatmap_max=float(np.max(heatmap)) if heatmap is not None else None,
                        boxes=boxes,
                        decode_ms=packet.decode_ms,
                        preprocess_ms=float(core.get("preprocess_ms", 0.0) or 0.0) + color_convert_ms,
                        feature_extraction_ms=float(core.get("feature_ms", 0.0) or 0.0),
                        ann_search_ms=float(core.get("memory_score_ms", 0.0) or 0.0),
                        anomaly_scoring_ms=(
                            float(core.get("defect_score_ms", 0.0) or 0.0)
                            + float(core.get("fusion_ms", 0.0) or 0.0)
                            + float(core.get("image_score_ms", 0.0) or 0.0)
                        ),
                        postprocess_ms=float(core.get("heatmap_resize_ms", 0.0) or 0.0) + extra_postprocess_ms,
                        runtime_revision=runtime_revision,
                        detection_context=dict(detection_context),
                    )
                    annotated = None
                    if options.generate_visualizations:
                        assert heatmap is not None and writer is not None
                        visual_started = time.perf_counter()
                        annotated = _annotate_frame(packet.bgr_frame, heatmap, result, overlay_enabled=options.save_overlay)
                        result.visualization_ms = float((time.perf_counter() - visual_started) * 1000.0)
                        result.visualisation_ms = result.visualization_ms
                        save_started = time.perf_counter()
                        if options.save_heatmap:
                            heatmap_path = output_dir / "heatmaps" / f"frame_{packet.frame_index:06d}_heatmap.png"
                            _atomic_write_image(heatmap_path, normalize_uint8(heatmap))
                            result.heatmap_path = str(heatmap_path)
                        if options.save_mask:
                            mask_path = output_dir / "masks" / f"frame_{packet.frame_index:06d}_mask.png"
                            _atomic_write_image(mask_path, binary_mask(heatmap, threshold))
                            result.mask_path = str(mask_path)
                        result.save_ms = float((time.perf_counter() - save_started) * 1000.0)
                        write_started = time.perf_counter()
                        writer.write(annotated)
                        result.encode_ms = float((time.perf_counter() - write_started) * 1000.0)
                        result.encode_write_ms = result.encode_ms
                    result.frame_pipeline_ms = (time.perf_counter() - frame_started) * 1000.0
                    result.frame_total_ms = float(packet.decode_ms + result.frame_pipeline_ms)
                    records.append(result)
                    if options.result_callback is not None:
                        options.result_callback(result)
                    if options.generate_visualizations and annotated is not None:
                        if "first_frame" not in keyframes:
                            keyframes["first_frame"] = annotated.copy()
                        if result.prediction_label == 0 and "normal_representative" not in keyframes:
                            keyframes["normal_representative"] = annotated.copy()
                        if result.prediction_label == 1 and "first_anomaly" not in keyframes:
                            keyframes["first_anomaly"] = annotated.copy()
                        if result.anomaly_score is not None and result.anomaly_score > highest_score:
                            highest_score = result.anomaly_score
                            keyframes["highest_anomaly_score"] = annotated.copy()
                        keyframes["last_frame"] = annotated.copy()
                    if options.generate_visualizations and annotated is not None and options.preview_callback is not None:
                        # A browser may briefly retain the previous JPEG while
                        # the backend promotes the next preview on Windows.
                        # Preview delivery is optional UI feedback, never part
                        # of the algorithm result: a transient preview write
                        # failure must not turn an otherwise inferred frame
                        # into a failed inspection record or drop it from the
                        # output video.
                        try:
                            options.preview_callback(annotated, result)
                        except Exception as preview_exc:
                            warnings.append(
                                f"frame {packet.frame_index}: 预览更新已跳过（{type(preview_exc).__name__}: {preview_exc}）"
                            )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    failed = FrameResult(
                        frame_index=packet.frame_index,
                        timestamp_ms=packet.timestamp_ms,
                        source_fps=metadata.source_fps,
                        original_width=metadata.width,
                        original_height=metadata.height,
                        inference_width=int(config.get("image_size", 0)),
                        inference_height=int(config.get("image_size", 0)),
                        inference_time_ms=None,
                        frame_pipeline_ms=(time.perf_counter() - frame_started) * 1000.0,
                        anomaly_score=None,
                        prediction="ERROR",
                        prediction_label=None,
                        threshold=None,
                        error=error,
                        status="failed",
                        decode_ms=packet.decode_ms,
                        frame_total_ms=float(packet.decode_ms + (time.perf_counter() - frame_started) * 1000.0),
                        runtime_revision=runtime_revision,
                        detection_context=dict(detection_context),
                    )
                    records.append(failed)
                    if options.result_callback is not None:
                        options.result_callback(failed)
                    warnings.append(f"frame {packet.frame_index}: {error}")
                    # Do not fabricate a result visualization for failed inference.
                if options.progress_callback is not None:
                    total = metadata.reported_frame_count
                    options.progress_callback(
                        {
                            "frame_index": packet.frame_index,
                            "processed_records": len(records),
                            "reported_total_frames": total,
                            "progress": (packet.frame_index + 1) / total if total > 0 else None,
                        }
                    )
            if metadata.reported_frame_count > 0 and not cancelled and options.max_frames is None and options.end_time is None:
                actual_decoded = len(records)
                if actual_decoded < metadata.reported_frame_count:
                    warnings.append("解码在报告总帧数之前结束；已保留已成功读取的帧，并在汇总中记录实际帧数。")
                elif actual_decoded > metadata.reported_frame_count:
                    warnings.append(
                        "OpenCV 报告的输入总帧数小于顺序实际解码帧数；进度和汇总使用实际解码帧数，不丢弃有效帧。"
                    )
            if options.generate_visualizations:
                assert writer is not None
                if writer.frame_count <= 0:
                    raise VideoOutputError("没有成功推理的帧，无法生成有效输出视频。")
                writer_info = writer.close_and_promote()
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        if hasattr(inferencer, "config") and isinstance(inferencer.config, dict):
            for key, (existed, value) in restore_inferencer_config.items():
                if existed:
                    inferencer.config[key] = value
                else:
                    inferencer.config.pop(key, None)
        if owns_inferencer and hasattr(inferencer, "visual_save_queue"):
            inferencer.visual_save_queue.close()

    processing_loop_wall_clock_ms = (time.perf_counter() - processing_started) * 1000.0
    paused_ms = max(
        0.0,
        float(options.paused_seconds_callback() if options.paused_seconds_callback is not None else 0.0) * 1000.0,
    )
    processing_loop_ms = max(0.0, processing_loop_wall_clock_ms - paused_ms)
    success = [record for record in records if record.status == "success"]
    failures = [record for record in records if record.status == "failed"]
    skipped = [record for record in records if record.status == "skipped_by_stride"]
    inference_times = [float(record.inference_time_ms) for record in success if record.inference_time_ms is not None]
    pipeline_times = [float(record.frame_pipeline_ms) for record in success if record.frame_pipeline_ms is not None]
    normal_count = sum(record.prediction_label == 0 for record in success)
    anomaly_count = sum(record.prediction_label == 1 for record in success)
    highest = max(success, key=lambda item: float(item.anomaly_score or float("-inf"))) if success else None
    keyframe_save_started = time.perf_counter()
    keyframe_paths = _write_keyframes(keyframes, keyframe_dir) if options.generate_visualizations and options.save_keyframes else {}
    keyframe_save_ms = (time.perf_counter() - keyframe_save_started) * 1000.0 if keyframe_paths else 0.0
    records_save_started = time.perf_counter()
    jsonl_output, csv_output = _write_records(records, jsonl_path, csv_path, options.save_jsonl, options.save_csv)
    records_save_ms = (time.perf_counter() - records_save_started) * 1000.0

    output_path: Path | None = None
    output_metadata_payload: dict[str, Any] | None = None
    output_sha256 = ""
    if options.generate_visualizations:
        assert writer_info is not None
        output_path = writer_info.final_path
        # CAP_PROP_FRAME_COUNT can be one frame short for some FMP4/MP4V files.
        # Validate the promoted output by actually decoding it sequentially and
        # retain the container-reported count separately for traceability.
        with VideoReader(output_path) as output_reader:
            output_metadata = output_reader.metadata
            assert output_metadata is not None
            output_decoded_frames = sum(1 for _ in output_reader.iter_frames())
        output_readable = output_decoded_frames > 0
        if not output_readable:
            raise VideoOutputError(f"输出视频最终重新读取验证失败：{output_path}")
        if output_decoded_frames != len(success):
            raise VideoOutputError(
                f"输出视频逐帧回读数量不一致：成功推理 {len(success)} 帧，实际可解码 {output_decoded_frames} 帧。"
            )
        output_metadata_payload = {
            "fps": output_metadata.source_fps,
            "width": output_metadata.width,
            "height": output_metadata.height,
            "frame_count": output_metadata.reported_frame_count,
            "decoded_frame_count": output_decoded_frames,
            "file_size_bytes": output_path.stat().st_size,
            "readback_verified": True,
        }
        output_sha256 = _sha256(output_path)

    input_sha256 = _sha256(input_path)
    wall_clock_end_to_end_ms = (time.perf_counter() - processing_started) * 1000.0
    paused_ms = max(
        0.0,
        float(options.paused_seconds_callback() if options.paused_seconds_callback is not None else 0.0) * 1000.0,
    )
    active_processing_ms = max(0.0, wall_clock_end_to_end_ms - paused_ms)
    end_to_end_ms = wall_clock_end_to_end_ms
    throughput = len(success) / (active_processing_ms / 1000.0) if active_processing_ms > 0 else 0.0
    inference_total_ms = sum(inference_times)
    pure_inference_fps = len(success) / (inference_total_ms / 1000.0) if inference_total_ms > 0 else 0.0
    realtime = bool(int(options.frame_stride) == 1 and not failures and throughput >= metadata.source_fps)
    stage_values: dict[str, list[float]] = {
        "decode_ms": [float(record.decode_ms) for record in records],
        "preprocess_ms": [float(record.preprocess_ms) for record in success],
        "feature_extraction_ms": [float(record.feature_extraction_ms) for record in success],
        "ann_search_ms": [float(record.ann_search_ms) for record in success],
        "anomaly_scoring_ms": [float(record.anomaly_scoring_ms) for record in success],
        "inference_ms": inference_times,
        "postprocess_ms": [float(record.postprocess_ms) for record in success],
        "visualization_ms": [float(record.visualization_ms) for record in success if record.visualization_ms is not None],
        "encode_ms": [float(record.encode_ms) for record in success if record.encode_ms is not None],
        "save_ms": [float(record.save_ms) for record in success if record.save_ms is not None],
        "frame_total_ms": [float(record.frame_total_ms) for record in records if record.frame_total_ms is not None],
    }
    total_result_save_ms = float(sum(stage_values["save_ms"]) + records_save_ms + keyframe_save_ms)
    timing = {
        "model_load_ms": options.model_load_ms if not options.model_reused else None,
        "memory_bank_load_ms": options.memory_bank_load_ms if not options.memory_bank_reused else None,
        "model_reused": bool(options.model_reused),
        "memory_bank_reused": bool(options.memory_bank_reused),
        "processing_loop_ms": processing_loop_ms,
        "processing_loop_wall_clock_ms": processing_loop_wall_clock_ms,
        "active_processing_ms": active_processing_ms,
        "paused_ms": paused_ms,
        "wall_clock_end_to_end_ms": wall_clock_end_to_end_ms,
        "records_save_ms": records_save_ms,
        "keyframe_save_ms": keyframe_save_ms if options.generate_visualizations else None,
        "save_ms": total_result_save_ms,
        "end_to_end_ms": end_to_end_ms,
        "pure_inference_fps": pure_inference_fps,
        "full_pipeline_fps": throughput,
        "stages": {name: _duration_stats(values) for name, values in stage_values.items()},
    }
    visualization_artifacts = []
    if output_path is not None:
        visualization_artifacts.append({"kind": "annotated_video", "path": str(output_path)})
    visualization_artifacts.extend(
        {"kind": f"keyframe:{name}", "path": path} for name, path in keyframe_paths.items()
    )
    summary = {
        "runtime_revision": runtime_revision,
        "detection_context": detection_context,
        "processing_mode": "offline_video_file_frame_by_frame_detection",
        "input_video_path": str(input_path),
        "output_video_path": str(output_path) if output_path is not None else "",
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "input_video_metadata": metadata.as_dict(),
        "output_video_metadata": output_metadata_payload,
        "requested_output_video_path": str(requested_output),
        "generate_visualizations": bool(options.generate_visualizations),
        "visualization_status": "generated" if options.generate_visualizations else "not_generated",
        "visualization_artifacts": visualization_artifacts,
        "timing_mode": "strict_total" if options.generate_visualizations else "no_save_total",
        "timing": timing,
        "reported_input_frames": metadata.reported_frame_count,
        "total_decoded_records": len(records),
        "successful_processed_frames": len(success),
        "failed_frames": len(failures),
        "skipped_by_stride_frames": len(skipped),
        "normal_frames": int(normal_count),
        "anomaly_frames": int(anomaly_count),
        "highest_anomaly_score": float(highest.anomaly_score) if highest and highest.anomaly_score is not None else None,
        "highest_anomaly_frame": int(highest.frame_index) if highest else None,
        "method_mode": str(config.get("method_mode_resolved") or (success[0].method_mode if success else "normal_memory")),
        "calibration_version": str(success[0].calibration_version) if success else "",
        "calibration_applied_frames": int(sum(bool(record.calibration_applied) for record in success)),
        "fallback_frames": int(sum(bool(record.fallback_used) for record in success)),
        "average_inference_ms": float(statistics.fmean(inference_times)) if inference_times else None,
        "p50_inference_ms": _percentile(inference_times, 50.0),
        "p95_inference_ms": _percentile(inference_times, 95.0),
        "average_frame_pipeline_ms": float(statistics.fmean(pipeline_times)) if pipeline_times else None,
        "total_processing_ms": end_to_end_ms,
        "model_initialization_ms": options.model_load_ms if options.model_load_ms is not None else model_initialization_ms,
        "pure_inference_fps": pure_inference_fps,
        "full_pipeline_fps": throughput,
        "processing_throughput_fps": throughput,
        "reached_input_fps": realtime,
        "source_fps": metadata.source_fps,
        "frame_stride": int(options.frame_stride),
        "start_time": options.start_time,
        "end_time": options.end_time,
        "max_frames": options.max_frames,
        "device": options.device or str(config.get("device", "")),
        "model": options.model_name,
        "memory_path": str(Path(options.memory_path).resolve()) if options.memory_path else "",
        "config_path": options.config_path,
        "dataset": str(config.get("dataset", "")),
        "category": str(config.get("category", "")),
        "memory_bank_id": str(config.get("domain_profile_id", "")),
        "inference_mode": "inference_only" if bool(config.get("inference_only", False)) else "standard_inference",
        "test_data_update_allowed": bool(config.get("test_data_update_allowed", False)),
        "image_size": int(config.get("image_size", 0)),
        "threshold": float(success[0].threshold) if success and success[0].threshold is not None else options.threshold,
        "codec": writer_info.codec if writer_info is not None else None,
        "codec_fallback_used": writer_info.fallback_used if writer_info is not None else None,
        "jsonl_path": jsonl_output,
        "csv_path": csv_output,
        "keyframes": keyframe_paths,
        "model_loaded_once": owns_inferencer,
        "model_load_count": 1 if owns_inferencer else 0,
        "cancelled": cancelled,
        "warnings": warnings,
    }
    summary_save_started = time.perf_counter()
    _atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    summary_save_ms = float((time.perf_counter() - summary_save_started) * 1000.0)
    final_end_to_end_ms = float((time.perf_counter() - processing_started) * 1000.0)
    final_paused_ms = max(
        0.0,
        float(options.paused_seconds_callback() if options.paused_seconds_callback is not None else 0.0) * 1000.0,
    )
    final_active_processing_ms = max(0.0, final_end_to_end_ms - final_paused_ms)
    final_throughput = (
        len(success) / (final_active_processing_ms / 1000.0)
        if final_active_processing_ms > 0
        else 0.0
    )
    timing["summary_save_ms"] = summary_save_ms
    timing["save_ms"] = float(timing["save_ms"] + summary_save_ms)
    timing["end_to_end_ms"] = final_end_to_end_ms
    timing["wall_clock_end_to_end_ms"] = final_end_to_end_ms
    timing["active_processing_ms"] = final_active_processing_ms
    timing["paused_ms"] = final_paused_ms
    timing["full_pipeline_fps"] = final_throughput
    summary["total_processing_ms"] = final_end_to_end_ms
    summary["active_processing_seconds"] = final_active_processing_ms / 1000.0
    summary["paused_seconds"] = final_paused_ms / 1000.0
    summary["full_pipeline_fps"] = final_throughput
    summary["processing_throughput_fps"] = final_throughput
    summary["reached_input_fps"] = bool(int(options.frame_stride) == 1 and not failures and final_throughput >= metadata.source_fps)
    _atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = ["VideoDetectionOptions", "VideoInputError", "VideoOutputError", "run_video_detection"]
