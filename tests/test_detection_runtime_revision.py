from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from backend.api.routes import _validate_inference_binding
from backend.adapters.patchcore_adapter import PatchCoreAdapter
from backend.migrations.detection_history_v0002_runtime_revision import MIGRATION_ID
from backend.services.job_service import JobService
from backend.services.result_storage import ResultStorage
from video.video_inference import VideoDetectionOptions, run_video_detection


class _Settings:
    def __init__(self, root: Path):
        self.project_root = root
        self.runtime_root = root / "runtime"
        self.jobs_root = self.runtime_root / "jobs"
        self.upload_root = self.runtime_root / "uploads"
        self.log_root = self.runtime_root / "logs"

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_root, self.jobs_root, self.upload_root, self.log_root):
            path.mkdir(parents=True, exist_ok=True)


def _result(revision: int | None, *, result_id: str, task_id: str, batch_id: str | None = None) -> dict:
    context = {
        "profile_id": "profile-qa",
        "memory_bank_id": "bank-qa",
        "config_id": "config-qa",
        "model_id": "model-qa",
        "threshold": 0.42,
        "precision": "fp16",
        "feature_layers": ["layer2"],
        "runtime_revision": revision,
        "algorithm_mode": "normal_memory_defect_guided_lightweight_calibration",
        "started_at": "2026-08-25T15:00:00+00:00",
    }
    return {
        "result_id": result_id,
        "runtime_revision": revision,
        "model_id": "model-qa",
        "prediction": "normal",
        "is_anomaly": False,
        "anomaly_score": 0.12,
        "threshold": 0.42,
        "inference_time_ms": 1.2,
        "device": "cuda",
        "metadata": {
            "dataset": "qa",
            "category": "revision",
            "filename": f"{result_id}.png",
            "source_path": f"QA_TEST_REVISION_{result_id}.png",
            "task_id": task_id,
            "runtime_revision": revision,
            "detection_context": context,
        },
    }


def _binding_services(revision_box: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_profiles=SimpleNamespace(
            context_snapshot=lambda: {
                "runtime_revision": revision_box["value"],
                "profile_id": "profile-qa",
                "profile_version": "2",
                "status": "ready",
                "method_mode": "normal_memory_defect_guided_lightweight_calibration",
            },
            configuration=lambda: {
                "profile_status": "ready",
                "memory_bank_id": "bank-qa",
                "model_id": "model-qa",
                "dataset": "qa",
                "category": "revision",
                "config_profile_id": "config-qa",
                "config_profile_version": 1,
                "device": "cuda",
            },
        ),
        domain_profiles=SimpleNamespace(
            get_profile=lambda *_args: {
                "frozen": True,
                "threshold": 0.42,
                "actual_precision": "fp16",
                "feature_layers": ["layer2"],
                "method_mode_resolved": "normal_memory_defect_guided_lightweight_calibration",
            }
        ),
        config_profiles=SimpleNamespace(
            compatibility=lambda *_args: {"compatible": True},
            get=lambda *_args: {
                "config_hash": "qa-config-hash",
                "threshold": 0.42,
                "config": {
                    "device": "cuda",
                    "precision": "fp16",
                    "feature_layers": ["layer2"],
                    "method_mode_resolved": "normal_memory_defect_guided_lightweight_calibration",
                },
            },
        ),
    )


def _binding(services: SimpleNamespace, revision: int) -> dict:
    return _validate_inference_binding(
        services,
        profile_id="profile-qa",
        runtime_revision=revision,
        memory_bank_id=None,
        model_id=None,
        dataset=None,
        category=None,
        config_profile_id=None,
        config_profile_version=None,
    )


def test_single_detection_runtime_revision_persisted(tmp_path: Path) -> None:
    storage = ResultStorage(_Settings(tmp_path))  # type: ignore[arg-type]
    storage.save_result(_result(41, result_id="single-1", task_id="single-task"), request_id="request-1")

    stored = storage.get_result("single-1")

    assert stored is not None
    assert stored["runtime_revision"] == 41
    assert stored["metadata"]["runtime_revision"] == 41
    assert stored["metadata"]["detection_context"]["runtime_revision"] == 41


def test_batch_parent_runtime_revision(tmp_path: Path) -> None:
    settings = _Settings(tmp_path)
    jobs = JobService(settings)  # type: ignore[arg-type]
    job = jobs.create("batch")
    context = _binding(_binding_services({"value": 52}), 52)["detection_context"]
    selection = PatchCoreAdapter._manifest_selection(
        {
            "model_id": "model-qa",
            "memory_bank_id": "bank-qa",
            "config_profile_id": "config-qa",
            "detection_context": context,
        }
    )
    jobs.update(
        job,
        status="running",
        runtime_revision=52,
        detection_context=context.as_dict(),
        selection=selection,
    )

    manifest = jobs.read_manifest(job)

    assert manifest["runtime_revision"] == 52
    assert manifest["detection_context"]["runtime_revision"] == 52
    assert manifest["selection"]["runtime_revision"] == 52


def test_batch_children_runtime_revision(tmp_path: Path) -> None:
    storage = ResultStorage(_Settings(tmp_path))  # type: ignore[arg-type]
    for index in range(3):
        storage.save_result(
            _result(52, result_id=f"batch-child-{index}", task_id="batch-task", batch_id="batch-task"),
            request_id="batch-request",
            batch_id="batch-task",
        )

    page = storage.list_results(page_size=10)
    children = [row for row in page["items"] if row["batch_id"] == "batch-task"]

    assert len(children) == 3
    assert {row["runtime_revision"] for row in children} == {52}
    assert {row["metadata"]["detection_context"]["runtime_revision"] for row in children} == {52}


class _FakeVideoInferencer:
    def infer_prepared(self, prepared, meta=None, save_outputs=False, generate_visuals=False):
        return {
            "score": 0.15,
            "threshold": 0.42,
            "pred_label": 0,
            "inference_time_ms": 1.0,
            "_heatmap_array": np.zeros((prepared.height, prepared.width), dtype=np.float32),
        }


def _tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 32))
    assert writer.isOpened()
    try:
        writer.write(np.full((32, 48, 3), 80, dtype=np.uint8))
        writer.write(np.full((32, 48, 3), 120, dtype=np.uint8))
    finally:
        writer.release()


def test_video_runtime_revision(tmp_path: Path) -> None:
    source = tmp_path / "QA_TEST_REVISION_VIDEO.mp4"
    _tiny_video(source)
    output_dir = tmp_path / "video-results"
    context = _result(63, result_id="unused", task_id="video-task")["metadata"]["detection_context"]
    summary = run_video_detection(
        VideoDetectionOptions(
            input_video=source,
            output_video=output_dir / "result.mp4",
            output_dir=output_dir,
            config={"dataset": "qa", "category": "revision", "image_size": 32, "device": "cpu"},
            generate_visualizations=False,
            max_frames=2,
            runtime_revision=63,
            detection_context=context,
        ),
        inferencer=_FakeVideoInferencer(),
    )

    jsonl_rows = [json.loads(line) for line in Path(summary["jsonl_path"]).read_text(encoding="utf-8").splitlines()]
    csv_rows = list(csv.DictReader(Path(summary["csv_path"]).open("r", encoding="utf-8-sig", newline="")))

    assert summary["runtime_revision"] == 63
    assert summary["detection_context"]["runtime_revision"] == 63
    assert {row["runtime_revision"] for row in jsonl_rows} == {63}
    assert {int(row["runtime_revision"]) for row in csv_rows} == {63}


def test_runtime_switch_after_task_start_does_not_change_revision() -> None:
    revision_box = {"value": 71}
    binding = _binding(_binding_services(revision_box), 71)
    frozen = binding["detection_context"]

    revision_box["value"] = 72

    assert frozen.runtime_revision == 71
    assert frozen.as_dict()["runtime_revision"] == 71
    assert frozen.profile_id == "profile-qa"
    assert frozen.memory_bank_id == "bank-qa"
    assert frozen.config_id == "config-qa"
    assert frozen.model_id == "model-qa"
    assert frozen.threshold == 0.42
    assert frozen.precision == "fp16"
    assert frozen.feature_layers == ("layer2",)
    assert frozen.algorithm_mode == "normal_memory_defect_guided_lightweight_calibration"
    assert frozen.started_at


def test_legacy_detection_null_runtime_revision(tmp_path: Path) -> None:
    storage = ResultStorage(_Settings(tmp_path))  # type: ignore[arg-type]
    legacy = _result(None, result_id="legacy-1", task_id="legacy-task")
    legacy.pop("runtime_revision", None)
    legacy["metadata"].pop("runtime_revision", None)
    legacy["metadata"].pop("detection_context", None)
    storage.save_result(legacy, request_id="legacy-request")

    stored = storage.get_result("legacy-1")

    assert stored is not None
    assert stored["runtime_revision"] is None
    assert stored["runtime_revision_display"] == "历史记录未记录"


def test_detection_update_preserves_runtime_revision(tmp_path: Path) -> None:
    settings = _Settings(tmp_path)
    jobs = JobService(settings)  # type: ignore[arg-type]
    job = jobs.create("batch")
    jobs.update(
        job,
        status="pending",
        runtime_revision=81,
        detection_context={"runtime_revision": 81},
        selection={"runtime_revision": 81, "detection_context": {"runtime_revision": 81}},
    )

    for status in ("running", "completed", "failed"):
        jobs.update(job, status=status, progress=1.0)

    manifest = jobs.read_manifest(job)
    assert manifest["runtime_revision"] == 81
    assert manifest["detection_context"]["runtime_revision"] == 81
    assert manifest["selection"]["runtime_revision"] == 81


def test_detection_history_runtime_revision_migration_is_versioned(tmp_path: Path) -> None:
    storage = ResultStorage(_Settings(tmp_path))  # type: ignore[arg-type]
    with storage._connect() as conn:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(detection_results)")}
        migration = conn.execute(
            "SELECT migration_id FROM schema_migrations WHERE migration_id = ?",
            (MIGRATION_ID,),
        ).fetchone()
    assert "runtime_revision" in columns
    assert migration is not None
