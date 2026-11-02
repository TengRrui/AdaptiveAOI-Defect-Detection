from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from backend.errors import ApiError
from backend.services.feedback_optimization_service import FeedbackOptimizationService
from backend.services.feedback_task_status import get_feedback_task_status_display
from core.memory_bank import MemoryBank


BASE_ID = "target_profile_v1"


class FakeCandidates:
    def __init__(self, rows: list[dict]):
        self.rows = {row["candidate_id"]: row for row in rows}

    def get_candidate(self, candidate_id: str) -> dict:
        if candidate_id not in self.rows:
            raise ApiError("NOT_FOUND", "candidate missing", 404)
        return dict(self.rows[candidate_id])

    @staticmethod
    def _transition_allowed(_previous: str, _next: str) -> bool:
        return True


class FakeSettings:
    def __init__(self, root: Path, current: str = BASE_ID):
        self.memory_banks_root = root / "memory_banks"
        self.jobs_root = root / "jobs"
        self.upload_root = root / "uploads"
        self.project_root = root
        self._dynamic = {"memory_bank_id": current}
        for path in (self.memory_banks_root, self.jobs_root, self.upload_root):
            path.mkdir(parents=True, exist_ok=True)

    def dynamic_settings(self) -> dict:
        return dict(self._dynamic)

    def update_dynamic_settings(self, values: dict) -> dict:
        self._dynamic.update(values)
        return dict(self._dynamic)


class FakeDomainProfiles:
    def __init__(self, base_dir: Path, metadata: dict):
        self.base_dir = base_dir
        self.metadata = metadata

    def resolve_for_inference(self, profile_id: str, model: str, dataset: str, category: str) -> dict:
        if profile_id != BASE_ID:
            raise ApiError("PROFILE_NOT_FOUND", "profile missing", 404)
        return {
            "memory_path": self.base_dir / "memory_bank.pt",
            "config": {"knn_k": 1},
            "method_mode_resolved": self.metadata["method_mode_resolved"],
            "config_profile_id": self.metadata["config_profile_id"],
            "config_profile_version": self.metadata["config_profile_version"],
        }

    def get_profile(self, profile_id: str, _dataset: str, _category: str, _model: str) -> dict:
        return {
            "memory_bank_id": profile_id,
            "config_profile_id": "cfg-base",
            "config_profile_version": 1,
            "threshold": 0.5,
            "device": "cpu",
        }

    def activate(self, *_args, **_kwargs) -> None:
        return None


class FakeConfigProfiles:
    def validate(self, *_args) -> dict:
        return {"valid": True}

    def get(self, *_args) -> dict:
        return {"config": {"image_size": 256, "knn_k": 1}}

    def frozen_snapshot(self, *_args) -> tuple[dict, str]:
        return ({"image_size": 256, "knn_k": 1}, "cfg-hash")

    def bind_and_freeze(self, *_args) -> None:
        return None

    def mark_applied(self, *_args) -> None:
        return None


class FakeRuntimeResources:
    def __init__(self):
        self.applied: list[str] = []
        self.apply_options: list[dict] = []
        self.apply_error: ApiError | None = None
        self.revisions = 0

    def rebuild_index(self) -> None:
        self.revisions += 1

    def list_resources(self, **_kwargs) -> list[dict]:
        return [
            {"profile_id": BASE_ID, "resource_id": "resource-base"},
            {"profile_id": "feedback_candidate_v2", "resource_id": "resource-candidate"},
        ]

    def apply_profile(self, resource_id: str, **_kwargs) -> dict:
        self.applied.append(resource_id)
        self.apply_options.append(dict(_kwargs))
        if self.apply_error is not None:
            raise self.apply_error
        return {"applied": True}


def _schema(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE feedback_candidates(candidate_id TEXT PRIMARY KEY,status TEXT,updated_at TEXT);
            CREATE TABLE feedback_audit_log(audit_id TEXT PRIMARY KEY,candidate_id TEXT,job_id TEXT,action TEXT,actor TEXT,from_status TEXT,to_status TEXT,details_json TEXT,created_at TEXT);
            CREATE TABLE feedback_optimization_jobs(
              job_id TEXT PRIMARY KEY,base_profile_id TEXT,base_config_profile_id TEXT,base_config_profile_version INTEGER,
              candidate_ids_json TEXT,candidate_count INTEGER DEFAULT 0,strategy TEXT,objective TEXT,
              update_threshold INTEGER DEFAULT 1,update_normal_memory INTEGER DEFAULT 0,update_defect_memory INTEGER DEFAULT 0,
              update_lightweight_calibrator INTEGER DEFAULT 0,status TEXT,progress REAL DEFAULT 0,stage TEXT,message TEXT,
              constraints_json TEXT,candidate_profile_id TEXT,candidate_config_profile_id TEXT,candidate_config_profile_version INTEGER,
              metrics_before_json TEXT DEFAULT '{}',metrics_after_json TEXT DEFAULT '{}',complexity_before_json TEXT DEFAULT '{}',
              complexity_after_json TEXT DEFAULT '{}',validation_json TEXT DEFAULT '{}',error_code TEXT,error_message TEXT,
              created_at TEXT,started_at TEXT,finished_at TEXT,applied_at TEXT,rolled_back_at TEXT,previous_profile_id TEXT,
              previous_config_profile_id TEXT,previous_config_profile_version INTEGER);
            CREATE TABLE feedback_optimization_items(
              job_id TEXT,candidate_id TEXT,role TEXT,status TEXT,message TEXT,
              source_memory_bank_id TEXT,target_memory_bank_id TEXT,target_memory_bank_version TEXT,
              strategy TEXT,used_at TEXT,usage_result TEXT,generated_profile_id TEXT,
              started_at TEXT,finished_at TEXT,updated_at TEXT,
              PRIMARY KEY(job_id,candidate_id));
            """
        )


def _base_metadata(mode: str = "normal_defect_dual_memory") -> dict:
    return {
        "memory_bank_id": BASE_ID,
        "dataset": "visa",
        "category": "candle",
        "model_id": "patchcore-visa",
        "config_profile_id": "cfg-base",
        "config_profile_version": 1,
        "image_size": 256,
        "feature_dimension": 4,
        "method_mode_resolved": mode,
        "frozen": True,
        "threshold": 0.5,
    }


def _reviewed_rows(*, mode: str = "normal_defect_dual_memory") -> list[dict]:
    rows = []
    for index in range(5):
        rows.append({
            "candidate_id": f"normal-{index}", "status": "accepted", "memory_bank_id": BASE_ID,
            "dataset": "visa", "category": "candle", "model_version": "patchcore-visa",
            "config_profile_id": "cfg-base", "config_profile_version": 1,
            "method_mode": mode,
            "corrected_label": "normal", "original_prediction": "anomaly", "original_score": 0.10 + index * 0.01,
            "source_snapshot_sha256": f"normal-sha-{index}", "context": {"image_size": 256, "feature_dimension": 4},
            "annotations": [], "has_mask": False, "mode": "single",
        })
        rows.append({
            "candidate_id": f"defect-{index}", "status": "accepted", "memory_bank_id": BASE_ID,
            "dataset": "visa", "category": "candle", "model_version": "patchcore-visa",
            "config_profile_id": "cfg-base", "config_profile_version": 1,
            "method_mode": mode,
            "corrected_label": "defect", "original_prediction": "normal", "original_score": 0.90 - index * 0.01,
            "source_snapshot_sha256": f"defect-sha-{index}", "context": {"image_size": 256, "feature_dimension": 4},
            "annotations": [{"mask_path": "reviewed-mask.png"}], "has_mask": True, "mode": "video" if index == 0 else "batch",
        })
    return rows


def _compatibility_service(tmp_path: Path, rows: list[dict], mode: str = "normal_defect_dual_memory") -> FeedbackOptimizationService:
    database = tmp_path / "feedback.sqlite3"
    _schema(database)
    base_dir = tmp_path / "memory_banks" / BASE_ID
    base_dir.mkdir(parents=True)
    metadata = _base_metadata(mode)
    (base_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (base_dir / "memory_bank.pt").write_bytes(b"immutable-base")
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    service.path = database
    service.settings = FakeSettings(tmp_path)
    service.candidates = FakeCandidates(rows)
    service.domain_profiles = FakeDomainProfiles(base_dir, metadata)
    return service


def _payload(rows: list[dict], strategy: str = "comprehensive", publish_mode: str = "save_new_version") -> dict:
    return {
        "base_profile_id": BASE_ID,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "strategy": strategy,
        "publish_mode": publish_mode,
        "update_threshold": True,
        "update_normal_memory": strategy in {"normal_memory", "comprehensive"},
        "update_defect_memory": strategy in {"defect_prototypes", "comprehensive"},
        "update_lightweight_calibrator": strategy in {"lightweight_calibrator", "comprehensive"},
    }


def test_reviewed_compatibility_and_method_scoped_comprehensive(tmp_path: Path) -> None:
    rows = _reviewed_rows()
    service = _compatibility_service(tmp_path, rows)
    result = service.compatibility(_payload(rows))
    assert result["compatible"] is True
    assert result["normal_count"] == result["defect_count"] == 5
    assert result["safe_normal_count"] == result["masked_defect_count"] == 5
    assert result["update_flags"] == {
        "update_threshold": True,
        "update_normal_memory": True,
        "update_defect_memory": True,
        "update_lightweight_calibrator": False,
    }


def test_incompatible_official_and_duplicate_evidence_are_blocked(tmp_path: Path) -> None:
    rows = _reviewed_rows()
    rows[0]["context"]["official_test_mode"] = True
    rows[1]["dataset"] = "other"
    rows[3]["method_mode"] = "normal_memory"
    service = _compatibility_service(tmp_path, rows)
    with sqlite3.connect(service.path) as conn:
        conn.execute("INSERT INTO feedback_optimization_jobs(job_id,base_profile_id,candidate_ids_json,status,created_at) VALUES(?,?,?,?,?)", ("old-job", BASE_ID, "[]", "applied", "now"))
        conn.execute("INSERT INTO feedback_optimization_items(job_id,candidate_id,role,status,message) VALUES(?,?,?,?,?)", ("old-job", rows[2]["candidate_id"], "normal", "applied", ""))
    result = service.compatibility(_payload(rows, "threshold_only"))
    codes = {item["code"] for item in result["errors"]}
    assert "OFFICIAL_TEST_FEEDBACK_LOCKED" in codes
    assert "FEEDBACK_PROFILE_COMPATIBILITY_FAILED" in codes
    assert "FEEDBACK_ALREADY_CONSUMED" not in codes
    assert "FEEDBACK_METHOD_MODE_MISMATCH" not in codes
    warning_codes = {item["code"] for item in result["warnings"]}
    assert "FEEDBACK_METHOD_MODE_VARIANT" in warning_codes
    assert result["compatible"] is False


def test_normal_memory_feedback_is_compatible_with_calibrated_profile(tmp_path: Path) -> None:
    rows = _reviewed_rows(mode="normal_memory")
    service = _compatibility_service(tmp_path, rows, mode="normal_defect_calibration")

    result = service.compatibility(_payload(rows, "threshold_only"))

    assert result["compatible"] is True
    assert result["base_method_mode"] == "normal_defect_calibration"
    assert result["target_method_mode"] == "normal_defect_calibration"
    assert result["mode_transition"] is False
    assert result["source_method_modes"] == ["normal_memory"]
    assert "FEEDBACK_METHOD_MODE_VARIANT" in {item["code"] for item in result["warnings"]}


def test_cross_memory_bank_reuse_still_requires_feature_compatibility(tmp_path: Path) -> None:
    rows = _reviewed_rows()
    for row in rows:
        row["memory_bank_id"] = "older-compatible-bank"
        row["source_memory_bank_id"] = "older-compatible-bank"
    service = _compatibility_service(tmp_path, rows)

    compatible = service.compatibility(_payload(rows, "threshold_only"))

    assert compatible["compatible"] is True
    assert "FEEDBACK_CROSS_MEMORY_BANK_SOURCE" in {item["code"] for item in compatible["warnings"]}

    rows[0]["model_version"] = "different-feature-model"
    incompatible = service.compatibility(_payload(rows, "threshold_only"))
    assert incompatible["compatible"] is False
    assert "FEEDBACK_PROFILE_COMPATIBILITY_FAILED" in {item["code"] for item in incompatible["errors"]}


def test_normal_memory_threshold_update_creates_calibrated_candidate_mode(tmp_path: Path) -> None:
    rows = _reviewed_rows(mode="normal_memory")
    service = _compatibility_service(tmp_path, rows, mode="normal_memory")

    result = service.compatibility(_payload(rows, "threshold_only"))

    assert result["compatible"] is True
    assert result["base_method_mode"] == "normal_only"
    assert result["target_method_mode"] == "normal_defect_calibration"
    assert result["mode_transition"] is True


def test_feedback_task_status_presentation_is_unambiguous() -> None:
    assert get_feedback_task_status_display("ready_to_apply")["task_status_display"] == "已生成，待应用"
    assert get_feedback_task_status_display("ready_to_apply")["can_apply"] is True
    assert get_feedback_task_status_display("cancelled")["task_status_display"] == "已取消"
    assert get_feedback_task_status_display("failed")["task_status_display"] == "更新失败"
    assert get_feedback_task_status_display("applied")["task_status_display"] == "已应用"
    assert get_feedback_task_status_display("completed")["task_status_display"] == "已完成"


def test_terminal_task_requires_finalized_items_and_failures_finalize_active_rows(tmp_path: Path) -> None:
    rows = _reviewed_rows()
    service = _compatibility_service(tmp_path, rows)
    service.logger = SimpleNamespace(error=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)
    with sqlite3.connect(service.path) as conn:
        conn.execute(
            "INSERT INTO feedback_optimization_jobs(job_id,base_profile_id,candidate_ids_json,status,stage,message,constraints_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
            ("state-job", BASE_ID, '["normal-0"]', "running", "running", "", "{}", "now"),
        )
        conn.execute(
            "INSERT INTO feedback_optimization_items(job_id,candidate_id,role,status,message) VALUES(?,?,?,?,?)",
            ("state-job", "normal-0", "normal", "processing", ""),
        )
    with pytest.raises(ApiError) as exc_info:
        service._update("state-job", status="ready_to_apply")
    assert exc_info.value.code == "FEEDBACK_ITEM_STATE_INCOMPLETE"
    failed = service._update("state-job", status="failed", message="safe failure")
    assert failed["task_status_display"] == "更新失败"
    with sqlite3.connect(service.path) as conn:
        assert conn.execute(
            "SELECT status FROM feedback_optimization_items WHERE job_id='state-job'"
        ).fetchone()[0] == "failed"


def test_config_variant_is_allowed_but_feature_space_mismatch_is_blocked(tmp_path: Path) -> None:
    rows = _reviewed_rows(mode="normal_memory")
    rows[0]["config_profile_id"] = "cfg-alternate"
    rows[0]["config_profile_version"] = 9
    service = _compatibility_service(tmp_path, rows, mode="normal_defect_calibration")

    compatible = service.compatibility(_payload(rows, "threshold_only"))

    assert compatible["compatible"] is True
    assert "FEEDBACK_CONFIG_VARIANT" in {item["code"] for item in compatible["warnings"]}

    rows[0]["context"]["feature_dimension"] = 8
    incompatible = service.compatibility(_payload(rows, "threshold_only"))
    assert incompatible["compatible"] is False
    assert "FEEDBACK_EMBEDDING_DIMENSION_MISMATCH" in {item["code"] for item in incompatible["errors"]}


def test_fixed_budget_normal_memory_never_ingests_defect(tmp_path: Path) -> None:
    base_path = tmp_path / "base" / "memory_bank.pt"
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    base_features = np.arange(400, dtype=np.float32).reshape(100, 4) / 100.0
    MemoryBank(knn_k=1, device="cpu", backend="auto").fit(base_features).save(base_path, {"knn_k": 1})
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    extracted = [
        {"candidate": {"corrected_label": "normal", "original_prediction": "anomaly"}, "patches": np.full((50, 4), 9.0, dtype=np.float32), "mask_path": None},
        {"candidate": {"corrected_label": "defect", "original_prediction": "normal"}, "patches": np.full((50, 4), 999.0, dtype=np.float32), "mask_path": None},
        {"candidate": {"corrected_label": "normal", "original_prediction": "anomaly"}, "patches": np.full((50, 4), 555.0, dtype=np.float32), "mask_path": "mask.png"},
    ]
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    result = service._update_normal_memory(extracted, candidate_path, {"memory_path": base_path, "config": {"knn_k": 1}}, {}, {"max_memory_growth": 0.05})
    candidate_bank = MemoryBank.load(candidate_path / "memory_bank.pt", device="cpu")[0]
    assert result["updated"] is True
    assert result["memory_budget_preserved"] is True
    assert len(candidate_bank.features) == len(base_features) == 100
    assert np.any(np.all(np.isclose(candidate_bank.features, 9.0), axis=1))
    assert not np.any(np.all(np.isclose(candidate_bank.features, 999.0), axis=1))
    assert not np.any(np.all(np.isclose(candidate_bank.features, 555.0), axis=1))
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == base_hash
    assert len(torch.load(candidate_path / "coreset_indices.pt", weights_only=True)) == 100


def test_masked_defect_builds_separate_prototype_only(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :16] = 255
    Image.fromarray(mask).save(mask_path)
    extracted = [
        {"candidate": {"corrected_label": "normal"}, "patches": np.random.default_rng(1).normal(size=(16, 4)).astype(np.float32), "patch_distances": np.full(16, 0.05, dtype=np.float32), "score": 0.1},
        {"candidate": {"corrected_label": "defect"}, "patches": np.random.default_rng(2).normal(size=(16, 4)).astype(np.float32), "patch_distances": np.linspace(0.5, 1.0, 16, dtype=np.float32), "patch_shape": (4, 4), "mask_path": mask_path, "score": 0.9},
    ]
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    metadata = {"method_mode_resolved": "normal_defect_dual_memory", "max_defect_prototypes": 8}
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    result = service._update_defect_memory(extracted, candidate_path, metadata)
    assert result["updated"] is True
    assert result["masked_defect_count"] == 1
    assert (candidate_path / "defect_prototypes.pt").is_file()
    assert not (candidate_path / "memory_bank.pt").exists()


def test_validation_metrics_report_false_negative_rate() -> None:
    result = FeedbackOptimizationService._metrics([0.2, 0.8, 0.4, 0.3], [0, 1, 1, 0], 0.5)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["false_negative_rate"] == 0.5


def test_optimizer_uses_verified_review_snapshot_and_rejects_tampering(tmp_path: Path) -> None:
    settings = FakeSettings(tmp_path)
    snapshot = settings.jobs_root / "feedback" / "snapshot_v1"
    snapshot.mkdir(parents=True)
    source = snapshot / "source.png"
    mask = snapshot / "reviewed_mask.png"
    source.write_bytes(b"reviewed-source")
    mask.write_bytes(b"reviewed-mask")
    files = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (source, mask)
    ]
    manifest = snapshot / "snapshot.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    candidate = {
        "source_snapshot_ref": "feedback/snapshot_v1",
        "source_snapshot_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "annotations": [],
    }
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    service.settings = settings
    resolved_source, resolved_mask = service._snapshot_evidence(candidate)
    assert resolved_source == source.resolve()
    assert resolved_mask == mask.resolve()
    assert service._source_image(candidate, tmp_path) == source.resolve()
    source.write_bytes(b"tampered")
    assert service._source_image(candidate, tmp_path) is None


def test_threshold_search_and_lightweight_calibrator(tmp_path: Path) -> None:
    rows = [
        {"original_score": score, "corrected_label": "normal" if index < 5 else "defect"}
        for index, score in enumerate([0.10, 0.15, 0.20, 0.25, 0.30, 0.75, 0.80, 0.85, 0.90, 0.95])
    ]
    threshold, metrics, options = FeedbackOptimizationService._choose_threshold(
        rows, 0.5, "f1", min_recall=0.95, max_fpr=0.10,
    )
    assert 0.30 < threshold <= 0.75
    assert metrics["recall"] == 1.0 and metrics["false_positive_rate"] == 0.0
    assert options
    extracted = []
    for index in range(5):
        extracted.append({"candidate": {"corrected_label": "normal"}, "patch_distances": np.linspace(0.01, 0.2, 16, dtype=np.float32), "score": 0.10 + index * 0.01})
        extracted.append({"candidate": {"corrected_label": "defect"}, "patch_distances": np.linspace(0.5, 1.0, 16, dtype=np.float32), "score": 0.90 - index * 0.01})
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    result = service._update_lightweight(extracted, tmp_path, {"threshold": threshold}, {"config": {"random_seed": 42}})
    assert result["updated"] is True
    assert result["parameter_count"] > 0
    assert (tmp_path / "lightweight_calibrator.json").is_file()


def test_update_current_requires_current_base_and_failed_job_cannot_apply(tmp_path: Path) -> None:
    rows = _reviewed_rows()
    service = _compatibility_service(tmp_path, rows)
    service.settings._dynamic["memory_bank_id"] = "another-profile"
    result = service.compatibility(_payload(rows, "threshold_only", "update_current"))
    assert "FEEDBACK_CURRENT_PROFILE_MISMATCH" in {item["code"] for item in result["errors"]}
    service.settings._dynamic["memory_bank_id"] = BASE_ID
    service.config_profiles = FakeConfigProfiles()
    service.registry = SimpleNamespace(invalidate=lambda *_args: None)
    service.runtime_resources = FakeRuntimeResources()
    service._lock = __import__("threading").RLock()
    with sqlite3.connect(service.path) as conn:
        conn.execute(
            "INSERT INTO feedback_optimization_jobs(job_id,base_profile_id,candidate_ids_json,status,stage,message,constraints_json,validation_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("failed-job", BASE_ID, "[]", "validation_failed", "failed", "failed", "{}", '{"status":"failed"}', "now"),
        )
    with pytest.raises(ApiError) as exc_info:
        service.apply("failed-job", confirm=True)
    assert exc_info.value.code == "FEEDBACK_JOB_NOT_READY"


def test_freeze_apply_and_rollback_keep_original_profile(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite3"
    _schema(database)
    settings = FakeSettings(tmp_path)
    candidate_id = "feedback_candidate_v2"
    hidden = settings.memory_banks_root / ".feedback_candidates" / candidate_id
    hidden.mkdir(parents=True)
    metadata = {
        **_base_metadata(), "memory_bank_id": candidate_id, "feedback_parent_profile_id": BASE_ID,
        "config_profile_id": "cfg-candidate", "config_profile_version": 2,
    }
    (hidden / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (hidden / "memory_bank.pt").write_bytes(b"candidate-memory")
    (hidden / "threshold.json").write_text(json.dumps({"image_threshold": 0.5}), encoding="utf-8")
    (hidden / "effective_config.yaml").write_text("image_size: 256\nknn_k: 1\n", encoding="utf-8")
    base_dir = settings.memory_banks_root / BASE_ID
    base_dir.mkdir()
    (base_dir / "memory_bank.pt").write_bytes(b"immutable-base")
    (base_dir / "metadata.json").write_text(json.dumps(_base_metadata()), encoding="utf-8")
    base_hash = hashlib.sha256((base_dir / "memory_bank.pt").read_bytes()).hexdigest()
    with sqlite3.connect(database) as conn:
        conn.execute("INSERT INTO feedback_candidates(candidate_id,status,updated_at) VALUES(?,?,?)", ("normal-0", "ready_to_apply", "now"))
        conn.execute(
            "INSERT INTO feedback_optimization_jobs(job_id,base_profile_id,candidate_ids_json,status,stage,message,constraints_json,candidate_profile_id,candidate_config_profile_id,candidate_config_profile_version,validation_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("job-1", BASE_ID, '[\"normal-0\"]', "ready_to_apply", "ready", "", '{\"publish_mode\":\"save_new_version\"}', candidate_id, "cfg-candidate", 2, '{\"status\":\"passed\",\"base_memory_unchanged\":true}', "now"),
        )
        conn.execute("INSERT INTO feedback_optimization_items(job_id,candidate_id,role,status,message) VALUES(?,?,?,?,?)", ("job-1", "normal-0", "normal", "ready_to_apply", ""))
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    service.path = database
    service.settings = settings
    service.domain_profiles = FakeDomainProfiles(base_dir, _base_metadata())
    service.config_profiles = FakeConfigProfiles()
    service.registry = SimpleNamespace(invalidate=lambda *_args: None)
    service.candidates = FakeCandidates([])
    service.runtime_resources = FakeRuntimeResources()
    frozen = service.freeze("job-1", confirm=True)
    assert frozen["status"] == "frozen"
    assert (settings.memory_banks_root / candidate_id).is_dir()
    assert settings.dynamic_settings()["memory_bank_id"] == BASE_ID
    assert hashlib.sha256((base_dir / "memory_bank.pt").read_bytes()).hexdigest() == base_hash
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT status FROM feedback_candidates WHERE candidate_id='normal-0'").fetchone()[0] == "accepted"
        usage = conn.execute("SELECT usage_result,generated_profile_id FROM feedback_optimization_items WHERE job_id='job-1' AND candidate_id='normal-0'").fetchone()
        assert usage == ("frozen", candidate_id)
    service.runtime_resources.apply_error = ApiError("RUNTIME_TASK_ACTIVE", "真实外部任务仍在运行", 409)
    with pytest.raises(ApiError) as exc_info:
        service.apply("job-1", confirm=True)
    assert exc_info.value.code == "RUNTIME_TASK_ACTIVE"
    assert service.status("job-1")["status"] == "frozen"
    assert not any(
        thread.name == "feedback-apply-heartbeat-job-1" and thread.is_alive()
        for thread in threading.enumerate()
    )
    service.runtime_resources.apply_error = None
    applied = service.apply("job-1", confirm=True)
    assert applied["status"] == "applied"
    assert service.runtime_resources.applied[-1] == "resource-candidate"
    assert service.runtime_resources.apply_options[-1]["feedback_job_id"] == "job-1"
    assert not any(
        thread.name == "feedback-apply-heartbeat-job-1" and thread.is_alive()
        for thread in threading.enumerate()
    )
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT status FROM feedback_candidates WHERE candidate_id='normal-0'").fetchone()[0] == "accepted"
        assert conn.execute("SELECT usage_result FROM feedback_optimization_items WHERE job_id='job-1' AND candidate_id='normal-0'").fetchone()[0] == "applied"
    rolled_back = service.rollback("job-1", confirm=True)
    assert rolled_back["status"] == "rolled_back"
    assert service.runtime_resources.applied[-1] == "resource-base"
    assert hashlib.sha256((base_dir / "memory_bank.pt").read_bytes()).hexdigest() == base_hash
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT status FROM feedback_candidates WHERE candidate_id='normal-0'").fetchone()[0] == "accepted"
        assert conn.execute("SELECT usage_result FROM feedback_optimization_items WHERE job_id='job-1' AND candidate_id='normal-0'").fetchone()[0] == "rolled_back"
