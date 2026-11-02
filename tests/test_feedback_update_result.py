from __future__ import annotations

import json

from backend.services.feedback_update_result import build_feedback_update_result
from backend.database import DatabaseManager
from backend.services.feedback_optimization_service import FeedbackOptimizationService


def test_validation_failure_preserves_actual_and_required_values() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "task-recall",
            "status": "validation_failed",
            "strategy": "threshold_only",
            "base_profile_id": "bank-a",
            "constraints": {
                "min_recall": 0.95,
                "max_false_positive_rate": 0.10,
                "max_memory_growth": 0.05,
                "max_latency_growth": 0.10,
            },
            "validation": {
                "status": "failed",
                "metrics_after": {"recall": 0.60, "false_positive_rate": 0.08},
                "complexity_estimate": {"p95_latency_growth": 0.02},
                "base_memory_unchanged": True,
                "failed_checks": [
                    {
                        "code": "RECALL_BELOW_MINIMUM",
                        "message": "Recall 0.600 < 要求 0.950",
                        "actual": 0.60,
                        "required": 0.95,
                    }
                ],
            },
        }
    ).as_dict()

    assert result["outcome_code"] == "VALIDATION_FAILED"
    assert result["failed_stage"] == "候选版本验证"
    assert result["failed_rules"][0]["actual"] == 0.60
    recall = next(item for item in result["validation_rules"] if item["key"] == "recall")
    assert recall["actual"] == 0.60
    assert recall["required"] == 0.95


def test_eligibility_failure_is_not_reported_as_validation_failure() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "task-samples",
            "status": "validation_failed",
            "strategy": "lightweight_calibrator",
            "validation": {
                "status": "eligibility_failed",
                "eligibility": {
                    "blockers": [
                        {
                            "code": "STRATEGY_NORMAL_SAMPLES_INSUFFICIENT",
                            "field": "normal",
                            "actual": 2,
                            "required": 3,
                            "message": "更新轻量校准器至少需要 3 条正常样本；当前 2 条。",
                        }
                    ]
                },
            },
        }
    ).as_dict()

    assert result["outcome_code"] == "ELIGIBILITY_FAILED"
    assert result["failed_stage"] == "纠错样本准入"
    assert result["failed_rules"][0]["actual"] == 2
    assert "3 条" in result["summary"]


def test_success_outcomes_have_explicit_apply_semantics() -> None:
    common = {
        "job_id": "task-success",
        "strategy": "normal_memory",
        "base_profile_id": "bank-a",
        "candidate_profile_id": "bank-b",
        "candidate_config_profile_id": "cfg-b",
        "candidate_config_profile_version": 2,
        "validation": {
            "status": "passed",
            "metrics_before": {"recall": 0.96, "false_positive_rate": 0.08},
            "metrics_after": {"recall": 0.97, "false_positive_rate": 0.06},
            "complexity_estimate": {"p95_latency_growth": 0.01},
            "base_memory_unchanged": True,
        },
    }
    ready = build_feedback_update_result({**common, "status": "ready_to_apply"}).as_dict()
    applied = build_feedback_update_result({**common, "status": "applied"}).as_dict()
    completed = build_feedback_update_result({**common, "status": "completed"}).as_dict()

    assert ready["outcome_code"] == "SUCCESS_READY_TO_APPLY"
    assert ready["can_apply"] is True
    assert "待应用" in ready["title"]
    assert applied["outcome_code"] == "SUCCESS_APPLIED"
    assert "已应用" in applied["title"]
    assert completed["outcome_code"] == "SUCCESS_READY_TO_APPLY"
    assert completed["can_apply"] is True


def test_replacement_result_explains_old_version_removal() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "replace-1",
            "status": "applied",
            "constraints": {"publish_mode": "replace_original"},
            "base_profile_id": "bank-a",
            "candidate_profile_id": "bank-a-v2",
            "replacement_cleanup": {
                "cleanup_status": "completed",
                "old_resource_id": "res_old",
                "new_resource_id": "res_new",
            },
            "validation": {"status": "passed"},
        },
        {"completed": 2},
    ).as_dict()
    assert result["outcome_code"] == "SUCCESS_REPLACED"
    assert "替换原版本" in result["title"]
    assert result["generated_resources"]["replacement"]["old_resource_id"] == "res_old"


def test_replacement_cleanup_pending_is_not_hidden() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "replace-2",
            "status": "applied",
            "constraints": {"publish_mode": "replace_original"},
            "base_profile_id": "bank-a",
            "candidate_profile_id": "bank-a-v2",
            "replacement_cleanup": {"cleanup_status": "archived_pending_delete"},
            "validation": {"status": "passed"},
        },
        {},
    ).as_dict()
    assert result["outcome_code"] == "SUCCESS_REPLACED_WITH_CLEANUP_PENDING"
    assert "物理清理" in result["summary"]


def test_cancelled_outcome_is_explicit_and_retryable() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "task-cancelled",
            "status": "cancelled",
            "message": "用户取消了本次更新。",
        }
    ).as_dict()
    assert result["outcome_code"] == "CANCELLED"
    assert "取消" in result["title"]
    assert result["retryable"] is True


def test_execution_failure_keeps_stage_and_retry_signal() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "task-exec",
            "status": "failed",
            "stage": "提取纠错样本特征",
            "current_stage_id": "feature_extraction",
            "error_code": "FEATURE_SOURCE_MISSING",
            "message": "第7个样本文件无法读取。",
        }
    ).as_dict()
    assert result["outcome_code"] == "EXECUTION_FAILED"
    assert result["failed_stage"] == "feature_extraction"
    assert result["retryable"] is True
    assert result["failed_rules"][0]["code"] == "FEATURE_SOURCE_MISSING"


def test_execution_failure_does_not_fake_validation_failures() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "task-image-error",
            "status": "failed",
            "stage": "提取纠错样本特征",
            "current_stage_id": "feature_extraction",
            "error_code": "FEATURE_SOURCE_INVALID",
            "error_message": "UnidentifiedImageError: cannot identify image file",
            "message": "反馈更新未完成，请查看系统日志。",
            "constraints": {"min_recall": 0.95, "max_false_positive_rate": 0.10},
        }
    ).as_dict()
    assert result["outcome_code"] == "EXECUTION_FAILED"
    assert result["summary"].startswith("纠错样本原始文件无法识别")
    assert result["validation_rules"] == []
    assert result["failed_rules"][0]["code"] == "FEATURE_SOURCE_INVALID"


def test_execution_failure_hides_traceback_from_ordinary_summary() -> None:
    result = build_feedback_update_result(
        {
            "job_id": "task-internal-error",
            "status": "failed",
            "current_stage_id": "feature_extraction",
            "error_code": "FEEDBACK_OPTIMIZATION_FAILED",
            "error_message": "",
            "message": "NameError: name 'base_id' is not defined",
        }
    ).as_dict()
    assert "NameError" not in result["summary"]
    assert "base_id" not in result["summary"]
    assert "内部异常" in result["summary"]


def test_terminal_result_is_persisted_for_restart(tmp_path) -> None:
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    service.path = tmp_path / "feedback.sqlite3"
    service.database = DatabaseManager(service.path)
    service.logger = type("Logger", (), {"error": lambda *_args, **_kwargs: None})()
    service._ensure_schema()
    with service._connect() as conn:
        conn.execute(
            "INSERT INTO feedback_optimization_jobs(job_id,base_profile_id,candidate_ids_json,strategy,objective,status,progress,stage,message,constraints_json,validation_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "persisted-task", "bank-a", '["sample-1"]', "threshold_only", "f1", "running", 0.5, "验证新版本", "",
                '{"min_recall":0.95}',
                json.dumps({
                    "status": "failed",
                    "metrics_after": {"recall": 0.60},
                    "failed_checks": [{"code": "RECALL_BELOW_MINIMUM", "message": "Recall 0.600 < 要求 0.950", "actual": 0.60, "required": 0.95}],
                }),
                "2026-08-15T00:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO feedback_optimization_items(job_id,candidate_id,role,status) VALUES(?,?,?,?)",
            ("persisted-task", "sample-1", "defect", "processing"),
        )
    service._update("persisted-task", status="validation_failed", stage="候选版本验证失败", message="Recall未达标", finished_at="2026-08-15T00:00:02+00:00")
    with service._connect() as conn:
        stored = conn.execute("SELECT result_json FROM feedback_optimization_jobs WHERE job_id=?", ("persisted-task",)).fetchone()[0]
    payload = json.loads(stored)
    assert payload["outcome_code"] == "VALIDATION_FAILED"
    assert payload["failed_rules"][0]["actual"] == 0.60
