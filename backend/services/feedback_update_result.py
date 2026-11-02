"""Durable, user-facing result semantics for feedback optimization jobs.

The worker keeps its internal state machine (``ready_to_apply``,
``validation_failed`` and so on) because those values are useful for APIs and
recovery.  This module translates that state into a stable result contract for
operators.  It deliberately consumes persisted job fields only; no metric is
recomputed in the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


STRATEGY_NAMES = {
    "threshold_only": "仅调整判定参数",
    "normal_memory": "更新正常记忆",
    "defect_prototypes": "更新缺陷原型",
    "lightweight_calibrator": "更新轻量校准器",
    "comprehensive": "综合更新",
    "safe_incremental": "安全增量更新",
}
RESULT_SCHEMA_VERSION = 2


@dataclass
class FeedbackUpdateResult:
    """Stable result object persisted with every terminal update task."""

    task_id: str
    outcome: str
    outcome_code: str
    title: str
    summary: str
    failed_stage: str = ""
    strategy: dict[str, Any] = field(default_factory=dict)
    source_memory_bank: dict[str, Any] = field(default_factory=dict)
    candidate_memory_bank: dict[str, Any] = field(default_factory=dict)
    used_samples: dict[str, Any] = field(default_factory=dict)
    skipped_samples: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    validation_rules: list[dict[str, Any]] = field(default_factory=list)
    failed_rules: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    generated_resources: dict[str, Any] = field(default_factory=dict)
    can_apply: bool = False
    retryable: bool = False
    result_version: int = RESULT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _rule(
    key: str,
    label: str,
    actual: Any,
    operator: str,
    required: Any,
    *,
    passed: bool | None = None,
    unit: str = "",
) -> dict[str, Any]:
    if passed is None:
        if actual is None:
            passed = False
        elif operator == ">=":
            passed = float(actual) >= float(required)
        elif operator == "<=":
            passed = float(actual) <= float(required)
        elif operator == "==":
            passed = bool(actual) == bool(required)
    return {
        "key": key,
        "label": label,
        "actual": actual,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
        "unit": unit,
    }


def _validation_rules(job: Mapping[str, Any], validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    constraints = _dict(job.get("constraints"))
    before = _dict(validation.get("metrics_before"))
    after = _dict(validation.get("metrics_after"))
    latency = _dict(validation.get("latency"))
    complexity = _dict(validation.get("complexity_estimate"))
    updates = _dict(validation.get("updates"))
    memory_update = _dict(updates.get("normal_memory"))
    rules: list[dict[str, Any]] = []

    min_recall = _float_or_none(constraints.get("min_recall"))
    if min_recall is not None:
        rules.append(_rule("recall", "缺陷召回率", after.get("recall"), ">=", min_recall))
    max_fpr = _float_or_none(constraints.get("max_false_positive_rate"))
    if max_fpr is not None:
        rules.append(_rule("false_positive_rate", "误报率", after.get("false_positive_rate"), "<=", max_fpr))

    max_latency = _float_or_none(constraints.get("max_latency_growth"))
    latency_actual = _float_or_none(complexity.get("p95_latency_growth"))
    if max_latency is not None and latency_actual is not None:
        rules.append(_rule("latency_growth", "P95 延迟增长", latency_actual, "<=", max_latency, unit="比例"))

    max_memory = _float_or_none(constraints.get("max_memory_growth"))
    memory_actual = _float_or_none(memory_update.get("memory_growth"))
    if memory_actual is None and not bool(job.get("update_normal_memory")):
        memory_actual = 0.0
    if max_memory is not None and memory_actual is not None:
        rules.append(_rule("memory_growth", "记忆库增长", memory_actual, "<=", max_memory, unit="比例"))

    if "base_memory_unchanged" in validation:
        rules.append(_rule(
            "base_memory_integrity",
            "基础记忆库完整性",
            validation.get("base_memory_unchanged"),
            "==",
            True,
            passed=bool(validation.get("base_memory_unchanged")),
        ))
    elif before or after:
        # Legacy validation records did not persist the hash comparison.  A
        # missing value is deliberately shown as unavailable, not as a pass.
        rules.append(_rule("base_memory_integrity", "基础记忆库完整性", None, "==", True, passed=False))

    return rules


def _failed_rules(job: Mapping[str, Any], validation: Mapping[str, Any], rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explicit = [dict(item) for item in _list(validation.get("failed_checks")) if isinstance(item, Mapping)]
    if explicit:
        return explicit
    eligibility = _dict(validation.get("eligibility"))
    blockers = [dict(item) for item in _list(eligibility.get("blockers")) if isinstance(item, Mapping)]
    if blockers:
        return blockers
    return [
        {
            "code": f"{rule.get('key')}_failed",
            "message": f"{rule.get('label') or rule.get('key')}未满足要求。",
            "actual": rule.get("actual"),
            "required": rule.get("required"),
        }
        for rule in rules
        if not bool(rule.get("passed"))
    ]


def _execution_summary(job: Mapping[str, Any]) -> str:
    """Turn a worker exception into a short operator-facing explanation.

    The full exception is still retained in ``error_message`` and exposed in
    the technical details.  The normal result summary must not force an
    operator to understand Python exception names or absolute artifact paths.
    """
    raw_error = str(job.get("error_message") or "").strip()
    message = str(job.get("message") or "").strip()
    diagnostic = raw_error or message
    lowered = diagnostic.lower()
    if "unidentifiedimageerror" in lowered or "cannot identify image" in lowered:
        return "纠错样本原始文件无法识别为有效图像，请检查该样本文件或从本次更新中移除。"
    if "filenotfounderror" in lowered or "no such file" in lowered or "不存在" in raw_error:
        return "纠错样本原始文件不存在或已被移动，请检查文件后重新运行。"
    if "permissionerror" in lowered or "permission denied" in lowered or "权限" in raw_error:
        return "纠错样本文件没有读取权限，请检查文件权限后重新运行。"
    generic = {"反馈更新未完成，请查看系统日志。", "后台执行阶段发生异常，候选版本未发布。"}
    if diagnostic and diagnostic not in generic:
        # Do not leak absolute paths or a Python traceback into the ordinary
        # result card.  The complete value remains available under technical
        # details for diagnosis.
        exception_name = diagnostic.split(":", 1)[0].strip()
        if exception_name in {
            "NameError", "UnboundLocalError", "TypeError", "KeyError",
            "AttributeError", "IndexError", "ValueError",
        }:
            return "后台执行阶段发生内部异常，候选版本未发布；请重新运行或查看技术详情。"
        if exception_name:
            return f"后台执行阶段发生异常（{exception_name}），候选版本未发布；请重新运行或查看技术详情。"
    if message and message not in generic:
        return message
    return "后台执行阶段发生异常，候选版本未发布。"


def _outcome(job: Mapping[str, Any], validation: Mapping[str, Any]) -> tuple[str, str, str, str, bool, bool]:
    status = str(job.get("status") or "").strip().lower()
    validation_status = str(validation.get("status") or "").strip().lower()
    publish_mode = str(_dict(job.get("constraints")).get("publish_mode") or "update_current")
    if status == "applied":
        if publish_mode == "replace_original":
            cleanup = _dict(job.get("replacement_cleanup"))
            if cleanup.get("cleanup_status") == "archived_pending_delete":
                return "success", "SUCCESS_REPLACED_WITH_CLEANUP_PENDING", "✅ Memory Bank 更新成功，原版本已安全替换", "新版本已通过验证并成为当前运行方案；原版本已归档，物理清理待维护任务完成。", False, False
            return "success", "SUCCESS_REPLACED", "✅ Memory Bank 更新成功，已替换原版本", "新版本已通过验证并成为当前运行方案；原版本已从普通资源列表移除，历史审计仍保留。", False, False
        return "success", "SUCCESS_APPLIED", "✅ Memory Bank 更新成功并已应用", "新版本已通过验证并切换为当前运行方案。", False, False
    if status == "rolled_back":
        return "rolled_back", "ROLLED_BACK", "↩ 更新版本已回滚", "原运行方案已恢复为当前方案，生成的新版本仍保留用于审计。", False, False
    if status in {"ready_to_apply", "frozen", "completed"}:
        if status == "frozen":
            return "success", "SUCCESS_READY_TO_APPLY", "✅ Memory Bank 更新成功，版本已冻结待应用", "新版本已生成、验证并冻结，当前运行方案尚未切换。", True, False
        return "success", "SUCCESS_READY_TO_APPLY", "✅ Memory Bank 更新成功，新版本已生成并通过验证（待应用）", "新版本已生成并通过验证，当前运行方案尚未切换。", True, False
    if status == "cancelled":
        return "cancelled", "CANCELLED", "— Memory Bank 更新已取消", "用户取消了本次更新；未发布候选版本，原运行方案不受影响。", False, True
    if status == "validation_failed":
        if validation_status in {"eligibility_failed", "invalid_source"}:
            if validation_status == "eligibility_failed":
                return "eligibility_failed", "ELIGIBILITY_FAILED", "❌ 样本条件不满足，无法开始更新", "所选纠错样本未满足当前策略的最低准入条件。", False, True
            return "execution_failed", "EXECUTION_FAILED", "❌ 更新执行失败", "反馈样本的原始文件或快照不可读取，候选版本未发布。", False, True
        return "validation_failed", "VALIDATION_FAILED", "❌ 候选版本验证未通过", "候选版本已生成或计算，但未通过 Recall、误报率、延迟、记忆库增长或完整性门禁。", False, True
    if status in {"failed", "interrupted"}:
        if status == "interrupted":
            return "execution_failed", "EXECUTION_FAILED", "❌ 更新执行中断", "后台任务心跳停止或服务重启，候选版本未发布；可以重新运行。", False, True
        return "execution_failed", "EXECUTION_FAILED", "❌ 更新执行失败", "后台执行阶段发生异常，候选版本未发布。", False, True
    return "in_progress", "IN_PROGRESS", "⏳ Memory Bank 更新进行中", "后台任务仍在执行，最终结果将在任务完成后生成。", False, False


def build_feedback_update_result(
    job: Mapping[str, Any],
    item_summary: Mapping[str, Any] | None = None,
) -> FeedbackUpdateResult:
    """Build a result from persisted fields without recalculating metrics."""
    validation = _dict(job.get("validation"))
    item_summary = _dict(item_summary or job.get("item_summary"))
    outcome, outcome_code, title, summary, can_apply, retryable = _outcome(job, validation)
    rules = _validation_rules(job, validation)
    # An execution error before offline validation is not a validation
    # failure.  Do not manufacture Recall/FPR failures merely because the
    # task carried default constraint values.
    if outcome == "execution_failed" and not (
        _dict(validation.get("metrics_before")) or _dict(validation.get("metrics_after"))
    ):
        rules = []
    failed = _failed_rules(job, validation, rules)
    status = str(job.get("status") or "").lower()

    if outcome == "eligibility_failed":
        blockers = failed
        if blockers:
            summary = "；".join(str(item.get("message") or "样本准入条件未满足") for item in blockers)
    elif outcome == "validation_failed" and failed:
        summary = "；".join(str(item.get("message") or "验证规则未通过") for item in failed)
    elif outcome == "execution_failed":
        summary = _execution_summary(job)
        if not failed:
            failed = [{
                "code": str(job.get("error_code") or "FEEDBACK_OPTIMIZATION_FAILED"),
                "message": summary,
                "technical_message": str(job.get("error_message") or ""),
            }]
    elif outcome == "cancelled":
        summary = str(job.get("message") or summary)

    current_stage = str(job.get("current_stage_id") or job.get("stage") or "").strip()
    failed_stage = ""
    if outcome == "eligibility_failed":
        failed_stage = "纠错样本准入"
    elif outcome == "validation_failed":
        failed_stage = "候选版本验证"
    elif outcome == "execution_failed":
        failed_stage = current_stage

    warnings = [dict(item) for item in _list(validation.get("warnings")) if isinstance(item, Mapping)]
    if int(item_summary.get("skipped") or 0):
        warnings.append({
            "code": "SAMPLES_SKIPPED",
            "message": f"有 {int(item_summary.get('skipped') or 0)} 条样本未参与本次更新。",
        })

    normal_count = validation.get("normal_count")
    defect_count = validation.get("defect_count")
    strategy_code = str(job.get("strategy") or "").strip()
    strategy = {"code": strategy_code, "name": STRATEGY_NAMES.get(strategy_code, strategy_code or "目标域资源更新")}
    source_id = str(job.get("base_profile_id") or "")
    candidate_id = str(job.get("candidate_profile_id") or "")
    generated = {
        key: value
        for key, value in {
            "candidate_profile_id": candidate_id,
            "candidate_config_profile_id": job.get("candidate_config_profile_id"),
            "candidate_config_profile_version": job.get("candidate_config_profile_version"),
        }.items()
        if value not in (None, "")
    }
    replacement_cleanup = _dict(job.get("replacement_cleanup"))
    if replacement_cleanup:
        generated["replacement"] = replacement_cleanup

    return FeedbackUpdateResult(
        task_id=str(job.get("job_id") or ""),
        outcome=outcome,
        outcome_code=outcome_code,
        title=title,
        summary=summary,
        failed_stage=failed_stage,
        strategy=strategy,
        source_memory_bank={"memory_bank_id": source_id} if source_id else {},
        candidate_memory_bank={"memory_bank_id": candidate_id} if candidate_id else {},
        used_samples={
            "total": int(item_summary.get("completed") or 0),
            "completed": int(item_summary.get("completed") or 0),
            "normal": normal_count,
            "defect": defect_count,
        },
        skipped_samples={
            "skipped": int(item_summary.get("skipped") or 0),
            "failed": int(item_summary.get("failed") or 0),
            "cancelled": int(item_summary.get("cancelled") or 0),
            "interrupted": int(item_summary.get("interrupted") or 0),
        },
        validation_metrics={
            "before": _dict(validation.get("metrics_before")),
            "after": _dict(validation.get("metrics_after")),
            "latency": _dict(validation.get("latency")),
            "complexity": _dict(validation.get("complexity_estimate")),
        },
        validation_rules=rules,
        failed_rules=failed,
        warnings=warnings,
        elapsed_seconds=float(job.get("elapsed_seconds") or 0.0),
        generated_resources=generated,
        can_apply=bool(can_apply and validation.get("status") == "passed"),
        retryable=retryable,
    )
