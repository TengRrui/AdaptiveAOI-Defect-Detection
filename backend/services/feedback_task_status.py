"""Canonical user-facing state semantics for feedback optimization jobs/items."""

from __future__ import annotations

from typing import Any


_TASK_STATUS = {
    "queued": ("等待开始", "pending", "○", "info", False, False, True, False),
    "pending": ("等待开始", "pending", "○", "info", False, False, True, False),
    "preparing": ("准备中", "active", "⏳", "info", False, False, True, False),
    "running": ("更新中", "active", "⏳", "info", False, False, True, False),
    "building": ("更新中", "active", "⏳", "info", False, False, True, False),
    "updating": ("更新中", "active", "⏳", "info", False, False, True, False),
    "optimizing": ("更新中", "active", "⏳", "info", False, False, True, False),
    "validating": ("验证中", "active", "⏳", "info", False, False, True, False),
    "ready_to_apply": ("已生成，待应用", "ready", "✅", "success", True, False, False, True),
    "applying": ("正在应用", "active", "⏳", "info", False, False, False, False),
    "frozen": ("已冻结，待应用", "ready", "✅", "success", True, False, False, True),
    "applied": ("已应用", "success", "✅", "success", True, False, False, False),
    "completed": ("已完成", "success", "✅", "success", True, False, False, False),
    "cancelling": ("正在取消", "active", "⏳", "warning", False, False, False, False),
    "cancelled": ("已取消", "cancelled", "—", "warning", True, True, False, False),
    "validation_failed": ("验证未通过", "failed", "❌", "error", True, True, False, False),
    "failed": ("更新失败", "failed", "❌", "error", True, True, False, False),
    "interrupted": ("已中断", "failed", "⚠️", "warning", True, True, False, False),
    "rolled_back": ("已回滚", "success", "↩", "info", True, False, False, False),
}

_ITEM_STATUS = {
    "queued": "等待处理",
    "validating": "正在校验",
    "processing": "正在处理",
    "completed": "已用于候选版本",
    "skipped": "本次未使用",
    "failed": "处理失败",
    "cancelled": "已取消",
    "interrupted": "已中断",
    "legacy_state_unknown": "历史任务状态信息不完整",
}

ACTIVE_ITEM_STATUSES = frozenset({"queued", "validating", "processing"})
TERMINAL_ITEM_STATUSES = frozenset(set(_ITEM_STATUS) - set(ACTIVE_ITEM_STATUSES))
TERMINAL_TASK_STATUSES = frozenset(
    status for status, definition in _TASK_STATUS.items() if bool(definition[4])
)
ACTIVE_TASK_STATUSES = frozenset(set(_TASK_STATUS) - set(TERMINAL_TASK_STATUSES))


def get_feedback_task_status_display(status: Any) -> dict[str, Any]:
    """Return the only supported task status presentation/permission mapping."""
    normalized = str(status or "queued").strip().lower()
    definition = _TASK_STATUS.get(
        normalized,
        ("状态未知", "unknown", "?", "warning", False, False, False, False),
    )
    text, group, icon, severity, terminal, retry, cancel, apply = definition
    return {
        "task_status": normalized,
        "task_status_display": text,
        "status_group": group,
        "status_icon": icon,
        "severity": severity,
        "is_terminal": bool(terminal),
        "can_retry": bool(retry),
        "can_cancel": bool(cancel),
        "can_apply": bool(apply),
    }


def get_feedback_item_status_display(status: Any) -> str:
    normalized = str(status or "queued").strip().lower()
    return _ITEM_STATUS.get(normalized, "状态未知")
