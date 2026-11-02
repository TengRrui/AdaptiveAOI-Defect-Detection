from __future__ import annotations

from typing import Any

from bridge.api_client import client


def load_statistics_dashboard(module: str, **filters: Any) -> dict[str, Any]:
    payload = client().statistics_dashboard(module=module, **filters)
    if not payload.get("success"):
        error = payload.get("error") or {}
        return {
            "ok": False,
            "error": str(error.get("message") or "统计数据加载失败，请稍后重试。"),
            "api_error": error,
        }
    return {"ok": True, "dashboard": payload.get("dashboard") or {}}


def load_statistics_revision() -> int:
    payload = client().statistics_revision()
    if not payload.get("success"):
        return 0
    try:
        return max(0, int(payload.get("statistics_revision") or 0))
    except (TypeError, ValueError):
        return 0


__all__ = ["load_statistics_dashboard", "load_statistics_revision"]
