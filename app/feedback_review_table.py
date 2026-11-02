from __future__ import annotations

from typing import Any


REVIEW_TABLE_HEADERS = (
    "序号",
    "选择",
    "检测时间",
    "样本名称",
    "原始判定",
    "人工反馈",
    "缺陷类别",
    "Mask",
    "来源记忆库",
    "审核状态",
    "操作",
)

# Ratios are intentionally compact. Long values are shortened in the table
# and remain available through the detail dialog/tooltips.
REVIEW_TABLE_WIDTHS = (0.34, 0.38, 1.18, 1.58, 0.64, 0.72, 0.72, 0.62, 1.02, 0.72, 1.12)


def compact_sample_name(value: Any, *, max_length: int = 28) -> str:
    """Shorten long filenames while preserving a useful suffix."""
    text = str(value or "检测样本").strip() or "检测样本"
    if len(text) <= max_length:
        return text
    suffix_length = min(9, max(5, max_length // 3))
    prefix_length = max(7, max_length - suffix_length - 1)
    return f"{text[:prefix_length]}…{text[-suffix_length:]}"


def corrected_label_text(value: Any) -> str:
    return {
        "normal": "正常",
        "defect": "缺陷",
    }.get(str(value or "").strip().lower(), "待确定")


__all__ = [
    "REVIEW_TABLE_HEADERS",
    "REVIEW_TABLE_WIDTHS",
    "compact_sample_name",
    "corrected_label_text",
]
