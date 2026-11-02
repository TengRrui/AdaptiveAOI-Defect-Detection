from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from feedback_review_table import (
    REVIEW_TABLE_HEADERS,
    REVIEW_TABLE_WIDTHS,
    compact_sample_name,
    corrected_label_text,
)


def test_review_table_keeps_only_primary_audit_columns() -> None:
    assert len(REVIEW_TABLE_HEADERS) == 11
    assert len(REVIEW_TABLE_WIDTHS) == len(REVIEW_TABLE_HEADERS)
    assert "来源配置" not in REVIEW_TABLE_HEADERS
    assert "异常分数" not in REVIEW_TABLE_HEADERS
    assert "阈值" not in REVIEW_TABLE_HEADERS
    assert "来源记忆库" in REVIEW_TABLE_HEADERS
    assert REVIEW_TABLE_HEADERS[-1] == "操作"


def test_long_sample_name_preserves_distinguishing_suffix() -> None:
    source = "product_20260816_shift_a_station_03_defect_000123.png"
    rendered = compact_sample_name(source, max_length=28)
    assert len(rendered) == 28
    assert "…" in rendered
    assert rendered.endswith("0123.png")


def test_review_label_is_always_user_facing() -> None:
    assert corrected_label_text("normal") == "正常"
    assert corrected_label_text("defect") == "缺陷"
    assert corrected_label_text("unknown") == "待确定"
    assert corrected_label_text(None) == "待确定"
