from __future__ import annotations

import pytest

from backend.services.feedback_optimization_service import (
    FeedbackOptimizationService,
    _metric_or_default,
    _parse_metric_float,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0.0),
        (0.0, 0.0),
        ("0", 0.0),
        ("0.000", 0.0),
        (None, None),
        ("", None),
        ("not-a-number", None),
    ],
)
def test_metric_parser_distinguishes_zero_from_missing(value, expected) -> None:
    assert _parse_metric_float(value) == expected


def test_metric_or_default_keeps_zero_and_only_defaults_missing() -> None:
    assert _metric_or_default(0.0, 1.0) == 0.0
    assert _metric_or_default("0", 1.0) == 0.0
    assert _metric_or_default(None, 1.0) == 1.0
    assert _metric_or_default("", 1.0) == 1.0


def _evidence(normal_count: int, false_positives: int, defect_count: int = 20) -> tuple[list[float], list[int]]:
    normal_scores = [0.9] * false_positives + [0.1] * (normal_count - false_positives)
    defect_scores = [0.9] * defect_count
    return normal_scores + defect_scores, [0] * normal_count + [1] * defect_count


@pytest.mark.parametrize(
    ("normal_count", "false_positives", "expected"),
    [(20, 0, 0.0), (20, 1, 0.05), (20, 2, 0.10), (100, 11, 0.11)],
)
def test_fpr_metrics_preserve_zero_and_boundary_values(normal_count, false_positives, expected) -> None:
    scores, labels = _evidence(normal_count, false_positives)
    metrics = FeedbackOptimizationService._metrics(scores, labels, 0.5)
    assert metrics["false_positive_rate"] == expected
    assert metrics["false_positive_rate_status"] == "ok"


@pytest.mark.parametrize(
    ("actual", "limit", "passes"),
    [(0.0, 0.10, True), (0.05, 0.10, True), (0.10, 0.10, True), (0.11, 0.10, False), (1.0, 0.10, False)],
)
def test_fpr_gate_boundary_semantics(actual, limit, passes) -> None:
    parsed = _parse_metric_float(actual)
    assert parsed is not None
    assert (parsed <= limit) is passes


def test_bad_fpr_case_is_rejected_with_actual_value() -> None:
    # Four true-negative normal samples and one false-positive normal sample
    # produce the deliberately failing FPR=0.200 case.
    scores, labels = _evidence(5, 1)
    metrics = FeedbackOptimizationService._metrics(scores, labels, 0.5)
    actual = _parse_metric_float(metrics["false_positive_rate"])
    assert actual == 0.2
    assert actual > 0.10
    assert f"误报率 {actual:.3f} > 上限 {0.10:.3f}" == "误报率 0.200 > 上限 0.100"


def test_recall_zero_is_preserved_as_a_valid_metric() -> None:
    scores, labels = [0.1, 0.1], [1, 1]
    metrics = FeedbackOptimizationService._metrics(scores, labels, 0.5)
    assert metrics["recall"] == 0.0
    assert metrics["false_positive_rate"] is None
    assert metrics["false_positive_rate_status"] == "not_evaluable"


def test_missing_normal_evidence_never_becomes_fpr_zero() -> None:
    scores, labels = [0.9, 0.8], [1, 1]
    metrics = FeedbackOptimizationService._metrics(scores, labels, 0.5)
    assert metrics["false_positive_rate"] is None
    assert metrics["false_positive_rate_status"] == "not_evaluable"
    threshold, selected, options = FeedbackOptimizationService._choose_threshold(
        [{"original_score": score, "corrected_label": "defect"} for score in scores],
        0.5,
        "f1",
        min_recall=0.95,
        max_fpr=0.10,
    )
    assert threshold == 0.5
    assert selected["eligible"] is False
    assert options == []


def test_zero_growth_values_are_not_replaced_by_defaults() -> None:
    assert _metric_or_default(0.0, 0.05) == 0.0
    assert _metric_or_default("0.0", 0.10) == 0.0
