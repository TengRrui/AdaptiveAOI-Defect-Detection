from backend.services.strategy_eligibility_policy import evaluate_strategy_eligibility, get_strategy_policy


def test_threshold_minimum_is_not_the_same_as_recommended_stability_set() -> None:
    result = evaluate_strategy_eligibility(
        "threshold_only",
        "normal_defect_calibration",
        {"normal": 1, "defect": 1, "safe_normal": 0, "reliable_mask_defect": 0},
    )
    assert result["can_start"] is True
    assert result["candidate_validation_ready"] is False
    assert result["policy"]["recommended_requirements"]["normal_samples"] == 5
    assert any(item["code"] == "CANDIDATE_VALIDATION_EVIDENCE_INSUFFICIENT" for item in result["warnings"])


def test_normal_memory_requires_safe_normal_not_defect_count() -> None:
    eligible = evaluate_strategy_eligibility(
        "normal_memory",
        "normal_defect_calibration",
        {"normal": 1, "defect": 0, "safe_normal": 1, "reliable_mask_defect": 0},
    )
    assert eligible["can_start"] is True
    assert eligible["policy"]["hard_requirements"]["defect_samples"] == 0

    blocked = evaluate_strategy_eligibility(
        "normal_memory",
        "normal_defect_calibration",
        {"normal": 1, "defect": 0, "safe_normal": 0, "reliable_mask_defect": 0},
    )
    assert blocked["can_start"] is False
    assert any(item["code"] == "STRATEGY_SAFE_NORMAL_INSUFFICIENT" for item in blocked["blockers"])


def test_defect_prototype_requires_masked_defect_not_normal_count() -> None:
    result = evaluate_strategy_eligibility(
        "defect_prototypes",
        "normal_defect_dual_memory",
        {"normal": 0, "defect": 1, "safe_normal": 0, "reliable_mask_defect": 1},
    )
    assert result["can_start"] is True
    assert result["policy"]["hard_requirements"]["normal_samples"] == 0


def test_lightweight_calibrator_uses_two_class_two_fold_minimum() -> None:
    eligible = evaluate_strategy_eligibility(
        "lightweight_calibrator",
        "normal_memory_defect_guided_lite",
        {"normal": 2, "defect": 2, "safe_normal": 0, "reliable_mask_defect": 0},
    )
    assert eligible["can_start"] is True
    assert eligible["candidate_validation_ready"] is False

    blocked = evaluate_strategy_eligibility(
        "lightweight_calibrator",
        "normal_memory_defect_guided_lite",
        {"normal": 1, "defect": 2, "safe_normal": 0, "reliable_mask_defect": 0},
    )
    assert blocked["can_start"] is False
    assert any(item["code"] == "STRATEGY_NORMAL_SAMPLES_INSUFFICIENT" for item in blocked["blockers"])


def test_comprehensive_merges_only_resources_supported_by_base_mode() -> None:
    dual = get_strategy_policy("comprehensive", "normal_defect_dual_memory")
    assert "defect_memory" in dual.updates
    assert dual.require_defect_mask is True

    lite = get_strategy_policy("comprehensive", "normal_memory_defect_guided_lite")
    assert "lightweight_calibrator" in lite.updates
    assert lite.require_defect_mask is False
