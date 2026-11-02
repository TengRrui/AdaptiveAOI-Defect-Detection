from __future__ import annotations

import json

import numpy as np

from methods.lightweight_calibration import (
    apply_lightweight_profile,
    fit_lightweight_profile,
    load_lightweight_profile,
    write_lightweight_profile,
)


def test_lightweight_calibration_is_frozen_monotonic_and_feature_free(tmp_path):
    generator = np.random.default_rng(20260805)
    normal_rows = [generator.normal(0.25, 0.03, 256).astype(np.float32) for _ in range(12)]
    defect_rows = [generator.normal(0.72, 0.05, 256).astype(np.float32) for _ in range(8)]
    normal_scores = [float(row.max()) for row in normal_rows]
    defect_scores = [float(row.max()) for row in defect_rows]

    profile = fit_lightweight_profile(
        normal_rows,
        defect_rows,
        normal_scores,
        defect_scores,
        threshold=0.60,
        config={"normal_bank_id": "normal-bank-fixture", "lightweight_cv_folds": 4},
        seed=20260805,
    )

    assert profile["frozen"] is True
    assert profile["test_time_update_allowed"] is False
    assert profile["defect_features_saved"] is False
    assert profile["additional_ann_count"] == 0
    assert all(weight >= 0.0 for weight in profile["calibrator"]["weights"])

    applied = apply_lightweight_profile(defect_rows[0], defect_scores[0], profile)
    assert applied["final_score"] >= applied["baseline_score"]
    assert applied["gate_status"] in {"calibration_applied", "below_normal_tail_fallback"}

    target = write_lightweight_profile(tmp_path / "lightweight_calibrator.json", profile)
    payload = json.loads(target.read_text(encoding="utf-8"))
    metrics = payload.get("calibration_metrics") or {}
    assert "pooled_features" not in metrics
    assert "signals" not in metrics
    assert "final_scores" not in metrics
    restored = load_lightweight_profile(target)
    assert restored is not None and restored["frozen"] is True
