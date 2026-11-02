from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.api.schemas import FeedbackOptimizationCreateRequest
from bridge.api_client import (
    FEEDBACK_OPTIMIZATION_OBJECTIVE_CODES,
    FEEDBACK_OPTIMIZATION_PUBLISH_MODE_CODES,
    FEEDBACK_OPTIMIZATION_STRATEGY_CODES,
    IntegrationApiClient,
)


_CANDIDATES = ["candidate-a", "candidate-b"]


@pytest.mark.parametrize("strategy", sorted(FEEDBACK_OPTIMIZATION_STRATEGY_CODES))
@pytest.mark.parametrize("publish_mode", sorted(FEEDBACK_OPTIMIZATION_PUBLISH_MODE_CODES))
def test_all_strategy_publish_combinations_match_request_schema(
    strategy: str,
    publish_mode: str,
) -> None:
    payload = IntegrationApiClient.build_feedback_optimization_payload(
        base_profile_id="target-profile-a",
        candidate_ids=_CANDIDATES,
        strategy=strategy,
        objective="f1",
        publish_mode=publish_mode,
        replace_confirm=publish_mode == "replace_original",
        version_name="QA contract",
    )

    parsed = FeedbackOptimizationCreateRequest.model_validate(payload)
    assert parsed.strategy == strategy
    assert parsed.publish_mode == publish_mode
    assert parsed.candidate_ids == _CANDIDATES


def test_builder_normalizes_ids_but_never_translates_display_labels() -> None:
    payload = IntegrationApiClient.build_feedback_optimization_payload(
        base_profile_id=" target-profile-a ",
        candidate_ids=["candidate-a", "", "candidate-a", "candidate-b"],
        strategy="threshold_only",
        publish_mode="save_new_version",
    )

    assert payload["base_profile_id"] == "target-profile-a"
    assert payload["candidate_ids"] == ["candidate-a", "candidate-b"]
    assert FeedbackOptimizationCreateRequest.model_validate(payload)

    with pytest.raises(ValueError, match="策略代码"):
        IntegrationApiClient.build_feedback_optimization_payload(
            base_profile_id="target-profile-a",
            candidate_ids=_CANDIDATES,
            strategy="A. 仅调整判定参数",
        )


def test_request_schema_rejects_unknown_fields_instead_of_silently_dropping_them() -> None:
    payload = IntegrationApiClient.build_feedback_optimization_payload(
        base_profile_id="target-profile-a",
        candidate_ids=_CANDIDATES,
        strategy="threshold_only",
    )
    payload["source_config_id"] = "legacy-path-field"
    with pytest.raises(ValidationError):
        FeedbackOptimizationCreateRequest.model_validate(payload)


def test_contract_code_sets_are_explicit() -> None:
    assert FEEDBACK_OPTIMIZATION_OBJECTIVE_CODES == {"f1", "recall_priority", "balanced"}
    assert FEEDBACK_OPTIMIZATION_PUBLISH_MODE_CODES == {
        "replace_original", "update_current", "save_new_version"
    }
    assert "threshold_only" in FEEDBACK_OPTIMIZATION_STRATEGY_CODES
