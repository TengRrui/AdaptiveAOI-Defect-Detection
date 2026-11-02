from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.api.schemas import DomainBuildRequest, DomainScanRequest
from backend.errors import ApiError
from backend.services.domain_profile_service import (
    CREATABLE_DOMAIN_METHOD_MODES,
    DomainProfileService,
)
from core.domain_build_contract import DEFAULT_CREATE_MODE, LIGHTWEIGHT_CREATE_MODE


def _draft_service(tmp_path: Path) -> DomainProfileService:
    service = DomainProfileService.__new__(DomainProfileService)
    service.draft_root = tmp_path / "drafts"
    service.draft_root.mkdir()
    service._draft_revision_path = service.draft_root / ".revision"
    service._draft_revision = 0
    service._lock = threading.RLock()
    service.config_profiles = SimpleNamespace(get=lambda *_args: {})
    return service


def test_public_build_schema_defaults_lightweight_fp16_and_service_owns_creation_mode_policy() -> None:
    assert CREATABLE_DOMAIN_METHOD_MODES == {
        "normal_only",
        "normal_memory_defect_guided_lite",
    }
    scan = DomainScanRequest()
    build = DomainBuildRequest(dataset="qa", category="part")
    assert DEFAULT_CREATE_MODE == LIGHTWEIGHT_CREATE_MODE == "normal_memory_defect_guided_lite"
    assert scan.method_mode == DEFAULT_CREATE_MODE
    assert build.method_mode == DEFAULT_CREATE_MODE
    assert build.precision == "fp16"
    assert build.use_fp16 is True
    assert build.layers == ["layer2"]

    for retired in ("normal_defect_calibration", "normal_defect_dual_memory", "auto"):
        # Schema parsing remains tolerant so the route can return a stable
        # business error instead of FastAPI's generic enum failure.
        request = DomainBuildRequest(dataset="qa", category="part", method_mode=retired)
        assert request.method_mode == retired


def test_precision_contract_supports_legacy_bool_and_rejects_conflicts() -> None:
    legacy_fp32 = DomainBuildRequest(dataset="qa", category="part", use_fp16=False)
    explicit_fp32 = DomainBuildRequest(dataset="qa", category="part", precision="fp32")
    assert legacy_fp32.precision == "fp32" and legacy_fp32.use_fp16 is False
    assert explicit_fp32.precision == "fp32" and explicit_fp32.use_fp16 is False
    with pytest.raises(ValidationError, match="precision.*use_fp16"):
        DomainBuildRequest(dataset="qa", category="part", precision="fp16", use_fp16=False)


def test_new_draft_defaults_to_lightweight_fp16_and_rejects_retired_mode(tmp_path: Path) -> None:
    service = _draft_service(tmp_path)
    saved = service.save_draft({
        "draft_id": "domain-draft-qa123456",
        "payload": {"friendly_name": "QA"},
    })
    assert saved["payload"]["method_mode"] == DEFAULT_CREATE_MODE
    assert saved["payload"]["precision"] == "fp16"
    assert saved["payload"]["use_fp16"] is True
    assert saved["payload"]["layers"] == ["layer2"]

    for index, retired in enumerate(("normal_defect_calibration", "normal_defect_dual_memory", "dual_memory", "auto")):
        with pytest.raises(ApiError) as raised:
            service.save_draft({
                "draft_id": f"domain-draft-old{index}654321",
                "payload": {"friendly_name": "旧方式", "method_mode": retired},
            })
        assert raised.value.code == "DOMAIN_BUILD_MODE_RETIRED"
        assert "已停止新建" in raised.value.message


def test_historical_draft_remains_readable_without_silent_mode_rewrite(tmp_path: Path) -> None:
    service = _draft_service(tmp_path)
    draft_id = "domain-draft-old12345"
    (service.draft_root / f"{draft_id}.json").write_text(
        json.dumps({
            "draft_id": draft_id,
            "current_step": 1,
            "payload": {"friendly_name": "历史草稿", "method_mode": "normal_defect_dual_memory"},
            "created_at": "2026-08-01T00:00:00+00:00",
            "updated_at": "2026-08-01T00:00:00+00:00",
        }),
        encoding="utf-8",
    )

    loaded = service.get_draft(draft_id)

    assert loaded["payload"]["method_mode"] == "normal_defect_dual_memory"


def test_explicit_historical_feature_layers_are_not_rewritten(tmp_path: Path) -> None:
    service = _draft_service(tmp_path)
    saved = service.save_draft({
        "draft_id": "domain-draft-layers23",
        "payload": {
            "friendly_name": "历史联合特征草稿",
            "layers": ["layer2", "layer3"],
        },
    })

    assert saved["payload"]["layers"] == ["layer2", "layer3"]


def test_build_conditions_make_defect_dynamic_and_report_real_precision_fallback(tmp_path: Path) -> None:
    service = DomainProfileService.__new__(DomainProfileService)
    service.root = tmp_path / "profiles"
    service.root.mkdir()
    service._lock = threading.RLock()
    service._tasks = {}
    service._active_job_id = None
    service.scan_domain_samples = lambda _payload: {
        "normal": {"valid_count": 1, "video_summary": {"valid_video_count": 0}},
        "defect": {"valid_count": 0, "video_summary": {"valid_video_count": 0}},
    }
    service._official_base_model = lambda **_kwargs: {
        "model_id": "official-model",
        "model_version": "1.0",
        "backbone": "wide_resnet50_2",
        "profile_sha256": "a" * 64,
        "ready": True,
        "weights": {"sha256": "b" * 64},
        "manifest": {"sha256": "c" * 64},
        "resource_status": {"reason_code": "ready", "source": "qa"},
        "device_compatibility": {"resolved": "cpu", "warnings": []},
    }
    payload = {
        "friendly_name": "QA新域",
        "dataset": "qa",
        "category": "part",
        "method_mode": "normal_only",
        "normal_target_count": 1,
        "defect_target_count": 0,
        "coreset_ratio": 0.1,
        "knn_k": 1,
        "image_size": 256,
        "layers": ["layer2", "layer3"],
        "max_memory_patches": 100,
        "use_fp16": True,
    }

    normal_only = service.build_conditions(payload)
    by_id = {item["id"]: item for item in normal_only["items"]}
    assert by_id["defect_samples"]["status"] == "not_applicable"
    assert by_id["precision"]["status"] == "warning"
    assert "将使用单精度模式" in by_id["precision"]["summary"]
    assert normal_only["can_create"] is True

    lightweight = service.build_conditions({
        **payload,
        "friendly_name": "QA轻量域",
        "method_mode": "normal_memory_defect_guided_lite",
        "threshold_method": "best_f1",
    })
    lightweight_by_id = {item["id"]: item for item in lightweight["items"]}
    assert lightweight_by_id["defect_samples"]["status"] == "failed"
    assert "已审核" not in lightweight_by_id["defect_samples"]["summary"]
    assert "已审核" not in lightweight_by_id["defect_samples"]["repair"]
    assert lightweight["can_create"] is False

    service.scan_domain_samples = lambda _payload: {
        "normal": {"valid_count": 0, "video_summary": {"valid_video_count": 0}},
        "defect": {"valid_count": 0, "video_summary": {"valid_video_count": 0}},
    }
    no_sources = service.build_conditions(payload)
    no_sources_by_id = {item["id"]: item for item in no_sources["items"]}
    assert "已审核" not in no_sources_by_id["normal_samples"]["summary"]
    assert "已审核" not in no_sources_by_id["normal_samples"]["repair"]


def test_target_domain_ui_only_exposes_two_modes_and_dynamic_defect_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    creation_source = (root / "app" / "domain_resource_creation_page.py").read_text(encoding="utf-8")
    page_source = (root / "app" / "pages" / "4_目标域适配.py").read_text(encoding="utf-8")

    selector_block = creation_source.split("def _config_form() -> bool:", 1)[1].split("def _job_panel()", 1)[0]
    assert "MODE_LABELS = dict(CREATE_MODE_LABELS)" in creation_source
    assert '"domain_rc_mode": DEFAULT_CREATE_MODE' in creation_source
    assert "normal_defect_calibration" not in selector_block
    assert "normal_defect_dual_memory" not in selector_block
    assert "自动推荐模式" not in selector_block
    assert '"构建精度"' in selector_block
    assert '"半精度（FP16）" if value else "单精度（FP32）"' in selector_block
    assert "if needs_defect:" in page_source
    assert "基础正常记忆只需要正常样本" in page_source
