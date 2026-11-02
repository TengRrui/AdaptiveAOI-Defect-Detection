from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import feature_extractor
from backend.errors import ApiError
from backend.services.domain_profile_service import DomainProfileService
from backend.services.model_registry import ModelRegistry


def test_packaged_backbone_status_verifies_manifest_weight_and_sha(monkeypatch, tmp_path: Path) -> None:
    payload = b"qa-packaged-weight"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    root = tmp_path / "backbones"
    root.mkdir()
    (root / "qa.pth").write_bytes(payload)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "weights": {
                    "qa_backbone": {
                        "filename": "qa.pth",
                        "size_bytes": len(payload),
                        "sha256": digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(feature_extractor, "_backbone_root", lambda: root)
    monkeypatch.setitem(
        feature_extractor._BACKBONE_SPECS,
        "qa_backbone",
        {"filename": "qa.pth", "size": len(payload), "sha256": digest, "factory": object()},
    )
    feature_extractor._verify_packaged_weight.cache_clear()

    result = feature_extractor.backbone_resource_status("qa_backbone")

    assert result["ready"] is True
    assert result["reason_code"] == "ready"


def test_packaged_backbone_status_reports_actionable_missing_weight(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "backbones"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "weights": {
                    "qa_missing": {
                        "filename": "missing.pth",
                        "size_bytes": 3,
                        "sha256": "a" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(feature_extractor, "_backbone_root", lambda: root)
    monkeypatch.setitem(
        feature_extractor._BACKBONE_SPECS,
        "qa_missing",
        {"filename": "missing.pth", "size": 3, "sha256": "a" * 64, "factory": object()},
    )

    result = feature_extractor.backbone_resource_status("qa_missing")

    assert result["ready"] is False
    assert result["reason_code"] == "weight_missing"
    assert "未找到" in str(result["reason"])
    assert "恢复" in str(result["resolution"])


def test_model_catalog_does_not_require_a_bound_dataset_root(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("backbone: wide_resnet50_2\ndataset: qa\ncategory: part\n", encoding="utf-8")

    class Settings:
        defaults = {"model_id": "model-a"}
        models = {"model-a": {"display_name": "QA 模型", "model_type": "patchcore"}}
        algorithm_outputs_root = tmp_path / "outputs"

        def model_definition(self, model_id):
            return "model-a", dict(self.models["model-a"])

        def profile_path(self, model_id):
            return profile

        def algorithm_profile_unbound(self, model_id):
            return {"backbone": "wide_resnet50_2", "dataset": "qa", "category": "part"}

        def algorithm_profile(self, model_id):
            raise AssertionError("模型目录不得要求已绑定的数据根目录")

        def active_calibration_path(self, dataset, category):
            return tmp_path / "missing_calibration.json"

    class RuntimeRegistry:
        @staticmethod
        def has_model(*_args):
            return False

    registry = ModelRegistry(Settings(), RuntimeRegistry())  # type: ignore[arg-type]

    model = registry.model_info("model-a")

    assert model["model_id"] == "model-a"
    assert model["profile_valid"] is True
    assert model["is_default"] is True


def test_official_base_model_ignores_runtime_model_binding(monkeypatch, tmp_path: Path) -> None:
    profile = tmp_path / "official.yaml"
    profile.write_text("backbone: wide_resnet50_2\nlayers: [layer2, layer3]\n", encoding="utf-8")

    class Settings:
        defaults = {"model_id": "official-model"}
        backend = {"version": "9.1"}
        project_root = tmp_path

        @staticmethod
        def model_definition(model_id):
            assert model_id == "official-model"
            return "official-model", {"display_name": "正式特征资源", "version": "9.1"}

        @staticmethod
        def profile_path(_model_id):
            return profile

        @staticmethod
        def algorithm_profile_unbound(_model_id):
            return {"backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"]}

    class RuntimeRegistry:
        @staticmethod
        def resolve_device(_device):
            return "cpu", []

        @staticmethod
        def has_model(*_args):
            return False

    monkeypatch.setattr(
        feature_extractor,
        "backbone_resource_status",
        lambda *_args, **_kwargs: {
            "ready": True,
            "reason_code": "ready",
            "manifest_present": True,
            "weight_present": True,
            "expected_sha256": "b" * 64,
            "expected_size": 123,
            "source": "qa-controlled-resource",
        },
    )
    registry = ModelRegistry(Settings(), RuntimeRegistry())  # type: ignore[arg-type]

    resolved = registry.official_base_model(device="auto", require_ready=True)

    assert resolved["model_id"] == "official-model"
    assert resolved["model_version"] == "9.1"
    assert resolved["backbone"] == "wide_resnet50_2"
    assert resolved["weights"]["sha256"] == "b" * 64
    assert resolved["source"] == "system_official_model_registry"


def test_official_base_model_fails_with_system_resource_message(monkeypatch, tmp_path: Path) -> None:
    profile = tmp_path / "official.yaml"
    profile.write_text("backbone: wide_resnet50_2\n", encoding="utf-8")
    settings = SimpleNamespace(
        defaults={"model_id": "official-model"},
        backend={"version": "1.0"},
        project_root=tmp_path,
        model_definition=lambda model_id: (model_id, {"version": "1.0"}),
        profile_path=lambda _model_id: profile,
        algorithm_profile_unbound=lambda _model_id: {"backbone": "wide_resnet50_2"},
    )
    runtime_registry = SimpleNamespace(
        resolve_device=lambda _device: ("cpu", []),
        has_model=lambda *_args: False,
    )
    monkeypatch.setattr(
        feature_extractor,
        "backbone_resource_status",
        lambda *_args, **_kwargs: {
            "ready": False,
            "reason_code": "weight_missing",
            "reason": "受控权重缺失",
            "resolution": "恢复正式资源",
            "manifest_present": True,
            "weight_present": False,
        },
    )
    registry = ModelRegistry(settings, runtime_registry)  # type: ignore[arg-type]

    with pytest.raises(ApiError) as raised:
        registry.official_base_model(require_ready=True)

    assert raised.value.code == "OFFICIAL_BASE_MODEL_UNAVAILABLE"
    assert raised.value.message == "正式模型资源缺失或完整性校验失败。"


def test_domain_build_ignores_legacy_client_model_selection(monkeypatch) -> None:
    service = DomainProfileService.__new__(DomainProfileService)
    service.models = SimpleNamespace(
        official_base_model=lambda **_kwargs: {
            "model_id": "official-model",
            "model_version": "2.0",
            "profile_sha256": "a" * 64,
            "profile": {
                "backbone": "wide_resnet50_2",
                "layers": ["layer2", "layer3"],
                "image_size": 256,
                "device": "cpu",
            },
            "backbone": "wide_resnet50_2",
            "weights": {"sha256": "b" * 64},
            "manifest": {"sha256": "c" * 64},
            "resource_status": {"reason_code": "ready"},
        }
    )
    service.settings = SimpleNamespace(backend={"version": "2.0"})
    service.registry = SimpleNamespace(resolve_device=lambda _device: ("cpu", []))
    service.config_profiles = None
    monkeypatch.setattr(
        "backend.services.domain_profile_service._domain_adaptation_module",
        lambda: SimpleNamespace(
            SUPPORTED_DOMAIN_THRESHOLD_METHODS={"best_f1", "normal_percentile", "mean_std", "recall_priority"},
            SUPPORTED_DOMAIN_METHOD_MODES={"normal_only", "normal_defect_calibration", "normal_memory_defect_guided_lite", "normal_defect_dual_memory", "auto"},
        ),
    )

    config, model_id, *_ = service._validated_build_config(
        {
            "model_id": "legacy-user-choice",
            "backbone": "resnet50",
            "dataset": "qa",
            "category": "part",
            "device": "cpu",
            "method_mode": "normal_only",
            "threshold_method": "best_f1",
        }
    )

    assert model_id == "official-model"
    assert config["model_id"] == "official-model"
    assert config["backbone"] == "wide_resnet50_2"
    assert config["model_version"] == "2.0"
    assert config["base_model_weights_sha256"] == "b" * 64


def test_domain_draft_hides_legacy_system_model_fields() -> None:
    public = DomainProfileService._public_draft(
        {
            "draft_id": "domain-draft-qa123456",
            "payload": {
                "friendly_name": "QA",
                "model_id": "legacy-model",
                "base_model_version": "legacy-v1",
                "backbone": "resnet50",
                "coreset_ratio": 0.1,
            },
        }
    )

    assert public["payload"] == {"friendly_name": "QA", "coreset_ratio": 0.1}


def test_target_domain_ui_has_no_base_model_selector_or_patchcore_label() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "domain_resource_creation_page.py").read_text(encoding="utf-8")

    assert 'selectbox("基础模型版本"' not in source
    assert 'key="domain_rc_model"' not in source
    assert 'key="domain_rc_backbone"' not in source
    assert "PatchCore" not in source
