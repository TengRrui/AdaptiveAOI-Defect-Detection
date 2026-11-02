from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from core.domain_build_contract import DEFAULT_FEATURE_LAYERS, default_feature_layers
from core.feature_extractor import PatchFeatureExtractor


def test_canonical_default_is_layer2_only_and_returns_fresh_lists() -> None:
    first = default_feature_layers()
    second = default_feature_layers()

    assert DEFAULT_FEATURE_LAYERS == ("layer2",)
    assert first == second == ["layer2"]
    assert first is not second


def test_feature_extractor_and_formal_profiles_share_layer2_default() -> None:
    root = Path(__file__).resolve().parents[1]
    signature = inspect.signature(PatchFeatureExtractor.__init__)
    assert signature.parameters["layers"].default == ("layer2",)

    for name in ("default.yaml", "final_balanced.yaml"):
        profile = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
        assert profile["layers"] == ["layer2"]


def test_target_domain_ui_initializes_layer2_only() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "app" / "domain_resource_creation_page.py").read_text(encoding="utf-8")

    assert '"domain_rc_layers": default_feature_layers()' in source
    assert '"仅 layer2"' in source
