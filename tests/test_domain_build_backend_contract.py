from __future__ import annotations

from pathlib import Path
import inspect
import sqlite3

import numpy as np
import pytest
import torch

from core.domain_build_contract import (
    CREATE_ALLOWED_MODES,
    DEFAULT_CREATE_MODE,
    LIGHTWEIGHT_CREATE_MODE,
    READ_COMPATIBLE_MODES,
    normalize_precision_request,
)
from core.memory_bank import MemoryBank
from backend.services.domain_profile_service import DomainProfileService
from backend.services.runtime_resource_service import RuntimeResourceService
from methods.patchcore import domain_adaptation


def test_creation_and_history_mode_sets_are_separate() -> None:
    assert DEFAULT_CREATE_MODE == LIGHTWEIGHT_CREATE_MODE == "normal_memory_defect_guided_lite"
    assert CREATE_ALLOWED_MODES == {
        "normal_only",
        "normal_memory_defect_guided_lite",
    }
    assert CREATE_ALLOWED_MODES < READ_COMPATIBLE_MODES
    assert {
        "normal_defect_calibration",
        "normal_defect_dual_memory",
        "dual_memory",
        "auto",
    } <= READ_COMPATIBLE_MODES
    assert domain_adaptation.SUPPORTED_DOMAIN_METHOD_MODES == CREATE_ALLOWED_MODES


@pytest.mark.parametrize(
    "retired",
    ["normal_defect_calibration", "normal_defect_dual_memory", "dual_memory", "auto"],
)
def test_memory_bank_builder_rejects_retired_creation_modes_before_loading_model(
    retired: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="已停止用于新建资源"):
        domain_adaptation.build_and_calibrate_domain(
            [],
            [],
            {"method_mode": retired},
            tmp_path / retired,
        )


def test_precision_contract_defaults_and_legacy_bool() -> None:
    assert normalize_precision_request({}) == "fp16"
    assert normalize_precision_request({"precision": "fp16"}) == "fp16"
    assert normalize_precision_request({"precision": "fp32"}) == "fp32"
    assert normalize_precision_request({"use_fp16": True}) == "fp16"
    assert normalize_precision_request({"use_fp16": False}) == "fp32"
    with pytest.raises(ValueError, match="不一致"):
        normalize_precision_request({"precision": "fp16", "use_fp16": False})


def test_direct_builder_without_mode_uses_formal_lightweight_default(tmp_path: Path) -> None:
    normal = tmp_path / "normal.png"
    normal.write_bytes(b"not-decoded-before-mode-check")
    with pytest.raises(ValueError, match="缺陷校准样本为空"):
        domain_adaptation.build_and_calibrate_domain(
            [normal],
            [],
            {},
            tmp_path / "default-mode",
        )


def test_new_profile_metadata_contract_is_explicit() -> None:
    source = inspect.getsource(DomainProfileService._run_build)
    for field in (
        '"algorithm_mode": resolved_mode',
        '"normal_memory_enabled": True',
        '"lightweight_calibration_enabled": resolved_mode == LIGHTWEIGHT_CREATE_MODE',
        '"defect_guidance_enabled": resolved_mode == LIGHTWEIGHT_CREATE_MODE',
        '"defect_samples_written_to_memory_bank": False',
    ):
        assert field in source


def test_memory_bank_and_cpu_index_intentionally_use_float32(tmp_path: Path) -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)
    bank = MemoryBank(knn_k=1, backend="faiss_cpu_flat", device="cpu").fit(source)
    assert bank.features is not None
    assert bank.features.dtype == np.float32
    path = tmp_path / "memory_bank.pt"
    bank.save(path, {"memory_bank_dtype": "float32", "index_dtype": "float32"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["features"].dtype == torch.float32
    assert payload["metadata"]["index_dtype"] == "float32"


def test_feature_extractor_uses_autocast_without_model_half() -> None:
    source = (Path(__file__).resolve().parents[1] / "core" / "feature_extractor.py").read_text(encoding="utf-8")
    assert "torch.autocast" in source
    assert ".half()" not in source
    assert '"backbone_output_dtype"' in source


def test_temporary_cuda_build_runtime_is_closed_before_worker_return() -> None:
    builder_source = inspect.getsource(domain_adaptation.build_and_calibrate_domain)
    assert "model.close_build_runtime()" in builder_source
    model_source = inspect.getsource(__import__("methods.patchcore.model", fromlist=["FastPatchCoreAOI"]).FastPatchCoreAOI.close_build_runtime)
    assert 'self.extractor.to("cpu")' in model_source
    assert ".half()" not in model_source


def test_build_publication_uses_lightweight_catalog_registration_check(tmp_path: Path) -> None:
    source = inspect.getsource(DomainProfileService._run_build)
    assert "scan(force=True, include_resources=False)" in source
    assert "profile_registered(" in source

    db_path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE runtime_resources (profile_id TEXT, resource_type TEXT, archived INTEGER)"
        )
        conn.execute(
            "INSERT INTO runtime_resources(profile_id,resource_type,archived) VALUES(?,?,?)",
            ("qa-profile", "target_profile", 0),
        )
    service = object.__new__(RuntimeResourceService)
    service._connect = lambda: sqlite3.connect(db_path)  # type: ignore[method-assign]
    assert service.profile_registered("qa-profile", resource_type="target_profile")
    assert not service.profile_registered("qa-profile", resource_type="candidate_profile")


def test_catalog_scan_keeps_full_resource_payload_as_default() -> None:
    parameters = inspect.signature(RuntimeResourceService.scan).parameters
    assert parameters["include_resources"].default is True


def test_domain_builds_use_one_persistent_execution_worker() -> None:
    constructor = inspect.getsource(DomainProfileService.__init__)
    starter = inspect.getsource(DomainProfileService.start_build)
    assert "ThreadPoolExecutor" in constructor
    assert "max_workers=1" in constructor
    assert "self._build_executor.submit(self._run_build, task)" in starter
