from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import subprocess

import pytest

from backend.adapters.patchcore_adapter import PatchCoreAdapter
from backend.config import IntegrationSettings
from backend.errors import ApiError
from backend.services.domain_profile_service import DomainProfileService


INVALID_SELECTIONS = [
    ("../external", "part"),
    (r"..\external", "part"),
    ("qa", "../external"),
    ("qa", r"..\external"),
    ("qa", "/external"),
    ("qa", r"E:\external"),
    ("qa", r"E:external"),
    ("qa", r"\\server\share"),
    ("qa", "part:stream"),
    ("qa", "."),
    ("qa", ".."),
    ("qa", ""),
    ("qa", "part\x00name"),
]


def make_settings(tmp_path: Path) -> IntegrationSettings:
    return IntegrationSettings(tmp_path / "configs" / "system_integration.yaml", {})


@pytest.mark.parametrize("dataset,category", INVALID_SELECTIONS)
def test_invalid_selection_is_rejected_before_model_or_dataset_configuration(
    tmp_path: Path, dataset: str, category: str,
) -> None:
    adapter = PatchCoreAdapter(make_settings(tmp_path), None, None, None, None, None, None)

    with pytest.raises(ApiError) as error:
        adapter._resolve(None, dataset, category, None, None, False)

    assert error.value.code == "INVALID_MODEL_SELECTION"
    assert error.value.status_code == 422


@pytest.mark.parametrize("dataset,category", INVALID_SELECTIONS)
def test_builtin_memory_lookup_rejects_path_components(
    tmp_path: Path, dataset: str, category: str,
) -> None:
    service = object.__new__(DomainProfileService)
    service.settings = make_settings(tmp_path)

    with pytest.raises(ApiError) as error:
        service._builtin_profile("model", dataset, category)

    assert error.value.code == "INVALID_MODEL_SELECTION"


@pytest.mark.parametrize("dataset,category", INVALID_SELECTIONS)
def test_default_calibration_lookup_rejects_path_components(
    tmp_path: Path, dataset: str, category: str,
) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(ApiError) as error:
        settings.active_calibration_path(dataset, category)

    assert error.value.code == "INVALID_MODEL_SELECTION"


def test_unicode_selection_keeps_builtin_memory_and_owned_calibration_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    memory = tmp_path / "outputs" / "checkpoints" / "custom" / "瓷瓶 01" / "memory_bank.pt"
    memory.parent.mkdir(parents=True)
    memory.write_bytes(b"local-memory-fixture")
    calibration = tmp_path / "outputs" / "system_runtime" / "jobs" / "calibration_1" / "calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    settings.activate_calibration("custom", "瓷瓶 01", calibration)
    service = object.__new__(DomainProfileService)
    service.settings = settings
    service.config_profiles = SimpleNamespace(builtin_binding=lambda *_args, **_kwargs: {})

    profile = service._builtin_profile("model", "custom", "瓷瓶 01")

    assert profile is not None
    assert profile["dataset"] == "custom"
    assert profile["category"] == "瓷瓶 01"
    assert settings.active_calibration_path("custom", "瓷瓶 01") == calibration


def test_folder_selection_keeps_valid_absolute_paths_outside_project(tmp_path: Path) -> None:
    selected = tmp_path / "operator_samples" / "正常件"
    selected.mkdir(parents=True)
    service = object.__new__(DomainProfileService)
    service.settings = make_settings(tmp_path / "project")
    service.root = service.settings.memory_banks_root

    assert service._source_directory(selected, "正常") == selected.resolve()


def make_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            pytest.skip("Directory junctions are unavailable in this test environment")
    else:
        link.symlink_to(target, target_is_directory=True)


def test_builtin_memory_lookup_rejects_directory_link_outside_checkpoint_root(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "project")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "memory_bank.pt").write_bytes(b"external-memory-fixture")
    dataset_dir = settings.algorithm_outputs_root / "checkpoints" / "qa"
    dataset_dir.mkdir(parents=True)
    make_directory_link(dataset_dir / "part", outside)
    service = object.__new__(DomainProfileService)
    service.settings = settings
    service.config_profiles = SimpleNamespace(builtin_binding=lambda *_args, **_kwargs: {})

    with pytest.raises(ApiError) as error:
        service._builtin_profile("model", "qa", "part")

    assert error.value.code == "MEMORY_BANK_PATH_UNSAFE"


def test_default_calibration_lookup_rejects_directory_link_outside_calibration_root(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "project")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "calibration.json").write_text("{}", encoding="utf-8")
    dataset_dir = settings.algorithm_outputs_root / "calibrations" / "qa"
    dataset_dir.mkdir(parents=True)
    make_directory_link(dataset_dir / "part", outside)

    with pytest.raises(ApiError) as error:
        settings.active_calibration_path("qa", "part")

    assert error.value.code == "CALIBRATION_PATH_UNSAFE"
