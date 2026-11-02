from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from backend.atomic_io import AtomicJsonStore
from backend.errors import ApiError
from backend.path_validation import validate_dataset_category
from tools.path_utils import deep_merge, load_config, load_yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _load_dotenv(path: Path) -> None:
    """Read a small local .env file without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _interpolate(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _interpolate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default or "")

    return ENV_PATTERN.sub(replace, value)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class IntegrationSettings:
    """Central path, profile, runtime and dynamic-setting resolver."""

    def __init__(self, config_path: Path, raw: dict[str, Any]):
        self.config_path = config_path.resolve()
        self.project_root = self.config_path.parents[1]
        self.raw = raw
        paths = raw.get("paths", {})
        configured_root = str(paths.get("project_root", "auto")).strip()
        if configured_root and configured_root.lower() != "auto":
            self.project_root = self.resolve_path(configured_root)
        self.project_root = self.project_root.resolve()

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "IntegrationSettings":
        default_path = Path(__file__).resolve().parents[1] / "configs" / "system_integration.yaml"
        path = Path(config_path or os.environ.get("ADAPTIVEAOI_SYSTEM_CONFIG") or default_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Integration configuration not found: {path}")
        _load_dotenv(path.parents[1] / ".env")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Integration configuration must be a mapping: {path}")
        return cls(path, _interpolate(raw))

    def resolve_path(self, value: str | Path | None, default: str | Path | None = None) -> Path:
        raw = value if value not in (None, "") else default
        if raw in (None, ""):
            return self.project_root
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @property
    def paths(self) -> dict[str, Any]:
        return dict(self.raw.get("paths", {}))

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self.raw.get("defaults", {}))

    @property
    def backend(self) -> dict[str, Any]:
        return dict(self.raw.get("backend", {}))

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.raw.get("runtime", {}))

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return {str(key): dict(value or {}) for key, value in dict(self.raw.get("models", {})).items()}

    @property
    def runtime_root(self) -> Path:
        return self.resolve_path(self.paths.get("runtime_root"), "outputs/system_runtime")

    @property
    def jobs_root(self) -> Path:
        return self.resolve_path(self.paths.get("jobs_root"), "outputs/system_runtime/jobs")

    @property
    def upload_root(self) -> Path:
        return self.resolve_path(self.paths.get("upload_root"), "outputs/system_runtime/uploads")

    @property
    def log_root(self) -> Path:
        return self.resolve_path(self.paths.get("log_root"), "outputs/system_runtime/logs")

    @property
    def experiment_root(self) -> Path:
        return self.resolve_path(self.paths.get("experiment_root"), "outputs/results")

    @property
    def algorithm_outputs_root(self) -> Path:
        return self.resolve_path(self.paths.get("algorithm_outputs_root"), "outputs")

    @property
    def state_path(self) -> Path:
        return self.runtime_root / "runtime_settings.json"

    @property
    def active_calibrations_path(self) -> Path:
        return self.runtime_root / "active_calibrations.json"

    @property
    def memory_banks_root(self) -> Path:
        """Immutable target-domain profiles created by the local adaptation UI."""
        return self.resolve_path(self.paths.get("memory_banks_root"), "artifacts/memory_banks")

    @property
    def config_profiles_root(self) -> Path:
        """Versioned user-facing configuration schemes; never accepts client paths."""
        return self.resolve_path(self.paths.get("config_profiles_root"), "configs/profiles")

    @property
    def memory_bank_registry_path(self) -> Path:
        return self.runtime_root / "memory_bank_registry.json"

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_root, self.jobs_root, self.upload_root, self.log_root, self.memory_banks_root, self.config_profiles_root):
            path.mkdir(parents=True, exist_ok=True)

    def model_definition(self, model_id: str | None = None) -> tuple[str, dict[str, Any]]:
        resolved_id = str(model_id or self.dynamic_settings().get("model_id") or self.defaults.get("model_id", "")).strip()
        model = self.models.get(resolved_id)
        if model is None:
            raise KeyError(f"Unknown model_id: {resolved_id}")
        return resolved_id, model

    def profile_path(self, model_id: str | None = None) -> Path:
        _, model = self.model_definition(model_id)
        profile = model.get("profile")
        if not profile:
            raise ValueError("Model profile is missing from integration configuration.")
        path = self.resolve_path(profile)
        if not path.exists():
            raise FileNotFoundError(f"Algorithm profile not found: {path}")
        return path

    def algorithm_profile(self, model_id: str | None = None) -> dict[str, Any]:
        return deepcopy(load_config(self.profile_path(model_id)))

    def algorithm_profile_unbound(self, model_id: str | None = None) -> dict[str, Any]:
        """Load a profile for target-domain folders without requiring a dataset root."""
        path = self.profile_path(model_id)
        profile = load_yaml(path)
        default_path = path.parent / "default.yaml"
        if path.name != "default.yaml" and default_path.exists():
            profile = deep_merge(load_yaml(default_path), profile)
        profile = deepcopy(profile)
        profile["project_root"] = str(self.project_root)
        output_root = Path(str(profile.get("output_root", "outputs")))
        if not output_root.is_absolute():
            output_root = self.project_root / output_root
        profile["output_root"] = str(output_root.resolve())
        configured_data_root = self.paths.get("data_root")
        profile["data_root"] = str(self.resolve_path(configured_data_root)) if configured_data_root not in (None, "", "auto") else ""
        profile["num_workers"] = int(profile.get("num_workers", 0))
        return profile

    def data_root_for(self, profile: dict[str, Any] | None = None) -> Path:
        configured = self.paths.get("data_root")
        if configured not in (None, ""):
            return self.resolve_path(configured)
        if profile and profile.get("data_root"):
            return Path(str(profile["data_root"])).expanduser().resolve()
        raise ValueError("Data root is not configured. Set ADAPTIVEAOI_DATA_ROOT or configure paths.data_root.")

    def dynamic_settings(self) -> dict[str, Any]:
        return AtomicJsonStore(self.state_path).load(recover=True)

    def update_dynamic_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "model_id", "dataset", "category", "config", "device", "threshold",
            "memory_bank_id", "inference_mode", "save_visuals", "save_original",
            "config_profile_id", "config_profile_version", "config_hash",
            # These flags are controlled by the local operator/runtime and are
            # never accepted as arbitrary file paths.  ``official_test_mode``
            # is the hard isolation switch for the feedback optimizer.
            "official_test_mode", "feedback_optimization_locked",
            "test_data_update_allowed", "evaluation_mode",
            # Durable lifecycle state owned by RuntimeProfileService.  It
            # prevents startup recovery from silently re-applying a profile
            # after the operator explicitly stops detection.
            "runtime_deactivated", "deactivated_profile_ids", "deactivated_at",
            # One durable revisioned lifecycle snapshot backs health, status,
            # configuration and detection readiness.  These fields never
            # contain file paths or client-controlled resource identities.
            "runtime_status", "runtime_revision", "runtime_updated_at",
        }
        current = self.dynamic_settings()
        nullable_clear = {"threshold", "deactivated_at"}
        current.update(
            {
                key: value
                for key, value in updates.items()
                if key in allowed and (value is not None or key in nullable_clear)
            }
        )
        self.ensure_runtime_dirs()
        return AtomicJsonStore(self.state_path).write(current)

    def active_calibrations(self) -> dict[str, str]:
        if not self.active_calibrations_path.exists():
            return {}
        try:
            data = json.loads(self.active_calibrations_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def calibration_key(self, dataset: str, category: str) -> str:
        dataset, category = validate_dataset_category(dataset, category)
        return f"{dataset}/{category}"

    def active_calibration_path(self, dataset: str, category: str) -> Path:
        dataset, category = validate_dataset_category(dataset, category)
        active = self.active_calibrations().get(self.calibration_key(dataset, category))
        if active:
            path = Path(active).expanduser().resolve()
            if path.exists():
                return path
        root = (self.algorithm_outputs_root / "calibrations").resolve()
        path = (root / dataset / category / "calibration.json").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ApiError("CALIBRATION_PATH_UNSAFE", "校准资源路径超出受控目录。", 422) from exc
        return path

    def activate_calibration(self, dataset: str, category: str, calibration_path: str | Path) -> Path:
        key = self.calibration_key(dataset, category)
        path = Path(calibration_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        try:
            path.relative_to(self.jobs_root)
        except ValueError as exc:
            raise ValueError("Only immutable calibration files generated in integration jobs can be activated.") from exc
        payload = self.active_calibrations()
        payload[key] = str(path)
        self.ensure_runtime_dirs()
        self.active_calibrations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def host(self) -> str:
        return str(self.backend.get("host", "127.0.0.1"))

    def port(self) -> int:
        return int(self.backend.get("port", 8000))

    def debug(self) -> bool:
        return _to_bool(self.backend.get("debug"), False)

    def supported_suffixes(self) -> set[str]:
        return {str(item).lower() for item in self.defaults.get("supported_image_formats", [])}

    def max_upload_bytes(self) -> int:
        return int(self.defaults.get("max_upload_mb", 25)) * 1024 * 1024

    def max_batch_files(self) -> int:
        return int(self.defaults.get("max_batch_files", 200))
