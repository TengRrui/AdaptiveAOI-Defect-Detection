from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for very small demo envs
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "software_config.yaml"
LOG_DIR = PROJECT_ROOT / "outputs" / "system_runtime" / "logs" / "frontend"
CURRENT_STATE_PATH = LOG_DIR / "current_state.json"

REQUIRED_FIELDS = [
    "algorithm_project_root",
    "data_root",
    "default_dataset",
    "default_category",
    "default_config",
    "output_root",
    "device",
]

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """Raised when the software config is missing required fields."""


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip().strip('"').strip("'")
    return payload


def _expand_environment(value: Any) -> Any:
    """Expand ${NAME:-default} in a local config without changing global environment."""
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        # A launcher child process can inherit an explicitly empty variable.
        # Treat that the same as an unset optional variable so ``:-auto``
        # retains its documented shell-style fallback semantics.
        current = os.environ.get(name)
        return current if current not in (None, "") else (default or "")

    return ENV_PATTERN.sub(replace, value)


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ConfigError(f"软件配置必须是键值映射：{path}")
    return {str(key): _expand_environment(value) for key, value in data.items()}


@lru_cache(maxsize=8)
def _cached_software_config(path_text: str, modified_ns: int, size: int) -> dict[str, Any]:
    del modified_ns, size
    path = Path(path_text)
    config = _read_yaml(path)
    missing = [field for field in REQUIRED_FIELDS if field not in config or config[field] in (None, "")]
    if missing:
        raise ConfigError(f"软件配置缺少必填项：{', '.join(missing)}")
    config["_project_root"] = str(PROJECT_ROOT)
    config["_software_config_path"] = str(path)
    return config


def load_software_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the software config without mutating algorithm files."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    path = path.resolve()
    if not path.exists():
        raise ConfigError(f"未找到软件配置文件：{path}")

    stat = path.stat()
    return dict(_cached_software_config(str(path), int(stat.st_mtime_ns), int(stat.st_size)))


def build_runtime_state(
    overrides: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Streamlit-friendly state payload with optional UI overrides."""
    config = load_software_config(config_path)
    overrides = overrides or {}
    state = {
        "dataset": overrides.get("dataset", config["default_dataset"]),
        "category": overrides.get("category", config["default_category"]),
        "config": overrides.get("config", config["default_config"]),
        "device": overrides.get("device", config["device"]),
        "model_id": overrides.get("model_id", ""),
        "memory_bank_id": overrides.get("memory_bank_id", ""),
        "inference_mode": overrides.get("inference_mode", "inference_only"),
        "threshold": overrides.get("threshold"),
        "frozen": bool(overrides.get("frozen", False)),
        "algorithm_project_root": config["algorithm_project_root"],
        "fallback_algorithm_project_root": config.get("fallback_algorithm_project_root", ""),
        "data_root": config["data_root"],
        "output_root": overrides.get("output_root", config["output_root"]),
        "project_root": str(PROJECT_ROOT),
        "software_config_path": config["_software_config_path"],
    }
    return state


def save_current_state(
    overrides: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the current UI/runtime state for the demo shell."""
    state = build_runtime_state(overrides=overrides, config_path=config_path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def load_current_state() -> dict[str, Any]:
    if not CURRENT_STATE_PATH.exists():
        return save_current_state()
    return json.loads(CURRENT_STATE_PATH.read_text(encoding="utf-8"))
