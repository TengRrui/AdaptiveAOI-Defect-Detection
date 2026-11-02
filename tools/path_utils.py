from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_environment(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default or "")

    return ENV_PATTERN.sub(replace, value)


def _central_data_root(config_path: Path) -> str:
    """Read the single external-data setting without coupling profiles to a machine path."""
    path_config = config_path.parent / "path_config.yaml"
    if not path_config.exists():
        return ""
    payload = load_yaml(path_config)
    return str(_expand_environment(payload.get("data_root", ""))).strip()


def project_root_from_file(file_path: str | Path) -> Path:
    path = Path(file_path).resolve()
    if path.name.lower() == "configs":
        return path.parent
    for parent in [path.parent, *path.parents]:
        if (parent / "configs").exists() and (parent / "scripts").exists():
            return parent
    return Path.cwd().resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = load_yaml(path)
    default_path = path.parent / "default.yaml"
    if path.name != "default.yaml" and default_path.exists():
        cfg = deep_merge(load_yaml(default_path), cfg)

    root = project_root_from_file(path)
    if str(cfg.get("project_root", "")).lower() in {"", "auto"}:
        cfg["project_root"] = str(root)
    else:
        cfg["project_root"] = str(Path(cfg["project_root"]).resolve())

    output_root = Path(str(cfg.get("output_root", "outputs")))
    if not output_root.is_absolute():
        output_root = Path(cfg["project_root"]) / output_root
    cfg["output_root"] = str(output_root)
    data_root = str(_expand_environment(cfg.get("data_root", ""))).strip()
    if not data_root or data_root.lower() == "auto":
        data_root = _central_data_root(path)
    if not data_root or data_root.lower() == "auto":
        raise ValueError(
            "未配置原始数据根目录。请在 configs/path_config.yaml 中设置 data_root，"
            "或设置本进程环境变量 ADAPTIVEAOI_DATA_ROOT。"
        )
    cfg["data_root"] = str(Path(data_root).expanduser().resolve())
    cfg["num_workers"] = int(cfg.get("num_workers", 0))
    return cfg


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def output_path(config: dict[str, Any], *parts: str) -> Path:
    return ensure_dir(Path(config["output_root"]) / Path(*parts))


def memory_dir(config: dict[str, Any], dataset: str | None = None, category: str | None = None) -> Path:
    dataset = dataset or str(config["dataset"])
    category = category or str(config["category"])
    return ensure_dir(Path(config["output_root"]) / "checkpoints" / dataset / category)


def append_error_report(config_or_root: dict[str, Any] | str | Path, title: str, message: str) -> Path:
    if isinstance(config_or_root, dict):
        output_root = Path(config_or_root["output_root"])
    else:
        root = Path(config_or_root)
        output_root = root if root.name == "outputs" else root / "outputs"
    logs_dir = ensure_dir(output_root / "logs")
    report_path = logs_dir / "error_report.md"
    with report_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} - {title}\n\n")
        f.write(str(message).strip() + "\n")
    return report_path


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
