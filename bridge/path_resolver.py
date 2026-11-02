from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_loader import LOG_DIR, PROJECT_ROOT, load_software_config


REQUIRED_ALGORITHM_DIRS = ["core", "methods", "datasets", "scripts", "configs", "outputs"]
EXPECTED_SCRIPTS = [
    "scripts/build_manifest.py",
    "scripts/build_memory.py",
    "scripts/calibrate.py",
    "scripts/infer_image.py",
    "scripts/infer_batch.py",
    "scripts/infer_video.py",
    "scripts/benchmark_speed.py",
]
PATH_STATUS_PATH = LOG_DIR / "path_status.json"


class AlgorithmPathError(FileNotFoundError):
    """Raised when neither configured algorithm project path exists."""


def _path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "auto":
        return PROJECT_ROOT.resolve()
    return Path(text).expanduser().resolve()


def resolve_algorithm_project(config: dict[str, Any] | None = None) -> Path:
    """Prefer the primary algorithm root, then the configured fallback root."""
    config = config or load_software_config()
    primary = _path(config.get("algorithm_project_root"))
    fallback = _path(config.get("fallback_algorithm_project_root"))

    if primary and primary.exists():
        return primary
    if fallback and fallback.exists():
        return fallback

    raise AlgorithmPathError(
        "未找到当前算法项目目录。请检查 software_config.yaml 中的 "
        f"algorithm_project_root；已检查：{primary or '未配置'}；备用目录：{fallback or '未配置'}。"
    )


def collect_path_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_software_config()
    status: dict[str, Any] = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "primary_algorithm_project_root": str(_path(config.get("algorithm_project_root")) or ""),
        "fallback_algorithm_project_root": str(_path(config.get("fallback_algorithm_project_root")) or ""),
        "selected_algorithm_project_root": None,
        "algorithm_project_exists": False,
        "required_dirs": {},
        "scripts": {},
        "found_scripts": [],
        "missing_scripts": [],
        "errors": [],
    }

    try:
        algorithm_root = resolve_algorithm_project(config)
        status["selected_algorithm_project_root"] = str(algorithm_root)
        status["algorithm_project_exists"] = True
    except AlgorithmPathError as exc:
        status["errors"].append(str(exc))
        return status

    for dirname in REQUIRED_ALGORITHM_DIRS:
        path = algorithm_root / dirname
        status["required_dirs"][dirname] = {"path": str(path), "exists": path.exists()}

    for rel_script in EXPECTED_SCRIPTS:
        path = algorithm_root / rel_script
        item = {"path": str(path), "exists": path.exists()}
        status["scripts"][rel_script] = item
        if item["exists"]:
            status["found_scripts"].append(rel_script)
        else:
            status["missing_scripts"].append(rel_script)

    return status


def write_path_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    status = collect_path_status(config)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PATH_STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


if __name__ == "__main__":
    payload = write_path_status()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
