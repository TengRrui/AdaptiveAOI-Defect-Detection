from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from bridge.config_loader import LOG_DIR, PROJECT_ROOT, load_software_config
from bridge.api_client import client
from bridge.path_resolver import resolve_algorithm_project, write_path_status


LATEST_INFERENCE_PATH = LOG_DIR / "latest_inference.json"


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip().strip('"').strip("'")
    return payload


def read_json(path: str | Path) -> dict[str, Any] | list[Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_text_tail(path: str | Path, max_lines: int = 80) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as exc:
        return f"读取日志失败: {type(exc).__name__}: {exc}"


def algorithm_root() -> Path:
    return resolve_algorithm_project(load_software_config())


def algorithm_python_path() -> Path:
    from bridge.algorithm_bridge import resolve_algorithm_python

    return resolve_algorithm_python(load_software_config())


def cuda_status() -> dict[str, Any]:
    python_path = algorithm_python_path()
    command = [
        str(python_path),
        "-c",
        "import torch, json; print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}))",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=20)
        if completed.returncode != 0:
            return {"ok": False, "python": str(python_path), "cuda_available": None, "error": completed.stderr.strip()}
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload["ok"] = True
        payload["python"] = str(python_path)
        return payload
    except Exception as exc:
        return {"ok": False, "python": str(python_path), "cuda_available": None, "error": f"{type(exc).__name__}: {exc}"}


def algorithm_config_path(config: str | Path | None = None) -> Path:
    cfg = load_software_config()
    root = algorithm_root()
    selected = Path(config or cfg["default_config"])
    if not selected.is_absolute():
        selected = root / selected
    return selected.resolve()


def load_algorithm_config(config: str | Path | None = None) -> dict[str, Any]:
    path = algorithm_config_path(config)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
        return loaded if isinstance(loaded, dict) else {}
    return _parse_simple_yaml(text)


def algorithm_output_root(config: str | Path | None = None) -> Path:
    root = algorithm_root()
    cfg = load_algorithm_config(config)
    project_root_value = str(cfg.get("project_root", "auto"))
    project_root = root if project_root_value.lower() == "auto" else Path(project_root_value)
    if not project_root.is_absolute():
        project_root = root / project_root
    output_root = Path(str(cfg.get("output_root", "outputs")))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return output_root.resolve()


def software_path(path_value: str | Path | None, default_relative: str | None = None) -> Path:
    value = Path(path_value or default_relative or "")
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def memory_bank_path(dataset: str, category: str, config: str | Path | None = None) -> Path:
    return algorithm_output_root(config) / "checkpoints" / str(dataset).lower() / str(category) / "memory_bank.pt"


def calibration_path(dataset: str, category: str, config: str | Path | None = None) -> Path:
    return algorithm_output_root(config) / "calibrations" / str(dataset).lower() / str(category) / "calibration.json"


def batch_result_path(dataset: str, category: str, config: str | Path | None = None) -> Path:
    return algorithm_output_root(config) / "results" / str(dataset).lower() / str(category) / "result.csv"


def speed_summary_path(config: str | Path | None = None) -> Path:
    return algorithm_output_root(config) / "speed_reports" / "speed_summary.json"


def latest_speed_report(config: str | Path | None = None) -> Path | None:
    speed_dir = algorithm_output_root(config) / "speed_reports"
    if not speed_dir.exists():
        return None
    candidates = sorted(speed_dir.glob("speed*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def path_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if exists and path.is_file() else 0.0,
    }


def discover_algorithm_configs() -> list[str]:
    config_dir = algorithm_root() / "configs"
    if not config_dir.exists():
        return [load_software_config()["default_config"]]
    configs = [f"configs/{path.name}" for path in sorted(config_dir.glob("*.yaml"))]
    default = load_software_config()["default_config"]
    if default not in configs:
        configs.insert(0, default)
    return configs


def discover_categories(dataset: str, config: str | Path | None = None) -> list[str]:
    dataset = str(dataset).lower()
    roots = [
        algorithm_output_root(config) / "checkpoints" / dataset,
        algorithm_output_root(config) / "calibrations" / dataset,
        algorithm_output_root(config) / "results" / dataset,
    ]
    categories: set[str] = set()
    for root in roots:
        if root.exists():
            categories.update(path.name for path in root.iterdir() if path.is_dir())
    default = str(load_software_config()["default_category"])
    if default:
        categories.add(default)
    return sorted(categories)


def _artifact_exists(value: Any) -> bool:
    """Check a saved result reference without issuing a browser/API request."""
    text = str(value or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost"}:
            return True
        prefix = "/artifacts/"
        if not parsed.path.startswith(prefix):
            return False
        relative = Path(unquote(parsed.path[len(prefix) :].lstrip("/")))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        return (PROJECT_ROOT / "outputs" / "system_runtime" / "jobs" / relative).is_file()
    try:
        return Path(text).expanduser().is_file()
    except (OSError, ValueError):
        return False


def latest_inference() -> dict[str, Any] | None:
    payload = read_json(LATEST_INFERENCE_PATH)
    if not isinstance(payload, dict):
        return None
    result = dict(payload)
    fields = ("image_path", "overlay_path", "heatmap_path", "mask_path", "result_json")
    availability: dict[str, bool] = {}
    referenced = 0
    available = 0
    for field in fields:
        value = result.get(field)
        if value:
            referenced += 1
        exists = _artifact_exists(value)
        availability[field] = exists
        if exists:
            available += 1
        elif value:
            # Keep the immutable history/JSON untouched, but do not let
            # Streamlit request a known-missing artifact and generate a 404.
            result[field] = None
    result["artifact_availability"] = availability
    result["artifact_status"] = (
        "available" if available and available == referenced else
        "partial" if available else
        "unavailable" if referenced else
        "not_recorded"
    )
    return result


def _api_service_state(payload: dict[str, Any]) -> tuple[str, str]:
    """Normalize liveness/readiness into a small set of UI-safe states."""
    if not isinstance(payload, dict) or not payload.get("success"):
        error = payload.get("error") if isinstance(payload, dict) else {}
        code = str((error or {}).get("code") or "") if isinstance(error, dict) else ""
        if code in {"API_UNAVAILABLE", "CONNECT_TIMEOUT", "READ_TIMEOUT"}:
            return "starting", "算法服务正在启动，请稍候。"
        return "error", "算法服务启动失败，请重新启动系统或查看启动日志。"
    state = str(payload.get("status") or "ready").lower()
    if state in {"starting", "pending", "loading"}:
        return "starting", "算法服务正在启动，请稍候。"
    if state in {"error", "failed"}:
        return "error", "算法服务启动失败，请重新启动系统或查看启动日志。"
    if state == "degraded":
        return "degraded", "算法服务已连接，但部分资源仍在初始化。"
    return "ready", "算法服务已就绪。"


def save_latest_inference(payload: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_INFERENCE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return LATEST_INFERENCE_PATH


def current_state() -> dict[str, Any]:
    # Runtime resource identity is backend-owned.  This local fallback contains
    # presentation defaults only and never a remembered Profile binding.
    config = load_software_config()
    return {
        "dataset": "",
        "category": "",
        "config": str(config.get("default_config") or ""),
        "device": str(config.get("device") or "auto"),
        "memory_bank_id": "",
        "config_profile_id": "",
    }


def system_status(dataset: str | None = None, category: str | None = None, config: str | Path | None = None) -> dict[str, Any]:
    cfg = load_software_config()
    state = current_state()
    dataset = dataset or state.get("dataset") or cfg["default_dataset"]
    category = category or state.get("category") or cfg["default_category"]
    config = config or state.get("config") or cfg["default_config"]
    path_status = write_path_status(cfg)
    if not path_status.get("algorithm_project_exists"):
        return {
            "ok": False,
            "error": "算法项目路径不存在，请检查 software_config.yaml。",
            "algorithm_project_root": "",
            "device": state.get("device", cfg["device"]),
            "dataset": dataset,
            "category": category,
            "config": str(config),
            "output_root": "",
            "memory_bank": {"path": "", "exists": False, "size_bytes": 0, "size_mb": 0.0},
            "calibration": {"path": "", "exists": False, "size_bytes": 0, "size_mb": 0.0},
            "latest_inference": latest_inference(),
            "latest_inference_path": str(LATEST_INFERENCE_PATH),
            "path_status": path_status,
            "error_tail": read_text_tail(LOG_DIR / "error.log", max_lines=8),
            "call_history_path": str(LOG_DIR / "call_history.log"),
            "error_log_path": str(LOG_DIR / "error.log"),
            "speed_summary": "",
            "latest_speed_report": "",
            "algorithm_python": "",
            "cuda": {"ok": False, "cuda_available": None},
        }
    api_health = client().health()
    api_state, api_message = _api_service_state(api_health)
    api_ok = api_state in {"ready", "degraded"}
    config_response = client().config(timeout=3.0) if api_ok else {"success": False}
    applied_runtime = ((config_response.get("config") or {}).get("runtime") or {}) if config_response.get("success") else {}
    if applied_runtime:
        dataset = str(applied_runtime.get("dataset") or dataset)
        category = str(applied_runtime.get("category") or category)
        config = str(applied_runtime.get("config") or config)
        state["device"] = str(applied_runtime.get("device") or state.get("device") or cfg["device"])
    api_summary = client().summary() if api_ok else {"success": False}
    runtime_profile = applied_runtime.get("memory_bank_profile") or {}
    if applied_runtime:
        memory = {
            "path": "",
            "exists": bool(applied_runtime.get("memory_bank_ready")),
            "size_bytes": 0,
            "size_mb": 0.0,
            "memory_bank_id": applied_runtime.get("memory_bank_id"),
            "friendly_name": runtime_profile.get("friendly_name") or "当前冻结记忆库",
            "normal_sample_count": runtime_profile.get("normal_sample_count"),
        }
        calibration = {
            "path": "",
            "exists": bool(applied_runtime.get("threshold_ready")),
            "size_bytes": 0,
            "size_mb": 0.0,
            "threshold": applied_runtime.get("threshold"),
            "method": runtime_profile.get("threshold_method"),
        }
    else:
        memory = path_info(memory_bank_path(dataset, category, config))
        calibration = path_info(calibration_path(dataset, category, config))
    latest = latest_inference()
    return {
        "ok": api_ok,
        "service_state": api_state,
        "service_message": api_message,
        "algorithm_project_root": str(algorithm_root()),
        "device": state.get("device", cfg["device"]),
        "dataset": dataset,
        "category": category,
        "config": str(config),
        "output_root": str(algorithm_output_root(config)),
        "memory_bank": memory,
        "calibration": calibration,
        "latest_inference": latest,
        "latest_inference_path": str(LATEST_INFERENCE_PATH),
        "path_status": path_status,
        "error_tail": read_text_tail(LOG_DIR / "error.log", max_lines=8),
        "call_history_path": str(LOG_DIR / "call_history.log"),
        "error_log_path": str(LOG_DIR / "error.log"),
        "speed_summary": str(speed_summary_path(config)),
        "latest_speed_report": str(latest_speed_report(config) or ""),
        "algorithm_python": str(algorithm_python_path()),
        "cuda": {
            "ok": api_ok,
            "cuda_available": api_health.get("cuda_available") if api_ok else None,
            "device": api_health.get("device") if api_ok else None,
            "gpu_name": api_health.get("gpu_name") if api_ok else None,
            "torch": api_health.get("torch_version") if api_ok else None,
            "cuda_version": api_health.get("cuda_version") if api_ok else None,
            "error": (api_health.get("error") or {}).get("message") if not api_ok else None,
        },
        "api": {
            "ok": api_ok,
            "base_url": load_software_config().get("integration_api_url"),
            "health": api_health if api_ok else None,
            "error": api_message if not api_ok else None,
        },
        "runtime": applied_runtime,
        "statistics": api_summary.get("summary") if api_summary.get("success") else None,
    }
