from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_loader import LOG_DIR, load_software_config
from .path_resolver import collect_path_status, resolve_algorithm_project, write_path_status


CALL_HISTORY_PATH = LOG_DIR / "call_history.log"
ERROR_LOG_PATH = LOG_DIR / "error.log"


def ensure_log_files() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CALL_HISTORY_PATH.touch(exist_ok=True)
    ERROR_LOG_PATH.touch(exist_ok=True)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    ensure_log_files()
    path.write_text("", encoding="utf-8") if not path.exists() else None
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _log_call(action: str, command: list[str], algorithm_root: Path, mode: str, execute: bool) -> None:
    _append_log(
        CALL_HISTORY_PATH,
        {
            "time": _timestamp(),
            "action": action,
            "mode": mode,
            "execute": bool(execute),
            "cwd": str(algorithm_root),
            "command": subprocess.list2cmdline(command),
            "argv": command,
        },
    )


def _log_error(action: str, error: BaseException | str) -> None:
    _append_log(
        ERROR_LOG_PATH,
        {
            "time": _timestamp(),
            "action": action,
            "error": str(error),
            "type": type(error).__name__ if isinstance(error, BaseException) else "Error",
        },
    )


def log_software_error(action: str, error: BaseException | str) -> None:
    _log_error(action, error)


def _as_path(path_value: str | Path, base: Path | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _config_path(config: str | Path, algorithm_root: Path) -> Path:
    return _as_path(config, algorithm_root)


def _software_output_path(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    root = Path(load_software_config()["_project_root"])
    return _as_path(output_dir, root)


def _script_path(algorithm_root: Path, script_name: str) -> Path:
    return algorithm_root / "scripts" / script_name


def resolve_algorithm_python(config: dict[str, Any] | None = None) -> Path:
    config = config or load_software_config()
    candidates = [
        os.environ.get("ADAPTIVEAOI_ALGORITHM_PYTHON"),
        os.environ.get("ADAPTIVEAOI_PYTHON"),
        config.get("algorithm_python"),
        sys.executable,
    ]
    for candidate in candidates:
        if not candidate or str(candidate).strip().lower() == "auto":
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path.resolve()
    return Path(sys.executable).resolve()


def _python_command(script_path: Path, args: list[str], python_executable: str | Path | None = None) -> list[str]:
    python_path = Path(python_executable or sys.executable)
    return [str(python_path), str(script_path), *args]


def _module_name(script_path: Path) -> str:
    return f"adaptiveaoi_algorithm_{script_path.stem}_{abs(hash(str(script_path)))}"


def _can_import_script(script_path: Path, algorithm_root: Path) -> tuple[bool, str | None]:
    try:
        if not script_path.exists():
            raise FileNotFoundError(script_path)
        root_text = str(algorithm_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        spec = importlib.util.spec_from_file_location(_module_name(script_path), script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load import spec for {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            raise AttributeError(f"{script_path} does not expose main()")
        return True, None
    except Exception as exc:  # import availability is advisory, so keep this quiet
        return False, f"{type(exc).__name__}: {exc}"


@contextmanager
def _temporary_argv(script_path: Path, args: list[str]):
    previous = sys.argv[:]
    sys.argv = [str(script_path), *args]
    try:
        yield
    finally:
        sys.argv = previous


def _run_import_main(script_path: Path, algorithm_root: Path, args: list[str]) -> Any:
    root_text = str(algorithm_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(_module_name(script_path), script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load import spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with _temporary_argv(script_path, args):
        return module.main()


def _invoke(
    action: str,
    script_name: str,
    args: list[str],
    execute: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_log_files()
    config = load_software_config()
    algorithm_root = resolve_algorithm_project(config)
    algorithm_python = resolve_algorithm_python(config)
    write_path_status(config)

    script_path = _script_path(algorithm_root, script_name)
    command = _python_command(script_path, args, algorithm_python)
    if not script_path.exists():
        error = f"算法脚本未找到: {script_path}"
        _log_error(action, error)
        _log_call(action, command, algorithm_root, "missing_script", execute)
        return {
            "ok": False,
            "action": action,
            "execute": bool(execute),
            "mode": "missing_script",
            "algorithm_project_root": str(algorithm_root),
            "script_path": str(script_path),
            "command": command,
            "command_line": subprocess.list2cmdline(command),
            "error": error,
            "metadata": metadata or {},
        }

    config_error = _validate_config_arg(args)
    if config_error:
        _log_error(action, config_error)
        _log_call(action, command, algorithm_root, "invalid_config", execute)
        return {
            "ok": False,
            "action": action,
            "execute": bool(execute),
            "mode": "invalid_config",
            "algorithm_project_root": str(algorithm_root),
            "script_path": str(script_path),
            "command": command,
            "command_line": subprocess.list2cmdline(command),
            "python_executable": str(algorithm_python),
            "error": config_error,
            "metadata": metadata or {},
        }

    use_import = Path(sys.executable).resolve() == algorithm_python
    import_available, import_error = _can_import_script(script_path, algorithm_root) if use_import else (False, None)
    mode = "import" if import_available else "subprocess"
    if import_error:
        _log_error(f"{action}:import_check", import_error)
    _log_call(action, command, algorithm_root, mode, execute)

    result = {
        "ok": True,
        "action": action,
        "execute": bool(execute),
        "mode": mode,
        "algorithm_project_root": str(algorithm_root),
        "script_path": str(script_path),
        "command": command,
        "command_line": subprocess.list2cmdline(command),
        "python_executable": str(algorithm_python),
        "import_available": import_available,
        "import_error": import_error,
        "metadata": metadata or {},
    }

    if not execute:
        return result

    try:
        if import_available:
            result["return_value"] = _run_import_main(script_path, algorithm_root, args)
        else:
            completed = subprocess.run(
                command,
                cwd=str(algorithm_root),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            result["returncode"] = completed.returncode
            result["stdout"] = completed.stdout
            result["stderr"] = completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(f"{action} failed with return code {completed.returncode}: {completed.stderr}")
        return result
    except Exception as exc:
        _log_error(action, exc)
        raise


def _validate_config_arg(args: list[str]) -> str | None:
    if "--config" not in args:
        return None
    idx = args.index("--config")
    if idx + 1 >= len(args):
        return "算法配置参数缺失: --config 后没有路径。"
    config_path = Path(args[idx + 1])
    if not config_path.exists():
        return f"算法配置文件不存在: {config_path}"
    return None


def build_manifest(dataset: str | None = None, execute: bool = False) -> dict[str, Any]:
    config = load_software_config()
    algorithm_root = resolve_algorithm_project(config)
    output_dir = algorithm_root / "outputs" / "results"
    args = ["--data_root", str(_as_path(config["data_root"])), "--output_dir", str(output_dir)]
    return _invoke("build_manifest", "build_manifest.py", args, execute, {"dataset": dataset})


def build_memory(dataset: str, category: str, config: str | Path, execute: bool = False) -> dict[str, Any]:
    algorithm_root = resolve_algorithm_project(load_software_config())
    args = ["--config", str(_config_path(config, algorithm_root)), "--category", category]
    return _invoke("build_memory", "build_memory.py", args, execute, {"dataset": dataset, "category": category})


def calibrate(
    dataset: str,
    category: str,
    config: str | Path,
    normal_shot: int,
    defect_shot: int,
    seed: int,
    use_defect_direction: bool,
    execute: bool = False,
) -> dict[str, Any]:
    algorithm_root = resolve_algorithm_project(load_software_config())
    args = [
        "--config",
        str(_config_path(config, algorithm_root)),
        "--dataset",
        dataset,
        "--category",
        category,
        "--normal_shot",
        str(int(normal_shot)),
        "--defect_shot",
        str(int(defect_shot)),
        "--seed",
        str(int(seed)),
    ]
    if use_defect_direction:
        args.append("--use_defect_direction")
    return _invoke("calibrate", "calibrate.py", args, execute, {"dataset": dataset, "category": category})


def infer_image(
    dataset: str,
    category: str,
    config: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    algorithm_root = resolve_algorithm_project(load_software_config())
    requested_output_dir = _software_output_path(output_dir)
    if requested_output_dir is not None:
        requested_output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--config",
        str(_config_path(config, algorithm_root)),
        "--category",
        category,
        "--image_path",
        str(_as_path(image_path)),
    ]
    return _invoke(
        "infer_image",
        "infer_image.py",
        args,
        execute,
        {
            "dataset": dataset,
            "category": category,
            "requested_output_dir": str(requested_output_dir) if requested_output_dir else None,
            "note": "Core infer_image.py saves outputs according to the algorithm config.",
        },
    )


def infer_batch(
    dataset: str,
    category: str,
    config: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    algorithm_root = resolve_algorithm_project(load_software_config())
    requested_output_dir = _software_output_path(output_dir)
    if requested_output_dir is not None:
        requested_output_dir.mkdir(parents=True, exist_ok=True)
    args = ["--config", str(_config_path(config, algorithm_root)), "--category", category, "--split", "test"]
    return _invoke(
        "infer_batch",
        "infer_batch.py",
        args,
        execute,
        {
            "dataset": dataset,
            "category": category,
            "requested_input_dir": str(_as_path(input_dir)) if input_dir else None,
            "requested_output_dir": str(requested_output_dir) if requested_output_dir else None,
            "note": "Core infer_batch.py reads the dataset split from the algorithm manifest/config.",
        },
    )


def benchmark_speed(
    dataset: str,
    category: str,
    config: str | Path,
    image_size: int,
    device: str,
    precision: str,
    execute: bool = False,
) -> dict[str, Any]:
    algorithm_root = resolve_algorithm_project(load_software_config())
    args = [
        "--config",
        str(_config_path(config, algorithm_root)),
        "--dataset",
        dataset,
        "--category",
        category,
        "--image_size",
        str(int(image_size)),
        "--device",
        device,
        "--precision",
        precision,
    ]
    return _invoke("benchmark_speed", "benchmark_speed.py", args, execute, {"dataset": dataset, "category": category})


def dry_run_all_default_calls() -> dict[str, Any]:
    config = load_software_config()
    status = collect_path_status(config)
    dataset = config["default_dataset"]
    category = config["default_category"]
    algorithm_config = config["default_config"]
    calls = [
        build_manifest(dataset=dataset, execute=False),
        build_memory(dataset, category, algorithm_config, execute=False),
        calibrate(dataset, category, algorithm_config, 10, 2, 0, False, execute=False),
        benchmark_speed(dataset, category, algorithm_config, 256, config["device"], "fp32", execute=False),
    ]
    status["dry_run_calls"] = calls
    return status


if __name__ == "__main__":
    payload = dry_run_all_default_calls()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
