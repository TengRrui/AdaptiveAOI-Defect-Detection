"""The stop entry must leave an unrelated process on a configured port alive."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
def test_stop_does_not_terminate_unrecorded_listener(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(root / "scripts" / "stop_system.bat", tmp_path / "scripts" / "stop_system.bat")
    shutil.copytree(root / "launchers", tmp_path / "launchers")
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); print(s.getsockname()[1],flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        port = process.stdout.readline().strip()
        assert port.isdigit()
        environment = {**os.environ, "ADAPTIVEAOI_BACKEND_PORT": port, "ADAPTIVEAOI_FRONTEND_PORT": port}
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(tmp_path / "scripts" / "stop_system.bat")],
            cwd=tmp_path, env=environment, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert process.poll() is None, "Stop script terminated a listener it did not start"
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
@pytest.mark.parametrize("mismatch", ["project_path", "creation_time"])
def test_stop_rejects_stale_or_foreign_process_record(tmp_path: Path, mismatch: str) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "launchers", tmp_path / "launchers")
    process_root = str(tmp_path) if mismatch == "creation_time" else "another-project"
    process = subprocess.Popen([sys.executable, "-u", "-c", "import time; time.sleep(60)", "streamlit", process_root])
    try:
        actual_started = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", f"(Get-Process -Id {process.pid}).StartTime.ToUniversalTime().ToString('o')"],
            text=True, timeout=10,
        ).strip()
        state = {"project_root": str(tmp_path), "frontend": {
            "started_by_launcher": True, "pid": process.pid,
            "process_started_at": "2000-01-01T00:00:00Z" if mismatch == "creation_time" else actual_started,
        }}
        log_root = tmp_path / "outputs" / "system_runtime" / "logs" / "launcher"
        log_root.mkdir(parents=True)
        (log_root / "launcher_state.json").write_text(json.dumps(state), encoding="utf-8")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tmp_path / "launchers" / "stop.ps1"), "-NoDialog"],
            capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert process.poll() is None, "Stop script trusted a stale PID or another project"
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
