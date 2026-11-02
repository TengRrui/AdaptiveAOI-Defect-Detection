from __future__ import annotations

from pathlib import Path


def test_launcher_treats_existing_watchdog_as_idempotent_start() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "launchers" / "start.ps1").read_text(encoding="utf-8")
    function = source.split("function Start-BackendWatchdog", 1)[1].split(
        'Write-LauncherLog \'INFO\' "T1 launchers开始执行。', 1
    )[0]

    assert "function Get-ExistingWatchdogProcessId" in source
    assert "$discoveredPid = Get-ExistingWatchdogProcessId" in function
    assert "未重复启动" in function
    assert "$watchdog.ExitCode -eq 0" in function
    assert "本次重复启动正常结束" in function
    assert "退出码=$($watchdog.ExitCode)" in function
