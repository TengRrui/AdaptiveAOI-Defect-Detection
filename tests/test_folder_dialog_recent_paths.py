from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from backend.services.folder_dialog_service import FolderDialogService
from backend.services import folder_dialog_helper


RECENT_KEY = "batch_last_browse_directory"


def test_recent_directory_persists_across_service_restart(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch_a = tmp_path / "BatchA"
    batch_a.mkdir()

    service = FolderDialogService(preferences)
    assert service.remember_directory(RECENT_KEY, batch_a) == str(batch_a.resolve())

    restarted = FolderDialogService(preferences)
    assert restarted.resolve_initial_directory("", RECENT_KEY) == str(batch_a.resolve())
    assert json.loads(preferences.read_text(encoding="utf-8"))[RECENT_KEY] == str(batch_a.resolve())


def test_current_valid_directory_has_priority_over_recent(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch_a = tmp_path / "BatchA"
    batch_b = tmp_path / "BatchB"
    batch_a.mkdir()
    batch_b.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch_a)

    assert service.resolve_initial_directory(str(batch_b), RECENT_KEY) == str(batch_b.resolve())


def test_recent_directory_has_priority_over_task_and_demo_fallbacks(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    recent = tmp_path / "Recent"
    task = tmp_path / "Task"
    demo = tmp_path / "Demo"
    for directory in (recent, task, demo):
        directory.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, recent)

    assert service.resolve_initial_directory("", RECENT_KEY, [str(task), str(demo)]) == str(recent.resolve())


def test_task_then_demo_fallback_order_without_recent_directory(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    task = tmp_path / "Task"
    demo = tmp_path / "Demo"
    task.mkdir()
    demo.mkdir()
    service = FolderDialogService(preferences)

    assert service.resolve_initial_directory("", RECENT_KEY, [str(task), str(demo)]) == str(task.resolve())
    assert service.resolve_initial_directory("", RECENT_KEY, [str(tmp_path / 'missing'), str(demo)]) == str(demo.resolve())


def test_invalid_current_path_uses_valid_recent_directory(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch_a = tmp_path / "BatchA"
    batch_a.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch_a)

    missing = tmp_path / "missing" / "BatchB"
    assert service.resolve_initial_directory(str(missing), RECENT_KEY) == str(batch_a.resolve())


def test_deleted_recent_directory_falls_back_to_nearest_existing_parent(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    parent = tmp_path / "Data"
    removed = parent / "BatchA"
    removed.mkdir(parents=True)
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, removed)
    removed.rmdir()

    restarted = FolderDialogService(preferences)
    assert restarted.resolve_initial_directory("", RECENT_KEY) == str(parent.resolve())


def test_invalid_manual_directory_does_not_overwrite_recent_preference(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch_a = tmp_path / "BatchA"
    batch_a.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch_a)

    with pytest.raises(ValueError, match="不存在"):
        service.remember_directory(RECENT_KEY, tmp_path / "does-not-exist")

    assert service.resolve_initial_directory("", RECENT_KEY) == str(batch_a.resolve())


def test_scoped_recent_directories_do_not_cross_contaminate(tmp_path: Path) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch = tmp_path / "Batch"
    target = tmp_path / "Target"
    batch.mkdir()
    target.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch)
    service.remember_directory("target_domain_last_browse_directory", target)

    restarted = FolderDialogService(preferences)
    assert restarted.resolve_initial_directory("", RECENT_KEY) == str(batch.resolve())
    assert restarted.resolve_initial_directory("", "target_domain_last_browse_directory") == str(target.resolve())


def test_dialog_status_records_resolved_initial_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch = tmp_path / "Batch"
    batch.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch)

    # Keep this test independent of the desktop session.  The production
    # thread still receives the same already-resolved initial directory.
    monkeypatch.setattr(service, "_run_dialog", lambda *args: None)
    started = service.start("", recent_key=RECENT_KEY)
    status = service.status(str(started["dialog_id"])) or {}

    assert status["initial_directory"] == str(batch.resolve())


class _FakeTkRoot:
    def withdraw(self) -> None:
        return None

    def attributes(self, *_args: object) -> None:
        return None

    def update_idletasks(self) -> None:
        return None

    def destroy(self) -> None:
        return None


class _FakePickerProcess:
    """Small Popen double used to exercise the production IPC lifecycle."""

    _next_pid = 40000

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self.returncode if self.returncode is not None else -15
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _write_picker_result(service: FolderDialogService, dialog_id: str, *, status: str, path: Path | None = None, request_id: str | None = None) -> Path:
    record = service._dialogs[dialog_id]
    result_path = Path(str(record["result_path"]))
    payload = {
        "request_id": request_id or dialog_id,
        "status": status,
        "path": str(path) if path else None,
        "paths": [str(path)] if path and status == "selected" else [],
        "message": None,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return result_path


def _write_picker_open(service: FolderDialogService, dialog_id: str, *, request_id: str | None = None) -> Path:
    record = service._dialogs[dialog_id]
    open_path = Path(str(record["open_path"]))
    open_path.write_text(
        json.dumps(
            {
                "request_id": request_id or dialog_id,
                "status": "open",
                "opened_at": 1000.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return open_path


def test_folder_picker_success_then_success_uses_fresh_request_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    first_path = tmp_path / "First"
    second_path = tmp_path / "Second"
    first_path.mkdir()
    second_path.mkdir()
    processes: list[_FakePickerProcess] = []
    helper_args: list[list[object]] = []

    def fake_popen(*_args: object, **_kwargs: object) -> _FakePickerProcess:
        process = _FakePickerProcess()
        processes.append(process)
        helper_args.append(list(_args[0]))
        return process

    monkeypatch.setattr("backend.services.folder_dialog_service.subprocess.Popen", fake_popen)
    service = FolderDialogService(preferences)

    first = service.start(str(first_path), recent_key=RECENT_KEY)
    first_id = str(first["dialog_id"])
    assert first["request_id"] == first_id
    assert helper_args[0][helper_args[0].index("--request-id") + 1] == first_id
    _write_picker_result(service, first_id, status="selected", path=first_path)
    assert service.status(first_id, consume=True)["status"] == "selected"

    second = service.start(str(second_path), recent_key=RECENT_KEY)
    second_id = str(second["dialog_id"])
    assert second_id != first_id
    assert second["request_id"] == second_id
    assert helper_args[1][helper_args[1].index("--request-id") + 1] == second_id
    _write_picker_result(service, second_id, status="selected", path=second_path)
    assert service.status(second_id, consume=True)["path"] == str(second_path.resolve())
    assert len(processes) == 2


@pytest.mark.parametrize("user_wait_seconds", [30.0, 60.0, 120.0])
def test_open_picker_has_no_user_selection_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_wait_seconds: float,
) -> None:
    clock = [1000.0]
    process = _FakePickerProcess()
    monkeypatch.setattr("backend.services.folder_dialog_service.time.time", lambda: clock[0])
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    service = FolderDialogService(tmp_path / "recent_paths.json")

    started = service.start(str(tmp_path), recent_key=RECENT_KEY)
    dialog_id = str(started["dialog_id"])
    assert started["status"] == "starting"
    _write_picker_open(service, dialog_id)
    opened = service.status(dialog_id, consume=False)
    assert opened and opened["status"] == "open"

    clock[0] += user_wait_seconds
    waiting = service.status(dialog_id, consume=False)

    assert waiting and waiting["status"] == "open"
    assert process.poll() is None
    assert "等待超时" not in str(waiting.get("message") or "")


def test_helper_startup_timeout_applies_only_before_open_handshake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [2000.0]
    process = _FakePickerProcess()
    monkeypatch.setattr("backend.services.folder_dialog_service.time.time", lambda: clock[0])
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    service = FolderDialogService(tmp_path / "recent_paths.json")
    started = service.start(str(tmp_path), recent_key=RECENT_KEY)
    dialog_id = str(started["dialog_id"])

    clock[0] += 10.1
    status = service.status(dialog_id, consume=False)

    assert status and status["status"] == "error"
    assert "启动超时" in str(status.get("message") or "")
    assert process.poll() is not None


def test_open_picker_reports_helper_crash_without_user_timeout_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePickerProcess()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    service = FolderDialogService(tmp_path / "recent_paths.json")
    started = service.start(str(tmp_path), recent_key=RECENT_KEY)
    dialog_id = str(started["dialog_id"])
    _write_picker_open(service, dialog_id)
    assert service.status(dialog_id, consume=False)["status"] == "open"

    process.returncode = 7
    failed = service.status(dialog_id, consume=False)

    assert failed and failed["status"] == "error"
    assert "服务异常" in str(failed.get("message") or "")
    assert "等待超时" not in str(failed.get("message") or "")


def test_helper_publishes_open_handshake_before_waiting_for_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "picker-result.json"
    open_output = tmp_path / "picker-open.json"
    selected = tmp_path / "Selected"
    selected.mkdir()

    def choose_directory(**_kwargs: object) -> str:
        handshake = json.loads(open_output.read_text(encoding="utf-8"))
        assert handshake["request_id"] == "qa-request"
        assert handshake["status"] == "open"
        return str(selected)

    fake_dialog = SimpleNamespace(askdirectory=choose_directory)
    fake_tk = SimpleNamespace(Tk=_FakeTkRoot, filedialog=fake_dialog)
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    arguments = SimpleNamespace(
        output=str(output),
        open_output=str(open_output),
        request_id="qa-request",
        dialog_type="folder",
        multiple="false",
        file_type=[],
        initial_dir=str(tmp_path),
        title="选择文件夹",
    )

    assert folder_dialog_helper.run(arguments) == 0
    final = json.loads(output.read_text(encoding="utf-8"))
    assert final["status"] == "selected"
    assert final["path"] == str(selected.resolve())


def test_helper_cancel_or_window_close_returns_normal_canceled_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "picker-result.json"
    open_output = tmp_path / "picker-open.json"
    fake_dialog = SimpleNamespace(askdirectory=lambda **_kwargs: "")
    fake_tk = SimpleNamespace(Tk=_FakeTkRoot, filedialog=fake_dialog)
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    arguments = SimpleNamespace(
        output=str(output),
        open_output=str(open_output),
        request_id="qa-cancel-request",
        dialog_type="folder",
        multiple="false",
        file_type=[],
        initial_dir=str(tmp_path),
        title="选择文件夹",
    )

    assert folder_dialog_helper.run(arguments) == 0
    handshake = json.loads(open_output.read_text(encoding="utf-8"))
    final = json.loads(output.read_text(encoding="utf-8"))
    assert handshake["status"] == "open"
    assert final["status"] == "canceled"
    assert final["message"] is None
    assert final["paths"] == []


def test_batch_status_failures_never_cancel_native_picker() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "pages" / "2_批量检测.py"
    ).read_text(encoding="utf-8")
    poll_body = source.split("def _poll_folder_dialog() -> None:", 1)[1].split(
        "def _run_current_batch(",
        1,
    )[0]

    assert "cancel_folder_dialog" not in poll_body
    assert "failures <= 5" not in poll_body
    assert '{"starting", "open", "busy"}' in poll_body


def test_target_domain_shared_picker_accepts_starting_and_open_states() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "pages" / "4_目标域适配.py"
    ).read_text(encoding="utf-8")

    assert source.count('{"starting", "open", "busy"}') >= 3
    assert source.count('if bool((response.get("error") or {}).get("retryable"))') >= 3


def test_folder_picker_cancel_then_success_cleans_cancelled_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    first_path = tmp_path / "First"
    second_path = tmp_path / "Second"
    first_path.mkdir()
    second_path.mkdir()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: _FakePickerProcess(),
    )
    service = FolderDialogService(preferences)

    first = service.start(str(first_path), recent_key=RECENT_KEY)
    first_id = str(first["dialog_id"])
    result_path = Path(str(service._dialogs[first_id]["result_path"]))
    result_path.write_text("stale", encoding="utf-8")
    cancelled = service.cancel(first_id)
    assert cancelled and cancelled["status"] == "canceled"
    assert not result_path.exists()
    assert service._active_dialog_id is None

    second = service.start(str(second_path), recent_key=RECENT_KEY)
    second_id = str(second["dialog_id"])
    _write_picker_result(service, second_id, status="selected", path=second_path)
    assert service.status(second_id, consume=True)["status"] == "selected"


def test_folder_picker_completed_result_releases_active_before_next_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    first_path = tmp_path / "First"
    second_path = tmp_path / "Second"
    first_path.mkdir()
    second_path.mkdir()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: _FakePickerProcess(),
    )
    service = FolderDialogService(preferences)

    first = service.start(str(first_path), recent_key=RECENT_KEY)
    first_id = str(first["dialog_id"])
    _write_picker_result(service, first_id, status="selected", path=first_path)
    # A user may click again before the next Streamlit polling tick.  The
    # server must observe and finalize the previous result instead of saying
    # that a stale picker is still busy.
    second = service.start(str(second_path), recent_key=RECENT_KEY)
    assert second.get("dialog_id") and second.get("status") == "starting"
    assert second["dialog_id"] != first_id


def test_folder_picker_close_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    first_path = tmp_path / "First"
    second_path = tmp_path / "Second"
    first_path.mkdir()
    second_path.mkdir()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: _FakePickerProcess(),
    )
    service = FolderDialogService(preferences)

    first = service.start(str(first_path), recent_key=RECENT_KEY)
    first_id = str(first["dialog_id"])
    _write_picker_result(service, first_id, status="canceled")
    assert service.status(first_id, consume=True)["status"] == "canceled"

    second = service.start(str(second_path), recent_key=RECENT_KEY)
    second_id = str(second["dialog_id"])
    _write_picker_result(service, second_id, status="selected", path=second_path)
    assert service.status(second_id, consume=True)["path"] == str(second_path.resolve())


def test_folder_picker_stale_result_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    selected = tmp_path / "Selected"
    selected.mkdir()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: _FakePickerProcess(),
    )
    service = FolderDialogService(preferences)
    started = service.start(str(selected), recent_key=RECENT_KEY)
    dialog_id = str(started["dialog_id"])
    _write_picker_result(service, dialog_id, status="selected", path=selected, request_id="old-request")

    status = service.status(dialog_id, consume=False)
    assert status and status["status"] == "error"
    assert "不匹配" in str(status.get("message") or "")
    assert service._active_dialog_id is None


def test_folder_picker_result_without_request_id_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    selected = tmp_path / "Selected"
    selected.mkdir()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: _FakePickerProcess(),
    )
    service = FolderDialogService(preferences)
    started = service.start(str(selected), recent_key=RECENT_KEY)
    dialog_id = str(started["dialog_id"])
    result_path = Path(str(service._dialogs[dialog_id]["result_path"]))
    result_path.write_text(
        json.dumps({"status": "selected", "path": str(selected), "paths": [str(selected)]}),
        encoding="utf-8",
    )

    status = service.status(dialog_id, consume=False)
    assert status and status["status"] == "error"
    assert "请求" in str(status.get("message") or "")
    assert service._active_dialog_id is None


def test_folder_picker_timeout_returns_to_idle_and_repeated_rounds_are_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    selected = tmp_path / "Selected"
    selected.mkdir()
    monkeypatch.setattr(
        "backend.services.folder_dialog_service.subprocess.Popen",
        lambda *_args, **_kwargs: _FakePickerProcess(),
    )
    service = FolderDialogService(preferences)
    for index in range(10):
        started = service.start(str(selected), recent_key=RECENT_KEY)
        dialog_id = str(started["dialog_id"])
        open_path = _write_picker_open(service, dialog_id)
        opened = service.status(dialog_id, consume=False)
        assert opened and opened["status"] == "open"
        assert not open_path.exists()
        if index % 2:
            _write_picker_result(service, dialog_id, status="canceled")
        else:
            _write_picker_result(service, dialog_id, status="selected", path=selected)
        final = service.status(dialog_id, consume=True)
        assert final and final["status"] in {"selected", "canceled"}
        assert service._active_dialog_id is None
        assert not Path(str(service._dialogs.get(dialog_id, {}).get("result_path") or "missing")).exists()


def test_cancel_does_not_overwrite_recent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch_a = tmp_path / "BatchA"
    batch_b = tmp_path / "BatchB"
    batch_a.mkdir()
    batch_b.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch_a)

    captured: dict[str, str] = {}

    def cancel_dialog(**kwargs: object) -> str:
        captured["initialdir"] = str(kwargs.get("initialdir") or "")
        return ""

    fake_dialog = SimpleNamespace(askdirectory=cancel_dialog)
    fake_tk = SimpleNamespace(Tk=_FakeTkRoot, filedialog=fake_dialog)
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    service._dialogs["cancel"] = {"dialog_id": "cancel", "status": "opening"}
    service._active_dialog_id = "cancel"

    service._run_dialog("cancel", str(batch_b), "folder", False, [], None, RECENT_KEY)

    assert captured["initialdir"] == str(batch_b.resolve())
    assert service.status("cancel")["status"] == "canceled"
    assert FolderDialogService(preferences).resolve_initial_directory("", RECENT_KEY) == str(batch_a.resolve())


def test_successful_native_selection_updates_recent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preferences = tmp_path / "recent_paths.json"
    batch_a = tmp_path / "BatchA"
    batch_b = tmp_path / "BatchB"
    batch_a.mkdir()
    batch_b.mkdir()
    service = FolderDialogService(preferences)
    service.remember_directory(RECENT_KEY, batch_a)

    fake_dialog = SimpleNamespace(askdirectory=lambda **_kwargs: str(batch_b))
    fake_tk = SimpleNamespace(Tk=_FakeTkRoot, filedialog=fake_dialog)
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    service._dialogs["selected"] = {"dialog_id": "selected", "status": "opening"}
    service._active_dialog_id = "selected"

    service._run_dialog("selected", str(batch_a), "folder", False, [], None, RECENT_KEY)

    assert service.status("selected")["status"] == "selected"
    assert FolderDialogService(preferences).resolve_initial_directory("", RECENT_KEY) == str(batch_b.resolve())
