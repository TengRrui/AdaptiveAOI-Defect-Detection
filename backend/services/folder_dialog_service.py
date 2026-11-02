from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class FolderDialogService:
    """Open one local Windows picker in a short-lived desktop helper process.

    Tk must own the thread that creates its root window.  The previous daemon
    thread ran Tk inside the FastAPI process, which is unreliable on Windows
    desktop sessions and left the API/UI stuck in ``opening`` forever.  The
    helper process keeps the picker responsive and gives the API a finite,
    pollable lifecycle without sharing GUI state with Uvicorn.
    """

    _FINAL_STATES = {"selected", "canceled", "error"}
    _PENDING_STATES = {"starting", "open"}
    _HELPER_START_TIMEOUT_SECONDS = 10.0

    _RECENT_KEY_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")

    def __init__(self, recent_paths_file: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._active_dialog_id: str | None = None
        self._dialogs: dict[str, dict[str, Any]] = {}
        self._recent_paths_file = Path(recent_paths_file).resolve() if recent_paths_file else None
        self._recent_paths: dict[str, str] = self._load_recent_paths()

    def start(
        self,
        initial_dir: str | None = None,
        *,
        dialog_type: str = "folder",
        multiple: bool = False,
        file_types: list[str] | None = None,
        title: str | None = None,
        recent_key: str | None = None,
        fallback_dirs: list[str] | None = None,
    ) -> dict[str, Any]:
        dialog_type = str(dialog_type or "folder").lower()
        if dialog_type not in {"folder", "file", "files"}:
            raise ValueError("不支持的本地选择窗口类型。")
        if dialog_type == "files":
            multiple = True
        recent_key = str(recent_key or "").strip().lower() or None
        if recent_key and not self._RECENT_KEY_PATTERN.fullmatch(recent_key):
            raise ValueError("本地选择窗口用途标识无效。")
        allowed_types = [str(item).lower().lstrip(".") for item in (file_types or []) if str(item).strip()]
        resolved_initial_dir = self.resolve_initial_directory(initial_dir, recent_key, fallback_dirs)
        result_root = self._recent_paths_file.parent if self._recent_paths_file is not None else Path.cwd()
        with self._lock:
            if self._active_dialog_id:
                active = self._dialogs.get(self._active_dialog_id)
                if active is not None:
                    # A previous browser rerun may happen between the native
                    # dialog closing and its next status poll.  Refresh it
                    # before declaring the picker busy.
                    self._refresh_dialog_locked(active)
                    if active.get("status") not in self._FINAL_STATES:
                        return {"status": "busy"}
                    self._release_active_locked(active)
            self._prune_locked()
            dialog_id = uuid.uuid4().hex
            result_path = result_root / f"folder_dialog_{dialog_id}.json"
            open_path = result_root / f"folder_dialog_{dialog_id}.open.json"
            started_at = time.time()
            self._dialogs[dialog_id] = {
                "dialog_id": dialog_id,
                "request_id": dialog_id,
                "status": "starting",
                "path": None,
                "paths": [],
                "message": None,
                "dialog_type": dialog_type,
                "initial_directory": resolved_initial_dir,
                "recent_key": recent_key,
                "started_at": started_at,
                "opened_at": None,
                # This deadline only covers helper startup and the OPEN
                # handshake.  It is cleared permanently once the native
                # picker is visible; user selection time is never bounded.
                "handshake_deadline": None,
                "result_path": str(result_path),
                "open_path": str(open_path),
            }
            self._active_dialog_id = dialog_id

        # A small compatibility seam keeps older unit/integration harnesses
        # that monkeypatch ``_run_dialog`` deterministic.  Production never
        # takes this branch: the instance has no override and the picker is
        # always isolated in the helper process below.
        legacy_runner = self.__dict__.get("_run_dialog")
        if callable(legacy_runner):
            legacy_runner(
                dialog_id,
                resolved_initial_dir,
                dialog_type,
                multiple,
                allowed_types,
                title,
                recent_key,
            )
            return {"dialog_id": dialog_id, "request_id": dialog_id, "status": "starting"}

        try:
            helper_args = [
                sys.executable,
                "-m",
                "backend.services.folder_dialog_helper",
                "--output",
                str(result_path),
                "--open-output",
                str(open_path),
                "--request-id",
                dialog_id,
                "--initial-dir",
                str(resolved_initial_dir or ""),
                "--dialog-type",
                dialog_type,
                "--title",
                str(title or ""),
                "--multiple",
                "true" if multiple else "false",
            ]
            for file_type in allowed_types:
                helper_args.extend(("--file-type", file_type))
            creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            process = subprocess.Popen(
                helper_args,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
                close_fds=True,
            )
            with self._lock:
                record = self._dialogs.get(dialog_id)
                if record is not None:
                    if record.get("status") in self._FINAL_STATES:
                        # A concurrent cancellation may have happened while
                        # the helper was being spawned.  Never orphan it.
                        self._terminate_process_locked({"_process": process})
                    else:
                        helper_started_at = time.time()
                        record.update({
                            "helper_pid": int(process.pid),
                            "_process": process,
                            "helper_started_at": helper_started_at,
                            "handshake_deadline": helper_started_at + self._HELPER_START_TIMEOUT_SECONDS,
                            "status": "starting",
                        })
        except Exception as exc:
            with self._lock:
                record = self._dialogs.get(dialog_id)
                if record is not None:
                    self._finalize_locked(record, "error", f"无法启动本地选择窗口：{exc}")
        return {"dialog_id": dialog_id, "request_id": dialog_id, "status": "starting"}

    def _run_dialog(
        self,
        dialog_id: str,
        initial_dir: str,
        dialog_type: str,
        multiple: bool,
        file_types: list[str],
        title: str | None,
        recent_key: str | None,
    ) -> None:
        """Legacy synchronous Tk adapter retained for isolated test doubles.

        The live service does not call this method; ``start`` launches
        :mod:`folder_dialog_helper` instead.  Keeping the tiny adapter avoids
        breaking downstream harnesses that exercise recent-directory
        semantics with an injected fake ``tkinter`` module.
        """

        import tkinter as tk
        from tkinter import filedialog

        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update_idletasks()
            if dialog_type == "folder":
                selected = filedialog.askdirectory(
                    parent=root,
                    title=str(title or "选择文件夹"),
                    initialdir=initial_dir,
                    mustexist=True,
                )
                paths = [selected] if selected else []
            elif multiple:
                paths = list(
                    filedialog.askopenfilenames(
                        parent=root,
                        title=str(title or "选择文件"),
                        initialdir=initial_dir,
                        filetypes=[("所有文件", "*.*")],
                    )
                )
            else:
                selected = filedialog.askopenfilename(
                    parent=root,
                    title=str(title or "选择文件"),
                    initialdir=initial_dir,
                    filetypes=[("所有文件", "*.*")],
                )
                paths = [selected] if selected else []
            normalized = [str(Path(path).expanduser().resolve()) for path in paths if str(path)]
            status = "selected" if normalized else "canceled"
            with self._lock:
                record = self._dialogs.get(dialog_id)
                if record is not None:
                    record.update({
                        "status": status,
                        "path": normalized[0] if normalized else None,
                        "paths": normalized,
                        "message": None,
                    })
                    if status == "selected" and recent_key and normalized:
                        selected_path = Path(normalized[0])
                        self._remember_directory(
                            recent_key,
                            selected_path if dialog_type == "folder" else selected_path.parent,
                        )
                if self._active_dialog_id == dialog_id:
                    self._active_dialog_id = None
        except Exception as exc:
            with self._lock:
                record = self._dialogs.get(dialog_id)
                if record is not None:
                    record.update({"status": "error", "message": str(exc), "path": None, "paths": []})
                if self._active_dialog_id == dialog_id:
                    self._active_dialog_id = None
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass

    def status(self, dialog_id: str, *, consume: bool = False) -> dict[str, Any] | None:
        with self._lock:
            record = self._dialogs.get(dialog_id)
            if record is None:
                return None
            self._refresh_dialog_locked(record)
            result = dict(record)
            result.pop("_process", None)
            result.pop("result_path", None)
            result.pop("open_path", None)
            result.pop("handshake_deadline", None)
            if consume and result.get("status") in self._FINAL_STATES:
                removed = self._dialogs.pop(dialog_id, None)
                if removed:
                    self._cleanup_result_file(removed)
                if self._active_dialog_id == dialog_id:
                    self._active_dialog_id = None
            return result

    def cancel(self, dialog_id: str) -> dict[str, Any] | None:
        """Cancel a running helper and return its final state."""

        with self._lock:
            record = self._dialogs.get(str(dialog_id))
            if record is None:
                return None
            self._refresh_dialog_locked(record)
            process = record.get("_process")
            if record.get("status") not in self._FINAL_STATES:
                if process is not None:
                    self._terminate_process_locked(record)
                self._finalize_locked(record, "canceled", "已取消本地选择窗口。")
            else:
                self._release_active_locked(record)
            self._cleanup_result_file(record)
            result = dict(record)
            result.pop("_process", None)
            result.pop("result_path", None)
            result.pop("open_path", None)
            result.pop("handshake_deadline", None)
            return result

    def remember_directory(self, recent_key: str, directory: str | Path) -> str:
        """Persist one valid UI browse directory without opening a dialog.

        This is used when a user types a directory manually and successfully
        starts a task.  It deliberately shares the same scoped preference as
        the native picker, so browser sessions never need a process-global
        Python variable to remember a path.
        """

        normalized_key = str(recent_key or "").strip().lower()
        if not self._RECENT_KEY_PATTERN.fullmatch(normalized_key):
            raise ValueError("本地选择窗口用途标识无效。")
        try:
            resolved = Path(directory).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ValueError("最近使用目录无效。") from exc
        if not resolved.is_dir():
            raise ValueError("最近使用目录不存在或当前无法访问。")
        self._remember_directory(normalized_key, resolved)
        return str(resolved)

    def resolve_initial_directory(
        self,
        value: str | None,
        recent_key: str | None = None,
        fallback_dirs: list[str] | None = None,
    ) -> str:
        """Resolve the native picker's start directory using safe fallbacks."""

        return self._existing_directory(value, recent_key, fallback_dirs)

    def _refresh_dialog_locked(self, record: dict[str, Any]) -> None:
        if record.get("status") in self._FINAL_STATES:
            self._release_active_locked(record)
            return
        result_path = Path(str(record.get("result_path") or ""))
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                status = str(payload.get("status") or "error")
                if status not in self._FINAL_STATES:
                    status = "error"
                payload_request_id = str(payload.get("request_id") or "").strip()
                expected_request_id = str(record.get("request_id") or record.get("dialog_id") or "")
                # Every helper result is scoped to the request that created
                # its IPC file.  A missing ID is just as unsafe as an old ID:
                # accepting either could apply a stale result after a rapid
                # close-and-reopen sequence.
                if not payload_request_id or payload_request_id != expected_request_id:
                    self._terminate_process_locked(record)
                    self._cleanup_result_file(record)
                    self._finalize_locked(record, "error", "本地选择窗口返回结果与当前请求不匹配。")
                    return
                record.update({
                    "status": status,
                    "path": payload.get("path"),
                    "paths": list(payload.get("paths") or []) if status == "selected" else [],
                    "message": payload.get("message"),
                })
                if status == "selected":
                    recent_key = str(record.get("recent_key") or "")
                    values = list(record.get("paths") or [])
                    if recent_key and values:
                        selected = Path(str(values[0]))
                        self._remember_directory(recent_key, selected if record.get("dialog_type") == "folder" else selected.parent)
                self._terminate_process_locked(record)
                self._cleanup_result_file(record)
                self._release_active_locked(record)
                return
            except (OSError, UnicodeError, json.JSONDecodeError):
                # The helper writes atomically; a transient read failure is
                # treated as pending until the helper exits.
                pass

        open_path = Path(str(record.get("open_path") or ""))
        if record.get("status") == "starting" and open_path.is_file():
            try:
                payload = json.loads(open_path.read_text(encoding="utf-8"))
                payload_request_id = str(payload.get("request_id") or "").strip()
                expected_request_id = str(record.get("request_id") or record.get("dialog_id") or "")
                if payload_request_id != expected_request_id or payload.get("status") != "open":
                    self._terminate_process_locked(record)
                    self._cleanup_result_file(record)
                    self._finalize_locked(record, "error", "本地选择窗口握手与当前请求不匹配。")
                    return
                record.update({
                    "status": "open",
                    "opened_at": payload.get("opened_at") or time.time(),
                    "handshake_deadline": None,
                    "message": None,
                })
                self._cleanup_path(open_path)
            except (OSError, UnicodeError, json.JSONDecodeError):
                # Atomic publication makes this unlikely, but a transient
                # filesystem read must not kill a native window.
                pass

        process = record.get("_process")
        handshake_deadline = float(record.get("handshake_deadline") or 0.0)
        if record.get("status") == "starting" and handshake_deadline and time.time() >= handshake_deadline:
            self._terminate_process_locked(record)
            self._cleanup_result_file(record)
            self._finalize_locked(record, "error", "本地文件夹选择服务启动超时，请重新打开。")
            return
        if process is not None and process.poll() is not None:
            self._terminate_process_locked(record)
            self._cleanup_result_file(record)
            self._finalize_locked(record, "error", "本地文件夹选择服务异常，请重新打开。")

    @staticmethod
    def _terminate_process_locked(record: dict[str, Any]) -> None:
        process = record.get("_process")
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except (OSError, AttributeError):
                pass
            try:
                process.wait(timeout=0.5)
            except (OSError, AttributeError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except (OSError, AttributeError, subprocess.TimeoutExpired):
                    pass

    def _finalize_locked(self, record: dict[str, Any], status: str, message: str | None = None) -> None:
        record.update({
            "status": status,
            "message": message,
            "path": None if status != "selected" else record.get("path"),
            "paths": [] if status != "selected" else list(record.get("paths") or []),
        })
        self._release_active_locked(record)

    def _release_active_locked(self, record: dict[str, Any]) -> None:
        dialog_id = str(record.get("dialog_id") or "")
        if dialog_id and self._active_dialog_id == dialog_id:
            self._active_dialog_id = None

    @staticmethod
    def _cleanup_result_file(record: dict[str, Any]) -> None:
        for field in ("result_path", "open_path"):
            raw_path = str(record.get(field) or "").strip()
            if raw_path:
                FolderDialogService._cleanup_path(Path(raw_path))

    @staticmethod
    def _cleanup_path(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _existing_directory(
        self,
        value: str | None,
        recent_key: str | None = None,
        fallback_dirs: list[str] | None = None,
    ) -> str:
        def existing_exact(raw: str | None) -> str | None:
            if not raw:
                return None
            try:
                candidate = Path(raw).expanduser()
                if candidate.is_file():
                    return str(candidate.parent)
                if candidate.exists() and candidate.is_dir():
                    return str(candidate)
            except OSError:
                return None
            return None

        def nearest_existing_parent(raw: str | None) -> str | None:
            if not raw:
                return None
            try:
                candidate = Path(raw).expanduser()
                if candidate.suffix:
                    candidate = candidate.parent
                while not candidate.exists() and candidate.parent != candidate:
                    candidate = candidate.parent
                if candidate.exists() and candidate.is_dir():
                    return str(candidate)
            except OSError:
                return None
            return None

        # The field's exact path wins only when it is still valid.  An old or
        # mistyped field must not drag the next selector back to a drive root;
        # the shared target-domain recent directory has the next priority.
        if existing := existing_exact(value):
            return existing
        if recent_key:
            remembered = str(self._recent_paths.get(recent_key) or "")
            if existing := existing_exact(remembered):
                return existing
        for fallback in fallback_dirs or []:
            if existing := existing_exact(fallback):
                return existing
        if existing := nearest_existing_parent(value):
            return existing
        if recent_key:
            remembered = str(self._recent_paths.get(recent_key) or "")
            if existing := nearest_existing_parent(remembered):
                return existing
        for fallback in fallback_dirs or []:
            if existing := nearest_existing_parent(fallback):
                return existing
        return str(Path.home())

    def _load_recent_paths(self) -> dict[str, str]:
        if self._recent_paths_file is None or not self._recent_paths_file.is_file():
            return {}
        try:
            payload = json.loads(self._recent_paths_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return {
            str(key): str(value)
            for key, value in (payload.items() if isinstance(payload, dict) else [])
            if self._RECENT_KEY_PATTERN.fullmatch(str(key)) and str(value)
        }

    def _remember_directory(self, recent_key: str, directory: Path) -> None:
        if self._recent_paths_file is None:
            return
        try:
            resolved = directory.resolve()
            if not resolved.is_dir():
                return
            with self._lock:
                self._recent_paths[recent_key] = str(resolved)
                self._recent_paths_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._recent_paths_file.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(self._recent_paths, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self._recent_paths_file)
        except OSError:
            # Remembering a convenience path must never make the native dialog
            # itself fail after the user has already selected a valid source.
            return

    def _prune_locked(self) -> None:
        completed = [
            dialog_id
            for dialog_id, record in self._dialogs.items()
            if record.get("status") in self._FINAL_STATES
        ]
        for dialog_id in completed[:-8]:
            record = self._dialogs.pop(dialog_id, None)
            if record:
                self._terminate_process_locked(record)
                self._cleanup_result_file(record)
