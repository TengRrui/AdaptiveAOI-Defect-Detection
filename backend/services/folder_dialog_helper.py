"""Desktop picker helper launched outside the FastAPI process.

The helper owns Tk's main thread and publishes one atomic JSON result.  It is
deliberately tiny: no application services, database, or Streamlit imports are
allowed here, so opening a local picker cannot contend with inference/runtime
work in the backend process.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(arguments: argparse.Namespace) -> int:
    output = Path(arguments.output).resolve()
    open_output = Path(arguments.open_output).resolve()
    request_id = str(arguments.request_id or "").strip()
    dialog_type = str(arguments.dialog_type or "folder").lower()
    multiple = str(arguments.multiple or "false").lower() == "true"
    file_types = [str(item).lower().lstrip(".") for item in (arguments.file_type or []) if str(item).strip()]
    selected_paths: list[str] = []
    status = "error"
    message: str | None = "无法打开本地选择窗口，请确认当前为 Windows 桌面会话。"
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        _write_result(
            open_output,
            {
                "request_id": request_id,
                "status": "open",
                "opened_at": time.time(),
            },
        )
        initial_dir = str(arguments.initial_dir or "")
        if dialog_type == "folder":
            selected = filedialog.askdirectory(
                parent=root,
                title=str(arguments.title or "选择文件夹"),
                initialdir=initial_dir,
                mustexist=True,
            )
            selected_paths = [selected] if selected else []
        else:
            patterns = (
                [(f"支持的文件（{', '.join('*.' + item for item in file_types)}）", tuple("*." + item for item in file_types))]
                if file_types
                else [("所有文件", "*.*")]
            )
            if multiple:
                selected_paths = list(
                    filedialog.askopenfilenames(
                        parent=root,
                        title=str(arguments.title or "选择文件"),
                        initialdir=initial_dir,
                        filetypes=patterns,
                    )
                )
            else:
                selected_one = filedialog.askopenfilename(
                    parent=root,
                    title=str(arguments.title or "选择文件"),
                    initialdir=initial_dir,
                    filetypes=patterns,
                )
                selected_paths = [selected_one] if selected_one else []
        if not selected_paths:
            status = "canceled"
            message = None
        else:
            candidates = [Path(item).expanduser().resolve() for item in selected_paths]
            valid = (
                all(candidate.is_dir() for candidate in candidates)
                if dialog_type == "folder"
                else all(candidate.is_file() for candidate in candidates)
            )
            if file_types and dialog_type != "folder":
                valid = valid and all(candidate.suffix.lower().lstrip(".") in file_types for candidate in candidates)
            if valid:
                status = "selected"
                selected_paths = [str(candidate) for candidate in candidates]
                message = None
            else:
                message = "所选文件或文件夹不存在、格式不支持或当前无法访问。"
    except Exception as exc:
        message = f"无法打开本地选择窗口：{exc}"
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        _write_result(
            output,
            {
                "request_id": request_id,
                "status": status,
                "path": selected_paths[0] if selected_paths else None,
                "paths": selected_paths if status == "selected" else [],
                "message": message,
            },
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--open-output", required=True)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--initial-dir", default="")
    parser.add_argument("--dialog-type", default="folder")
    parser.add_argument("--multiple", default="false")
    parser.add_argument("--title", default="")
    parser.add_argument("--file-type", action="append", default=[])
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
