from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


class AtomicJsonStore:
    """Crash-safe JSON storage with a last-known-good backup."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value

    def load(self, *, recover: bool = True) -> dict[str, Any]:
        if self.path.exists():
            try:
                return self._read(self.path)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if recover and self.backup_path.exists():
            value = self._read(self.backup_path)
            self.write(value, preserve_backup=False)
            return value
        return {}

    @staticmethod
    def _flush_file(path: Path, payload: bytes) -> None:
        with path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def write(self, value: dict[str, Any], *, preserve_backup: bool = True) -> dict[str, Any]:
        payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        backup_temporary: Path | None = None
        try:
            if preserve_backup and self.path.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{self.backup_path.name}.", suffix=".tmp", dir=self.path.parent, delete=False
                ) as handle:
                    backup_temporary = Path(handle.name)
                shutil.copyfile(self.path, backup_temporary)
                with backup_temporary.open("r+b") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(backup_temporary, self.backup_path)
                backup_temporary = None

            with tempfile.NamedTemporaryFile(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
            self._flush_file(temporary, payload)
            os.replace(temporary, self.path)
            temporary = None
            return dict(value)
        finally:
            for candidate in (temporary, backup_temporary):
                if candidate is not None:
                    candidate.unlink(missing_ok=True)
