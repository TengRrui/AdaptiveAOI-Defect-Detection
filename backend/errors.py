from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Expected client-visible error with a stable code and safe detail payload."""

    def __init__(self, code: str, message: str, status_code: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
