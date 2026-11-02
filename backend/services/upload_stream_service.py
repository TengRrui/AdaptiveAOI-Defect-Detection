from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.errors import ApiError


@dataclass(frozen=True)
class StreamedUpload:
    path: Path
    size_bytes: int
    sha256: str


async def stream_upload_to_path(
    upload: Any,
    target: str | Path,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> StreamedUpload:
    """Stream an UploadFile to an atomic, size-bounded local file."""
    destination = Path(target).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.uploading")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("wb") as handle:
            while True:
                chunk = await upload.read(max(64 * 1024, int(chunk_size)))
                if not chunk:
                    break
                written += len(chunk)
                if written > int(max_bytes):
                    raise ApiError(
                        "FILE_TOO_LARGE",
                        "上传文件超过当前系统允许的大小限制。",
                        413,
                        {"max_bytes": int(max_bytes)},
                    )
                digest.update(chunk)
                handle.write(chunk)
            if written <= 0:
                raise ApiError("EMPTY_UPLOAD", "上传文件为空，无法继续处理。", 422)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return StreamedUpload(destination, written, digest.hexdigest())
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(upload, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
