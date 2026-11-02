from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


ProgressCallback = Callable[[int, int], None]


def _file_size(file_object: Any) -> int:
    size = getattr(file_object, "size", None)
    if size is not None:
        return int(size)
    if hasattr(file_object, "getbuffer"):
        return int(file_object.getbuffer().nbytes)
    current = file_object.tell()
    file_object.seek(0, 2)
    size = int(file_object.tell())
    file_object.seek(current)
    return size


class StreamingMultipartEncoder:
    """Minimal requests-compatible multipart body that never copies the file."""

    def __init__(
        self,
        file_object: Any,
        *,
        filename: str,
        fields: Mapping[str, Any] | None = None,
        field_name: str = "file",
        content_type: str | None = None,
        chunk_size: int = 1024 * 1024,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.file_object = file_object
        self.filename = Path(str(filename)).name
        self.fields = {str(key): str(value) for key, value in dict(fields or {}).items()}
        self.field_name = str(field_name)
        self.content_type = content_type or mimetypes.guess_type(self.filename)[0] or "application/octet-stream"
        self.chunk_size = max(64 * 1024, int(chunk_size))
        self.progress = progress
        self.boundary = f"AdaptiveAOI-{uuid.uuid4().hex}"
        self.file_size = _file_size(file_object)
        self._prefix = self._build_prefix()
        self._suffix = f"\r\n--{self.boundary}--\r\n".encode("utf-8")
        self.content_length = len(self._prefix) + self.file_size + len(self._suffix)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": f"multipart/form-data; boundary={self.boundary}",
            "Content-Length": str(self.content_length),
        }

    def __len__(self) -> int:
        return self.content_length

    def _build_prefix(self) -> bytes:
        parts: list[bytes] = []
        for key, value in self.fields.items():
            parts.append(
                f'--{self.boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8")
            )
        escaped_name = self.filename.replace('"', "'")
        parts.append(
            (
                f'--{self.boundary}\r\nContent-Disposition: form-data; name="{self.field_name}"; '
                f'filename="{escaped_name}"\r\nContent-Type: {self.content_type}\r\n\r\n'
            ).encode("utf-8")
        )
        return b"".join(parts)

    def __iter__(self) -> Iterator[bytes]:
        if hasattr(self.file_object, "seek"):
            self.file_object.seek(0)
        yield self._prefix
        sent = 0
        while True:
            chunk = self.file_object.read(self.chunk_size)
            if not chunk:
                break
            sent += len(chunk)
            if self.progress is not None:
                self.progress(sent, self.file_size)
            yield chunk
        yield self._suffix
