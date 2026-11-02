from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest

from backend.atomic_io import AtomicJsonStore
from backend.database import DatabaseManager
from backend.errors import ApiError
from backend.services.manifest_cache import FileManifestCache
from backend.services.upload_stream_service import stream_upload_to_path
from bridge.api_client import client
from bridge.streaming_upload import StreamingMultipartEncoder


class AsyncUpload:
    def __init__(self, value: bytes) -> None:
        self.source = io.BytesIO(value)
        self.closed = False

    async def read(self, size: int) -> bytes:
        return self.source.read(size)

    async def close(self) -> None:
        self.closed = True


def test_database_manager_enforces_foreign_keys_and_rolls_back(tmp_path: Path) -> None:
    manager = DatabaseManager(tmp_path / "state.sqlite3")
    with manager.connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        conn.executescript(
            "CREATE TABLE parent(id TEXT PRIMARY KEY);"
            "CREATE TABLE child(id TEXT PRIMARY KEY,parent_id TEXT NOT NULL REFERENCES parent(id));"
        )
        conn.execute("INSERT INTO parent(id) VALUES('p1')")
        conn.execute("INSERT INTO child(id,parent_id) VALUES('c1','p1')")
    with pytest.raises(Exception):
        with manager.connect() as conn:
            conn.execute("DELETE FROM parent WHERE id='p1'")
    with manager.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM parent").fetchone()[0] == 1
    with pytest.raises(RuntimeError):
        with manager.transaction(immediate=True) as conn:
            conn.execute("INSERT INTO parent(id) VALUES('p2')")
            raise RuntimeError("rollback")
    with manager.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM parent WHERE id='p2'").fetchone()[0] == 0


def test_atomic_json_write_backup_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AtomicJsonStore(tmp_path / "runtime_settings.json")
    store.write({"profile": "a"})
    store.write({"profile": "b"})
    assert store.backup_path.exists()
    store.path.write_text("{broken", encoding="utf-8")
    assert store.load(recover=True) == {"profile": "a"}
    assert json.loads(store.path.read_text(encoding="utf-8")) == {"profile": "a"}

    old = store.path.read_bytes()
    original_replace = __import__("backend.atomic_io", fromlist=["os"]).os.replace

    def fail_primary(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == store.path:
            raise OSError("simulated interruption")
        original_replace(source, destination)

    monkeypatch.setattr("backend.atomic_io.os.replace", fail_primary)
    with pytest.raises(OSError):
        store.write({"profile": "interrupted"})
    assert store.path.read_bytes() == old


def test_stream_upload_is_bounded_and_cleans_failed_partial(tmp_path: Path) -> None:
    target = tmp_path / "upload.bin"
    source = AsyncUpload(b"x" * (3 * 1024 * 1024 + 7))
    result = asyncio.run(stream_upload_to_path(source, target, max_bytes=4 * 1024 * 1024, chunk_size=128 * 1024))
    assert result.size_bytes == target.stat().st_size
    assert source.closed is True
    too_large = AsyncUpload(b"z" * (1024 * 1024 + 1))
    with pytest.raises(ApiError):
        asyncio.run(stream_upload_to_path(too_large, target, max_bytes=1024 * 1024, chunk_size=64 * 1024))
    assert not target.exists()
    assert not target.with_name(f".{target.name}.uploading").exists()


def test_streaming_multipart_does_not_materialize_source() -> None:
    source = io.BytesIO(b"video" * 1000)
    progress: list[tuple[int, int]] = []
    body = StreamingMultipartEncoder(source, filename="测试 视频.mp4", progress=lambda sent, total: progress.append((sent, total)))
    encoded = b"".join(body)
    assert b"filename=" in encoded
    assert encoded.endswith(f"--{body.boundary}--\r\n".encode())
    assert progress[-1] == (5000, 5000)


def test_http_client_and_manifest_hash_are_reused(tmp_path: Path) -> None:
    assert client() is client()
    manager = DatabaseManager(tmp_path / "cache.sqlite3")
    cache = FileManifestCache(manager)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable-content")
    first = cache.digest(artifact)
    second = cache.digest(artifact)
    assert first == second
    assert cache.metrics() == {"hits": 1, "misses": 1, "hit_rate": 0.5}
