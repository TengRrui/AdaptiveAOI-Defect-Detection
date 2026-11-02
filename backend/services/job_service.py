from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import IntegrationSettings
from backend.services.task_state_service import TaskStateService


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    kind: str
    root: Path
    input_dir: Path
    results_dir: Path
    manifest_path: Path


class JobService:
    def __init__(self, settings: IntegrationSettings):
        self.settings = settings
        self.settings.ensure_runtime_dirs()
        self.states = TaskStateService(self.settings.runtime_root / "task_states.sqlite3")
        recovered = set(self.states.recover_unfinished())
        self._recover_manifests(recovered)

    def _recover_manifests(self, recovered: set[str]) -> None:
        """Reconcile pre-database manifests and make restart interruption explicit."""
        active = {
            "queued", "pending", "initializing", "loading_model", "running",
            "pausing", "paused", "resuming", "stopping", "cancelling",
            "calibrating", "building", "validating", "optimizing",
        }
        # Unfinished rows are already known from SQLite, so reconcile their
        # corresponding manifest directly instead of enumerating the complete
        # job-history directory on every process start.
        for task_id in recovered:
            manifest_path = self.settings.jobs_root / task_id / "job_manifest.json"
            self._reconcile_manifest(manifest_path, force_interrupted=True)

        # The full filesystem walk is only a compatibility migration for jobs
        # created before the durable task index existed.  Once that migration
        # is recorded, subsequent cold starts remain independent of the number
        # of historical job directories.
        if self.states.legacy_manifest_reconciliation_complete():
            return
        indexed_task_ids = self.states.known_task_ids()
        for manifest_path in self.settings.jobs_root.glob("*/job_manifest.json"):
            directory_task_id = str(manifest_path.parent.name)
            if directory_task_id in indexed_task_ids:
                continue
            self._reconcile_manifest(manifest_path, active_statuses=active)
        self.states.mark_legacy_manifest_reconciliation_complete()

    def _reconcile_manifest(
        self,
        manifest_path: Path,
        *,
        force_interrupted: bool = False,
        active_statuses: set[str] | None = None,
    ) -> None:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            job_id = str(payload.get("job_id") or manifest_path.parent.name)
            kind = str(payload.get("kind") or "job")
            raw_status = str(payload.get("status") or "pending").lower()
            if force_interrupted or raw_status in (active_statuses or set()):
                now = utc_now()
                payload.update(
                    status="interrupted",
                    stage="interrupted",
                    updated_at=now,
                    finished_at=now,
                    interruption_reason="service_restart",
                    rerunnable=True,
                )
                manifest_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            self.states.upsert(job_id, kind, payload.get("status"), payload)
        except (OSError, ValueError, json.JSONDecodeError):
            return

    def create(self, kind: str) -> JobPaths:
        job_id = f"{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
        root = self.settings.jobs_root / job_id
        input_dir = root / "input"
        results_dir = root / "results"
        for path in (input_dir, results_dir):
            path.mkdir(parents=True, exist_ok=False)
        job = JobPaths(job_id=job_id, kind=kind, root=root, input_dir=input_dir, results_dir=results_dir, manifest_path=root / "job_manifest.json")
        now = utc_now()
        self.write_manifest(
            job,
            {
                "job_id": job_id,
                "kind": kind,
                "status": "pending",
                "created_at": now,
                "updated_at": now,
                "heartbeat_at": now,
                "owner_process_id": os.getpid(),
            },
        )
        self.states.upsert(job_id, kind, "pending", self.read_manifest(job))
        return job

    def get(self, job_id: str, kind: str | None = None) -> JobPaths:
        root = (self.settings.jobs_root / job_id).resolve()
        root.relative_to(self.settings.jobs_root.resolve())
        manifest_path = root / "job_manifest.json"
        payload: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = {}
        resolved_kind = str(kind or payload.get("kind") or "job")
        return JobPaths(
            job_id=job_id,
            kind=resolved_kind,
            root=root,
            input_dir=root / "input",
            results_dir=root / "results",
            manifest_path=manifest_path,
        )

    def read_manifest(self, job: JobPaths) -> dict[str, Any]:
        if not job.manifest_path.exists():
            return {}
        try:
            payload = json.loads(job.manifest_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def write_manifest(self, job: JobPaths, payload: dict[str, Any]) -> Path:
        job.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        job.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return job.manifest_path

    def update(self, job: JobPaths, **updates: Any) -> dict[str, Any]:
        payload = self.read_manifest(job)
        payload.update(updates)
        now = utc_now()
        payload["updated_at"] = now
        payload.setdefault("owner_process_id", os.getpid())
        if str(payload.get("status") or "").lower() in {
            "queued", "initializing", "loading_model", "running", "pausing", "paused", "resuming",
            "stopping", "cancelling", "calibrating", "building", "validating", "optimizing",
        }:
            payload["heartbeat_at"] = now
        self.write_manifest(job, payload)
        self.states.upsert(job.job_id, job.kind, payload.get("status"), payload)
        return payload

    def fail(self, job: JobPaths, code: str, message: str) -> dict[str, Any]:
        return self.update(job, status="failed", error={"code": code, "message": message})

    def durable_state(self, job_id: str) -> dict[str, Any] | None:
        return self.states.get(job_id)
