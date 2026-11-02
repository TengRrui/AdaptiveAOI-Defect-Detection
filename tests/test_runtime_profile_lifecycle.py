from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.errors import ApiError
from backend.services.config_profile_service import ConfigProfileService
from backend.services.runtime_profile_service import RuntimeProfileService
from backend.services.runtime_resource_service import RuntimeResourceService


PROFILE_ID = "qa_bank_a"
BANK_RESOURCE_ID = "res_111111111111111111111111"
CONFIG_RESOURCE_ID = "res_222222222222222222222222"


class FakeSettings:
    def __init__(self, runtime: dict | None = None):
        self._runtime = dict(runtime or {})
        self.defaults = {"model_id": "model", "dataset": "qa", "category": "part"}

    def dynamic_settings(self) -> dict:
        return dict(self._runtime)

    def update_dynamic_settings(self, updates: dict) -> dict:
        self._runtime.update(updates)
        return dict(self._runtime)


class FakeResourcesForProfile:
    def __init__(self):
        self.apply_calls: list[str] = []
        self.apply_options: list[dict] = []
        self.deactivate_calls: list[str] = []
        self.scan_calls = 0

    def scan(self, force: bool = False) -> None:
        self.scan_calls += int(force)

    @staticmethod
    def startup_catalog_ready() -> bool:
        return False

    def list_resources(self, **_kwargs) -> list[dict]:
        return [{
            "resource_id": "res_" + "a" * 24,
            "profile_id": PROFILE_ID,
            "display_name": "QA A",
            "version": "1",
            "dataset": "qa",
            "category": "part",
            "frozen": True,
            "archived": False,
            "integrity_status": "normal",
            "metadata": {"model_id": "model", "deployment_scope": "formal"},
        }]

    def get_resource(self, resource_id: str) -> dict:
        return self.list_resources()[0]

    def apply_profile(self, resource_id: str, *_args, **_kwargs) -> dict:
        self.apply_calls.append(resource_id)
        self.apply_options.append(dict(_kwargs))
        return {"runtime": {"memory_bank_id": PROFILE_ID}}

    def deactivate_current_profile(self, actor: str) -> dict:
        self.deactivate_calls.append(actor)
        return {"deactivated": True}


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args, **_kwargs):
        return self

    def fetchall(self):
        return []


class FakeRegistry:
    def __init__(self, active_count: int = 0):
        self.active_count = active_count
        self.invalidations = 0

    def status(self) -> dict:
        return {"runtimes": [{"active_count": self.active_count}]}

    def invalidate(self, **_kwargs) -> int:
        if self.active_count:
            raise ApiError("RUNTIME_IN_USE", "busy", 409)
        self.invalidations += 1
        return 1


class FakeDomainProfiles:
    def __init__(self):
        self.deactivated: list[str] = []
        self.activated: list[str] = []

    def deactivate(self, profile_id: str) -> None:
        self.deactivated.append(profile_id)

    def activate(self, profile_id: str, *_args) -> None:
        self.activated.append(profile_id)


class FakeConfigProfiles:
    def __init__(self):
        self.applied: list[tuple[str, int, bool]] = []

    def mark_applied(self, profile_id: str, version: int, applied: bool) -> None:
        self.applied.append((profile_id, version, applied))


class LifecycleSettings(FakeSettings):
    def __init__(self, root: Path):
        super().__init__({"runtime_deactivated": True, "memory_bank_id": ""})
        self.project_root = root
        self.runtime_root = root / "runtime"
        self.jobs_root = self.runtime_root / "jobs"
        self.memory_banks_root = root / "memory_banks"
        self.algorithm_outputs_root = root / "outputs"
        self.config_profiles_root = root / "configs"

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.runtime_root,
            self.jobs_root,
            self.memory_banks_root,
            self.algorithm_outputs_root,
            self.config_profiles_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, value: str | Path | None, default: str) -> Path:
        candidate = Path(value or default)
        return candidate.resolve() if candidate.is_absolute() else (self.project_root / candidate).resolve()


class LifecycleDomainProfiles:
    def __init__(self, root: Path):
        self.root = root
        self.archive_root = root / "_archive"
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._tasks: dict = {}
        self.forgotten: list[str] = []

    def reservations(self, _profile_id: str) -> list[str]:
        return []

    def archive(self, profile_id: str) -> dict:
        source = self.root / profile_id
        target = self.archive_root / profile_id
        shutil.move(str(source), str(target))
        return {"profile_id": profile_id, "archived": True}

    def restore(self, profile_id: str) -> dict:
        source = self.archive_root / profile_id
        target = self.root / profile_id
        shutil.move(str(source), str(target))
        return {"profile_id": profile_id, "archived": False}

    def archived_profile_directory(self, profile_id: str) -> Path:
        return self.archive_root / profile_id

    @staticmethod
    def _read_target_metadata(path: Path, verify_artifacts: bool = False) -> dict:
        assert path.is_dir()
        assert (path / "metadata.json").is_file()
        if verify_artifacts:
            assert (path / "memory_bank.pt").is_file()
        return json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    def forget_archived(self, profile_id: str, *_args, **_kwargs) -> None:
        self.forgotten.append(profile_id)


class LifecycleConfigProfiles:
    def __init__(self, root: Path):
        self.root = root
        self.archive_root = root / "_archive"
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.forgotten: list[tuple[str, int]] = []

    def _version_dir(self, profile_id: str, version: int) -> Path:
        return self.root / profile_id / f"v{version:04d}"

    def archived_version_directory(self, profile_id: str, version: int) -> Path:
        return self.archive_root / profile_id / f"v{version:04d}"

    def archive_version(self, profile_id: str, version: int) -> dict:
        source = self._version_dir(profile_id, version)
        target = self.archived_version_directory(profile_id, version)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return {"config_profile_id": profile_id, "version": version, "archived": True}

    def restore_version(self, profile_id: str, version: int) -> dict:
        source = self.archived_version_directory(profile_id, version)
        target = self._version_dir(profile_id, version)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return {"config_profile_id": profile_id, "version": version, "archived": False}

    def forget_archived(self, profile_id: str, version: int, *_args, **_kwargs) -> None:
        self.forgotten.append((profile_id, version))

    def mark_applied(self, *_args, **_kwargs) -> None:
        return None


def _resource_service(active_count: int = 0) -> RuntimeResourceService:
    service = object.__new__(RuntimeResourceService)
    service.settings = FakeSettings({
        "model_id": "model",
        "dataset": "qa",
        "category": "part",
        "memory_bank_id": PROFILE_ID,
        "config_profile_id": "cfg-a",
        "config_profile_version": 1,
        "threshold": 0.5,
        "runtime_deactivated": False,
    })
    service.runtime_profiles = None
    service.registry = FakeRegistry(active_count)
    service.domain_profiles = FakeDomainProfiles()
    service.config_profiles = FakeConfigProfiles()
    service.logger = SimpleNamespace(exception=lambda *_args, **_kwargs: None)
    service.warmup = SimpleNamespace(status=lambda: {"state": "ready"}, restart=lambda: None)
    service._apply_lock = threading.RLock()
    service._activity_lock = threading.RLock()
    service._active_profile_validations = 0
    service._running_tasks = lambda: []
    service._domain_build_references = lambda: []
    service._connect = lambda: FakeConnection()
    service._record_audit = lambda *_args, **_kwargs: None
    service._bump_revision = lambda **_kwargs: {"revision": 2}
    return service


def test_explicit_deactivation_survives_bootstrap_without_default_reapply():
    settings = FakeSettings({"memory_bank_id": "", "runtime_deactivated": True})
    resources = FakeResourcesForProfile()
    service = RuntimeProfileService(settings, resources, SimpleNamespace(info=lambda *_args, **_kwargs: None))

    current = service.bootstrap()

    assert current["profile_status"] == "missing"
    assert current["explicitly_deactivated"] is True
    assert resources.apply_calls == []


def test_runtime_context_publishes_complete_empty_binding_during_deactivation():
    settings = FakeSettings({
        "model_id": "model",
        "dataset": "qa",
        "category": "part",
        "memory_bank_id": PROFILE_ID,
        "config_profile_id": "cfg-a",
        "config_profile_version": 1,
        "runtime_deactivated": False,
        "runtime_revision": 4,
    })
    service = RuntimeProfileService(settings, FakeResourcesForProfile(), SimpleNamespace())

    runtime = service.persist({
        "model_id": "",
        "dataset": "",
        "category": "",
        "memory_bank_id": "",
        "config_profile_id": "",
        "config_profile_version": 0,
        "runtime_deactivated": True,
    }, runtime_status="deactivating")
    context = service.context_snapshot()

    assert runtime["runtime_revision"] == 5
    assert context["runtime_status"] == "deactivating"
    assert context["current_profile"] is None
    assert context["current_profile_id"] is None
    assert context["current_memory_bank_id"] is None
    assert context["current_config_id"] is None
    assert context["dataset"] is None
    assert context["category"] is None


def test_stale_warmup_cannot_republish_old_profile_after_deactivation():
    settings = FakeSettings({
        "model_id": "model",
        "dataset": "qa",
        "category": "part",
        "memory_bank_id": PROFILE_ID,
        "config_profile_id": "cfg-a",
        "config_profile_version": 1,
        "runtime_deactivated": False,
        "runtime_revision": 9,
    })
    service = RuntimeProfileService(settings, FakeResourcesForProfile(), SimpleNamespace())
    stale = service.context_snapshot()
    service.persist({
        "model_id": "",
        "dataset": "",
        "category": "",
        "memory_bank_id": "",
        "config_profile_id": "",
        "config_profile_version": 0,
        "runtime_deactivated": True,
    }, runtime_status="deactivating")

    changed = service.set_runtime_status(
        "ready",
        expected_revision=int(stale["runtime_revision"]),
        expected_profile_id=str(stale["current_profile_id"]),
    )

    assert changed is False
    assert service.context_snapshot()["runtime_status"] == "deactivating"
    assert service.context_snapshot()["current_profile"] is None


def test_deactivate_clears_complete_binding_and_preserves_resource_identity():
    service = _resource_service()

    result = service.deactivate_current_profile("qa")

    runtime = result["runtime"]
    assert runtime["memory_bank_id"] == ""
    assert runtime["config_profile_id"] == ""
    assert runtime["config_profile_version"] == 0
    assert runtime["threshold"] is None
    assert runtime["runtime_deactivated"] is True
    assert PROFILE_ID in runtime["deactivated_profile_ids"]
    assert service.domain_profiles.deactivated == [PROFILE_ID]
    assert service.config_profiles.applied == [("cfg-a", 1, False)]


def test_deactivate_is_rejected_while_runtime_has_an_active_lease():
    service = _resource_service(active_count=1)
    before = service.settings.dynamic_settings()

    with pytest.raises(ApiError) as error:
        service.deactivate_current_profile("qa")

    assert error.value.code == "RUNTIME_IN_USE"
    assert service.settings.dynamic_settings() == before


def test_deactivate_is_rejected_while_a_detection_task_is_running():
    service = _resource_service()
    service._running_tasks = lambda: [{"job_id": "single-qa", "kind": "single", "status": "running"}]
    before = service.settings.dynamic_settings()

    with pytest.raises(ApiError) as error:
        service.deactivate_current_profile("qa")

    assert error.value.code == "RUNTIME_TASK_ACTIVE"
    assert service.settings.dynamic_settings() == before


def test_can_switch_runtime_excludes_only_the_initiating_feedback_job():
    service = _resource_service()
    service._running_tasks = lambda **_kwargs: [
        {"job_id": "feedback-self", "kind": "feedback_optimization", "status": "applying"},
        {"job_id": "batch-other", "kind": "batch", "status": "running"},
    ]

    blocked = service.can_switch_runtime(exclude_feedback_job_id="feedback-self")

    assert blocked["allowed"] is False
    assert [item["job_id"] for item in blocked["blockers"]] == ["batch-other"]
    assert "批量检测任务正在运行" in blocked["summary"]

    service._running_tasks = lambda **_kwargs: [
        {"job_id": "feedback-self", "kind": "feedback_optimization", "status": "applying"},
    ]
    allowed = service.can_switch_runtime(exclude_feedback_job_id="feedback-self")
    assert allowed["allowed"] is True
    assert allowed["blockers"] == []


def test_ordinary_runtime_switch_never_excludes_feedback_activity():
    service = _resource_service()
    service._running_tasks = lambda **_kwargs: [
        {"job_id": "feedback-self", "kind": "feedback_optimization", "status": "applying"},
    ]

    result = service.can_switch_runtime()

    assert result["allowed"] is False
    assert result["blockers"][0]["job_id"] == "feedback-self"
    assert "反馈候选版本正在应用" in result["summary"]


@pytest.mark.parametrize(
    "activity",
    [
        {"job_id": "feedback-self", "kind": "feedback_optimization", "status": "running"},
        {"job_id": "feedback-self", "kind": "batch", "status": "applying"},
        {"job_id": "different-feedback", "kind": "feedback_optimization", "status": "applying"},
    ],
)
def test_feedback_exclusion_requires_exact_id_kind_and_applying_state(activity: dict):
    service = _resource_service()
    service._running_tasks = lambda **_kwargs: [dict(activity)]

    result = service.can_switch_runtime(exclude_feedback_job_id="feedback-self")

    assert result["allowed"] is False
    assert result["blockers"][0]["job_id"] == activity["job_id"]


def test_runtime_registry_execution_lease_is_a_switch_blocker():
    service = _resource_service(active_count=1)

    result = service.can_switch_runtime()

    assert result["allowed"] is False
    assert any(item["kind"] == "runtime_lease" for item in result["blockers"])
    assert "检测任务正在使用当前运行资源" in result["summary"]


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("single", "单图检测任务正在运行"),
        ("batch", "批量检测任务正在运行"),
        ("video", "视频检测任务正在运行"),
        ("feedback_optimization", "反馈优化任务正在运行"),
    ],
)
def test_each_real_active_task_kind_blocks_runtime_switch(kind: str, message: str):
    service = _resource_service()
    service._running_tasks = lambda **_kwargs: [
        {"job_id": f"{kind}-active", "kind": kind, "status": "running"},
    ]

    result = service.can_switch_runtime()

    assert result["allowed"] is False
    assert message in result["summary"]


def test_domain_build_blocks_until_it_reaches_a_terminal_state():
    service = _resource_service()
    service._domain_build_references = lambda: [
        {"job_id": "domain-active", "kind": "目标域构建", "status": "building"},
    ]
    assert service.can_switch_runtime()["allowed"] is False

    service._domain_build_references = lambda: []
    assert service.can_switch_runtime()["allowed"] is True


def test_stale_job_is_interrupted_and_does_not_block_runtime_switch():
    stale_at = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(timespec="seconds")
    interrupted: list[str] = []

    class FakeStates:
        @staticmethod
        def list(*, statuses):
            assert statuses == {"pending", "running", "paused"}
            return [{
                "task_id": "stale-batch",
                "task_kind": "batch",
                "runtime_status": "running",
                "updated_at": stale_at,
                "payload": {
                    "job_id": "stale-batch",
                    "status": "running",
                    "heartbeat_at": stale_at,
                    "owner_process_id": os.getpid(),
                },
            }]

        @staticmethod
        def mark_interrupted(task_id, _reason):
            interrupted.append(task_id)

    service = _resource_service()
    del service._running_tasks
    service.jobs = SimpleNamespace(states=FakeStates())
    service.settings.jobs_root = None

    result = service.can_switch_runtime()

    assert result["allowed"] is True
    assert interrupted == ["stale-batch"]


def test_runtime_profile_wrapper_forwards_feedback_job_exclusion():
    settings = FakeSettings({"memory_bank_id": PROFILE_ID})
    resources = FakeResourcesForProfile()
    service = RuntimeProfileService(settings, resources, SimpleNamespace())
    service.current_profile = lambda: {"profile_status": "ready"}

    service.apply_profile("res_" + "a" * 24, feedback_job_id="feedback-self")

    assert resources.apply_options[-1]["feedback_job_id"] == "feedback-self"


def test_switching_to_another_memory_bank_replaces_the_complete_runtime_binding():
    service = _resource_service()
    service.test_lock = None
    target_profile_id = "qa_bank_b"
    target_resource = {
        "resource_id": "res_bbbbbbbbbbbbbbbbbbbbbbbb",
        "resource_type": "target_profile",
        "profile_id": target_profile_id,
        "dataset": "qa",
        "category": "part",
        "frozen": True,
        "archived": False,
        "integrity_status": "normal",
        "metadata": {"model_id": "model"},
    }
    service._get = lambda _resource_id: dict(target_resource)
    service.validate_profile = lambda _resource_id: {"valid": True}
    service.domain_profiles.resolve_for_inference = lambda *_args: {
        "memory_bank_id": target_profile_id,
        "model_id": "model",
        "dataset": "qa",
        "category": "part",
        "config_profile_id": "cfg-b",
        "config_profile_version": 2,
        "threshold": 0.7,
    }
    service.config_profiles.get = lambda *_args: {
        "config_profile_id": "cfg-b",
        "version": 2,
        "threshold": 0.7,
        "config_hash": "cfg-b-hash",
        "config": {"device": "cpu"},
    }
    service.config_profiles.compatibility = lambda *_args: {"compatible": True}

    result = service.apply_profile(target_resource["resource_id"], "qa")

    runtime = result["runtime"]
    assert runtime["memory_bank_id"] == target_profile_id
    assert runtime["config_profile_id"] == "cfg-b"
    assert runtime["config_profile_version"] == 2
    assert runtime["threshold"] == 0.7
    assert runtime["runtime_deactivated"] is False
    assert PROFILE_ID in runtime["deactivated_profile_ids"]
    assert target_profile_id not in runtime["deactivated_profile_ids"]
    assert service.config_profiles.applied[-1] == ("cfg-b", 2, True)


def test_memory_bank_must_keep_one_configuration():
    service = object.__new__(ConfigProfileService)
    service.list_for_memory_bank = lambda *_args, **_kwargs: [{
        "config_profile_id": "cfg-a", "version": 1, "archived": False,
    }]

    with pytest.raises(ApiError) as error:
        service.remove_from_memory_bank(PROFILE_ID, "cfg-a", 1)

    assert error.value.code == "LAST_MEMORY_BANK_CONFIG"


def test_archived_memory_bank_can_restore_and_then_delete_owned_configuration(tmp_path: Path):
    settings = LifecycleSettings(tmp_path)
    domain_profiles = LifecycleDomainProfiles(settings.memory_banks_root)
    config_profiles = LifecycleConfigProfiles(settings.config_profiles_root)
    registry = FakeRegistry()
    logger = SimpleNamespace(
        exception=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    service = RuntimeResourceService(settings, config_profiles, domain_profiles, SimpleNamespace(), logger, registry)
    service.scan = lambda force=False: None
    feedback_release_calls = []
    history_revision_calls = []
    service.feedback_candidates = SimpleNamespace(
        mark_source_memory_bank_deleted=lambda **kwargs: (
            feedback_release_calls.append(kwargs)
            or {"affected": 1, "released": 1, "still_protected": 0, "active_jobs": 0}
        )
    )
    service.history_management = SimpleNamespace(
        resource_dependency_changed=lambda reason: history_revision_calls.append(reason) or 7
    )

    bank_dir = settings.memory_banks_root / PROFILE_ID
    bank_dir.mkdir(parents=True)
    (bank_dir / "metadata.json").write_text(json.dumps({"memory_bank_id": PROFILE_ID}), encoding="utf-8")
    (bank_dir / "memory_bank.pt").write_bytes(b"qa-memory")
    config_dir = config_profiles._version_dir("cfg-a", 1)
    config_dir.mkdir(parents=True)
    (config_dir / "profile.yaml").write_text("dataset: qa\ncategory: part\n", encoding="utf-8")
    (config_dir / "metadata.json").write_text(json.dumps({"profile_id": "cfg-a", "version": 1}), encoding="utf-8")

    now = "2026-08-10T00:00:00+00:00"
    with service._connect() as conn:
        conn.execute(
            "INSERT INTO runtime_resources(resource_id,resource_key,resource_type,display_name,version,profile_id,dataset,category,method_mode,artifact_ref,frozen,active,archived,source_type,metadata_json,integrity_status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (BANK_RESOURCE_ID, f"target_profile:{PROFILE_ID}", "target_profile", "QA A", "1", PROFILE_ID, "qa", "part", "normal_memory", str(bank_dir.relative_to(tmp_path)), 1, 0, 0, "target_domain", "{}", "normal", now, now),
        )
        conn.execute(
            "INSERT INTO runtime_resources(resource_id,resource_key,resource_type,display_name,version,profile_id,dataset,category,method_mode,artifact_ref,frozen,active,archived,source_type,metadata_json,integrity_status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (CONFIG_RESOURCE_ID, "configuration:cfg-a:v1", "configuration", "QA Config", "1", "cfg-a:v1", "qa", "part", "normal_memory", str((config_dir / 'profile.yaml').relative_to(tmp_path)), 1, 0, 0, "target_domain_snapshot", "{}", "normal", now, now),
        )
        conn.execute(
            "INSERT INTO runtime_resource_dependencies(relation_id,parent_resource_id,child_resource_id,relation_type,required,created_at) VALUES(?,?,?,?,?,?)",
            ("rel-config", BANK_RESOURCE_ID, CONFIG_RESOURCE_ID, "配置快照", 1, now),
        )

    archived = service.archive(BANK_RESOURCE_ID, actor="qa")
    assert archived["archived"] is True
    assert domain_profiles.archived_profile_directory(PROFILE_ID).is_dir()
    assert config_profiles.archived_version_directory("cfg-a", 1).is_dir()

    restored = service.restore(BANK_RESOURCE_ID, actor="qa")
    assert restored["restored"] is True
    assert bank_dir.is_dir()
    assert config_dir.is_dir()

    service.archive(BANK_RESOURCE_ID, actor="qa")
    preview = service.delete_preview(BANK_RESOURCE_ID)
    assert preview["allowed"] is True
    assert preview["owned_configuration_count"] == 1
    deleted = service.delete(
        BANK_RESOURCE_ID,
        preview["preview_token"],
        actor="qa",
        acknowledged=True,
        reason="isolated lifecycle test",
    )
    assert deleted["deleted"] is True
    assert deleted["feedback_dependency_release"]["released"] == 1
    assert deleted["statistics_revision"] == 7
    assert feedback_release_calls[0]["profile_id"] == PROFILE_ID
    assert feedback_release_calls[0]["connection"] is not None
    assert history_revision_calls == ["memory_bank_deleted_history_protection_recalculated"]
    assert deleted["deleted_configuration_ids"] == [CONFIG_RESOURCE_ID]
    assert domain_profiles.forgotten == [PROFILE_ID]
    assert config_profiles.forgotten == [("cfg-a", 1)]
    with service._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE resource_id IN (?,?)", (BANK_RESOURCE_ID, CONFIG_RESOURCE_ID)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM runtime_resource_tombstones WHERE resource_id=?", (BANK_RESOURCE_ID,)).fetchone()[0] == 1
