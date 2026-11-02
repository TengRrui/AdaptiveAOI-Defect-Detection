from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from backend.errors import ApiError


TEST_ONLY_SCOPES = {"test", "test_only", "development", "development_only"}
FORMAL_PROFILE_INTEGRITY = {"normal", "valid", "passed"}
RUNTIME_BINDING_FIELDS = {
    "model_id",
    "dataset",
    "category",
    "config",
    "device",
    "threshold",
    "memory_bank_id",
    "config_profile_id",
    "config_profile_version",
    "config_hash",
    "inference_mode",
    "frozen",
    "runtime_deactivated",
    "deactivated_at",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def deployment_scope(profile: dict[str, Any] | None) -> str:
    """Return the deployment scope without changing immutable profile files."""

    item = dict(profile or {})
    metadata = dict(item.get("metadata") or {})
    explicit = str(
        metadata.get("deployment_scope")
        or item.get("deployment_scope")
        or metadata.get("profile_scope")
        or ""
    ).strip().lower()
    if explicit in TEST_ONLY_SCOPES:
        return "test_only"
    dataset = str(item.get("dataset") or metadata.get("dataset") or "").strip().lower()
    category = str(item.get("category") or metadata.get("category") or "").strip().lower()
    if (dataset, category) == ("test_data", "test_domain"):
        return "test_only"
    return "formal"


class RuntimeProfileService:
    """The only service allowed to resolve and persist the active runtime profile.

    ``runtime_settings.json`` remains a small durable state file, but callers no
    longer assemble a model/config/memory-bank tuple themselves.  They select a
    complete frozen profile and this service persists the resolved binding.
    """

    def __init__(self, settings: Any, resources: Any, logger: Any, *, test_lock: Any | None = None):
        self.settings = settings
        self.resources = resources
        self.logger = logger
        self.test_lock = test_lock
        self._context_lock = threading.RLock()
        self._context: dict[str, Any] = {}
        initial_runtime = dict(self.settings.dynamic_settings() or {})
        self._context = self._build_context(initial_runtime)

    def raw_runtime(self) -> dict[str, Any]:
        """Return the runtime portion of the single published context."""

        with self._context_lock:
            if self._context:
                return deepcopy(dict(self._context.get("runtime") or {}))
        return dict(self.settings.dynamic_settings() or {})

    def persist(
        self,
        updates: dict[str, Any],
        *,
        runtime_status: str | None = None,
        bump_revision: bool = True,
    ) -> dict[str, Any]:
        """Internal persistence boundary for runtime state.

        Resource identity fields are accepted only from profile application
        paths.  UI-facing code calls ``apply_profile``/``apply_configuration``.
        """

        requested = dict(updates or {})
        with self._context_lock:
            durable = dict(self.settings.dynamic_settings() or {})
            changed_binding = any(key in requested for key in RUNTIME_BINDING_FIELDS)
            revision = int(durable.get("runtime_revision") or 0)
            if bump_revision and changed_binding:
                revision += 1
            requested["runtime_revision"] = revision
            requested["runtime_updated_at"] = _utc_now()
            projected = {**durable, **requested}
            requested["runtime_status"] = str(runtime_status or self._status_for_runtime(projected))
            persisted = self.settings.update_dynamic_settings(requested)
            self._context = self._build_context(persisted, runtime_status=requested["runtime_status"])
            return deepcopy(dict(self._context.get("runtime") or {}))

    @staticmethod
    def _status_for_runtime(runtime: dict[str, Any]) -> str:
        if bool(runtime.get("runtime_deactivated")) and not runtime.get("memory_bank_id"):
            return "inactive"
        if runtime.get("memory_bank_id"):
            return "loading"
        return "not_ready"

    def _profile_state_from_runtime(self, runtime: dict[str, Any]) -> dict[str, Any]:
        current_id = str(runtime.get("memory_bank_id") or "")
        if not current_id:
            explicitly_deactivated = bool(runtime.get("runtime_deactivated"))
            return {
                "profile_status": "missing",
                "status_reason": (
                    "检测服务已启动，当前未加载目标域记忆库。"
                    if explicitly_deactivated
                    else "尚未应用正式冻结 Profile。"
                ),
                "resource": None,
                "runtime": runtime,
                "inference_mode": "inference_only",
                "explicitly_deactivated": explicitly_deactivated,
            }
        item = next(
            (row for row in self.list_profiles() if str(row.get("profile_id") or "") == current_id),
            None,
        )
        if item is None:
            return {
                "profile_status": "invalid",
                "status_reason": "运行设置引用的 Profile 不存在或已归档。",
                "resource": None,
                "runtime": runtime,
                "inference_mode": "inference_only",
            }
        if deployment_scope(item) == "test_only":
            return {
                "profile_status": "invalid",
                "status_reason": "测试 Profile 不能作为正式启动资源。",
                "resource": item,
                "runtime": runtime,
                "inference_mode": "inference_only",
            }
        if not item.get("runtime_eligible"):
            return {
                "profile_status": "invalid",
                "status_reason": "当前 Profile 未冻结、已归档或完整性异常。",
                "resource": item,
                "runtime": runtime,
                "inference_mode": "inference_only",
            }
        return {
            "profile_status": "ready",
            "status_reason": "正式冻结 Profile 已绑定。",
            "resource": item,
            "runtime": runtime,
            "inference_mode": "inference_only",
        }

    @staticmethod
    def _profile_summary(profile_state: dict[str, Any]) -> dict[str, Any] | None:
        resource = dict(profile_state.get("resource") or {})
        runtime = dict(profile_state.get("runtime") or {})
        if not resource or not runtime.get("memory_bank_id"):
            return None
        return {
            "profile_id": resource.get("profile_id") or runtime.get("memory_bank_id"),
            "name": resource.get("display_name") or resource.get("profile_id"),
            "version": resource.get("version"),
            "memory_bank_id": resource.get("profile_id") or runtime.get("memory_bank_id"),
            "config_profile_id": runtime.get("config_profile_id"),
            "config_profile_version": runtime.get("config_profile_version"),
            "dataset": resource.get("dataset") or runtime.get("dataset"),
            "category": resource.get("category") or runtime.get("category"),
            "deployment_scope": resource.get("deployment_scope") or (resource.get("metadata") or {}).get("deployment_scope"),
            "frozen": bool(resource.get("frozen")),
        }

    def _build_context(self, runtime: dict[str, Any], *, runtime_status: str | None = None) -> dict[str, Any]:
        runtime_copy = deepcopy(dict(runtime or {}))
        state = self._profile_state_from_runtime(runtime_copy)
        resource = deepcopy(state.get("resource")) if state.get("resource") else None
        status = str(runtime_status or runtime_copy.get("runtime_status") or self._status_for_runtime(runtime_copy))
        metadata = dict((resource or {}).get("metadata") or {})
        profile_id = str((resource or {}).get("profile_id") or runtime_copy.get("memory_bank_id") or "") or None
        profile_version = str((resource or {}).get("version") or metadata.get("profile_version") or "") or None
        model_id = str(runtime_copy.get("model_id") or metadata.get("model_id") or "") or None
        method_mode = str(
            (resource or {}).get("method_mode")
            or metadata.get("method_mode")
            or metadata.get("method_mode_resolved")
            or ""
        ) or None
        algorithm_mode = str(metadata.get("algorithm_mode") or method_mode or "") or None
        return {
            # Canonical inference binding.  The shorter aliases are consumed by
            # new clients; the ``current_*`` names remain for compatibility.
            "profile_id": profile_id,
            "profile_version": profile_version,
            "memory_bank_id": str(runtime_copy.get("memory_bank_id") or "") or None,
            "config_profile_id": str(runtime_copy.get("config_profile_id") or "") or None,
            "config_version": int(runtime_copy.get("config_profile_version") or 0) or None,
            "model_id": model_id,
            "model_version": model_id,
            "method_mode": method_mode,
            "algorithm_mode": algorithm_mode,
            "requested_precision": metadata.get("requested_precision") or metadata.get("precision"),
            "actual_precision": metadata.get("actual_precision"),
            "normal_memory_enabled": bool(metadata.get("normal_memory_enabled", bool(profile_id))),
            "lightweight_calibration_enabled": bool(
                metadata.get("lightweight_calibration_enabled", metadata.get("lightweight_calibration", False))
            ),
            "defect_guidance_enabled": bool(
                metadata.get("defect_guidance_enabled", metadata.get("lightweight_calibration", False))
            ),
            "status": status,
            "updated_at": str(runtime_copy.get("runtime_updated_at") or "") or None,
            "current_profile_id": profile_id,
            "current_memory_bank_id": str(runtime_copy.get("memory_bank_id") or "") or None,
            "current_config_id": str(runtime_copy.get("config_profile_id") or "") or None,
            "current_config_version": int(runtime_copy.get("config_profile_version") or 0) or None,
            "dataset": str(runtime_copy.get("dataset") or "") or None,
            "category": str(runtime_copy.get("category") or "") or None,
            "runtime_status": status,
            "runtime_revision": int(runtime_copy.get("runtime_revision") or 0),
            "profile_status": str(state.get("profile_status") or "missing"),
            "status_reason": str(state.get("status_reason") or ""),
            "explicitly_deactivated": bool(state.get("explicitly_deactivated")),
            "current_profile": self._profile_summary(state),
            "resource": resource,
            "runtime": runtime_copy,
        }

    def refresh_context(self, *, runtime_status: str | None = None) -> dict[str, Any]:
        """Re-resolve the durable binding and publish one coherent snapshot."""

        with self._context_lock:
            runtime = dict(self.settings.dynamic_settings() or {})
            self._context = self._build_context(runtime, runtime_status=runtime_status)
            return deepcopy(self._context)

    def context_snapshot(self) -> dict[str, Any]:
        with self._context_lock:
            return deepcopy(self._context)

    def set_runtime_status(
        self,
        status: str,
        *,
        expected_revision: int | None = None,
        expected_profile_id: str | None = None,
    ) -> bool:
        """Update lifecycle state only when the caller still owns the context."""

        with self._context_lock:
            current_revision = int(self._context.get("runtime_revision") or 0)
            current_profile_id = str(self._context.get("current_profile_id") or "")
            if expected_revision is not None and current_revision != int(expected_revision):
                return False
            if expected_profile_id is not None and current_profile_id != str(expected_profile_id or ""):
                return False
            persisted = self.settings.update_dynamic_settings(
                {
                    "runtime_status": str(status),
                    "runtime_revision": current_revision,
                    "runtime_updated_at": _utc_now(),
                }
            )
            self._context = self._build_context(persisted, runtime_status=str(status))
            return True

    @staticmethod
    def _decorate(item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        metadata = dict(result.get("metadata") or {})
        scope = deployment_scope(result)
        metadata["deployment_scope"] = scope
        metadata.setdefault("official_test_mode", False)
        metadata["test_data_update_allowed"] = False
        if scope == "test_only":
            metadata["is_default"] = False
        result["metadata"] = metadata
        result["deployment_scope"] = scope
        result["runtime_eligible"] = bool(
            scope == "formal"
            and result.get("frozen")
            and not result.get("archived")
            and str(result.get("integrity_status") or "").lower() in FORMAL_PROFILE_INTEGRITY
        )
        return result

    def list_profiles(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        return [
            self._decorate(item)
            for item in self.resources.list_resources(
                resource_type="target_profile",
                include_archived=include_archived,
            )
        ]

    def current_profile(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = deepcopy(context) if context is not None else self.context_snapshot()
        return {
            "profile_status": context.get("profile_status"),
            "status_reason": context.get("status_reason"),
            "resource": context.get("resource"),
            "runtime": context.get("runtime") or {},
            "inference_mode": "inference_only",
            "explicitly_deactivated": bool(context.get("explicitly_deactivated")),
            "runtime_status": context.get("runtime_status"),
            "runtime_revision": context.get("runtime_revision"),
        }

    def _formal_default(self) -> dict[str, Any] | None:
        profiles = [item for item in self.list_profiles() if item.get("runtime_eligible")]
        if not profiles:
            return None
        defaults = self.settings.defaults
        preferred = next(
            (
                item
                for item in profiles
                if str(item.get("dataset") or "").lower() == str(defaults.get("dataset") or "").lower()
                and str(item.get("category") or "") == str(defaults.get("category") or "")
                and str((item.get("metadata") or {}).get("model_id") or "") == str(defaults.get("model_id") or "")
            ),
            None,
        )
        if preferred is not None:
            return preferred
        marked = [item for item in profiles if bool((item.get("metadata") or {}).get("is_default"))]
        candidates = marked or profiles
        return sorted(candidates, key=lambda item: str(item.get("created_at") or ""), reverse=True)[0]

    def bootstrap(self) -> dict[str, Any]:
        """Converge stale/test bindings before model warm-up starts."""

        if self.resources.startup_catalog_ready():
            self.logger.info("runtime_profile_bootstrap_catalog_reused scope=active_formal_graph")
        else:
            # The active graph is missing or its fast mtime/size signature is
            # stale.  Only this exceptional path pays the full authoritative
            # catalog scan before warm-up; archived/candidate resources stay
            # outside the Runtime Ready critical path.
            self.logger.info("runtime_profile_bootstrap_catalog_refresh reason=active_graph_stale")
            self.resources.scan(force=True)
        self.refresh_context()
        current = self.current_profile()
        if current.get("profile_status") == "ready":
            resource = current.get("resource") or {}
            self.logger.info(
                "runtime_profile_bootstrap_ready name=%s version=%s memory_bank=%s config=%s:v%s status=ready",
                resource.get("display_name") or resource.get("profile_id"),
                resource.get("version") or "1",
                resource.get("profile_id") or "",
                current.get("runtime", {}).get("config_profile_id") or "",
                current.get("runtime", {}).get("config_profile_version") or "",
            )
            return current
        # A user initiated deactivation is durable runtime state, not a broken
        # binding that startup recovery should silently replace.  This is what
        # makes "停用" survive a backend or computer restart.
        if bool((current.get("runtime") or {}).get("runtime_deactivated")):
            self.logger.info("runtime_profile_bootstrap_explicitly_deactivated")
            return current
        fallback = self._formal_default()
        if fallback is None:
            self.persist(
                {
                    "memory_bank_id": "",
                    "config_profile_id": "",
                    "config_profile_version": 0,
                    "config_hash": "",
                    "threshold": None,
                    "inference_mode": "inference_only",
                    "runtime_deactivated": False,
                }
            )
            self.logger.error("runtime_profile_bootstrap_missing_formal_profile")
            return self.current_profile()
        previous = current.get("resource") or {}
        if deployment_scope(previous) == "test_only":
            self.logger.warning(
                "runtime_profile_test_binding_rejected profile=%s dataset=%s category=%s",
                previous.get("profile_id"),
                previous.get("dataset"),
                previous.get("category"),
            )
        self.apply_profile(str(fallback["resource_id"]), actor="startup_recovery")
        return self.current_profile()

    def apply_profile(
        self,
        resource_id: str,
        actor: str = "operator",
        *,
        config_profile_id: str | None = None,
        config_profile_version: int | None = None,
        feedback_job_id: str | None = None,
    ) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("profile_apply")
        item = self._decorate(self.resources.get_resource(resource_id))
        if deployment_scope(item) == "test_only":
            raise ApiError(
                "TEST_PROFILE_NOT_APPLICABLE",
                "测试 Profile 已隔离，不能设为当前正式运行方案。",
                409,
                {"deployment_scope": "test_only", "profile_id": item.get("profile_id")},
            )
        result = self.resources.apply_profile(
            resource_id,
            actor,
            config_profile_id=config_profile_id,
            config_profile_version=config_profile_version,
            feedback_job_id=feedback_job_id,
        )
        current = self.current_profile()
        if current.get("profile_status") != "ready":
            raise ApiError("RUNTIME_PROFILE_APPLY_INCONSISTENT", "运行方案写入后未形成一致绑定，已拒绝继续推理。", 500)
        return {**result, "current": current}

    def deactivate(self, actor: str = "operator") -> dict[str, Any]:
        """Stop new inference without deleting or archiving any resource."""

        result = self.resources.deactivate_current_profile(actor)
        current = self.current_profile()
        if current.get("profile_status") != "missing" or not current.get("explicitly_deactivated"):
            raise ApiError("RUNTIME_PROFILE_DEACTIVATE_INCONSISTENT", "运行方案停用后状态未能保持一致。", 500)
        return {**result, "current": current}

    def apply_configuration(self, updates: dict[str, Any], actor: str = "runtime_config_ui") -> dict[str, Any]:
        requested = dict(updates or {})
        current = self.current_profile()
        bank_id = str(requested.get("memory_bank_id") or current.get("runtime", {}).get("memory_bank_id") or "")
        item = next((row for row in self.list_profiles() if str(row.get("profile_id") or "") == bank_id), None)
        if item is None:
            raise ApiError("RUNTIME_PROFILE_MISSING", "所选完整运行 Profile 不存在或已归档。", 409)
        for key in ("dataset", "category", "model_id"):
            supplied = str(requested.get(key) or "")
            expected = str(
                (item.get("metadata") or {}).get("model_id")
                if key == "model_id"
                else item.get(key)
                or ""
            )
            if supplied and supplied.lower() != expected.lower():
                raise ApiError("RUNTIME_PROFILE_BINDING_MISMATCH", "运行配置必须整体切换，不能拼接其他 Profile 的资源字段。", 409)
        result = self.apply_profile(
            str(item["resource_id"]),
            actor,
            config_profile_id=requested.get("config_profile_id"),
            config_profile_version=requested.get("config_profile_version"),
        )
        mutable = {key: requested[key] for key in ("save_visuals", "save_original") if key in requested}
        if mutable:
            result["runtime"] = self.persist(mutable)
        return result

    def rollback(self, actor: str = "operator") -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("profile_rollback")
        result = self.resources.rollback(actor)
        current = self.current_profile()
        if current.get("profile_status") != "ready":
            raise ApiError("RUNTIME_ROLLBACK_INCONSISTENT", "回滚结果不是可用的正式冻结 Profile。", 409)
        return {**result, "current": current}

    def restore_default(self, actor: str = "operator") -> dict[str, Any]:
        default = self._formal_default()
        if default is None:
            raise ApiError("DEFAULT_PROFILE_NOT_FOUND", "系统没有可验证的默认冻结方案。", 409)
        return self.apply_profile(str(default["resource_id"]), actor)

    def test_lock_state(self) -> dict[str, Any]:
        runtime = self.raw_runtime()
        state = self.test_lock.state() if self.test_lock is not None else {
            "locked": bool(runtime.get("official_test_mode")),
            "official_test_mode": bool(runtime.get("official_test_mode")),
            "test_only_source": False,
            "test_data_update_allowed": False,
            "inference_only": True,
        }
        return {**state, "profile_status": self.current_profile().get("profile_status")}

    def set_official_test_mode(self, enabled: bool, actor: str = "operator") -> dict[str, Any]:
        """Atomically enter or leave the evaluation write lock.

        Disabling the lock never enables test-data learning; it only restores
        offline construction/calibration using explicitly non-test data.
        """

        if bool(enabled) and self.current_profile().get("profile_status") != "ready":
            raise ApiError("OFFICIAL_TEST_PROFILE_REQUIRED", "开启正式测试锁前必须应用完整、冻结且有效的正式 Profile。", 409)
        before = self.raw_runtime()
        after = self.persist(
            {
                "official_test_mode": bool(enabled),
                "test_data_update_allowed": False,
                "feedback_optimization_locked": bool(enabled),
                "evaluation_mode": "official_test" if enabled else "calibration_only",
            }
        )
        self.logger.info(
            "official_test_lock_changed enabled=%s actor=%s profile=%s",
            bool(enabled),
            actor,
            after.get("memory_bank_id") or before.get("memory_bank_id") or "",
        )
        return {"before": before, "runtime": after, "test_lock": self.test_lock_state()}

    def configuration(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = deepcopy(context) if context is not None else self.context_snapshot()
        current = self.current_profile(context)
        runtime = dict(current.get("runtime") or {})
        item = current.get("resource") or {}
        if current.get("profile_status") != "ready":
            return {
                "dataset": "",
                "category": "",
                "config": "",
                "device": str(runtime.get("device") or "auto"),
                "model_id": "",
                "memory_bank_id": "",
                "threshold": None,
                "inference_mode": "inference_only",
                "frozen": False,
                "memory_bank_ready": False,
                "threshold_ready": False,
                "memory_bank_profile": {},
                "config_profile_id": "",
                "config_profile_version": None,
                "config_hash": "",
                "config_profile": {},
                "config_profile_name": "未配置正式运行方案",
                "profile_status": current.get("profile_status"),
                "profile_status_reason": current.get("status_reason"),
            }
        bank = self.resources.domain_profiles.resolve_for_inference(
            str(item.get("profile_id") or ""),
            str((item.get("metadata") or {}).get("model_id") or ""),
            str(item.get("dataset") or ""),
            str(item.get("category") or ""),
        )
        config_id = str(runtime.get("config_profile_id") or bank.get("config_profile_id") or "")
        config_version = int(runtime.get("config_profile_version") or bank.get("config_profile_version") or 0)
        config_profile = self.resources.config_profiles.get(config_id, config_version)
        values = dict(config_profile.get("config") or {})
        threshold = config_profile.get("threshold") if config_profile.get("threshold") is not None else bank.get("threshold")
        normal_count = int(bank.get("normal_sample_count") or 0)
        defect_count = int(bank.get("defect_calibration_sample_count") or 0)
        return {
            "dataset": str(bank.get("dataset") or "").lower(),
            "category": str(bank.get("category") or ""),
            "config": str(values.get("config") or runtime.get("config") or ""),
            "device": str(values.get("device") or runtime.get("device") or "auto"),
            "model_id": str(bank.get("model_id") or values.get("model_id") or ""),
            "memory_bank_id": str(bank.get("memory_bank_id") or ""),
            "threshold": threshold,
            "inference_mode": "inference_only",
            "official_test_mode": bool(runtime.get("official_test_mode")),
            "test_data_update_allowed": False,
            "evaluation_mode": str(runtime.get("evaluation_mode") or ("official_test" if runtime.get("official_test_mode") else "calibration_only")),
            "feedback_optimization_locked": bool(runtime.get("feedback_optimization_locked")),
            "frozen": bool(bank.get("frozen")),
            "memory_bank_ready": bool(bank.get("frozen") and threshold is not None),
            "threshold_ready": threshold is not None,
            "memory_bank_profile": bank,
            "evaluation_readiness": {
                "required_normal_sample_count": 100,
                "required_defect_sample_count": 30,
                "normal_sample_count": normal_count,
                "defect_sample_count": defect_count,
                "sample_budget_ready": normal_count == 100 and defect_count == 30,
                "profile_frozen": bool(bank.get("frozen")),
                "threshold_ready": threshold is not None,
                "ready": bool(bank.get("frozen") and threshold is not None and normal_count == 100 and defect_count == 30),
            },
            "config_profile_id": config_id,
            "config_profile_version": config_version,
            "config_hash": str(config_profile.get("config_hash") or bank.get("config_hash") or ""),
            "config_profile": config_profile,
            "config_profile_name": str(config_profile.get("friendly_name") or "当前配置快照"),
            "profile_status": "ready",
            "profile_status_reason": current.get("status_reason"),
            "deployment_scope": "formal",
        }


__all__ = ["RuntimeProfileService", "deployment_scope"]
