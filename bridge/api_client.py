from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse
import json

import requests
from requests.adapters import HTTPAdapter

from bridge.config_loader import load_software_config
from bridge.streaming_upload import ProgressCallback, StreamingMultipartEncoder
from core.domain_build_contract import DEFAULT_CREATE_MODE, normalize_precision_request, precision_uses_fp16


DEFAULT_TIMEOUT_SECONDS = 600
STATUS_TIMEOUT_SECONDS = 1.5
FEEDBACK_STATUS_TIMEOUT_SECONDS = 3.0
READ_MAX_ATTEMPTS = 3
READ_RETRY_BACKOFF_SECONDS = (0.2, 0.6)

# These are the wire-level values accepted by
# ``backend.api.schemas.FeedbackOptimizationCreateRequest``.  Keep the
# display labels in the Streamlit page separate from these codes so a
# translated label can never accidentally be sent as an enum value.
FEEDBACK_OPTIMIZATION_STRATEGY_CODES = frozenset(
    {
        "safe_incremental",
        "comprehensive",
        "threshold_only",
        "normal_memory",
        "defect_prototypes",
        "lightweight_calibrator",
    }
)
FEEDBACK_OPTIMIZATION_PUBLISH_MODE_CODES = frozenset(
    {"replace_original", "update_current", "save_new_version"}
)
FEEDBACK_OPTIMIZATION_OBJECTIVE_CODES = frozenset({"f1", "recall_priority", "balanced"})


_LOGGER = logging.getLogger("adaptive_aoi.frontend.api")


class IntegrationApiClient:
    """Small Streamlit-side client for the AdaptiveAOI integration API."""

    def __init__(self, base_url: str | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        config = load_software_config()
        self.base_url = str(base_url or config.get("integration_api_url") or "http://127.0.0.1:8000").rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0, pool_block=False)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    @staticmethod
    def _domain_build_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["method_mode"] = str(normalized.get("method_mode") or DEFAULT_CREATE_MODE)
        precision = normalize_precision_request(normalized, default="fp16")
        normalized["precision"] = precision
        normalized["use_fp16"] = precision_uses_fp16(precision)
        return normalized

    def artifact_url(self, path: str | None) -> str | None:
        if not path:
            return None
        if str(path).startswith(("http://", "https://")):
            return str(path)
        return self._url(str(path))

    def artifact_bytes(self, path: str | None, *, max_bytes: int = 50 * 1024 * 1024) -> dict[str, Any]:
        """Read a controlled local artifact without exposing its server path."""
        url = self.artifact_url(path)
        if not url:
            return {"success": False, "error": {"code": "ARTIFACT_NOT_FOUND", "message": "样本图像不可用。"}}
        base = urlparse(self.base_url)
        target = urlparse(url)
        if (target.scheme, target.hostname, target.port) != (base.scheme, base.hostname, base.port):
            return {"success": False, "error": {"code": "ARTIFACT_ORIGIN_INVALID", "message": "样本图像来源无效。"}}
        try:
            response = self.session.get(url, timeout=(2.0, 30.0))
            response.raise_for_status()
        except requests.RequestException:
            return {"success": False, "error": {"code": "ARTIFACT_UNAVAILABLE", "message": "样本图像暂时无法读取。"}}
        content = response.content
        if len(content) > max(1, int(max_bytes)):
            return {"success": False, "error": {"code": "ARTIFACT_TOO_LARGE", "message": "样本图像超过编辑器允许的大小。"}}
        return {
            "success": True,
            "content": content,
            "content_type": str(response.headers.get("content-type") or "application/octet-stream"),
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        retry_read: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call the local API without leaking transport exceptions to Streamlit.

        Only explicitly marked, idempotent reads are retried.  Detection,
        feedback and profile-mutating requests therefore remain single-shot,
        so a slow or restarting service cannot create duplicate tasks.
        """
        method = str(method or "GET").upper()
        request_timeout = float(timeout if timeout is not None else self.timeout)
        is_idempotent_read = method in {"GET", "HEAD", "OPTIONS"}
        attempts = READ_MAX_ATTEMPTS if retry_read and is_idempotent_read else 1
        response = None
        started_at = time.perf_counter()
        last_exception: requests.RequestException | None = None
        for attempt in range(attempts):
            try:
                # Keep connection establishment short while allowing a long
                # read for genuine inference/statistics requests.
                response = self.session.request(
                    method,
                    self._url(path),
                    timeout=(min(2.0, max(0.2, request_timeout)), request_timeout),
                    **kwargs,
                )
                if response.status_code not in {502, 503, 504} or attempt >= attempts - 1:
                    break
            except requests.RequestException as exc:
                last_exception = exc
                if attempt >= attempts - 1:
                    _LOGGER.debug("AdaptiveAOI API request failed method=%s path=%s: %s", method, path, exc)
                    if isinstance(exc, requests.ReadTimeout) and is_idempotent_read:
                        code = "API_STATUS_TIMEOUT"
                        message = "状态刷新稍慢，请稍后重试。"
                    elif isinstance(exc, requests.ReadTimeout):
                        code = "API_REQUEST_TIMEOUT"
                        message = "请求超时，系统未自动重试；请先刷新任务状态，确认后再操作。"
                    elif isinstance(exc, (requests.ConnectTimeout, requests.ConnectionError)):
                        code = "API_CONNECTION_LOST"
                        message = "后台服务连接中断。"
                    else:
                        code = "API_TRANSPORT_ERROR"
                        message = "后台服务通信异常，请稍后重试。"
                    return {
                        "success": False,
                        "error": {
                            "code": code,
                            "message": message,
                            "retryable": bool(is_idempotent_read),
                        },
                        "transport": {
                            "attempts": attempt + 1,
                            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                            "exception": type(exc).__name__,
                        },
                    }
            if attempt < attempts - 1:
                time.sleep(READ_RETRY_BACKOFF_SECONDS[min(attempt, len(READ_RETRY_BACKOFF_SECONDS) - 1)])
        if response is None:
            return {
                "success": False,
                "error": {"code": "API_UNAVAILABLE", "message": "算法服务正在启动，请稍候。", "retryable": True},
                "transport": {
                    "attempts": attempts,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                    "exception": type(last_exception).__name__ if last_exception else None,
                },
            }
        try:
            payload = response.json()
        except ValueError:
            return {
                "success": False,
                "error": {"code": "INVALID_API_RESPONSE", "message": "算法服务暂时返回了无法读取的响应，请稍后重试。", "retryable": response.status_code in {502, 503, 504}},
            }
        if not response.ok and payload.get("success", True):
            payload["success"] = False
        payload.setdefault("transport", {
            "attempts": attempt + 1,
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
            "http_status": response.status_code,
        })
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health", timeout=STATUS_TIMEOUT_SECONDS, retry_read=True)

    def liveness(self) -> dict[str, Any]:
        return self._request("GET", "/api/health/live", timeout=STATUS_TIMEOUT_SECONDS, retry_read=True)

    def readiness(self) -> dict[str, Any]:
        return self._request("GET", "/api/health/ready", timeout=STATUS_TIMEOUT_SECONDS, retry_read=True)

    def retry_readiness(self) -> dict[str, Any]:
        return self._request("POST", "/api/health/ready/retry", timeout=STATUS_TIMEOUT_SECONDS)

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/api/models")

    def load_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/models/load", json=payload)

    def config(self, *, timeout: float = STATUS_TIMEOUT_SECONDS) -> dict[str, Any]:
        return self._request("GET", "/api/config", timeout=timeout, retry_read=True)

    def runtime_context(self, *, timeout: float = STATUS_TIMEOUT_SECONDS) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/context", timeout=timeout, retry_read=True)

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", "/api/config", json=payload)

    def detect_upload(
        self,
        uploaded_file: Any,
        runtime: dict[str, Any],
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        filename = str(getattr(uploaded_file, "name", "upload.png"))
        data = {
            **self._inference_binding_fields(runtime),
            "generate_visualizations": str(bool(runtime.get("generate_visualizations", True))).lower(),
            "mode": "inference_only",
        }
        body = StreamingMultipartEncoder(uploaded_file, filename=filename, fields=data, progress=progress)
        return self._request("POST", "/api/detect", data=body, headers=body.headers)

    def detect_directory(self, input_dir: str, runtime: dict[str, Any], *, async_job: bool = False) -> dict[str, Any]:
        data = {
            "input_dir": str(input_dir),
            **self._inference_binding_fields(runtime),
            "generate_visualizations": str(bool(runtime.get("generate_visualizations", True))).lower(),
            "mode": "inference_only",
            "async_job": str(bool(async_job)).lower(),
        }
        return self._request("POST", "/api/detect/batch", data=data, timeout=10.0 if async_job else None)

    @staticmethod
    def _inference_binding_fields(runtime: dict[str, Any]) -> dict[str, str]:
        """Serialize one coherent runtime snapshot.

        New clients send only the optimistic-lock identity.  The API resolves
        model, Memory Bank and Config from its authoritative RuntimeContext.
        The full-field branch remains only for legacy callers that do not yet
        understand runtime revisions.
        """

        context = dict(runtime.get("runtime_context") or runtime)
        profile_id = str(context.get("profile_id") or context.get("current_profile_id") or "")
        revision = context.get("runtime_revision")
        if profile_id and revision is not None:
            return {"profile_id": profile_id, "runtime_revision": str(int(revision))}
        return {
            "model_id": str(runtime.get("model_id") or ""),
            "dataset": str(runtime.get("dataset") or ""),
            "category": str(runtime.get("category") or ""),
            "device": str(runtime.get("device") or "auto"),
            "memory_bank_id": str(runtime.get("memory_bank_id") or ""),
            "config_profile_id": str(runtime.get("config_profile_id") or ""),
            "config_profile_version": str(runtime.get("config_profile_version") or ""),
        }

    def job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/jobs/{job_id}", timeout=3.0, retry_read=True)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/jobs/{job_id}/cancel", timeout=3.0)

    def rerun_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/jobs/{job_id}/rerun", timeout=5.0)

    def start_folder_dialog(
        self,
        initial_dir: str = "",
        *,
        dialog_type: str = "folder",
        multiple: bool = False,
        file_types: list[str] | None = None,
        recent_key: str | None = None,
        fallback_dirs: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/local/folder-dialog",
            timeout=3.0,
            json={
                "initial_dir": str(initial_dir or ""),
                "dialog_type": str(dialog_type or "folder"),
                "multiple": bool(multiple),
                "file_types": list(file_types or []),
                "recent_key": str(recent_key or "") or None,
                "fallback_dirs": [str(path) for path in (fallback_dirs or []) if str(path)],
            },
        )

    def folder_dialog_status(self, dialog_id: str, *, consume: bool = True) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/local/folder-dialog/{dialog_id}",
            timeout=2.0,
            retry_read=True,
            params={"consume": str(bool(consume)).lower()},
        )

    def cancel_folder_dialog(self, dialog_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/local/folder-dialog/{dialog_id}/cancel",
            timeout=2.0,
        )

    def remember_folder_dialog_directory(self, recent_key: str, directory: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/local/folder-dialog/recent-directory",
            timeout=3.0,
            json={"recent_key": str(recent_key), "directory": str(directory)},
        )

    def domain_adaptation_defaults(self, *, model_id: str, dataset: str, category: str, device: str) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/domain-adaptation/defaults",
            params={"model_id": model_id, "dataset": dataset, "category": category, "device": device},
        )

    def scan_domain_samples(self, normal_dir: str = "", defect_dir: str = "", **kwargs: Any) -> dict[str, Any]:
        payload = {"normal_dir": normal_dir, "defect_dir": defect_dir, **kwargs}
        return self._request(
            "POST",
            "/api/domain-adaptation/scan",
            json=payload,
        )

    def check_domain_build(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/domain-adaptation/build-readiness", json=self._domain_build_payload(payload))

    def domain_build_readiness(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/domain-adaptation/build-readiness", json=self._domain_build_payload(payload))

    def domain_drafts(self) -> dict[str, Any]:
        return self._request("GET", "/api/domain-adaptation/drafts")

    def save_domain_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["payload"] = self._domain_build_payload(dict(normalized.get("payload") or {}))
        return self._request("PUT", "/api/domain-adaptation/drafts", json=normalized)

    def domain_draft_delete_preview(self, draft_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/domain-adaptation/drafts/{draft_id}/delete-preview", timeout=20.0)

    def delete_domain_draft(self, draft_id: str, *, acknowledged: bool = False, plan_id: str = "", preview_token: str = "") -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/api/domain-adaptation/drafts/{draft_id}",
            params={"acknowledged": bool(acknowledged), "plan_id": plan_id, "preview_token": preview_token},
        )

    def domain_feedback_candidates(self) -> dict[str, Any]:
        return self._request("GET", "/api/domain-adaptation/feedback-candidates")

    def feedback_correction_pool(
        self,
        *,
        memory_bank_id: str | None = None,
        source_profile_id: str | None = None,
        source_target_domain_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/feedback/correction-pool/available",
            params={
                key: value
                for key, value in {
                    "memory_bank_id": memory_bank_id,
                    "source_profile_id": source_profile_id,
                    "source_target_domain_id": source_target_domain_id,
                }.items()
                if value
            },
            retry_read=True,
        )

    def preview_domain_videos(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/domain-adaptation/video-preview", json=dict(payload))

    def start_domain_adaptation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/domain-adaptation/jobs", json=self._domain_build_payload(payload), timeout=5.0)

    def domain_adaptation_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/domain-adaptation/jobs/{job_id}", timeout=3.0)

    def current_domain_adaptation_job(self) -> dict[str, Any]:
        return self._request("GET", "/api/domain-adaptation/jobs/current", timeout=3.0, retry_read=True)

    def cancel_domain_adaptation(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/domain-adaptation/jobs/{job_id}/cancel", timeout=3.0)

    def rerun_domain_adaptation(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/domain-adaptation/jobs/{job_id}/rerun", timeout=5.0)

    def domain_candidate(self, profile_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/domain-adaptation/candidates/{profile_id}", timeout=5.0, retry_read=True)

    def create_domain_parameter_version(self, profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/domain-adaptation/profiles/{profile_id}/parameter-version",
            json=dict(payload),
            timeout=120.0,
        )

    def validate_domain_candidate(self, profile_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/domain-adaptation/candidates/{profile_id}/validate", timeout=10.0)

    def freeze_domain_candidate(self, profile_id: str, *, confirm: bool = True) -> dict[str, Any]:
        return self._request("POST", f"/api/domain-adaptation/candidates/{profile_id}/freeze", json={"confirm": confirm}, timeout=15.0)

    def apply_domain_candidate(self, profile_id: str, *, confirm: bool = True) -> dict[str, Any]:
        # Applying performs validation, Runtime creation, warm-up and one
        # atomic switch.  Keep the single request alive for the full semantic
        # operation instead of making the operator click again after a client
        # timeout while the backend continues working.
        return self._request(
            "POST",
            f"/api/domain-adaptation/candidates/{profile_id}/apply",
            json={"confirm": confirm},
            timeout=120.0,
        )

    def memory_banks(self, *, dataset: str | None = None, category: str | None = None, model_id: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/memory-banks",
            params={key: value for key, value in {"dataset": dataset, "category": category, "model_id": model_id}.items() if value not in (None, "")},
        )

    def config_profiles(self, *, dataset: str | None = None, category: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/config-profiles",
            params={key: value for key, value in {"dataset": dataset, "category": category}.items() if value not in (None, "")},
        )

    def config_profile(self, profile_id: str, version: int) -> dict[str, Any]:
        return self._request("GET", f"/api/config-profiles/{profile_id}/versions/{int(version)}")

    def create_config_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/config-profiles", json=payload)

    def update_config_profile(self, profile_id: str, version: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/api/config-profiles/{profile_id}/versions/{int(version)}", json=payload)

    def copy_config_profile(self, profile_id: str, version: int, friendly_name: str | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/config-profiles/{profile_id}/versions/{int(version)}/copy",
            json={"friendly_name": friendly_name},
        )

    def validate_config_profile(self, profile_id: str, version: int) -> dict[str, Any]:
        return self._request("POST", f"/api/config-profiles/{profile_id}/versions/{int(version)}/validate")

    def archive_config_profile(self, profile_id: str, version: int) -> dict[str, Any]:
        return self._request("POST", f"/api/config-profiles/{profile_id}/versions/{int(version)}/archive")

    def import_config_profile(self, friendly_name: str, yaml_text: str) -> dict[str, Any]:
        return self._request("POST", "/api/config-profiles/import", json={"friendly_name": friendly_name, "yaml_text": yaml_text})

    def export_config_profile(self, profile_id: str, version: int) -> dict[str, Any]:
        return self._request("GET", f"/api/config-profiles/{profile_id}/versions/{int(version)}/export")

    def memory_bank(self, memory_bank_id: str, *, dataset: str, category: str, model_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/memory-banks/{memory_bank_id}",
            params={"dataset": dataset, "category": category, "model_id": model_id},
        )

    def memory_bank_configs(self, memory_bank_id: str, *, include_archived: bool = False) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/memory-banks/{memory_bank_id}/configs",
            params={"include_archived": str(bool(include_archived)).lower()},
            timeout=30.0,
            retry_read=True,
        )

    def create_memory_bank_config(self, memory_bank_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/memory-banks/{memory_bank_id}/configs", json=payload, timeout=60.0)

    def revise_memory_bank_config(self, memory_bank_id: str, profile_id: str, version: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/memory-banks/{memory_bank_id}/configs/{profile_id}/versions/{int(version)}/revise",
            json=payload,
            timeout=60.0,
        )

    def delete_memory_bank_config(self, memory_bank_id: str, profile_id: str, version: int, *, acknowledged: bool = False) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/memory-banks/{memory_bank_id}/configs/{profile_id}/versions/{int(version)}/delete",
            json={"acknowledged": bool(acknowledged)},
            timeout=60.0,
        )

    def activate_memory_bank(self, memory_bank_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/memory-banks/{memory_bank_id}/activate", json=payload)

    def rename_memory_bank(self, memory_bank_id: str, friendly_name: str) -> dict[str, Any]:
        return self._request("POST", f"/api/memory-banks/{memory_bank_id}/rename", json={"friendly_name": friendly_name})

    def archive_memory_bank(self, memory_bank_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/memory-banks/{memory_bank_id}/archive")

    # Unified runtime resource registry.  Resource mutations use stable IDs;
    # the browser/Streamlit layer never sends an artifact path.
    def runtime_resources(self, **params: Any) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/runtime/resources",
            params={key: value for key, value in params.items() if value not in (None, "")},
            timeout=30.0,
            retry_read=True,
        )

    def runtime_resource(self, resource_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/runtime/resources/{resource_id}", timeout=20.0, retry_read=True)

    def runtime_resource_usage(self, resource_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/runtime/resources/{resource_id}/usage", timeout=20.0, retry_read=True)

    def runtime_resource_revision(self, since: int | None = None) -> dict[str, Any]:
        params = {"since": int(since)} if since is not None else None
        return self._request("GET", "/api/runtime/resources/revision", params=params, timeout=10.0, retry_read=True)

    def runtime_resource_tombstones(self, limit: int = 200) -> dict[str, Any]:
        response = self._request("GET", "/api/runtime/resources/deletion-records", params={"limit": int(limit)}, timeout=20.0, retry_read=True)
        if response.get("success"):
            response["tombstones"] = list(response.get("deletion_records") or [])
        return response

    def runtime_resource_scan(self) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/resources/scan", timeout=60.0)

    def runtime_resource_validate(self, resource_id: str | None = None) -> dict[str, Any]:
        if resource_id:
            return self._request("POST", f"/api/runtime/resources/{resource_id}/validate", timeout=60.0)
        return self._request("POST", "/api/runtime/resources/validate", timeout=120.0)

    def runtime_resource_archive(
        self,
        resource_id: str,
        *,
        confirm: bool,
        actor: str = "operator",
        current_profile_action: str = "cancel",
        replacement_resource_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/runtime/resources/{resource_id}/archive",
            json={
                "confirm": confirm,
                "actor": actor,
                "current_profile_action": current_profile_action,
                "replacement_resource_id": replacement_resource_id,
            },
            timeout=30.0,
        )

    def runtime_resource_restore(self, resource_id: str, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", f"/api/runtime/resources/{resource_id}/restore", json={"confirm": confirm, "actor": actor}, timeout=30.0)

    def runtime_resource_delete_preview(self, resource_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/runtime/resources/{resource_id}/delete-preview", timeout=20.0, retry_read=True)

    def runtime_resource_delete(self, resource_id: str, *, confirm: bool = True, acknowledged: bool = False, preview_token: str, delete_artifacts: bool = True, reason: str = "", actor: str = "operator", current_profile_action: str = "cancel", replacement_resource_id: str | None = None) -> dict[str, Any]:
        payload = {
            "confirm": bool(confirm),
            "acknowledged": bool(acknowledged),
            "preview_token": preview_token,
            "delete_artifacts": delete_artifacts,
            "reason": reason,
            "actor": actor,
            "current_profile_action": current_profile_action,
            "replacement_resource_id": replacement_resource_id,
        }
        return self._request(
            "POST",
            f"/api/runtime/resources/{resource_id}/delete",
            json=payload,
            timeout=60.0,
        )

    def runtime_resource_batch_archive_preview(self, resource_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/resources/batch-archive-preview", json={"resource_ids": resource_ids}, timeout=30.0)

    def runtime_resource_batch_archive(self, resource_ids: list[str], *, confirm: bool, current_profile_action: str = "cancel", replacement_resource_id: str | None = None, actor: str = "local_operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/resources/batch-archive", json={"resource_ids": resource_ids, "confirm": confirm, "current_profile_action": current_profile_action, "replacement_resource_id": replacement_resource_id, "actor": actor}, timeout=120.0)

    def runtime_resource_batch_restore(self, resource_ids: list[str], *, confirm: bool, actor: str = "local_operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/resources/batch-restore", json={"resource_ids": resource_ids, "confirm": confirm, "actor": actor}, timeout=120.0)

    def runtime_resource_batch_delete_preview(self, resource_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/resources/batch-delete-preview", json={"resource_ids": resource_ids}, timeout=60.0)

    def runtime_resource_batch_delete(self, resource_ids: list[str], *, preview_token: str, acknowledged: bool = False, confirm: bool = True, reason: str = "", current_profile_action: str = "cancel", replacement_resource_id: str | None = None, actor: str = "local_operator") -> dict[str, Any]:
        payload = {"resource_ids": resource_ids, "preview_token": preview_token, "confirm": bool(confirm), "acknowledged": bool(acknowledged), "reason": reason, "current_profile_action": current_profile_action, "replacement_resource_id": replacement_resource_id, "actor": actor}
        return self._request("POST", "/api/runtime/resources/batch-delete", json=payload, timeout=180.0)

    def runtime_target_cleanup_summary(self) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/target-domains/cleanup-summary", timeout=60.0, retry_read=True)

    def runtime_target_batch_archive_preview(self, resource_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/target-domains/batch-archive-preview", json={"resource_ids": resource_ids}, timeout=30.0)

    def runtime_target_batch_archive(self, resource_ids: list[str], *, confirm: bool, current_profile_action: str = "cancel", replacement_resource_id: str | None = None, actor: str = "local_operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/target-domains/batch-archive", json={"resource_ids": resource_ids, "confirm": confirm, "current_profile_action": current_profile_action, "replacement_resource_id": replacement_resource_id, "actor": actor}, timeout=120.0)

    def runtime_target_batch_restore(self, resource_ids: list[str], *, confirm: bool, actor: str = "local_operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/target-domains/batch-restore", json={"resource_ids": resource_ids, "confirm": confirm, "actor": actor}, timeout=120.0)

    def runtime_target_batch_delete_preview(self, resource_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/target-domains/batch-delete-preview", json={"resource_ids": resource_ids}, timeout=60.0)

    def runtime_target_batch_delete(self, resource_ids: list[str], *, preview_token: str, acknowledged: bool = False, confirm: bool = True, reason: str = "", current_profile_action: str = "cancel", replacement_resource_id: str | None = None, actor: str = "local_operator") -> dict[str, Any]:
        payload = {"resource_ids": resource_ids, "preview_token": preview_token, "confirm": bool(confirm), "acknowledged": bool(acknowledged), "reason": reason, "current_profile_action": current_profile_action, "replacement_resource_id": replacement_resource_id, "actor": actor}
        return self._request("POST", "/api/runtime/target-domains/batch-delete", json=payload, timeout=180.0)

    def runtime_target_clear_all_preview(self) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/target-domains/clear-all-preview", timeout=120.0)

    def runtime_target_clear_all(self, *, plan_id: str, preview_token: str, acknowledged: bool, actor: str = "local_operator") -> dict[str, Any]:
        payload = {"plan_id": plan_id, "preview_token": preview_token, "acknowledged": bool(acknowledged), "actor": actor}
        return self._request(
            "POST",
            "/api/runtime/target-domains/clear-all",
            json=payload,
            timeout=300.0,
        )

    def runtime_configuration_batch_archive_preview(self, configuration_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/configurations/batch-archive-preview", json={"configuration_ids": configuration_ids}, timeout=30.0)

    def runtime_configuration_batch_archive(self, configuration_ids: list[str], *, confirm: bool, current_profile_action: str = "cancel", replacement_resource_id: str | None = None, actor: str = "local_operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/configurations/batch-archive", json={"configuration_ids": configuration_ids, "confirm": confirm, "current_profile_action": current_profile_action, "replacement_resource_id": replacement_resource_id, "actor": actor}, timeout=120.0)

    def runtime_configuration_batch_restore(self, configuration_ids: list[str], *, confirm: bool, actor: str = "local_operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/configurations/batch-restore", json={"configuration_ids": configuration_ids, "confirm": confirm, "actor": actor}, timeout=120.0)

    def runtime_configuration_batch_delete_preview(self, configuration_ids: list[str]) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/configurations/batch-delete-preview", json={"configuration_ids": configuration_ids}, timeout=60.0)

    def runtime_configuration_batch_delete(self, configuration_ids: list[str], *, preview_token: str, acknowledged: bool = False, confirm: bool = True, reason: str = "", current_profile_action: str = "cancel", replacement_resource_id: str | None = None, actor: str = "local_operator") -> dict[str, Any]:
        payload = {"configuration_ids": configuration_ids, "preview_token": preview_token, "confirm": bool(confirm), "acknowledged": bool(acknowledged), "reason": reason, "current_profile_action": current_profile_action, "replacement_resource_id": replacement_resource_id, "actor": actor}
        return self._request("POST", "/api/runtime/configurations/batch-delete", json=payload, timeout=180.0)

    def runtime_resource_dependencies(self, resource_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/runtime/resources/{resource_id}/dependencies", timeout=20.0, retry_read=True)

    def runtime_resource_references(self, resource_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/runtime/resources/{resource_id}/references", timeout=20.0, retry_read=True)

    def runtime_profiles(self) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/profiles", timeout=30.0, retry_read=True)

    def runtime_test_lock(self) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/test-lock", timeout=5.0, retry_read=True)

    def set_runtime_test_lock(self, enabled: bool, *, actor: str = "operator") -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/runtime/test-lock",
            json={"enabled": bool(enabled), "actor": actor},
            timeout=10.0,
        )

    def runtime_profile(self, resource_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/runtime/profiles/{resource_id}", timeout=20.0, retry_read=True)

    def runtime_profile_validate(self, resource_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/runtime/profiles/{resource_id}/validate", timeout=60.0)

    def runtime_profile_apply(
        self,
        resource_id: str,
        *,
        actor: str = "operator",
        config_profile_id: str | None = None,
        config_profile_version: int | None = None,
    ) -> dict[str, Any]:
        payload = {"resource_id": resource_id, "actor": actor}
        if config_profile_id and config_profile_version:
            payload.update(config_profile_id=config_profile_id, config_profile_version=int(config_profile_version))
        return self._request("POST", "/api/runtime/profiles/apply", json=payload, timeout=60.0)

    def runtime_profile_deactivate(self, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/runtime/profiles/deactivate",
            json={"confirm": bool(confirm), "actor": actor},
            timeout=60.0,
        )

    def runtime_profile_rollback(self, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/profiles/rollback", json={"confirm": confirm, "actor": actor}, timeout=60.0)

    def runtime_maintenance_summary(self) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/maintenance/summary", timeout=60.0, retry_read=True)

    def runtime_maintenance_rebuild(self) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/maintenance/rebuild-index", timeout=120.0)

    def runtime_maintenance_orphans(self) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/maintenance/find-orphans", timeout=60.0)

    def runtime_maintenance_cleanup_preview(self) -> dict[str, Any]:
        return self._request("POST", "/api/runtime/maintenance/cleanup-preview", timeout=60.0)

    def runtime_maintenance_cleanup(self, resource_ids: list[str], *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/maintenance/cleanup", json={"resource_ids": resource_ids, "confirm": confirm, "actor": actor}, timeout=60.0)

    def runtime_maintenance_backup_index(self, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/maintenance/backup-index", json={"confirm": confirm, "actor": actor}, timeout=60.0)

    def runtime_maintenance_restore_default(self, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", "/api/runtime/maintenance/restore-default", json={"confirm": confirm, "actor": actor}, timeout=60.0)

    def runtime_maintenance_audit(self, *, limit: int = 50) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/maintenance/audit", params={"limit": int(limit)}, timeout=30.0, retry_read=True)

    def runtime_maintenance_pending_cleanup(self) -> dict[str, Any]:
        return self._request("GET", "/api/runtime/maintenance/resource-trash", timeout=30.0, retry_read=True)

    def runtime_maintenance_retry_cleanup(self, operation_id: str, *, confirm: bool = True, actor: str = "operator") -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/runtime/maintenance/resource-trash/{operation_id}/cleanup",
            json={"confirm": bool(confirm), "actor": actor},
            timeout=60.0,
        )

    def upload_video(self, uploaded_file: Any, *, progress: ProgressCallback | None = None) -> dict[str, Any]:
        filename = Path(str(getattr(uploaded_file, "name", "upload.mp4"))).name
        body = StreamingMultipartEncoder(uploaded_file, filename=filename, progress=progress)
        return self._request("POST", "/api/video/uploads", data=body, headers=body.headers)

    def video_upload(self, upload_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/video/uploads/{upload_id}", timeout=3.0, retry_read=True)

    def start_video_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/video/jobs", json=payload)

    def video_job(self, job_id: str, heartbeat: bool = False) -> dict[str, Any]:
        return self._request("GET", f"/api/video/jobs/{job_id}", params={"heartbeat": str(bool(heartbeat)).lower()})

    def pause_video_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/pause")

    def resume_video_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/resume")

    def stop_video_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/stop")

    def cancel_video_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/cancel")

    def rerun_video_job(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/rerun")

    def open_video_output(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/open-output")

    def open_video_result(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/video/jobs/{job_id}/open-video")

    def results(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "/api/results", params={key: value for key, value in params.items() if value not in (None, "")})

    def summary(self) -> dict[str, Any]:
        return self._request("GET", "/api/statistics/summary", timeout=STATUS_TIMEOUT_SECONDS, retry_read=True)

    def statistics_revision(self, since: int | None = None) -> dict[str, Any]:
        params = {"since": int(since)} if since is not None else None
        return self._request(
            "GET",
            "/api/statistics/revision",
            params=params,
            timeout=STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def statistics_dashboard(self, **params: Any) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/statistics/dashboard",
            params={key: value for key, value in params.items() if value not in (None, "")},
            timeout=max(STATUS_TIMEOUT_SECONDS, 30.0),
            retry_read=True,
        )

    def history_management_summary(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        params = {"filters": json.dumps(filters or {}, ensure_ascii=False, separators=(",", ":"))}
        return self._request(
            "GET",
            "/api/history/management/summary",
            params=params,
            timeout=60.0,
            retry_read=True,
        )

    def history_management_records(
        self,
        filters: dict[str, Any] | None = None,
        *,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        params = {
            "filters": json.dumps(filters or {}, ensure_ascii=False, separators=(",", ":")),
            "page": page,
            "page_size": page_size,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }
        return self._request(
            "GET",
            "/api/history/management/records",
            params=params,
            timeout=60.0,
            retry_read=True,
        )

    def history_management_task_items(
        self,
        task_id: str,
        *,
        mode: str,
        page: int = 1,
        page_size: int = 20,
        query: str | None = None,
        prediction: str | None = None,
        feedback_status: str | None = None,
        time_from_ms: float | None = None,
        time_to_ms: float | None = None,
    ) -> dict[str, Any]:
        params = {
            "mode": mode,
            "page": page,
            "page_size": page_size,
            "query": query,
            "prediction": prediction,
            "feedback_status": feedback_status,
            "time_from_ms": time_from_ms,
            "time_to_ms": time_to_ms,
        }
        return self._request(
            "GET",
            f"/api/history/management/tasks/{task_id}/items",
            params={key: value for key, value in params.items() if value not in (None, "")},
            timeout=30.0,
            retry_read=True,
        )

    def history_management_delete_task_items(
        self,
        task_id: str,
        item_ids: list[str],
        *,
        acknowledged: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/history/management/tasks/{task_id}/items/delete",
            json={
                "mode": "batch",
                "item_ids": item_ids,
                "acknowledged": bool(acknowledged),
            },
            timeout=30.0,
        )

    def history_management_delete_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/history/management/delete-preview", json=payload, timeout=120.0)

    def history_management_delete(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/history/management/delete", json=payload, timeout=10.0)

    def history_management_delete_all_preview(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request(
            "POST", "/api/history/management/delete-all-preview", json=payload or {}, timeout=120.0
        )

    def history_management_delete_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/history/management/delete-all", json=payload, timeout=10.0)

    def history_management_cleanup_jobs(self, limit: int = 20) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/history/management/cleanup-jobs",
            params={"limit": limit},
            timeout=STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def history_management_cleanup_job(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/history/management/cleanup-jobs/{job_id}",
            timeout=STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def history_management_cancel_cleanup_job(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/history/management/cleanup-jobs/{job_id}/cancel",
            timeout=STATUS_TIMEOUT_SECONDS,
        )

    def history_management_audit(self, limit: int = 50) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/history/management/audit",
            params={"limit": limit},
            timeout=STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def history_management_retry_cleanup(self, cleanup_ids: list[str] | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/history/management/retry-file-cleanup",
            json={"cleanup_ids": cleanup_ids or []},
            timeout=30.0,
        )

    def experiments(self) -> dict[str, Any]:
        return self._request("GET", "/api/experiments")

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/experiments/{experiment_id}")

    def calibrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/calibrations", json=payload)

    def activate_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/calibrations/activate", json=payload)

    def save_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/feedback", json=payload)

    def feedbacks(self, *, task_id: str, mode: str | None = None, sample_id: str | None = None, frame_index: int | None = None) -> dict[str, Any]:
        params = {
            "task_id": task_id,
            "mode": mode,
            "sample_id": sample_id,
            "frame_index": frame_index,
        }
        return self._request("GET", "/api/feedback", params={key: value for key, value in params.items() if value not in (None, "")}, retry_read=True)

    def video_feedback_interval(self, job_id: str, *, start_ms: float, end_ms: float, max_frames: int = 8) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/video/jobs/{job_id}/feedback-interval",
            params={"start_ms": start_ms, "end_ms": end_ms, "max_frames": max_frames},
            retry_read=True,
        )

    def save_bulk_feedback(self, feedbacks: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/api/feedback/bulk", json={"feedbacks": feedbacks})

    def feedback_candidates(self, **params: Any) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/feedback/candidates",
            params={key: value for key, value in params.items() if value not in (None, "")},
            retry_read=True,
        )

    def feedback_revision(self, since: int | None = None) -> dict[str, Any]:
        params = {"since": int(since)} if since is not None else None
        return self._request(
            "GET",
            "/api/feedback/revision",
            params=params,
            timeout=10.0,
            retry_read=True,
        )

    def feedback_resources(self) -> dict[str, Any]:
        return self._request("GET", "/api/feedback/resources", timeout=30.0, retry_read=True)

    def feedback_candidate(self, candidate_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/feedback/candidates/{candidate_id}", retry_read=True)

    def feedback_candidates_delete_preview(self, candidate_ids: list[str]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/feedback/candidates/delete-preview",
            json={"candidate_ids": candidate_ids, "actor": "local_operator"},
        )

    def delete_feedback_candidates(self, candidate_ids: list[str], *, acknowledged: bool = False, preview_token: str | None = None) -> dict[str, Any]:
        payload = {"candidate_ids": candidate_ids, "acknowledged": bool(acknowledged), "actor": "local_operator"}
        if preview_token:
            payload["preview_token"] = preview_token
        return self._request(
            "POST",
            "/api/feedback/candidates/delete",
            json=payload,
            timeout=30.0,
        )

    def review_feedback_candidate(self, candidate_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/candidates/{candidate_id}/review", json=payload)

    def review_feedback_candidates_bulk(self, candidate_ids: list[str], *, action: str, reviewer: str = "local_operator", note: str = "", require_snapshot: bool = True) -> dict[str, Any]:
        return self._request("POST", "/api/feedback/candidates/review-bulk", json={"candidate_ids": candidate_ids, "action": action, "reviewer": reviewer, "note": note, "require_snapshot": require_snapshot})

    def feedback_candidate_snapshot(self, candidate_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/candidates/{candidate_id}/snapshot", timeout=60.0)

    def feedback_candidate_protection(self, candidate_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/feedback/candidates/{candidate_id}/protection-status", retry_read=True)

    def feedback_annotation_history(self, candidate_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/feedback/candidates/{candidate_id}/annotation/history", retry_read=True)

    def feedback_annotation_latest(self, candidate_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/feedback/candidates/{candidate_id}/annotation", retry_read=True)

    def feedback_candidate_source(self, candidate_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/feedback/candidates/{candidate_id}/source", retry_read=True)

    def feedback_candidate_source_preview(self, candidate_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/feedback/candidates/{candidate_id}/source-preview", retry_read=True)

    def save_feedback_annotation(self, candidate_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/api/feedback/candidates/{candidate_id}/annotation", json=payload)

    def feedback_statistics(self, memory_bank_id: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/api/feedback/statistics", params={"memory_bank_id": memory_bank_id} if memory_bank_id else None, retry_read=True)

    def feedback_audit(self, **params: Any) -> dict[str, Any]:
        return self._request("GET", "/api/feedback/audit", params={key: value for key, value in params.items() if value not in (None, "")}, retry_read=True)

    def create_feedback_optimization(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/feedback/optimization/jobs", json=payload, timeout=10.0)

    @staticmethod
    def build_feedback_optimization_payload(
        *,
        base_profile_id: str,
        candidate_ids: list[str] | tuple[str, ...],
        strategy: str,
        objective: str = "f1",
        update_threshold: bool = True,
        update_normal_memory: bool = False,
        update_defect_memory: bool = False,
        update_lightweight_calibrator: bool = False,
        min_recall: float = 0.95,
        max_false_positive_rate: float = 0.10,
        max_memory_growth: float = 0.05,
        max_latency_growth: float = 0.10,
        publish_mode: str = "update_current",
        replace_confirm: bool = False,
        version_name: str = "",
        note: str | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Build the one canonical feedback-update request body.

        The page may render translated names and different strategy cards, but
        it must call this builder before both the compatibility preview and the
        mutating create request.  This keeps the two requests identical and
        rejects display labels/unknown enum values locally instead of sending
        a payload that can only fail with an opaque HTTP 422.
        """

        profile_id = str(base_profile_id or "").strip()
        if not profile_id:
            raise ValueError("base_profile_id 不能为空。")
        normalized_ids = list(dict.fromkeys(
            str(value).strip() for value in candidate_ids or () if str(value).strip()
        ))
        if not normalized_ids:
            raise ValueError("candidate_ids 至少需要一个反馈样本。")
        strategy_code = str(strategy or "").strip()
        if strategy_code not in FEEDBACK_OPTIMIZATION_STRATEGY_CODES:
            raise ValueError(f"不支持的反馈更新策略代码：{strategy_code or '空值'}。")
        objective_code = str(objective or "").strip()
        if objective_code not in FEEDBACK_OPTIMIZATION_OBJECTIVE_CODES:
            raise ValueError(f"不支持的反馈优化目标：{objective_code or '空值'}。")
        publish_code = str(publish_mode or "").strip()
        if publish_code not in FEEDBACK_OPTIMIZATION_PUBLISH_MODE_CODES:
            raise ValueError(f"不支持的版本生成方式代码：{publish_code or '空值'}。")
        return {
            "base_profile_id": profile_id,
            "candidate_ids": normalized_ids,
            "strategy": strategy_code,
            "objective": objective_code,
            "update_threshold": bool(update_threshold),
            "update_normal_memory": bool(update_normal_memory),
            "update_defect_memory": bool(update_defect_memory),
            "update_lightweight_calibrator": bool(update_lightweight_calibrator),
            "min_recall": float(min_recall),
            "max_false_positive_rate": float(max_false_positive_rate),
            "max_memory_growth": float(max_memory_growth),
            "max_latency_growth": float(max_latency_growth),
            "publish_mode": publish_code,
            "replace_confirm": bool(replace_confirm),
            "version_name": str(version_name or "").strip(),
            "note": None if note is None else str(note),
            "idempotency_key": str(idempotency_key or "").strip(),
        }

    def feedback_optimization_compatibility(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/feedback/optimization/compatibility", json=payload, timeout=20.0)

    def feedback_optimization_strategies(self, method_mode: str | None = None) -> dict[str, Any]:
        params = {"method_mode": method_mode} if method_mode else None
        return self._request("GET", "/api/feedback/optimization/strategies", params=params, retry_read=True)

    def feedback_optimization_jobs(self, limit: int = 100) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/feedback/optimization/jobs",
            params={"limit": limit, "summary": True},
            timeout=FEEDBACK_STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def feedback_optimization_job(self, job_id: str, *, compact: bool = False) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/feedback/optimization/jobs/{job_id}",
            params={"compact": compact, "include_items": not compact},
            timeout=FEEDBACK_STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def validate_feedback_optimization(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/optimization/jobs/{job_id}/validate", timeout=10.0)

    def feedback_candidate_profile(self, job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/feedback/optimization/jobs/{job_id}/candidate-profile",
            timeout=FEEDBACK_STATUS_TIMEOUT_SECONDS,
            retry_read=True,
        )

    def cancel_feedback_optimization(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/optimization/jobs/{job_id}/cancel", timeout=3.0)

    def rerun_feedback_optimization(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/optimization/jobs/{job_id}/rerun", timeout=10.0)

    def apply_feedback_optimization(self, job_id: str, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/optimization/jobs/{job_id}/apply", json={"confirm": confirm, "actor": actor}, timeout=20.0)

    def freeze_feedback_optimization(self, job_id: str, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/optimization/jobs/{job_id}/freeze", json={"confirm": confirm, "actor": actor}, timeout=30.0)

    def rollback_feedback_optimization(self, job_id: str, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        return self._request("POST", f"/api/feedback/optimization/jobs/{job_id}/rollback", json={"confirm": confirm, "actor": actor}, timeout=20.0)


@lru_cache(maxsize=4)
def client(base_url: str | None = None) -> IntegrationApiClient:
    return IntegrationApiClient(base_url=base_url)
