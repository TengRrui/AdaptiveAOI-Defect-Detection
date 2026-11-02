from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest

from backend.api.routes import _validate_inference_binding
from backend.errors import ApiError
from bridge.api_client import IntegrationApiClient
from service import batch_service, inference_service, video_service


def _context(config_id: str = "cfg-initial", revision: int = 21) -> dict:
    return {
        "runtime_revision": revision,
        "profile_id": "bank-a",
        "profile_version": "2",
        "memory_bank_id": "bank-a",
        "config_profile_id": config_id,
        "config_version": 1,
        "model_id": "formal-model",
        "model_version": "formal-model",
        "dataset": "dataset-a",
        "category": "part-a",
        "method_mode": "normal_defect_calibration",
        "status": "ready",
        "updated_at": "2026-08-14T09:00:00+00:00",
    }


class _Upload(BytesIO):
    name = "qa.png"


class _InferenceClient:
    def __init__(self):
        self.runtime = None

    def detect_upload(self, _uploaded_file, runtime):
        self.runtime = runtime
        context = runtime["runtime_context"]
        return {
            "success": True,
            "request_id": "req-single",
            "job_id": "job-single",
            "result": {
                "result_id": "result-single",
                "anomaly_score": 0.25,
                "is_anomaly": False,
                "model_id": context["model_id"],
                "metadata": {
                    "dataset": context["dataset"],
                    "category": context["category"],
                    "memory_bank_id": context["memory_bank_id"],
                    "config_profile_id": context["config_profile_id"],
                },
            },
        }

    @staticmethod
    def artifact_url(value):
        return value


class _BatchClient:
    def __init__(self):
        self.runtime = None

    def detect_directory(self, _input_dir, runtime, *, async_job=False):
        self.runtime = runtime
        assert async_job is True
        return {"success": True, "job_id": "job-batch", "batch_id": "batch-a", "status": "queued", "total": 1}


class _VideoClient:
    def __init__(self):
        self.payload = None

    def start_video_job(self, payload):
        self.payload = payload
        return {"success": True, "video_job": {"job_id": "job-video"}}


@pytest.mark.parametrize("config_id,revision", [("cfg-initial", 21), ("cfg-2", 22)])
def test_single_detection_uses_active_runtime_context_not_page_selection(monkeypatch, config_id, revision):
    api = _InferenceClient()
    context = _context(config_id, revision)
    monkeypatch.setattr(inference_service, "client", lambda: api)
    monkeypatch.setattr(inference_service, "load_runtime_context", lambda: {"ok": True, "context": context})
    monkeypatch.setattr(inference_service, "save_latest_inference", lambda _result: None)

    result = inference_service.run_image_inference(
        "stale-page-dataset",
        "stale-page-category",
        "",
        uploaded_file=_Upload(b"image"),
        memory_bank_id="stale-page-bank",
    )

    assert result["ok"] is True
    assert result["config"] == config_id
    assert result["model_id"] == "formal-model"
    assert api.runtime["runtime_context"] == context
    assert "patchcore-default" not in repr(api.runtime)


def test_batch_detection_uses_active_runtime_context_not_page_selection(monkeypatch, tmp_path):
    api = _BatchClient()
    context = _context("cfg-2", 22)
    monkeypatch.setattr(batch_service, "client", lambda: api)
    monkeypatch.setattr(batch_service, "load_runtime_context", lambda: {"ok": True, "context": context})

    result = batch_service.start_batch_inference(
        "stale-page-dataset",
        "stale-page-category",
        "",
        tmp_path,
        memory_bank_id="stale-page-bank",
    )

    assert result["ok"] is True
    assert result["config"] == "cfg-2"
    assert api.runtime["runtime_context"] == context
    assert "patchcore-default" not in repr(api.runtime)


def test_video_detection_sends_only_runtime_identity(monkeypatch):
    api = _VideoClient()
    context = _context("cfg-2", 22)
    monkeypatch.setattr(video_service, "client", lambda: api)
    monkeypatch.setattr(video_service, "load_runtime_context", lambda: {"ok": True, "context": context})

    result = video_service.start_video_detection(
        "upload-a",
        {"config": "", "model_id": "stale-model", "memory_bank_id": "stale-bank"},
        {"frame_stride": 2, "generate_visualizations": False},
    )

    assert result["ok"] is True
    assert api.payload["profile_id"] == "bank-a"
    assert api.payload["runtime_revision"] == 22
    assert "model_id" not in api.payload
    assert "memory_bank_id" not in api.payload
    assert "config_profile_id" not in api.payload


def test_new_client_serializes_only_profile_and_revision():
    fields = IntegrationApiClient._inference_binding_fields(
        {"runtime_context": _context("cfg-2", 22), "config": "", "model_id": "stale-model"}
    )

    assert fields == {"profile_id": "bank-a", "runtime_revision": "22"}


def test_backend_resolves_full_binding_and_keeps_revision_guard():
    context = _context("cfg-2", 22)
    runtime = {
        "profile_status": "ready",
        "memory_bank_id": "bank-a",
        "model_id": "formal-model",
        "dataset": "dataset-a",
        "category": "part-a",
        "config_profile_id": "cfg-2",
        "config_profile_version": 1,
        "device": "cuda",
    }
    services = SimpleNamespace(
        runtime_profiles=SimpleNamespace(
            context_snapshot=lambda: context,
            configuration=lambda: runtime,
        ),
        domain_profiles=SimpleNamespace(
            get_profile=lambda *_args: {"frozen": True, "threshold": 0.7}
        ),
        config_profiles=SimpleNamespace(
            compatibility=lambda *_args: {"compatible": True},
            get=lambda *_args: {
                "config_hash": "hash-cfg-2",
                "threshold": 0.8,
                "config": {"device": "cuda", "model_id": "formal-model"},
            },
        ),
    )

    binding = _validate_inference_binding(
        services,
        profile_id="bank-a",
        runtime_revision=22,
        memory_bank_id=None,
        model_id=None,
        dataset=None,
        category=None,
        config_profile_id=None,
        config_profile_version=None,
    )

    assert binding["model_id"] == "formal-model"
    assert binding["memory_bank_id"] == "bank-a"
    assert binding["config_profile_id"] == "cfg-2"

    with pytest.raises(ApiError) as error:
        _validate_inference_binding(
            services,
            profile_id="bank-a",
            runtime_revision=21,
            memory_bank_id=None,
            model_id=None,
            dataset=None,
            category=None,
            config_profile_id=None,
            config_profile_version=None,
        )
    assert error.value.code == "INFERENCE_RUNTIME_BINDING_MISMATCH"


def test_backend_rejects_detection_until_new_runtime_is_ready():
    context = {**_context("cfg-2", 22), "status": "loading"}
    services = SimpleNamespace(
        runtime_profiles=SimpleNamespace(
            context_snapshot=lambda: context,
            configuration=lambda: {"profile_status": "ready"},
        )
    )

    with pytest.raises(ApiError) as error:
        _validate_inference_binding(
            services,
            profile_id="bank-a",
            runtime_revision=22,
            memory_bank_id=None,
            model_id=None,
            dataset=None,
            category=None,
            config_profile_id=None,
            config_profile_version=None,
        )
    assert error.value.code == "RUNTIME_PROFILE_NOT_READY"
