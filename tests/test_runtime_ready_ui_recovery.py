from __future__ import annotations

from pathlib import Path

from service.runtime_config_service import (
    RuntimeStatusSnapshot,
    merge_runtime_status,
    runtime_status_snapshot_from_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_loading_readiness_has_priority_over_ready_looking_runtime_context() -> None:
    snapshot = runtime_status_snapshot_from_readiness(
        {
            "success": True,
            "status": "loading",
            "phase": "runtime",
            "current_stage": "正在加载目标域资源",
            "progress": 62,
            "runtime_context": {
                "runtime_revision": 8,
                "status": "ready",
                "profile_id": "profile-a",
                "memory_bank_id": "bank-a",
                "config_profile_id": "config-a",
                "model_id": "model-a",
                "dataset": "mvtec",
                "category": "bottle",
            },
        }
    )

    assert snapshot.status == "loading"
    assert snapshot.runtime_revision == 8
    assert snapshot.binding == (8, "profile-a", "bank-a", "config-a", "model-a")


def test_live_snapshot_overrides_cached_ready_status_without_discarding_layout_data() -> None:
    cached_runtime = {
        "runtime_status": "ready",
        "memory_bank_id": "old-bank",
        "config_profile_id": "old-config",
        "model_id": "old-model",
        "dataset": "mvtec",
        "category": "bottle",
        "memory_bank_profile": {"friendly_name": "旧名称", "presentation_hint": "保留用于稳定布局"},
    }
    snapshot = RuntimeStatusSnapshot(
        runtime_revision=9,
        status="loading",
        profile_id="profile-b",
        memory_bank_id="bank-b",
        config_id="config-b",
        model_id="model-b",
        dataset="visa",
        category="capsule",
    )

    merged = merge_runtime_status(cached_runtime, snapshot)

    assert merged["runtime_status"] == "loading"
    assert merged["runtime_revision"] == 9
    assert merged["memory_bank_id"] == "bank-b"
    assert merged["config_profile_id"] == "config-b"
    assert merged["memory_bank_profile"]["friendly_name"] == "profile-b"
    assert merged["memory_bank_profile"]["presentation_hint"] == "保留用于稳定布局"


def test_runtime_ready_transition_is_owned_by_page_local_fragments() -> None:
    common = _source("app/ui_common.py")
    single = _source("app/pages/1_单图检测.py")
    batch = _source("app/pages/2_批量检测.py")
    video = _source("app/pages/9_视频文件检测.py")

    monitor_body = common.split("def render_detection_resource_monitor", 1)[1].split(
        "def runtime_header_items", 1
    )[0]
    assert "@st.fragment" not in common.split("def render_detection_resource_monitor", 1)[0][-80:]
    assert 'st.rerun(scope="app")' not in monitor_body
    assert 'render_detection_resource_monitor("single")' in single
    assert 'render_detection_resource_monitor("batch")' in batch
    assert 'render_detection_resource_monitor("video")' in video
    assert '@st.fragment(run_every="1.5s")' in single
    assert '@st.fragment(run_every="1.5s")' in batch
    assert '@st.fragment(run_every="1.5s")' in video


def test_detection_inputs_are_not_keyed_to_runtime_revision() -> None:
    single = _source("app/pages/1_单图检测.py")
    batch = _source("app/pages/2_批量检测.py")
    video = _source("app/pages/9_视频文件检测.py")

    assert 'key="single_selected_file"' in single
    assert "SINGLE_INPUT_SNAPSHOT_KEY" in single
    assert "运行配置已更新，请重新检测当前图片" in single
    assert 'BATCH_INPUT_KEY = "batch_input_dir"' in batch
    assert "批量输入路径与筛选状态已保留" in batch
    assert 'key="video_selected_file"' in video
    assert 'st.session_state.pop("video_upload_state", None)' not in video
    assert "当前视频仍然保留" in video
