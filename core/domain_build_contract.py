from __future__ import annotations

from typing import Any, Mapping


# Formal target-domain creation baseline.  Keep this value in one place so
# Streamlit, API schemas, services and the algorithm adapter cannot silently
# drift to different defaults.
LIGHTWEIGHT_CREATE_MODE = "normal_memory_defect_guided_lite"
DEFAULT_CREATE_MODE = LIGHTWEIGHT_CREATE_MODE

# Formal default feature space for newly created resources.  Keep the tuple
# immutable and return a fresh list at API/config boundaries so one request
# cannot mutate the process-wide default.  Historical profiles with an
# explicit ``layers`` field remain authoritative and are not migrated.
DEFAULT_FEATURE_LAYERS = ("layer2",)


def default_feature_layers() -> list[str]:
    return list(DEFAULT_FEATURE_LAYERS)


# New target-domain resources may only be created with these modes.  Historical
# resources use the broader read-compatible set below and are never rewritten
# merely because the creation policy changes.
CREATE_ALLOWED_MODES = frozenset(
    {
        "normal_only",
        "normal_memory_defect_guided_lite",
    }
)

READ_COMPATIBLE_MODES = frozenset(
    {
        *CREATE_ALLOWED_MODES,
        "normal_memory",
        "normal_defect_calibration",
        "normal_defect_dual_memory",
        "dual_memory",
        "auto",
        "lightweight_calibration",
    }
)

CREATE_MODE_LABELS = {
    "normal_only": "基础正常记忆",
    LIGHTWEIGHT_CREATE_MODE: "单正常记忆 · 缺陷引导轻量校准",
}

PRECISION_MODES = frozenset({"fp16", "fp32"})


def normalize_precision_request(payload: Mapping[str, Any], *, default: str = "fp16") -> str:
    """Return the public precision request while supporting the old bool field.

    ``precision`` is the canonical public contract.  ``use_fp16`` remains a
    compatibility alias for clients and drafts created before that contract
    existed.  Supplying both with conflicting values is rejected rather than
    silently selecting one.
    """

    explicit = str(payload.get("precision") or "").strip().lower()
    if explicit and explicit not in PRECISION_MODES:
        raise ValueError("构建精度只支持 fp16 或 fp32。")

    legacy_present = "use_fp16" in payload and payload.get("use_fp16") is not None
    legacy = "fp16" if bool(payload.get("use_fp16")) else "fp32"
    if explicit and legacy_present and explicit != legacy:
        raise ValueError("precision 与兼容字段 use_fp16 的取值不一致。")

    resolved = explicit or (legacy if legacy_present else str(default).strip().lower())
    if resolved not in PRECISION_MODES:
        raise ValueError("默认构建精度配置无效，只支持 fp16 或 fp32。")
    return resolved


def precision_uses_fp16(precision: Any) -> bool:
    return str(precision or "").strip().lower() == "fp16"


def dtype_name(value: Any) -> str:
    """Normalize torch/numpy dtype text for manifests and API responses."""

    text = str(value or "unknown").strip().lower()
    for prefix in ("torch.", "numpy.", "np."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text or "unknown"
