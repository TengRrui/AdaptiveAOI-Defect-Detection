from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _feature_layers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """Immutable identity of the exact Runtime used by one detection task.

    The context is created once after the request binding guard succeeds.  It
    is then copied into durable task/result artifacts; inference workers must
    not refresh these fields from the process-global Runtime.
    """

    profile_id: str
    memory_bank_id: str
    config_id: str
    model_id: str
    threshold: float | None
    precision: str
    feature_layers: tuple[str, ...]
    runtime_revision: int
    algorithm_mode: str
    started_at: str

    def __post_init__(self) -> None:
        if int(self.runtime_revision) < 0:
            raise ValueError("runtime_revision must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_layers"] = list(self.feature_layers)
        return payload

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        memory_bank_id: str,
        config_id: str,
        model_id: str,
        threshold: float | int | None,
        precision: str,
        feature_layers: Any,
        runtime_revision: int,
        algorithm_mode: str,
        started_at: str | None = None,
    ) -> "DetectionContext":
        return cls(
            profile_id=str(profile_id or ""),
            memory_bank_id=str(memory_bank_id or ""),
            config_id=str(config_id or ""),
            model_id=str(model_id or ""),
            threshold=float(threshold) if threshold is not None else None,
            precision=str(precision or "unknown").strip().lower(),
            feature_layers=_feature_layers(feature_layers),
            runtime_revision=int(runtime_revision),
            algorithm_mode=str(algorithm_mode or "unknown"),
            started_at=str(started_at or _utc_now()),
        )

    @classmethod
    def from_value(cls, value: "DetectionContext | Mapping[str, Any]") -> "DetectionContext":
        if isinstance(value, cls):
            return value
        return cls.create(
            profile_id=str(value.get("profile_id") or ""),
            memory_bank_id=str(value.get("memory_bank_id") or ""),
            config_id=str(value.get("config_id") or value.get("config_profile_id") or ""),
            model_id=str(value.get("model_id") or ""),
            threshold=value.get("threshold"),
            precision=str(value.get("precision") or value.get("actual_precision") or "unknown"),
            feature_layers=value.get("feature_layers"),
            runtime_revision=int(value.get("runtime_revision") or 0),
            algorithm_mode=str(value.get("algorithm_mode") or value.get("method_mode") or "unknown"),
            started_at=str(value.get("started_at") or _utc_now()),
        )


def detection_context_dict(value: DetectionContext | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return DetectionContext.from_value(value).as_dict()


__all__ = ["DetectionContext", "detection_context_dict"]
