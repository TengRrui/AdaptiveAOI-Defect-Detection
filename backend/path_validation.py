"""Validate resource names without restricting operator-selected directories."""
from __future__ import annotations

from backend.errors import ApiError


def validate_resource_segment(value: str, field: str) -> str:
    """Return one directory name, rejecting POSIX and Windows path syntax."""
    token = str(value).strip() if value is not None else ""
    if (
        not token
        or token in {".", ".."}
        or any(character in token for character in ("/", "\\", ":"))
        or any(ord(character) < 32 or ord(character) == 127 for character in token)
    ):
        raise ApiError(
            "INVALID_MODEL_SELECTION",
            "数据集和类别必须是有效的单个名称，不能包含路径。",
            422,
            {"field": field},
        )
    return token


def validate_dataset_category(dataset: str, category: str) -> tuple[str, str]:
    """Normalize a dataset/category pair used below managed resource roots."""
    return (
        validate_resource_segment(dataset, "dataset").lower(),
        validate_resource_segment(category, "category"),
    )
