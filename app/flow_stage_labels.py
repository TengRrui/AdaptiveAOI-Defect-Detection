from __future__ import annotations

from typing import Any


FLOW_STAGE_LABELS = {
    "input": "输入数据",
    "video_file": "读取与解码",
    "image_read": "读取与解码",
    "decode": "读取与解码",
    "tune": "图像预处理",
    "preprocess": "图像预处理",
    "neurology": "特征提取",
    "feature_extraction": "特征提取",
    "manage_search": "近邻检索",
    "ann_search": "近邻检索",
    "rule": "评分与判定",
    "anomaly_scoring": "评分与判定",
    "save": "可视化与保存",
    "visualization": "可视化与保存",
}


def flow_stage_label(stage: Any) -> str:
    """Return the public Chinese label without exposing an internal stage key."""
    key = str(stage or "").strip().lower()
    return FLOW_STAGE_LABELS.get(key, "其他处理")


__all__ = ["FLOW_STAGE_LABELS", "flow_stage_label"]
