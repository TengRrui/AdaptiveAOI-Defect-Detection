from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class ROIProposal:
    """A source-image ROI proposed by the coarse anomaly map."""

    box: TileBox
    score: float
    area: int
    source: str


@dataclass(frozen=True)
class ROISelection:
    proposals: tuple[ROIProposal, ...]
    component_count: int
    retained_component_count: int
    fallback_count: int
    reason: str


def _positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def schedule_tiles(width: int, height: int, tile_size: int = 768, overlap: int = 128) -> list[TileBox]:
    tile_size = int(tile_size)
    overlap = int(overlap)
    xs = _positions(int(width), tile_size, overlap)
    ys = _positions(int(height), tile_size, overlap)
    return [TileBox(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in ys for x in xs]


def crop_tile(image: Image.Image, box: TileBox) -> Image.Image:
    return image.crop((box.x0, box.y0, box.x1, box.y1))


def init_fusion(height: int, width: int, mode: str):
    mode = mode.lower()
    if mode == "max":
        acc = np.full((height, width), -np.inf, dtype=np.float32)
        weight = None
    elif mode in {"average", "avg", "mean"}:
        acc = np.zeros((height, width), dtype=np.float32)
        weight = np.zeros((height, width), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported tile fusion mode: {mode}")
    return acc, weight


def fuse_tile(acc, weight, tile_heatmap: np.ndarray, box: TileBox, mode: str):
    resized = cv2.resize(tile_heatmap.astype(np.float32), (box.width, box.height), interpolation=cv2.INTER_LINEAR)
    target = acc[box.y0:box.y1, box.x0:box.x1]
    mode = mode.lower()
    if mode == "max":
        np.maximum(target, resized, out=target)
    else:
        target += resized
        weight[box.y0:box.y1, box.x0:box.x1] += 1.0


def finalize_fusion(acc, weight, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "max":
        acc[~np.isfinite(acc)] = 0.0
        return acc.astype(np.float32)
    weight = np.maximum(weight, 1.0)
    return (acc / weight).astype(np.float32)


def box_iou(left: TileBox, right: TileBox) -> float:
    x0 = max(left.x0, right.x0)
    y0 = max(left.y0, right.y0)
    x1 = min(left.x1, right.x1)
    y1 = min(left.y1, right.y1)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = left.width * left.height + right.width * right.height - intersection
    return float(intersection / union) if union > 0 else 0.0


def _clip_box(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> TileBox:
    x0 = int(max(0, min(width - 1, np.floor(x0))))
    y0 = int(max(0, min(height - 1, np.floor(y0))))
    x1 = int(max(x0 + 1, min(width, np.ceil(x1))))
    y1 = int(max(y0 + 1, min(height, np.ceil(y1))))
    return TileBox(x0=x0, y0=y0, x1=x1, y1=y1)


def expand_box(box: TileBox, width: int, height: int, expand_ratio: float, min_size: int = 0) -> TileBox:
    """Expand a proposal around its center while retaining enough fine-detail context."""
    target_width = max(float(box.width) * (1.0 + 2.0 * max(0.0, float(expand_ratio))), float(min_size or 0))
    target_height = max(float(box.height) * (1.0 + 2.0 * max(0.0, float(expand_ratio))), float(min_size or 0))
    center_x = (box.x0 + box.x1) / 2.0
    center_y = (box.y0 + box.y1) / 2.0
    return _clip_box(
        center_x - target_width / 2.0,
        center_y - target_height / 2.0,
        center_x + target_width / 2.0,
        center_y + target_height / 2.0,
        width=width,
        height=height,
    )


def _union_box(left: TileBox, right: TileBox) -> TileBox:
    return TileBox(
        x0=min(left.x0, right.x0),
        y0=min(left.y0, right.y0),
        x1=max(left.x1, right.x1),
        y1=max(left.y1, right.y1),
    )


def _component_proposals(heatmap: np.ndarray, threshold: float, min_area: int) -> tuple[list[ROIProposal], list[ROIProposal]]:
    """Return all threshold components and the subset that clears min_area."""
    score = np.asarray(heatmap, dtype=np.float32)
    binary = np.greater_equal(score, float(threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    all_components: list[ROIProposal] = []
    retained: list[ROIProposal] = []
    for component_id in range(1, int(count)):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        if area <= 0:
            continue
        local_labels = labels[y:y + height, x:x + width]
        local_score = score[y:y + height, x:x + width]
        component_score = float(local_score[local_labels == component_id].max())
        proposal = ROIProposal(
            box=TileBox(x0=x, y0=y, x1=x + width, y1=y + height),
            score=component_score,
            area=area,
            source="threshold_component",
        )
        all_components.append(proposal)
        if area >= int(min_area):
            retained.append(proposal)
    return all_components, retained


def _top_peak_proposals(heatmap: np.ndarray, topk: int, fine_tile_size: int) -> list[ROIProposal]:
    """Use NMS-separated global peaks as a bounded fallback for small defects."""
    topk = max(0, int(topk))
    if topk == 0:
        return []
    score = np.asarray(heatmap, dtype=np.float32)
    if score.ndim != 2 or score.size == 0 or not np.isfinite(score).any():
        return []
    height, width = score.shape
    flat = score.reshape(-1)
    candidate_count = min(flat.size, max(256, topk * 32))
    indices = np.argpartition(flat, -candidate_count)[-candidate_count:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    min_distance = max(1, int(fine_tile_size) // 2)
    selected: list[ROIProposal] = []
    centers: list[tuple[int, int]] = []
    for index in indices:
        y, x = divmod(int(index), width)
        if any((x - other_x) ** 2 + (y - other_y) ** 2 < min_distance ** 2 for other_x, other_y in centers):
            continue
        half = max(1, int(fine_tile_size) // 2)
        box = _clip_box(x - half, y - half, x + half, y + half, width=width, height=height)
        selected.append(ROIProposal(box=box, score=float(flat[index]), area=1, source="global_topk_fallback"))
        centers.append((x, y))
        if len(selected) >= topk:
            break
    return selected


def _merge_proposals(proposals: list[ROIProposal], merge_iou: float) -> list[ROIProposal]:
    merged: list[ROIProposal] = []
    for proposal in sorted(proposals, key=lambda item: item.score, reverse=True):
        matched_index = next(
            (index for index, existing in enumerate(merged) if box_iou(existing.box, proposal.box) >= float(merge_iou)),
            None,
        )
        if matched_index is None:
            merged.append(proposal)
            continue
        existing = merged[matched_index]
        source = "+".join(sorted(set((existing.source + "+" + proposal.source).split("+"))))
        merged[matched_index] = ROIProposal(
            box=_union_box(existing.box, proposal.box),
            score=max(existing.score, proposal.score),
            area=existing.area + proposal.area,
            source=source,
        )
    return sorted(merged, key=lambda item: item.score, reverse=True)


def propose_rois(
    coarse_heatmap: np.ndarray,
    coarse_threshold: float,
    roi_topk: int,
    roi_min_area: int,
    roi_expand_ratio: float,
    roi_merge_iou: float,
    fine_tile_size: int,
    max_fine_rois: int,
    min_fine_rois: int = 1,
) -> ROISelection:
    """Create high-resolution fine-pass ROIs without discarding the coarse map.

    Components above the coarse threshold are kept when large enough. The best
    `roi_topk` components are also retained even when tiny, so an isolated small
    defect is not removed solely by the area filter. Global NMS-separated peaks
    are used only when the retained component set is below `min_fine_rois`.
    """
    heatmap = np.asarray(coarse_heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"coarse_heatmap must be 2D, got {heatmap.shape}")
    height, width = heatmap.shape
    if not np.isfinite(heatmap).any():
        return ROISelection((), 0, 0, 0, "coarse_heatmap_has_no_finite_score")

    all_components, area_components = _component_proposals(
        heatmap,
        threshold=float(coarse_threshold),
        min_area=max(1, int(roi_min_area)),
    )
    protected_small = sorted(all_components, key=lambda item: item.score, reverse=True)[:max(0, int(roi_topk))]
    selected = list(area_components)
    for proposal in protected_small:
        if proposal not in selected:
            selected.append(
                ROIProposal(box=proposal.box, score=proposal.score, area=proposal.area, source="threshold_small_topk")
            )

    fallback_count = 0
    if len(selected) < max(0, int(min_fine_rois)) and int(roi_topk) > 0:
        needed = max(int(roi_topk), int(min_fine_rois) - len(selected))
        fallback = _top_peak_proposals(heatmap, topk=needed, fine_tile_size=fine_tile_size)
        fallback_count = len(fallback)
        selected.extend(fallback)

    if not selected:
        reason = "no_threshold_component_and_global_topk_disabled" if int(roi_topk) <= 0 else "no_candidate_after_global_topk"
        return ROISelection((), len(all_components), len(area_components), 0, reason)

    expanded = [
        ROIProposal(
            box=expand_box(
                proposal.box,
                width=width,
                height=height,
                expand_ratio=roi_expand_ratio,
                min_size=max(1, int(fine_tile_size)),
            ),
            score=proposal.score,
            area=proposal.area,
            source=proposal.source,
        )
        for proposal in selected
    ]
    merged = _merge_proposals(expanded, merge_iou=max(0.0, min(1.0, float(roi_merge_iou))))
    limited = tuple(merged[:max(0, int(max_fine_rois))])
    reason = "threshold_components" if all_components else "global_topk_fallback"
    if not limited:
        reason = "all_candidates_removed_by_max_fine_rois"
    return ROISelection(
        proposals=limited,
        component_count=len(all_components),
        retained_component_count=len(area_components),
        fallback_count=fallback_count,
        reason=reason,
    )
