"""Strategy-specific admission rules for feedback-driven updates.

The policy in this module answers one question only: *may the selected
reviewed evidence start an update task?*  It deliberately does not decide
whether a generated candidate may be frozen or applied.  That decision stays
with the candidate validation gate (Recall/FPR/latency/memory/integrity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


_MODE_ALIASES = {
    "normal_memory": "normal_only",
    "normal_only": "normal_only",
    "normal_defect_calibration": "normal_defect_calibration",
    "lightweight_calibration": "normal_memory_defect_guided_lite",
    "normal_memory_defect_guided_lite": "normal_memory_defect_guided_lite",
    "dual_memory": "normal_defect_dual_memory",
    "normal_defect_dual_memory": "normal_defect_dual_memory",
}


def normalize_method_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return _MODE_ALIASES.get(raw, raw or "normal_only")


@dataclass(frozen=True)
class StrategyEligibilityPolicy:
    strategy_code: str
    display_name: str
    updates: tuple[str, ...]
    min_normal_samples: int
    min_defect_samples: int
    require_both_classes: bool
    require_safe_normal: bool
    min_safe_normal_samples: int
    require_defect_mask: bool
    min_reliable_mask_samples: int
    allow_normal_only: bool
    allow_defect_only: bool
    allowed_label_types: tuple[str, ...]
    mask_requirement: str
    description: str
    recommended_normal_samples: int
    recommended_defect_samples: int
    recommended_safe_normal_samples: int
    recommended_reliable_mask_samples: int
    validation_min_total_samples: int
    validation_require_both_classes: bool
    validation_requirements: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["updates"] = list(self.updates)
        value["allowed_label_types"] = list(self.allowed_label_types)
        value["validation_requirements"] = list(self.validation_requirements)
        value["hard_requirements"] = {
            "normal_samples": self.min_normal_samples,
            "defect_samples": self.min_defect_samples,
            "safe_normal_samples": self.min_safe_normal_samples if self.require_safe_normal else 0,
            "reliable_mask_defect_samples": self.min_reliable_mask_samples if self.require_defect_mask else 0,
        }
        value["recommended_requirements"] = {
            "normal_samples": self.recommended_normal_samples,
            "defect_samples": self.recommended_defect_samples,
            "safe_normal_samples": self.recommended_safe_normal_samples,
            "reliable_mask_defect_samples": self.recommended_reliable_mask_samples,
        }
        return value


_POLICIES: dict[str, StrategyEligibilityPolicy] = {
    # _metrics/_choose_threshold can calculate a boundary with one evidence
    # point per class.  Five per class remains a recommendation for useful
    # stability evidence; the final validation gate also requires five total
    # labelled points and both classes.
    "threshold_only": StrategyEligibilityPolicy(
        strategy_code="threshold_only",
        display_name="仅调整判定参数",
        updates=("threshold",),
        min_normal_samples=1,
        min_defect_samples=1,
        require_both_classes=True,
        require_safe_normal=False,
        min_safe_normal_samples=0,
        require_defect_mask=False,
        min_reliable_mask_samples=0,
        allow_normal_only=False,
        allow_defect_only=False,
        allowed_label_types=("normal", "defect"),
        mask_requirement="不要求 Mask",
        description="用两类人工标签重新估计图像级判定边界，不改变特征记忆库。",
        recommended_normal_samples=5,
        recommended_defect_samples=5,
        recommended_safe_normal_samples=0,
        recommended_reliable_mask_samples=0,
        validation_min_total_samples=5,
        validation_require_both_classes=True,
        validation_requirements=("候选 Recall/FPR 需要两类标签", "至少 5 条总证据才进行稳定性估计"),
    ),
    # The memory replacement code consumes only safe false-positive normals.
    # Defect evidence is useful for a later validation, but is not required to
    # construct the normal-bank candidate itself.
    "normal_memory": StrategyEligibilityPolicy(
        strategy_code="normal_memory",
        display_name="更新正常记忆",
        updates=("normal_memory",),
        min_normal_samples=1,
        min_defect_samples=0,
        require_both_classes=False,
        require_safe_normal=True,
        min_safe_normal_samples=1,
        require_defect_mask=False,
        min_reliable_mask_samples=0,
        allow_normal_only=True,
        allow_defect_only=False,
        allowed_label_types=("normal",),
        mask_requirement="正常记忆候选不能带缺陷 Mask",
        description="只用人工确认的安全正常误检样本替换受限预算内的正常特征。",
        recommended_normal_samples=5,
        recommended_defect_samples=5,
        recommended_safe_normal_samples=3,
        recommended_reliable_mask_samples=0,
        validation_min_total_samples=5,
        validation_require_both_classes=True,
        validation_requirements=("记忆更新可只用正常样本开始", "冻结/应用仍需正常与缺陷证据完成 Recall/FPR 验证"),
    ),
    # Prototype extraction only reads masked defect regions.  Normal evidence
    # belongs to the later false-positive/Recall validation layer.
    "defect_prototypes": StrategyEligibilityPolicy(
        strategy_code="defect_prototypes",
        display_name="更新缺陷原型",
        updates=("defect_memory",),
        min_normal_samples=0,
        min_defect_samples=1,
        require_both_classes=False,
        require_safe_normal=False,
        min_safe_normal_samples=0,
        require_defect_mask=True,
        min_reliable_mask_samples=1,
        allow_normal_only=False,
        allow_defect_only=True,
        allowed_label_types=("defect",),
        mask_requirement="至少 1 条带可靠 Mask 的审核缺陷",
        description="只从确认缺陷区域构建缺陷参考特征，不写入正常记忆库。",
        recommended_normal_samples=5,
        recommended_defect_samples=5,
        recommended_safe_normal_samples=0,
        recommended_reliable_mask_samples=3,
        validation_min_total_samples=5,
        validation_require_both_classes=True,
        validation_requirements=("缺陷原型可先独立构建", "冻结/应用仍需正常样本完成 FPR 验证"),
    ),
    # fit_lightweight_profile requires both arrays to be non-empty and its
    # stratified CV needs at least two rows in each class for two folds.
    "lightweight_calibrator": StrategyEligibilityPolicy(
        strategy_code="lightweight_calibrator",
        display_name="更新轻量校准器",
        updates=("threshold", "lightweight_calibrator"),
        min_normal_samples=2,
        min_defect_samples=2,
        require_both_classes=True,
        require_safe_normal=False,
        min_safe_normal_samples=0,
        require_defect_mask=False,
        min_reliable_mask_samples=0,
        allow_normal_only=False,
        allow_defect_only=False,
        allowed_label_types=("normal", "defect"),
        mask_requirement="不强制要求 Mask",
        description="用两类监督证据拟合轻量评分校准器和判定参数，不更新骨干或正常记忆库。",
        recommended_normal_samples=5,
        recommended_defect_samples=5,
        recommended_safe_normal_samples=0,
        recommended_reliable_mask_samples=0,
        validation_min_total_samples=5,
        validation_require_both_classes=True,
        validation_requirements=("轻量校准至少支持 2 折分层交叉验证", "候选版本仍需通过 Recall/FPR 门禁"),
    ),
}


def _merge_policies(strategy_code: str, policies: list[StrategyEligibilityPolicy], *, description: str) -> StrategyEligibilityPolicy:
    updates: list[str] = []
    for policy in policies:
        for update in policy.updates:
            if update not in updates:
                updates.append(update)
    validation_notes: list[str] = []
    for policy in policies:
        for note in policy.validation_requirements:
            if note not in validation_notes:
                validation_notes.append(note)
    return StrategyEligibilityPolicy(
        strategy_code=strategy_code,
        display_name="综合更新",
        updates=tuple(updates),
        min_normal_samples=max(policy.min_normal_samples for policy in policies),
        min_defect_samples=max(policy.min_defect_samples for policy in policies),
        require_both_classes=any(policy.require_both_classes for policy in policies),
        require_safe_normal=any(policy.require_safe_normal for policy in policies),
        min_safe_normal_samples=max(policy.min_safe_normal_samples for policy in policies),
        require_defect_mask=any(policy.require_defect_mask for policy in policies),
        min_reliable_mask_samples=max(policy.min_reliable_mask_samples for policy in policies),
        allow_normal_only=all(policy.allow_normal_only for policy in policies),
        allow_defect_only=all(policy.allow_defect_only for policy in policies),
        allowed_label_types=tuple(sorted({label for policy in policies for label in policy.allowed_label_types})),
        mask_requirement="；".join(dict.fromkeys(policy.mask_requirement for policy in policies)),
        description=description,
        recommended_normal_samples=max(policy.recommended_normal_samples for policy in policies),
        recommended_defect_samples=max(policy.recommended_defect_samples for policy in policies),
        recommended_safe_normal_samples=max(policy.recommended_safe_normal_samples for policy in policies),
        recommended_reliable_mask_samples=max(policy.recommended_reliable_mask_samples for policy in policies),
        validation_min_total_samples=max(policy.validation_min_total_samples for policy in policies),
        validation_require_both_classes=any(policy.validation_require_both_classes for policy in policies),
        validation_requirements=tuple(validation_notes),
    )


def get_strategy_policy(strategy_code: Any, base_method_mode: Any = None) -> StrategyEligibilityPolicy:
    code = str(strategy_code or "threshold_only").strip().lower()
    if code != "comprehensive":
        return _POLICIES.get(code, _POLICIES["threshold_only"])

    mode = normalize_method_mode(base_method_mode)
    components = [_POLICIES["threshold_only"], _POLICIES["normal_memory"]]
    if mode == "normal_defect_dual_memory":
        components.append(_POLICIES["defect_prototypes"])
    elif mode == "normal_memory_defect_guided_lite":
        components.append(_POLICIES["lightweight_calibrator"])
    return _merge_policies(
        "comprehensive",
        components,
        description="按基础方法实际支持的组件动态合并判定参数、正常记忆、缺陷原型或轻量校准要求。",
    )


def list_strategy_policies(base_method_mode: Any = None) -> list[dict[str, Any]]:
    codes = ["threshold_only", "normal_memory", "defect_prototypes", "lightweight_calibrator", "comprehensive"]
    return [get_strategy_policy(code, base_method_mode).as_dict() for code in codes]


def evaluate_strategy_eligibility(
    strategy_code: Any,
    base_method_mode: Any,
    counts: Mapping[str, Any],
) -> dict[str, Any]:
    policy = get_strategy_policy(strategy_code, base_method_mode)

    def count(name: str) -> int:
        try:
            return max(0, int(counts.get(name, 0) or 0))
        except (TypeError, ValueError):
            return 0

    current = {
        "normal": count("normal"),
        "defect": count("defect"),
        "safe_normal": count("safe_normal"),
        "reliable_mask_defect": count("reliable_mask_defect"),
    }
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def blocker(code: str, message: str, actual: int, required: int, field: str) -> None:
        blockers.append({"code": code, "message": message, "actual": actual, "required": required, "field": field})

    if current["normal"] < policy.min_normal_samples:
        blocker(
            "STRATEGY_NORMAL_SAMPLES_INSUFFICIENT",
            f"{policy.display_name}至少需要 {policy.min_normal_samples} 条人工确认正常样本；当前 {current['normal']} 条。",
            current["normal"], policy.min_normal_samples, "normal",
        )
    if current["defect"] < policy.min_defect_samples:
        blocker(
            "STRATEGY_DEFECT_SAMPLES_INSUFFICIENT",
            f"{policy.display_name}至少需要 {policy.min_defect_samples} 条人工确认缺陷样本；当前 {current['defect']} 条。",
            current["defect"], policy.min_defect_samples, "defect",
        )
    if policy.require_safe_normal and current["safe_normal"] < policy.min_safe_normal_samples:
        blocker(
            "STRATEGY_SAFE_NORMAL_INSUFFICIENT",
            f"{policy.display_name}至少需要 {policy.min_safe_normal_samples} 条可安全写入正常记忆的误检样本（原判异常、人工正常、无缺陷 Mask）；当前 {current['safe_normal']} 条。",
            current["safe_normal"], policy.min_safe_normal_samples, "safe_normal",
        )
    if policy.require_defect_mask and current["reliable_mask_defect"] < policy.min_reliable_mask_samples:
        blocker(
            "STRATEGY_RELIABLE_MASK_INSUFFICIENT",
            f"{policy.display_name}至少需要 {policy.min_reliable_mask_samples} 条带可靠 Mask 的审核缺陷样本；当前 {current['reliable_mask_defect']} 条。",
            current["reliable_mask_defect"], policy.min_reliable_mask_samples, "reliable_mask_defect",
        )

    def recommendation(field: str, label: str, recommended: int) -> None:
        if recommended <= 0 or current[field] >= recommended or current[field] < getattr(policy, f"min_{field}_samples", 0):
            return
        warnings.append({
            "code": f"STRATEGY_{field.upper()}_RECOMMENDED",
            "message": f"{label}当前 {current[field]} 条，已达到启动最低条件；建议至少 {recommended} 条以提高候选版本验证稳定性。",
            "actual": current[field],
            "recommended": recommended,
            "field": field,
        })

    recommendation("normal", "正常纠错样本", policy.recommended_normal_samples)
    recommendation("defect", "缺陷纠错样本", policy.recommended_defect_samples)
    if policy.require_safe_normal and current["safe_normal"] < policy.recommended_safe_normal_samples and current["safe_normal"] >= policy.min_safe_normal_samples:
        warnings.append({
            "code": "STRATEGY_SAFE_NORMAL_RECOMMENDED",
            "message": f"安全正常样本当前 {current['safe_normal']} 条；建议至少 {policy.recommended_safe_normal_samples} 条以降低正常记忆更新偏差。",
            "actual": current["safe_normal"],
            "recommended": policy.recommended_safe_normal_samples,
            "field": "safe_normal",
        })
    if policy.require_defect_mask and current["reliable_mask_defect"] < policy.recommended_reliable_mask_samples and current["reliable_mask_defect"] >= policy.min_reliable_mask_samples:
        warnings.append({
            "code": "STRATEGY_MASK_RECOMMENDED",
            "message": f"可靠 Mask 缺陷样本当前 {current['reliable_mask_defect']} 条；建议至少 {policy.recommended_reliable_mask_samples} 条以提高缺陷原型覆盖。",
            "actual": current["reliable_mask_defect"],
            "recommended": policy.recommended_reliable_mask_samples,
            "field": "reliable_mask_defect",
        })

    total = current["normal"] + current["defect"]
    validation_ready = total >= policy.validation_min_total_samples and (
        not policy.validation_require_both_classes or (current["normal"] > 0 and current["defect"] > 0)
    )
    if not validation_ready:
        missing: list[str] = []
        if policy.validation_require_both_classes and current["normal"] == 0:
            missing.append("正常证据")
        if policy.validation_require_both_classes and current["defect"] == 0:
            missing.append("缺陷证据")
        if total < policy.validation_min_total_samples:
            missing.append(f"总证据至少 {policy.validation_min_total_samples} 条")
        warnings.append({
            "code": "CANDIDATE_VALIDATION_EVIDENCE_INSUFFICIENT",
            "message": "当前样本可以开始本策略计算，但候选版本暂不具备完整 Recall/FPR 准入验证：" + "、".join(missing) + "。",
            "field": "candidate_validation",
        })

    return {
        "strategy_code": policy.strategy_code,
        "display_name": policy.display_name,
        "policy": policy.as_dict(),
        "current_counts": current,
        "eligibility_requirements": policy.as_dict()["hard_requirements"],
        "recommended_requirements": policy.as_dict()["recommended_requirements"],
        "blockers": blockers,
        "warnings": warnings,
        "can_start": not blockers,
        "candidate_validation_ready": validation_ready,
        "can_auto_apply": not blockers and validation_ready,
    }
