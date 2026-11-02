"""Low-complexity defect-guided calibration on top of one normal memory bank.

The package deliberately stores only robust scalar statistics and a tiny
monotone calibrator.  It never stores defect embeddings, image bytes, or a
second nearest-neighbour index.

The public names remain compatible with existing build/evaluation callers, but
the implementation modules are resolved lazily.  In particular, training-only
scikit-learn metrics must not be imported when a frozen inference profile is
being read during backend cold start.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "apply_gate": ("calibration_gate", "apply_gate"),
    "fit_gate": ("calibration_gate", "fit_gate"),
    "DISPLAY_NAME": ("lightweight_profile", "DISPLAY_NAME"),
    "MODE": ("lightweight_profile", "MODE"),
    "VERSION": ("lightweight_profile", "VERSION"),
    "apply_lightweight_profile": ("lightweight_profile", "apply_lightweight_profile"),
    "fit_lightweight_profile": ("lightweight_profile", "fit_lightweight_profile"),
    "load_lightweight_profile": ("lightweight_profile", "load_lightweight_profile"),
    "write_lightweight_profile": ("lightweight_profile", "write_lightweight_profile"),
    "apply_monotonic_calibrator": ("monotonic_calibrator", "apply_monotonic_calibrator"),
    "fit_monotonic_calibrator": ("monotonic_calibrator", "fit_monotonic_calibrator"),
    "fit_robust_normalizer": ("robust_score_normalizer", "fit_robust_normalizer"),
    "normalize_patch_scores": ("robust_score_normalizer", "normalize_patch_scores"),
    "extract_pooling_features": ("spatial_score_pooling", "extract_pooling_features"),
    "select_pooling_ratio": ("spatial_score_pooling", "select_pooling_ratio"),
}


__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value

__all__ = [
    "apply_gate",
    "apply_lightweight_profile",
    "apply_monotonic_calibrator",
    "extract_pooling_features",
    "fit_gate",
    "fit_lightweight_profile",
    "fit_monotonic_calibrator",
    "fit_robust_normalizer",
    "load_lightweight_profile",
    "write_lightweight_profile",
    "MODE",
    "DISPLAY_NAME",
    "VERSION",
    "normalize_patch_scores",
    "select_pooling_ratio",
]
