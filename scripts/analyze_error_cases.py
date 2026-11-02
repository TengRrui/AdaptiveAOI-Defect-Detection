from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.path_utils import load_config


def reason_for(row) -> str:
    defect = str(row.get("defect_type", "")).lower()
    score = float(row.get("score", 0.0))
    threshold = float(row.get("threshold", 0.0))
    margin = abs(score - threshold)
    hints = []
    if margin < max(0.05, threshold * 0.1):
        hints.append("threshold-close")
    if any(k in defect for k in ["scratch", "crack", "thread", "poke", "small"]):
        hints.append("small-defect")
    if any(k in defect for k in ["color", "print", "logical", "swap"]):
        hints.append("appearance-or-logic")
    if not hints:
        hints.append("feature-overlap-or-structure-change")
    return "|".join(hints)


def main():
    parser = argparse.ArgumentParser(description="Analyze false positive and false negative cases.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    rows = []
    for result_csv in (output_root / "results").rglob("*result.csv"):
        try:
            df = pd.read_csv(result_csv)
        except Exception:
            continue
        required = {"label_gt", "pred_label", "score", "threshold", "image_path"}
        if not required.issubset(set(df.columns)):
            continue
        errors = df[df["label_gt"].astype(int) != df["pred_label"].astype(int)].copy()
        for item in errors.to_dict("records"):
            rows.append(
                {
                    "dataset": item.get("dataset", ""),
                    "category": item.get("category", ""),
                    "case_type": "false_positive" if int(item["label_gt"]) == 0 else "false_negative",
                    "image_path": item.get("image_path", ""),
                    "defect_type": item.get("defect_type", ""),
                    "score": item.get("score", ""),
                    "threshold": item.get("threshold", ""),
                    "heatmap_path": item.get("heatmap_path", ""),
                    "mask_path": item.get("mask_path", ""),
                    "overlay_path": item.get("overlay_path", ""),
                    "possible_reason": reason_for(item),
                    "source_result_csv": str(result_csv),
                }
            )
    out_csv = output_root / "results" / "error_cases.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    notes = [
        "# Error Case Notes",
        "",
        f"Total error cases collected: {len(rows)}",
        "",
        "- `threshold-close`: score is close to threshold; threshold calibration likely matters.",
        "- `small-defect`: defect type suggests small local defect such as scratch/crack/poke/thread.",
        "- `appearance-or-logic`: defect may involve color, print, swap, or logical structure changes.",
        "- `feature-overlap-or-structure-change`: normal and anomaly features may overlap under the current backbone/memory bank.",
    ]
    (output_root / "results" / "error_case_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"error_cases: {out_csv}")


if __name__ == "__main__":
    main()
