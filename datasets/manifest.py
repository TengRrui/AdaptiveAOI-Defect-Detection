from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = [
    "dataset",
    "category",
    "split",
    "image_path",
    "label",
    "label_name",
    "defect_type",
    "mask_path",
    "height",
    "width",
]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if is_image(p)], key=lambda p: str(p).lower())


def read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return int(height), int(width)


def row(dataset: str, category: str, split: str, image_path: Path, label: int, label_name: str, defect_type: str, mask_path: Path | None):
    height, width = read_size(image_path)
    mask_text = "" if mask_path is None or not mask_path.exists() else str(mask_path)
    return {
        "dataset": dataset,
        "category": category,
        "split": split.lower(),
        "image_path": str(image_path),
        "label": int(label),
        "label_name": label_name,
        "defect_type": defect_type,
        "mask_path": mask_text,
        "height": height,
        "width": width,
    }


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    if not mask_dir.exists():
        return None
    candidates = [
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}_mask.PNG",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.PNG",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}.JPG",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(mask_dir.glob(f"{stem}*"))
    return matches[0] if matches else None


def category_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    excluded = {"split_csv", "visa_pytorch", "visa_pytorch_old", "__pycache__"}
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name not in excluded],
        key=lambda p: p.name.lower(),
    )


def find_data_child(data_root: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    if not data_root.exists():
        return None
    for child in data_root.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return child
    return None


def build_mvtec_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for cat_dir in category_dirs(root):
        train_good = cat_dir / "train" / "good"
        test_root = cat_dir / "test"
        if not train_good.exists() or not test_root.exists():
            continue
        for image_path in iter_images(train_good):
            rows.append(row("mvtec", cat_dir.name, "train", image_path, 0, "good", "good", None))
        for defect_dir in category_dirs(test_root):
            is_good = defect_dir.name.lower() == "good"
            for image_path in iter_images(defect_dir):
                if is_good:
                    rows.append(row("mvtec", cat_dir.name, "test", image_path, 0, "good", "good", None))
                else:
                    mask_dir = cat_dir / "ground_truth" / defect_dir.name
                    mask_path = find_mask(mask_dir, image_path.stem)
                    rows.append(row("mvtec", cat_dir.name, "test", image_path, 1, "anomaly", defect_dir.name, mask_path))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_visa_from_split_csv(root: Path) -> pd.DataFrame:
    split_path = root / "split_csv" / "1cls.csv"
    rows = []
    csv_df = pd.read_csv(split_path)
    for item in csv_df.to_dict("records"):
        category = str(item["object"])
        split = str(item["split"]).lower()
        label_text = str(item["label"]).lower()
        label = 0 if label_text in {"normal", "good"} else 1
        label_name = "good" if label == 0 else "anomaly"
        defect_type = "good" if label == 0 else "anomaly"
        image_path = root / str(item["image"])
        mask_value = "" if pd.isna(item.get("mask", "")) else str(item.get("mask", "")).strip()
        mask_path = root / mask_value if mask_value else None
        if image_path.exists():
            rows.append(row("visa", category, split, image_path, label, label_name, defect_type, mask_path))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_visa_from_pytorch(root: Path) -> pd.DataFrame:
    rows = []
    for cat_dir in category_dirs(root):
        train_good = cat_dir / "train" / "good"
        test_root = cat_dir / "test"
        if not train_good.exists() or not test_root.exists():
            continue
        for image_path in iter_images(train_good):
            rows.append(row("visa", cat_dir.name, "train", image_path, 0, "good", "good", None))
        for defect_dir in category_dirs(test_root):
            is_good = defect_dir.name.lower() == "good"
            for image_path in iter_images(defect_dir):
                if is_good:
                    rows.append(row("visa", cat_dir.name, "test", image_path, 0, "good", "good", None))
                else:
                    mask_path = find_mask(cat_dir / "ground_truth" / defect_dir.name, image_path.stem)
                    rows.append(row("visa", cat_dir.name, "test", image_path, 1, "anomaly", defect_dir.name, mask_path))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_visa_manifest(root: Path, data_root: Path | None = None) -> pd.DataFrame:
    if (root / "split_csv" / "1cls.csv").exists():
        return build_visa_from_split_csv(root)
    if (root / "visa_pytorch").exists():
        return build_visa_from_pytorch(root / "visa_pytorch")
    if data_root is not None and (data_root / "visa_pytorch").exists():
        return build_visa_from_pytorch(data_root / "visa_pytorch")
    return build_visa_from_pytorch(root)


def parse_dagm_labels(labels_path: Path) -> dict[str, dict[str, str | int]]:
    labels = {}
    if not labels_path.exists():
        return labels
    for line in labels_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        image_name = parts[2]
        label = int(parts[1])
        mask_name = ""
        if len(parts) >= 5 and parts[4] not in {"0", "None", "none", "-"}:
            mask_name = parts[4]
        labels[image_name] = {"label": label, "mask_name": mask_name}
    return labels


def build_dagm_manifest(root: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    report_lines = ["# DAGM Structure Report", "", f"Root: `{root}`", ""]
    for cat_dir in category_dirs(root):
        if not (cat_dir / "Train").exists() and not (cat_dir / "Test").exists():
            continue
        report_lines.append(f"## {cat_dir.name}")
        for split_dir_name, split in [("Train", "train"), ("Test", "test")]:
            split_dir = cat_dir / split_dir_name
            labels_dir = split_dir / "Label"
            label_map = parse_dagm_labels(labels_dir / "Labels.txt")
            images = [p for p in iter_images(split_dir) if "label" not in p.parts and p.parent.name != "Label"]
            missing_label_count = 0
            anomaly_count = 0
            for image_path in images:
                info = label_map.get(image_path.name)
                if info is None:
                    missing_label_count += 1
                    label = 0
                    mask_name = ""
                else:
                    label = int(info["label"])
                    mask_name = str(info["mask_name"])
                mask_path = labels_dir / mask_name if label == 1 and mask_name else None
                anomaly_count += int(label == 1)
                rows.append(
                    row(
                        "dagm",
                        cat_dir.name,
                        split,
                        image_path,
                        label,
                        "anomaly" if label == 1 else "good",
                        "defect" if label == 1 else "good",
                        mask_path,
                    )
                )
            report_lines.append(
                f"- {split}: images={len(images)}, labels={len(label_map)}, anomalies={anomaly_count}, missing_labels={missing_label_count}"
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS), "\n".join(report_lines) + "\n"


def build_custom_manifest(root: Path, dataset_name: str = "custom") -> pd.DataFrame:
    rows = []
    for image_path in iter_images(root):
        if "mask" in {p.lower() for p in image_path.parts}:
            continue
        rows.append(row(dataset_name, root.name, "test", image_path, 0, "unknown", "unknown", None))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_all_manifests(data_root: Path) -> tuple[dict[str, pd.DataFrame], str]:
    data_root = Path(data_root)
    manifests = {}
    dagm_report = ""
    mvtec_root = find_data_child(data_root, ["mvtec", "mvtec_ad"])
    visa_root = find_data_child(data_root, ["visa", "VisA"])
    dagm_root = find_data_child(data_root, ["dagm"])

    if mvtec_root:
        manifests["mvtec"] = build_mvtec_manifest(mvtec_root)
    if visa_root:
        manifests["visa"] = build_visa_manifest(visa_root, data_root=data_root)
    elif (data_root / "visa_pytorch").exists():
        manifests["visa"] = build_visa_from_pytorch(data_root / "visa_pytorch")
    if dagm_root:
        manifests["dagm"], dagm_report = build_dagm_manifest(dagm_root)
    return manifests, dagm_report


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = []
    for (dataset, category), group in df.groupby(["dataset", "category"], sort=True):
        grouped.append(
            {
                "dataset": dataset,
                "category": category,
                "train_normal": int(((group["split"] == "train") & (group["label"] == 0)).sum()),
                "train_anomaly": int(((group["split"] == "train") & (group["label"] == 1)).sum()),
                "test_normal": int(((group["split"] == "test") & (group["label"] == 0)).sum()),
                "test_anomaly": int(((group["split"] == "test") & (group["label"] == 1)).sum()),
                "mask_count": int(group["mask_path"].fillna("").astype(str).ne("").sum()),
                "min_height": int(group["height"].min()),
                "max_height": int(group["height"].max()),
                "min_width": int(group["width"].min()),
                "max_width": int(group["width"].max()),
            }
        )
    return pd.DataFrame(grouped)


def write_manifest_outputs(data_root: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests, dagm_report = build_all_manifests(data_root)
    output_paths = {}
    all_frames = []
    for name, df in manifests.items():
        path = output_dir / f"manifest_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        output_paths[f"manifest_{name}"] = path
        all_frames.append(df)
    summary = summarize_manifest(pd.concat(all_frames, ignore_index=True)) if all_frames else pd.DataFrame()
    summary_csv = output_dir / "dataset_summary.csv"
    summary_json = output_dir / "dataset_summary.json"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    payload = {
        "data_root": str(data_root),
        "datasets": {
            name: {
                "category_count": int(df["category"].nunique()) if not df.empty else 0,
                "sample_count": int(len(df)),
                "categories": sorted(df["category"].unique().tolist()) if not df.empty else [],
            }
            for name, df in manifests.items()
        },
        "summary": summary.to_dict("records"),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_paths["dataset_summary_csv"] = summary_csv
    output_paths["dataset_summary_json"] = summary_json
    if dagm_report:
        report_path = output_dir / "dagm_structure_report.md"
        report_path.write_text(dagm_report, encoding="utf-8")
        output_paths["dagm_structure_report"] = report_path
    return output_paths


def load_manifest(manifest_path: Path, dataset: str | None = None, category: str | None = None, split: str | None = None, label: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if dataset is not None:
        df = df[df["dataset"].astype(str).str.lower() == dataset.lower()]
    if category is not None:
        df = df[df["category"].astype(str) == category]
    if split is not None:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    if label is not None:
        df = df[df["label"].astype(int) == int(label)]
    return df.reset_index(drop=True)
