import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_DIR_HINTS = {"ground_truth", "mask", "masks", "label", "labels", "gt"}
NORMAL_HINTS = {"good", "normal", "ok", "negative"}
ANOMALY_HINTS = {
    "anomaly",
    "anomalies",
    "defect",
    "defects",
    "defective",
    "bad",
    "positive",
}
EXCLUDED_CATEGORY_DIRS = {
    ".git",
    "__pycache__",
    "split_csv",
    "visa_pytorch",
    "visa_pytorch_old",
    "images",
    "masks",
    "data",
    "train",
    "test",
    "val",
    "validation",
    "ground_truth",
}
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_environment(value):
    if not isinstance(value, str):
        return value
    return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(2) or ""), value)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是键值映射：{config_path}")
    payload = {key: _expand_environment(value) for key, value in payload.items()}
    project_root = config_path.resolve().parents[1]
    for key in ("data_root", "output_root"):
        value = str(payload.get(key, "")).strip()
        if not value:
            raise ValueError(f"配置缺少 {key}：{config_path}")
        path = Path(value).expanduser()
        payload[key] = str((project_root / path).resolve() if not path.is_absolute() else path.resolve())
    return payload


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def has_mask_hint(path: Path) -> bool:
    return any(part.lower() in MASK_DIR_HINTS for part in path.parts)


def iter_images(root: Path):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if is_image_file(p)]


def image_resolution_stats(image_files):
    widths = []
    heights = []
    failed = []
    for image_file in image_files:
        try:
            with Image.open(image_file) as img:
                w, h = img.size
            widths.append(int(w))
            heights.append(int(h))
        except Exception as exc:
            failed.append({"path": str(image_file), "error": f"{type(exc).__name__}: {exc}"})

    if not widths:
        return {
            "resolution_sample_count": 0,
            "min_width": None,
            "max_width": None,
            "mean_width": None,
            "min_height": None,
            "max_height": None,
            "mean_height": None,
            "failed_images": failed,
        }

    return {
        "resolution_sample_count": len(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": round(mean(widths), 2),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": round(mean(heights), 2),
        "failed_images": failed,
    }


def make_category_row(dataset_name, dataset_type, root, category, normal_files, anomaly_files, mask_files, extra=None):
    input_files = list(normal_files) + list(anomaly_files)
    stats = image_resolution_stats(input_files)
    row = {
        "dataset": dataset_name,
        "dataset_type": dataset_type,
        "root": str(root),
        "category": category,
        "normal_count": len(normal_files),
        "anomaly_count": len(anomaly_files),
        "total_image_count": len(input_files),
        "mask_count": len(mask_files),
        "has_mask": len(mask_files) > 0,
        **{k: v for k, v in stats.items() if k != "failed_images"},
        "failed_image_count": len(stats["failed_images"]),
    }
    if extra:
        row.update(extra)
    return row, stats["failed_images"]


def child_dirs(root: Path):
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def summarize_mvtec(root: Path):
    rows = []
    failures = []
    categories = [
        p
        for p in child_dirs(root)
        if (p / "train").exists() and (p / "test").exists()
    ]

    for category_dir in sorted(categories, key=lambda p: p.name.lower()):
        train_good = iter_images(category_dir / "train" / "good")
        test_good = iter_images(category_dir / "test" / "good")
        normal_files = train_good + test_good

        anomaly_files = []
        test_root = category_dir / "test"
        for defect_dir in child_dirs(test_root):
            if defect_dir.name.lower() == "good":
                continue
            anomaly_files.extend(iter_images(defect_dir))

        mask_files = iter_images(category_dir / "ground_truth")
        defect_types = sorted(
            [p.name for p in child_dirs(test_root) if p.name.lower() != "good"],
            key=str.lower,
        )
        row, row_failures = make_category_row(
            "MVTec",
            "mvtec_ad",
            root,
            category_dir.name,
            normal_files,
            anomaly_files,
            mask_files,
            extra={"defect_types": "|".join(defect_types)},
        )
        rows.append(row)
        failures.extend(row_failures)

    return rows, failures


def summarize_visa(root: Path):
    rows = []
    failures = []
    categories = [
        p
        for p in child_dirs(root)
        if p.name.lower() not in EXCLUDED_CATEGORY_DIRS
        and ((p / "Data" / "Images").exists() or any(is_image_file(x) for x in p.rglob("*")))
    ]

    for category_dir in sorted(categories, key=lambda p: p.name.lower()):
        images_root = category_dir / "Data" / "Images"
        masks_root = category_dir / "Data" / "Masks"
        if images_root.exists():
            normal_files = iter_images(images_root / "Normal")
            anomaly_files = iter_images(images_root / "Anomaly")
            if not normal_files and not anomaly_files:
                normal_files, anomaly_files = split_generic_images(images_root)
        else:
            normal_files, anomaly_files = split_generic_images(category_dir)

        mask_files = iter_images(masks_root) if masks_root.exists() else [
            p for p in iter_images(category_dir) if has_mask_hint(p)
        ]

        defect_types = sorted(
            [p.name for p in child_dirs(images_root) if p.name.lower() not in NORMAL_HINTS],
            key=str.lower,
        ) if images_root.exists() else []
        row, row_failures = make_category_row(
            "VisA",
            "visa",
            root,
            category_dir.name,
            normal_files,
            anomaly_files,
            mask_files,
            extra={"defect_types": "|".join(defect_types)},
        )
        rows.append(row)
        failures.extend(row_failures)

    return rows, failures


def split_generic_images(root: Path):
    normal_files = []
    anomaly_files = []
    unknown_files = []
    for image_file in iter_images(root):
        if has_mask_hint(image_file):
            continue
        rel_parts = {part.lower() for part in image_file.relative_to(root).parts}
        if rel_parts & NORMAL_HINTS:
            normal_files.append(image_file)
        elif rel_parts & ANOMALY_HINTS:
            anomaly_files.append(image_file)
        else:
            unknown_files.append(image_file)
    return normal_files + unknown_files, anomaly_files


def summarize_generic(root: Path, dataset_name: str):
    rows = []
    failures = []
    categories = [
        p
        for p in child_dirs(root)
        if p.name.lower() not in EXCLUDED_CATEGORY_DIRS and any(is_image_file(x) for x in p.rglob("*"))
    ]

    for category_dir in sorted(categories, key=lambda p: p.name.lower()):
        normal_files, anomaly_files = split_generic_images(category_dir)
        mask_files = [p for p in iter_images(category_dir) if has_mask_hint(p)]
        row, row_failures = make_category_row(
            dataset_name,
            "generic",
            root,
            category_dir.name,
            normal_files,
            anomaly_files,
            mask_files,
            extra={"defect_types": ""},
        )
        rows.append(row)
        failures.extend(row_failures)

    return rows, failures


def find_child_case_insensitive(root: Path, names):
    if not root.exists():
        return None
    wanted = {name.lower() for name in names}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in wanted:
            return child
    return None


def valid_config_path(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper().startswith("TODO"):
        return None
    path = Path(text)
    return path if path.exists() else None


def discover_roots(config):
    data_root = Path(config["data_root"])
    roots = []

    mvtec_root = valid_config_path(config.get("mvtec_root")) or find_child_case_insensitive(
        data_root, ["mvtec", "mvtec_ad", "MVTec"]
    )
    visa_root = valid_config_path(config.get("visa_root")) or find_child_case_insensitive(
        data_root, ["visa", "VisA"]
    )
    dagm_root = find_child_case_insensitive(data_root, ["DAGM"])

    if mvtec_root:
        roots.append(("MVTec", mvtec_root))
    else:
        print("[WARN] MVTec root was not detected. Please confirm configs/path_config.yaml.")

    if visa_root:
        roots.append(("VisA", visa_root))
    else:
        print("[WARN] VisA root was not detected. Please confirm configs/path_config.yaml.")

    if dagm_root:
        roots.append(("DAGM", dagm_root))

    return roots


def write_outputs(rows, failures, config):
    output_root = Path(config["output_root"])
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_groups = {}
    for row in rows:
        dataset_groups.setdefault(row["dataset"], []).append(row)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(Path(config["data_root"])),
        "datasets": [
            {
                "name": dataset_name,
                "category_count": len(group_rows),
                "categories": [r["category"] for r in group_rows],
                "rows": group_rows,
            }
            for dataset_name, group_rows in sorted(dataset_groups.items())
        ],
        "failed_images": failures,
    }

    json_path = results_dir / "dataset_summary.json"
    csv_path = results_dir / "dataset_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "dataset",
        "dataset_type",
        "root",
        "category",
        "normal_count",
        "anomaly_count",
        "total_image_count",
        "mask_count",
        "has_mask",
        "resolution_sample_count",
        "min_width",
        "max_width",
        "mean_width",
        "min_height",
        "max_height",
        "mean_height",
        "failed_image_count",
        "defect_types",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path, dataset_groups


def main():
    parser = argparse.ArgumentParser(description="Check AOI anomaly datasets without modifying raw files.")
    parser.add_argument("--config", default="configs/path_config.yaml", help="Path config yaml.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    roots = discover_roots(config)

    all_rows = []
    all_failures = []
    for dataset_name, root in roots:
        if dataset_name == "MVTec":
            rows, failures = summarize_mvtec(root)
        elif dataset_name == "VisA":
            rows, failures = summarize_visa(root)
        else:
            rows, failures = summarize_generic(root, dataset_name)
        all_rows.extend(rows)
        all_failures.extend(failures)

    json_path, csv_path, dataset_groups = write_outputs(all_rows, all_failures, config)

    print("Dataset check finished.")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")
    for dataset_name in sorted(dataset_groups):
        rows = dataset_groups[dataset_name]
        normal_total = sum(r["normal_count"] for r in rows)
        anomaly_total = sum(r["anomaly_count"] for r in rows)
        mask_total = sum(r["mask_count"] for r in rows)
        print(
            f"{dataset_name}: categories={len(rows)}, "
            f"normal={normal_total}, anomaly={anomaly_total}, masks={mask_total}"
        )

    if all_failures:
        print(f"[WARN] Failed to read {len(all_failures)} image files. See dataset_summary.json.")


if __name__ == "__main__":
    main()
