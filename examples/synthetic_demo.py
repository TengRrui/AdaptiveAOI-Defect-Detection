"""Exercise real model build and inference with generated textures, without external image data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.domain_adaptation import build_and_calibrate_domain
from methods.patchcore.infer import PatchCoreInferencer
from tools.path_utils import load_yaml
from video.video_inference import VideoDetectionOptions, run_video_detection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    _, x = np.mgrid[:128, :128]
    groups: dict[str, list[Path]] = {}
    for group, count in [("normal", 12), ("calibration_defect", 8), ("inspection", 4)]:
        folder = destination / "inputs" / group
        folder.mkdir(parents=True, exist_ok=True)
        groups[group] = []
        for index in range(count):
            texture = 120 + 22 * np.sin(x / 5) + rng.normal(0, 2, (128, 128))
            image = np.repeat(texture[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)
            if group == "calibration_defect" or (group == "inspection" and index % 2):
                image[25:100, 40 + index:45 + index] = (245, 245, 245)
            path = folder / f"texture_{index:02d}.png"
            Image.fromarray(image).save(path)
            groups[group].append(path)

    config = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    config.update(dataset="synthetic", category="texture", device=args.device,
                  project_root=str(PROJECT_ROOT), data_root=str(destination / "inputs"),
                  image_size=128, batch_size=2, use_fp16=False, async_prefetch=False,
                  async_save_outputs=False, coreset_ratio=0.05, max_memory_patches=1024,
                  memory_backend="torch_gpu_flat" if args.device == "cuda" else "sklearn")
    results = {"purpose": "software_smoke_test", "device": args.device, "modes": {}}
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for mode in ("normal_only", "normal_memory_defect_guided_lite"):
        model_dir = destination / mode
        mode_config = {**config, "output_root": str(model_dir / "results"), "method_mode": mode,
                       "threshold_method": "normal_percentile" if mode == "normal_only" else "best_f1"}
        print(f"Building {mode}", flush=True)
        built = build_and_calibrate_domain(groups["normal"], groups["calibration_defect"], mode_config, model_dir)
        mode_config.update(calibration_path=str(model_dir / "threshold.json"), use_calibrated_threshold=True,
                           lightweight_calibrator_path=str(model_dir / "lightweight_calibrator.json"))
        inferencer = PatchCoreInferencer(mode_config, memory_path=built["memory_path"])
        records = []
        for path in groups["inspection"]:
            result = inferencer.infer_image(path, save_outputs=True)
            assert np.isfinite(result["score"])
            for key in ("heatmap_path", "mask_path", "overlay_path"):
                assert Path(result[key]).is_file(), key
            records.append({key: result[key] for key in ("score", "threshold", "pred_label", "heatmap_path", "mask_path", "overlay_path")})
        results["modes"][mode] = {"memory": str(built["memory_path"]), "images": records}
        if mode == "normal_only":
            video_path = destination / "inputs" / "generated.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (128, 128))
            if not writer.isOpened():
                raise RuntimeError("MJPG video writer is unavailable")
            try:
                for path in groups["inspection"]:
                    writer.write(cv2.imread(str(path)))
            finally:
                writer.release()
            results["video"] = run_video_detection(VideoDetectionOptions(
                input_video=video_path, output_video=destination / "annotated.avi",
                config=mode_config, memory_path=built["memory_path"],
                device=args.device, max_frames=4, overwrite=True,
            ), inferencer=inferencer)
        inferencer.model.close_build_runtime()
    result_path = destination / "smoke_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Completed two build modes, eight images and four video frames: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
