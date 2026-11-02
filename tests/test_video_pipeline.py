"""Small, self-contained regression tests for the offline video-file pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_demo_video import build_demo_video
from video.video_inference import VideoDetectionOptions, run_video_detection
from video.video_reader import VideoInputError, VideoReader


def write_png(path: Path, image: np.ndarray) -> None:
    ok, payload = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test PNG encoding failed")
    path.write_bytes(payload.tobytes())


class FakeInferencer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def infer_prepared(self, prepared, meta=None, save_outputs=False, generate_visuals=False):
        frame_index = int(prepared.image_path.stem.rsplit("_", 1)[-1])
        self.calls.append(frame_index)
        heatmap = np.zeros((prepared.height, prepared.width), dtype=np.float32)
        score = 0.25 if frame_index % 2 == 0 else 0.85
        if score >= 0.5:
            heatmap[10:30, 12:38] = score
        return {
            "score": score,
            "threshold": 0.5,
            "pred_label": int(score >= 0.5),
            "inference_time_ms": 1.0,
            "_heatmap_array": heatmap,
        }


class VideoPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="AdaptiveAOI_视频测试_")
        self.root = Path(self.temporary.name)
        self.source_dir = self.root / "源图像"
        self.source_dir.mkdir()
        first = np.full((72, 96, 3), (40, 120, 210), dtype=np.uint8)
        second = np.full((72, 96, 3), (210, 80, 35), dtype=np.uint8)
        write_png(self.source_dir / "01.png", first)
        write_png(self.source_dir / "02.png", second)
        self.input_video = self.root / "输入 视频.mp4"
        self.build = build_demo_video(
            self.source_dir,
            self.input_video,
            fps=10,
            max_duration=2.2,
            resolution=(320, 240),
            seed=7,
            sample_count=2,
            seconds_per_sample=0.35,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_generation_metadata_inference_and_readback(self) -> None:
        self.assertTrue(Path(self.build["output_video"]).is_file())
        self.assertTrue(self.build["actual"]["readback_verified"])
        with VideoReader(self.build["output_video"]) as reader:
            self.assertEqual(reader.metadata.width, 320)
            self.assertEqual(reader.metadata.height, 240)
            self.assertGreater(reader.metadata.source_fps, 0)
        created: list[FakeInferencer] = []

        def factory(config, memory):
            fake = FakeInferencer()
            created.append(fake)
            return fake

        output_dir = self.root / "检测 输出 中文"
        summary = run_video_detection(
            VideoDetectionOptions(
                input_video=self.build["output_video"],
                output_video=output_dir / "检测结果.mp4",
                output_dir=output_dir,
                config={"dataset": "test", "category": "fixture", "image_size": 64, "device": "cpu"},
                frame_stride=1,
                overwrite=False,
            ),
            inferencer_factory=factory,
        )
        self.assertEqual(len(created), 1, "model factory must be called exactly once")
        self.assertEqual(summary["model_load_count"], 1)
        self.assertEqual(summary["successful_processed_frames"], summary["total_decoded_records"])
        self.assertEqual(len(created[0].calls), summary["successful_processed_frames"])
        self.assertGreater(summary["normal_frames"], 0)
        self.assertGreater(summary["anomaly_frames"], 0)
        self.assertTrue(Path(summary["output_video_path"]).is_file())
        self.assertTrue(Path(summary["jsonl_path"]).is_file())
        self.assertTrue(Path(summary["csv_path"]).is_file())
        line_count = len(Path(summary["jsonl_path"]).read_text(encoding="utf-8").splitlines())
        self.assertEqual(line_count, summary["successful_processed_frames"])
        rows = [json.loads(line) for line in Path(summary["jsonl_path"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["anomaly_score"], 0.25, "video frame must use the same fake single-image inference result")
        capture = cv2.VideoCapture(summary["output_video_path"])
        try:
            self.assertTrue(capture.isOpened())
            self.assertGreater(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()
        renamed = Path(summary["output_video_path"]).with_name("资源释放验证.mp4")
        os.replace(summary["output_video_path"], renamed)
        self.assertTrue(renamed.is_file(), "writer must release its handle before returning")

    def test_stride_cancellation_and_invalid_inputs(self) -> None:
        fake = FakeInferencer()
        stride = run_video_detection(
            VideoDetectionOptions(
                input_video=self.build["output_video"],
                output_video=self.root / "stride.mp4",
                output_dir=self.root / "stride",
                config={"dataset": "test", "category": "fixture", "image_size": 64},
                frame_stride=2,
            ),
            inferencer=fake,
        )
        self.assertGreater(stride["skipped_by_stride_frames"], 0)
        self.assertEqual(stride["successful_processed_frames"], len(fake.calls))

        # A browser can lock the previous preview image while the job writes
        # the next one on Windows.  That optional callback must not convert a
        # real inference result into a failed frame.
        preview_failure = run_video_detection(
            VideoDetectionOptions(
                input_video=self.build["output_video"],
                output_video=self.root / "preview_failure.mp4",
                output_dir=self.root / "preview_failure",
                config={"dataset": "test", "category": "fixture", "image_size": 64},
                max_frames=3,
                preview_callback=lambda frame, result: (_ for _ in ()).throw(PermissionError("preview locked")),
            ),
            inferencer=FakeInferencer(),
        )
        self.assertEqual(preview_failure["failed_frames"], 0)
        self.assertEqual(preview_failure["successful_processed_frames"], 3)
        self.assertTrue(any("预览更新已跳过" in item for item in preview_failure["warnings"]))
        calls = {"count": 0}

        def cancel() -> bool:
            calls["count"] += 1
            return calls["count"] > 3

        cancelled = run_video_detection(
            VideoDetectionOptions(
                input_video=self.build["output_video"],
                output_video=self.root / "cancel.mp4",
                output_dir=self.root / "cancel",
                config={"dataset": "test", "category": "fixture", "image_size": 64},
                cancel_callback=cancel,
            ),
            inferencer=FakeInferencer(),
        )
        self.assertTrue(cancelled["cancelled"])
        self.assertGreater(cancelled["successful_processed_frames"], 0)
        with self.assertRaises(VideoInputError):
            with VideoReader(self.root / "missing.mp4"):
                pass
        invalid = self.root / "invalid.mp4"
        invalid.write_text("not a video", encoding="utf-8")
        with self.assertRaises(VideoInputError):
            with VideoReader(invalid):
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
