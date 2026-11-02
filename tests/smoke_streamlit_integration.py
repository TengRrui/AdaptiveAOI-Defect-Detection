from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class LocalUpload:
    def __init__(self, path: Path):
        self.name = "UI 中文 空格 sample.png"
        self._data = path.read_bytes()

    def getbuffer(self):
        return memoryview(self._data)


def page_check(path: Path) -> dict[str, object]:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(path)).run(timeout=120)
    return {"exception_count": len(app.exception), "exceptions": [str(item.value) for item in app.exception]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise Streamlit services and render every main page against the live integration API.")
    parser.add_argument("--software-root", required=True)
    parser.add_argument("--good", required=True)
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    software_root = Path(args.software_root).resolve()
    sys.path.insert(0, str(software_root))
    sys.path.insert(0, str(software_root / "app"))

    from service.batch_service import run_batch_inference
    from service.inference_service import run_image_inference

    single = run_image_inference("mvtec", "bottle", "configs/default.yaml", uploaded_file=LocalUpload(Path(args.good)), device="cuda")
    batch = run_batch_inference("mvtec", "bottle", "configs/default.yaml", args.batch_dir, "ignored", device="cuda")

    pages = [
        software_root / "app" / "系统总览.py",
        software_root / "app" / "pages" / "1_单图检测.py",
        software_root / "app" / "pages" / "2_批量检测.py",
        software_root / "app" / "pages" / "3_自学习校准.py",
        software_root / "app" / "pages" / "4_用户反馈.py",
        software_root / "app" / "pages" / "5_系统状态.py",
        software_root / "app" / "pages" / "6_模型管理.py",
        software_root / "app" / "pages" / "7_统计分析.py",
        software_root / "app" / "pages" / "8_实验结果.py",
    ]
    page_results = {path.name: page_check(path) for path in pages}
    summary = {"single_service_ok": single.get("ok"), "single_result_id": single.get("result_id"), "batch_service_ok": batch.get("ok"), "batch_rows": batch.get("row_count"), "pages": page_results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    page_failures = [name for name, result in page_results.items() if result["exception_count"]]
    if not single.get("ok") or not batch.get("ok") or page_failures:
        print(f"STREAMLIT SMOKE FAILED: {', '.join(page_failures) or 'service call'}", file=sys.stderr)
        return 1
    print("STREAMLIT SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
