from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.manifest import write_manifest_outputs
from tools.path_utils import append_error_report


def main():
    parser = argparse.ArgumentParser(description="Build unified manifests for MVTec, VisA, and DAGM.")
    parser.add_argument("--data_root", required=True, help="Raw data root.")
    parser.add_argument("--output_dir", required=True, help="Directory for manifest and summary outputs.")
    args = parser.parse_args()

    try:
        paths = write_manifest_outputs(Path(args.data_root), Path(args.output_dir))
        print("Manifest build finished.")
        for name, path in paths.items():
            print(f"{name}: {path}")
    except Exception as exc:
        append_error_report(Path(args.output_dir).parent, "build_manifest failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
