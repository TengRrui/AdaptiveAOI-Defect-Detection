"""Fetch requested torchvision weights and verify the existing resource specification."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extractor import _BACKBONE_SPECS, _verify_packaged_weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path, help="Local resource directory; weights are excluded from Git.")
    parser.add_argument("--backbones", nargs="+", choices=sorted(_BACKBONE_SPECS), default=["wide_resnet50_2"])
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = PROJECT_ROOT / "artifacts" / "models" / "backbones" / "manifest.json"
    manifest = destination / "manifest.json"
    if manifest.resolve() != source_manifest.resolve():
        if manifest.exists() and manifest.read_bytes() != source_manifest.read_bytes():
            raise ValueError("Destination has a different model manifest; select an empty resource directory.")
        shutil.copyfile(source_manifest, manifest)
    for name in dict.fromkeys(args.backbones):
        spec = _BACKBONE_SPECS[name]
        target = destination / str(spec["filename"])
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            try:
                with urlopen(f"https://download.pytorch.org/models/{spec['filename']}", timeout=60) as response, partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                _verify_packaged_weight(str(partial), int(spec["size"]), str(spec["sha256"]))
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        _verify_packaged_weight(str(target), int(spec["size"]), str(spec["sha256"]))
        print(f"Verified {name}: {target}")
    print(f"Set ADAPTIVEAOI_BACKBONE_ROOT={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
