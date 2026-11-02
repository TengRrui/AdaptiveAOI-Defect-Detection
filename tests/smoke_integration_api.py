from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def call(method: str, url: str, **kwargs):
    response = requests.request(method, url, timeout=900, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{method} {url} did not return JSON: {response.text[:300]}") from exc
    return response.status_code, payload


def _runtime_fields(config_payload: dict, *, device_override: str | None = None) -> dict[str, str]:
    config = config_payload.get("config") or {}
    runtime = config.get("runtime") or config.get("dynamic") or {}
    fields = {
        "model_id": str(runtime.get("model_id") or "patchcore-default"),
        "dataset": str(runtime.get("dataset") or "mvtec"),
        "category": str(runtime.get("category") or "bottle"),
        "device": str(device_override or runtime.get("device") or "auto"),
    }
    for key in ("memory_bank_id", "config_profile_id", "config_profile_version", "inference_mode"):
        value = runtime.get(key)
        if value not in (None, ""):
            fields[key] = str(value)
    return fields


def _result_summary(payload: dict) -> dict:
    result = payload.get("result") or {}
    metadata = result.get("metadata") or {}
    return {
        "prediction": result.get("prediction"),
        "anomaly_score": result.get("anomaly_score"),
        "threshold": result.get("threshold"),
        "inference_time_ms": result.get("inference_time_ms"),
        "memory_bank_id": metadata.get("memory_bank_id"),
        "config_profile_id": metadata.get("config_profile_id"),
        "method_mode": metadata.get("method_mode"),
        "cache_hit": metadata.get("cache_hit"),
    }


def detect(base_url: str, image_path: Path, runtime_fields: dict[str, str], filename: str | None = None) -> tuple[int, dict]:
    with image_path.open("rb") as handle:
        return call(
            "POST",
            f"{base_url}/api/detect",
            data=runtime_fields,
            files={"file": (filename or image_path.name, handle, "image/png")},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real, small AdaptiveAOI integration smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--good", required=True)
    parser.add_argument("--bad", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--invalid-file", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    good = Path(args.good)
    bad = Path(args.bad)

    summary: dict[str, object] = {"base_url": base_url, "checks": {}}
    code, health = call("GET", f"{base_url}/api/health")
    summary["checks"]["health"] = {"status": code, "success": health.get("success"), "cuda_available": health.get("cuda_available")}

    config_code, config_payload = call("GET", f"{base_url}/api/config")
    runtime_fields = _runtime_fields(config_payload)
    summary["checks"]["runtime_config"] = {
        "status": config_code,
        "success": config_payload.get("success"),
        "runtime": runtime_fields,
    }

    code, loaded = call("POST", f"{base_url}/api/models/load", json=runtime_fields)
    summary["checks"]["model_load"] = {"status": code, "success": loaded.get("success"), "model": loaded.get("model")}

    code, good_result = detect(base_url, good, runtime_fields, filename="中文 空格 good.png")
    summary["checks"]["single_good"] = {"status": code, "success": good_result.get("success"), "result": _result_summary(good_result)}

    code, bad_result = detect(base_url, bad, runtime_fields)
    summary["checks"]["single_anomaly"] = {"status": code, "success": bad_result.get("success"), "result": _result_summary(bad_result)}

    code, cache_result = detect(base_url, good, runtime_fields)
    cache_hit = ((cache_result.get("result") or {}).get("metadata") or {}).get("cache_hit")
    summary["checks"]["cache_reuse"] = {"status": code, "success": cache_result.get("success"), "cache_hit": cache_hit}

    code, batch_result = call(
        "POST",
        f"{base_url}/api/detect/batch",
        data={"input_dir": args.batch_dir, **runtime_fields},
    )
    summary["checks"]["batch"] = {"status": code, "success": batch_result.get("success"), "total": batch_result.get("total"), "successful": batch_result.get("successful"), "failed": batch_result.get("failed")}

    with Path(args.invalid_file).open("rb") as handle:
        code, invalid_result = call(
            "POST",
            f"{base_url}/api/detect",
            data=runtime_fields,
            files={"file": ("invalid.txt", handle, "text/plain")},
        )
    summary["checks"]["invalid_file"] = {"status": code, "success": invalid_result.get("success"), "error_code": (invalid_result.get("error") or {}).get("code")}

    code, stats = call("GET", f"{base_url}/api/statistics/summary")
    stats_summary = stats.get("summary") or {}
    summary["checks"]["statistics"] = {
        "status": code,
        "success": stats.get("success"),
        "summary": {key: stats_summary.get(key) for key in ("total", "normal", "anomaly", "pass_rate", "average_inference_time_ms", "source")},
    }

    code, experiments = call("GET", f"{base_url}/api/experiments")
    summary["checks"]["experiments"] = {"status": code, "success": experiments.get("success"), "count": len(experiments.get("experiments", []))}

    if args.cpu:
        cpu_fields = {**runtime_fields, "device": "cpu"}
        code, cpu_result = detect(base_url, good, cpu_fields)
        summary["checks"]["cpu_single"] = {"status": code, "success": cpu_result.get("success"), "result": _result_summary(cpu_result)}

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    required = ["health", "runtime_config", "model_load", "single_good", "single_anomaly", "cache_reuse", "batch", "statistics", "experiments"]
    failures = [name for name in required if not summary["checks"][name].get("success")]
    if summary["checks"]["invalid_file"].get("success") or summary["checks"]["invalid_file"].get("error_code") != "UNSUPPORTED_FILE_TYPE":
        failures.append("invalid_file")
    if cache_hit is not True:
        failures.append("cache_reuse")
    if args.cpu and not summary["checks"]["cpu_single"].get("success"):
        failures.append("cpu_single")
    if failures:
        print(f"SMOKE FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
