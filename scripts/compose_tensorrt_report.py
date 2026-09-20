"""Compose the portable TensorRT release report from container evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from inference_artifact_lab import load_manifest
from inference_artifact_lab.release import run_release_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("examples/squeezenet11-tensorrt.manifest.json"))
    parser.add_argument("--engine", type=Path, default=Path("artifacts/squeezenet1.1-fp32.engine"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/squeezenet11-reference.npy"))
    parser.add_argument("--container-output", type=Path, default=Path("artifacts/squeezenet11-tensorrt-output.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase-1/squeezenet11-tensorrt.json"))
    parser.add_argument("--trtexec-log", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    payload = json.loads(args.container_output.read_text(encoding="utf-8"))
    expected_fixture_sha256 = hashlib.sha256(args.reference.with_name("squeezenet11-fixture.npy").read_bytes()).hexdigest()
    if payload.get("fixture_sha256") != expected_fixture_sha256:
        raise SystemExit("TensorRT output fixture digest does not match the reference fixture")
    target = payload["outputs"]["output"]
    reference = np.load(args.reference, allow_pickle=False)
    runtime = payload["runtime"]
    log = args.trtexec_log.read_text(encoding="utf-8")
    performance = re.search(
        r"Throughput:\s+(?P<throughput>[0-9.]+) qps.*?Latency:.*?mean = (?P<mean>[0-9.]+) ms, median = (?P<median>[0-9.]+) ms,.*?percentile\(95%\) = (?P<p95>[0-9.]+) ms",
        log,
        re.S,
    )
    if not performance:
        raise SystemExit("trtexec log does not contain a performance summary")
    benchmark_evidence = {
        "source": "trtexec",
        "warmup_duration_ms": 200,
        "measured_runs": 100,
        "transfer_scope": "device_only",
        "latency_mean_seconds": float(performance["mean"]) / 1000,
        "latency_median_seconds": float(performance["median"]) / 1000,
        "latency_p95_seconds": float(performance["p95"]) / 1000,
        "throughput_runs_per_second": float(performance["throughput"]),
        "memory_scope": "not_peak; trtexec benchmark log does not expose process peak GPU memory",
    }
    environment = dict(payload["environment"])
    environment["tensorrt"] = runtime["tensorrt_version"].rsplit(".", 1)[0]
    report = run_release_gate(
        manifest,
        args.engine,
        observed_contract=payload["contract"],
        observed_environment=environment,
        reference_outputs=reference,
        target_outputs=target,
        benchmark_evidence=benchmark_evidence,
    )
    report_data = report.to_dict()
    report_data["evidence"] = {
        "fixture_sha256": expected_fixture_sha256,
        "reference_output_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "container_runtime": runtime,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report_data, indent=2, sort_keys=True))
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
