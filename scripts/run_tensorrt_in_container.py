"""Run the TensorRT adapter against the public Phase 1 fixture.

This script is intentionally executed inside the pinned NVIDIA container. It
does not fetch or inspect any local model outside the mounted public project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from inference_artifact_lab.adapters import TensorRTAdapter
from inference_artifact_lab.gate import load_manifest, run_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("examples/squeezenet11-tensorrt.manifest.json"))
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().split(",", 1)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import tensorrt
    environment = {
        "container_digest": os.environ.get("MODEL_RELEASE_GATE_CONTAINER_DIGEST"),
        "gpu_name": gpu[0].strip(),
        "compute_capability": gpu[1].strip() if len(gpu) > 1 else None,
        "cuda_driver": driver,
        "tensorrt": tensorrt.__version__.rsplit(".", 1)[0],
    }
    manifest = load_manifest(args.manifest)
    preflight = run_gate(manifest, args.engine, observed_environment=environment)
    failures = [check for check in preflight.checks if check.status.value == "fail"]
    if failures:
        print(json.dumps({"status": "fail", "checks": [check.to_dict() for check in failures]}, sort_keys=True))
        return 1
    adapter = TensorRTAdapter(args.engine)
    fixture = np.load(args.fixture, allow_pickle=False)
    outputs = adapter.run({"data": fixture})
    payload = {
        "contract": adapter.contract(),
        "runtime": adapter.runtime_info(),
        "fixture_shape": list(fixture.shape),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "environment": {key: value for key, value in environment.items() if key != "tensorrt"},
        "outputs": outputs,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"contract": payload["contract"], "runtime": payload["runtime"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
