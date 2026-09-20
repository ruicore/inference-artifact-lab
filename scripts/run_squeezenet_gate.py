"""Run the pinned CPU ONNX smoke gate and write its JSON evidence report."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from inference_artifact_lab import load_manifest
from inference_artifact_lab.adapters import OnnxRuntimeAdapter
from inference_artifact_lab.release import run_release_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("examples/squeezenet11-onnx.manifest.json"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/squeezenet1.1-7.onnx"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase-1/squeezenet11-onnx-cpu.json"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    adapter = OnnxRuntimeAdapter(args.artifact, providers=["CPUExecutionProvider"])
    rng = np.random.Generator(np.random.PCG64(20260920))
    inputs = {"data": rng.uniform(-1.0, 1.0, size=(1, 3, 224, 224)).astype(np.float32)}
    reference = adapter.run(inputs)["squeezenet0_flatten0_reshape0"]
    target = adapter.run(inputs)["squeezenet0_flatten0_reshape0"]
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "onnxruntime": __import__("onnxruntime").__version__,
        "provider": "CPUExecutionProvider",
    }
    report = run_release_gate(
        manifest,
        args.artifact,
        observed_contract=adapter.contract(),
        observed_environment=environment,
        reference_outputs=reference,
        target_outputs=target,
        benchmark_call=lambda: adapter.run(inputs),
    )
    report = replace(
        report,
        limitations=(
            "Reference and target outputs currently use the same ONNX Runtime CPU adapter; cross-runtime equivalence is not verified.",
            "TensorRT engine compatibility is not verified in this environment.",
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
