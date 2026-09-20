"""Run the public torchvision reference against the exported ONNX artifact."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from inference_artifact_lab import load_manifest
from inference_artifact_lab.adapters import OnnxRuntimeAdapter
from inference_artifact_lab.release import run_release_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("examples/squeezenet11-torchvision.manifest.json"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/squeezenet1.1-torchvision.onnx"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase-1/squeezenet11-torchvision-onnx-cpu.json"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    reference_model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT).eval()
    generator = torch.Generator().manual_seed(20260920)
    tensor = torch.rand((1, 3, 224, 224), generator=generator)
    with torch.no_grad():
        reference = reference_model(tensor).numpy()

    adapter = OnnxRuntimeAdapter(args.artifact, providers=["CPUExecutionProvider"])
    target = adapter.run({"data": tensor.numpy()})["output"]
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
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
        benchmark_call=lambda: adapter.run({"data": tensor.numpy()}),
    )
    report = replace(report, limitations=("TensorRT engine compatibility is not verified in this environment.",))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
