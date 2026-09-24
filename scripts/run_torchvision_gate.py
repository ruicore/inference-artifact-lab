"""Run the public torchvision reference against the exported ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
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
from inference_artifact_lab.release import run_bound_adapter_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("examples/squeezenet11-torchvision.manifest.json"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/squeezenet1.1-torchvision.onnx"))
    parser.add_argument("--report", type=Path, default=Path("reports/phase-1/squeezenet11-torchvision-onnx-cpu.json"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
    cache_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if manifest.source != weights.url or not manifest.source_sha256:
        raise SystemExit("manifest does not pin the selected public weights")
    reference_model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT).eval()
    observed_weight_sha256 = hashlib.sha256(cache_file.read_bytes()).hexdigest()
    if observed_weight_sha256 != manifest.source_sha256:
        raise SystemExit("public source weight bytes do not match manifest")
    generator = torch.Generator().manual_seed(20260920)
    tensor = torch.rand((1, 3, 224, 224), generator=generator)
    with torch.no_grad():
        reference = reference_model(tensor).numpy()

    # Keep the exact public fixture and reference output available to the
    # TensorRT container. They are generated artifacts and remain ignored by
    # Git; the container evidence can hash and verify them before use.
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = args.artifact.parent / "squeezenet11-fixture.npy"
    reference_path = args.artifact.parent / "squeezenet11-reference.npy"
    np.save(fixture_path, tensor.numpy())
    np.save(reference_path, reference)

    adapter = OnnxRuntimeAdapter(args.artifact, providers=["CPUExecutionProvider"])
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "provider": "CPUExecutionProvider",
    }
    report = run_bound_adapter_gate(
        manifest, args.artifact, adapter=adapter, fixture_path=fixture_path,
        reference_output_path=reference_path, observed_environment=environment,
    )
    report = replace(
        report,
        limitations=(
            "This report covers only the declared ONNX Runtime CPU baseline; TensorRT is an independent optional preview and is not validated by a CPU pass.",
            "The benchmark peak memory scope is Python allocations only; native process and GPU peak memory are not observed or claimed.",
            "A passing runtime report does not itself establish committed-checkout AC-11 or authorize publication.",
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_data = report.to_dict()
    report_data["evidence"] = {
        "source_weight_sha256": observed_weight_sha256,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "reference_output_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    args.report.write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report_data, indent=2, sort_keys=True))
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
