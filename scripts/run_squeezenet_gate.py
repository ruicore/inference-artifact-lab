"""Run the pinned CPU ONNX smoke gate and write its JSON evidence report."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

from inference_artifact_lab import CheckResult, GateStatus, load_manifest, run_gate
from inference_artifact_lab.adapters import OnnxRuntimeAdapter


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
    adapter.run(inputs)
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "onnxruntime": __import__("onnxruntime").__version__,
        "provider": "CPUExecutionProvider",
    }
    core = run_gate(manifest, args.artifact, observed_contract=adapter.contract(), observed_environment=environment)
    report = replace(
        core,
        status=GateStatus.NOT_VERIFIED,
        checks=core.checks + (CheckResult("runtime.equivalence", GateStatus.NOT_VERIFIED, "same-runtime smoke cannot establish cross-runtime equivalence", {}),),
        limitations=(
            "Legacy smoke only: one ONNX Runtime CPU adapter loaded and executed the artifact; no independent reference comparison was made.",
            "This report is not Phase 1 AC-05 evidence or a release decision.",
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
