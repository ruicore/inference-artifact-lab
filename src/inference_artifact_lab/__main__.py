"""Command line entry point: ``python -m inference_artifact_lab``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .gate import run_gate_from_file
from .release import run_release_gate
from .runtime import benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run core Model Release Gate checks")
    parser.add_argument("manifest", help="path to a JSON manifest")
    parser.add_argument("--artifact", help="artifact path (defaults to manifest artifact.path)")
    parser.add_argument("--observed-contract", help="JSON file containing observed inputs and outputs")
    parser.add_argument("--environment", help="JSON file containing the observed environment fingerprint")
    parser.add_argument("--reference-output", help="JSON file containing reference runtime output")
    parser.add_argument("--target-output", help="JSON file containing target runtime output")
    parser.add_argument("--benchmark-evidence", help="JSON file containing benchmark evidence")
    parser.add_argument("--report", help="write the release report JSON to this path")
    parser.add_argument("--runtime", choices=["onnx-cpu"], help="run a real package adapter before composing the release report")
    parser.add_argument("--inputs-npy", help="NumPy .npy input for the selected runtime adapter")
    args = parser.parse_args(argv)
    observed = None
    if args.observed_contract:
        with open(args.observed_contract, encoding="utf-8") as handle:
            observed = json.load(handle)
    environment = None
    if args.environment:
        with open(args.environment, encoding="utf-8") as handle:
            environment = json.load(handle)
    def load_json(path: str | None):
        if not path:
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    try:
        reference = load_json(args.reference_output)
        target = load_json(args.target_output)
        benchmark_evidence = load_json(args.benchmark_evidence)
        if args.runtime:
            if not args.inputs_npy or reference is None or environment is None:
                raise ValueError("--runtime requires --inputs-npy, --reference-output, and --environment")
            if args.runtime != "onnx-cpu":  # argparse currently prevents this; retain an explicit guard.
                raise ValueError(f"unsupported runtime {args.runtime}")
            import numpy as np
            from .adapters import OnnxRuntimeAdapter
            from .gate import load_manifest
            manifest = load_manifest(args.manifest)
            manifest_file = Path(args.manifest)
            artifact = Path(args.artifact) if args.artifact else Path(manifest.artifact.path)
            if not artifact.is_absolute():
                artifact = Path(os.path.normpath(str(manifest_file.parent / artifact)))
            values = np.load(args.inputs_npy, allow_pickle=False)
            adapter = OnnxRuntimeAdapter(artifact, providers=["CPUExecutionProvider"])
            output_name = adapter.contract()["outputs"][0]["name"]
            input_name = adapter.contract()["inputs"][0]["name"]
            target = adapter.run({input_name: values})[output_name]
            reference_value = reference.get(output_name) if isinstance(reference, dict) else reference
            measured = benchmark(lambda: adapter.run({input_name: values}), warmup_runs=1, measured_runs=5)
            report = run_release_gate(manifest, artifact, observed_contract=adapter.contract(), observed_environment=environment, reference_outputs=reference_value, target_outputs=target, benchmark_evidence=measured.to_dict())
        elif reference is not None or target is not None or benchmark_evidence is not None:
            from .gate import load_manifest
            manifest = load_manifest(args.manifest)
            report = run_release_gate(manifest, args.artifact, observed_contract=observed, observed_environment=environment, reference_outputs=reference, target_outputs=target, benchmark_evidence=benchmark_evidence)
        else:
            report = run_gate_from_file(args.manifest, args.artifact, observed_contract=observed, observed_environment=environment)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stdout)
        return 2
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
