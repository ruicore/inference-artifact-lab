"""Command line entry point: ``python -m inference_artifact_lab``."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
import platform
import sys
from pathlib import Path

from .gate import load_manifest, run_gate, run_gate_from_file
from .release import run_bound_adapter_gate, run_release_gate


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
    parser.add_argument("--reference-npy", help="pinned NumPy .npy reference output for the selected runtime adapter")
    args = parser.parse_args(argv)
    if args.runtime and (args.environment or args.reference_output or args.target_output or args.benchmark_evidence or args.observed_contract):
        print(json.dumps({"status": "fail", "error": "--runtime collects its own observations; supplied observation files are not allowed"}), file=sys.stdout)
        return 2
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
            if not args.inputs_npy or not args.reference_npy:
                raise ValueError("--runtime requires --inputs-npy and --reference-npy")
            if args.runtime != "onnx-cpu":  # argparse currently prevents this; retain an explicit guard.
                raise ValueError(f"unsupported runtime {args.runtime}")
            manifest = load_manifest(args.manifest)
            manifest_file = Path(args.manifest)
            artifact = Path(args.artifact) if args.artifact else Path(manifest.artifact.path)
            if not artifact.is_absolute():
                artifact = Path(os.path.normpath(str(manifest_file.parent / artifact)))
            observed_environment = {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "provider": "CPUExecutionProvider",
            }
            for package in ("torch", "torchvision", "onnxruntime"):
                try:
                    observed_environment[package] = import_module(package).__version__
                except ImportError:
                    observed_environment[package] = None
            preflight = run_gate(manifest, artifact, observed_environment=observed_environment)
            if any(check.status.value == "fail" for check in preflight.checks):
                report = preflight
            else:
                from .adapters import OnnxRuntimeAdapter
                fixture_path = Path(args.inputs_npy)
                reference_path = Path(args.reference_npy)
                fixture_bytes = fixture_path.read_bytes()
                reference_bytes = reference_path.read_bytes()
                binding = run_release_gate(
                    manifest, artifact, observed_environment=observed_environment,
                    fixture_bytes=fixture_bytes, reference_output_bytes=reference_bytes,
                )
                if any(check.status.value == "fail" for check in binding.checks) or next(check for check in binding.checks if check.check_id == "fixture.binding").status.value != "pass":
                    report = binding
                else:
                    adapter = OnnxRuntimeAdapter(artifact, providers=["CPUExecutionProvider"])
                    report = run_bound_adapter_gate(
                        manifest, artifact, adapter=adapter, fixture_path=fixture_path,
                        reference_output_path=reference_path,
                        observed_environment=observed_environment,
                    )
        elif reference is not None or target is not None or benchmark_evidence is not None:
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
