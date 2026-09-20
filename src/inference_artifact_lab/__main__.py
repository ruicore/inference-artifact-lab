"""Command line entry point: ``python -m inference_artifact_lab``."""

from __future__ import annotations

import argparse
import json
import sys

from .gate import run_gate_from_file
from .release import run_release_gate


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
        if reference is not None or target is not None or benchmark_evidence is not None:
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
