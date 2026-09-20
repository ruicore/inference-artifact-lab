"""Command line entry point: ``python -m inference_artifact_lab``."""

from __future__ import annotations

import argparse
import json
import sys

from .gate import run_gate_from_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run core Model Release Gate checks")
    parser.add_argument("manifest", help="path to a JSON manifest")
    parser.add_argument("--artifact", help="artifact path (defaults to manifest artifact.path)")
    parser.add_argument("--observed-contract", help="JSON file containing observed inputs and outputs")
    parser.add_argument("--environment", help="JSON file containing the observed environment fingerprint")
    args = parser.parse_args(argv)
    observed = None
    if args.observed_contract:
        with open(args.observed_contract, encoding="utf-8") as handle:
            observed = json.load(handle)
    environment = None
    if args.environment:
        with open(args.environment, encoding="utf-8") as handle:
            environment = json.load(handle)
    try:
        report = run_gate_from_file(args.manifest, args.artifact, observed_contract=observed, observed_environment=environment)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}), file=sys.stdout)
        return 2
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.status.value == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
