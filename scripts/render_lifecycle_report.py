"""Render lifecycle-report/v1; Phase 1 reports keep their existing renderer."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_artifact_lab.lifecycle_experiment import render_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.report.with_suffix(".md")
    output.write_text(render_report(json.loads(args.report.read_text(encoding="utf-8"))), encoding="utf-8")
    print(output.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
