"""Render a Model Release Gate JSON report as a reviewable Markdown file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(report: dict) -> str:
    lines = [
        "# Model Release Gate report",
        "",
        f"- Status: **{report.get('status', 'unknown')}**",
        f"- Schema: `{report.get('schema_version', 'unknown')}`",
        f"- Manifest digest: `{report.get('manifest_digest')}`",
        f"- Artifact: `{report.get('artifact', {}).get('path', '')}`",
        f"- Runtime profile: `{report.get('scope', {}).get('runtime', {}).get('profile', 'not_declared')}`",
        f"- Acceptance role: `{report.get('scope', {}).get('runtime', {}).get('acceptance_role', 'not_declared')}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Message |",
        "|---|---|---|",
    ]
    for check in report.get("checks", []):
        message = str(check.get("message", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{check.get('id', '')}` | **{check.get('status', '')}** | {message} |")
    lines.extend(["", "## Limitations", ""])
    limitations = report.get("limitations", []) or ["None recorded."]
    lines.extend(f"- {item}" for item in limitations)
    if report.get("evidence"):
        lines.extend(["", "## Evidence", "", "```json", json.dumps(report["evidence"], indent=2, sort_keys=True), "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.report.with_suffix(".md")
    output.write_text(render(json.loads(args.report.read_text(encoding="utf-8"))), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
