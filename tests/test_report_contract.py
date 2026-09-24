"""Public report-schema and renderer regression checks (PH1-R07 / AC-10)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import runpy

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "phase-1"


def test_checked_in_reports_validate_against_v1_schema() -> None:
    schema = json.loads((ROOT / "schemas" / "release-report.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    paths = sorted(REPORTS.glob("*.json"))
    assert paths
    for path in paths:
        validator.validate(json.loads(path.read_text(encoding="utf-8")))


def test_checked_in_reports_contain_no_obvious_private_paths_or_credentials() -> None:
    forbidden = re.compile(r"[A-Za-z]:\\|/Users/|/home/|-----BEGIN .*PRIVATE KEY-----|Bearer\s+\S+|\b(?:api[_-]?key|secret|access[_-]?token)\b", re.I)
    for path in REPORTS.glob("*.json"):
        assert forbidden.search(path.read_text(encoding="utf-8")) is None, path.name


def test_human_report_has_status_checks_and_limitations() -> None:
    render = runpy.run_path(str(ROOT / "scripts" / "render_report.py"))["render"]

    report = json.loads((REPORTS / "squeezenet11-torchvision-onnx-cpu.json").read_text(encoding="utf-8"))
    output = render(report)
    assert "# Model Release Gate report" in output
    assert f"Status: **{report['status']}**" in output
    assert "## Checks" in output
    assert "## Limitations" in output
    assert next(check for check in report["checks"] if check["id"] == "model.source_pin")["status"] == "pass"
    assert report["evidence"]["source_weight_sha256"] == next(check for check in report["checks"] if check["id"] == "model.source_pin")["evidence"]["sha256"]
    assert next(check for check in report["checks"] if check["id"] == "runtime.provenance")["status"] == "pass"


def test_tensorrt_report_does_not_claim_peak_memory_or_full_acceptance() -> None:
    report = json.loads((REPORTS / "squeezenet11-tensorrt.json").read_text(encoding="utf-8"))
    benchmark = next(check for check in report["checks"] if check["id"] == "benchmark.workload")
    assert benchmark["status"] == "not_verified"
    assert report["status"] == "not_verified"
    assert any("peak GPU memory" in item for item in report["limitations"])


def test_cpu_baseline_and_optional_preview_keep_independent_scope_and_status() -> None:
    from inference_artifact_lab import canonical_manifest_digest, load_manifest

    render = runpy.run_path(str(ROOT / "scripts" / "render_report.py"))["render"]
    for stem, manifest_name, profile, role, status in (
        ("squeezenet11-torchvision-onnx-cpu", "squeezenet11-torchvision", "onnx-cpu-windows-py311", "cpu_baseline", "pass"),
        ("squeezenet11-tensorrt", "squeezenet11-tensorrt", "tensorrt-gpu", "optional_preview", "not_verified"),
    ):
        manifest = load_manifest(ROOT / "examples" / f"{manifest_name}.manifest.json")
        report = json.loads((REPORTS / f"{stem}.json").read_text(encoding="utf-8"))
        assert report["scope"] == {"runtime": dict(manifest.runtime), "environment": dict(manifest.environment)}
        assert report["manifest_digest"] == canonical_manifest_digest(manifest)
        assert report["scope"]["runtime"]["profile"] == profile
        assert report["scope"]["runtime"]["acceptance_role"] == role
        assert report["status"] == status
        assert f"Runtime profile: `{profile}`" in render(report)
        assert f"Acceptance role: `{role}`" in render(report)


def test_legacy_same_adapter_smoke_does_not_claim_ac05() -> None:
    report = json.loads((REPORTS / "squeezenet11-onnx-cpu.json").read_text(encoding="utf-8"))
    assert report["status"] == "not_verified"
    runtime = next(check for check in report["checks"] if check["id"] == "runtime.equivalence")
    assert runtime["status"] == "not_verified"
