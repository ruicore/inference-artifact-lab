"""Reproduce PH2 from a clean committed clone and a fresh Python environment.

The receipt proves this reproduction procedure. Raw invocation checks that need
external unit-test/checkout evidence keep their own not_verified status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


def run(command: list[str], cwd: Path, log: Path, *, allowed_codes: tuple[int, ...] = (0,)) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(result.stdout + result.stderr)
    if result.returncode not in allowed_codes:
        raise RuntimeError(f"reproduction step {Path(command[0]).name} failed ({result.returncode}); inspect the local reproduction log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if status:
        raise SystemExit("committed-checkout reproduction requires a clean source worktree")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    destination = (args.destination or root / ".manifest" / f"lifecycle-repro-{uuid.uuid4().hex}").resolve()
    if destination.exists() or destination == root or root.is_relative_to(destination):
        raise SystemExit("destination must be a new independent clone directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-hardlinks", str(root), str(destination)], check=True, capture_output=True)
    clone_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
    if clone_revision != revision:
        raise RuntimeError("clone revision differs from the selected source commit")
    artifacts = destination / "artifacts"
    artifacts.mkdir(exist_ok=True)
    log = artifacts / "lifecycle-reproduction.log"
    run(["uv", "venv", "--python", args.python, ".venv"], destination, log)
    python = destination / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run(["uv", "sync", "--frozen", "--python", str(python), "--extra", "test", "--extra", "runtime-cpu", "--extra", "lifecycle"], destination, log)
    run(["uv", "pip", "install", "--python", str(python), "-r", "scripts/requirements-reference-public.txt"], destination, log)
    run([str(python), "-c", "import sys; assert sys.prefix != sys.base_prefix; print('fresh-venv',sys.version_info[:3])"], destination, log)
    run([str(python), "-m", "pytest", "-q", "--junitxml=artifacts/lifecycle-tests.xml"], destination, log)
    run([str(python), "scripts/export_torchvision_onnx.py"], destination, log)
    run([str(python), "scripts/run_torchvision_gate.py", "--report", "artifacts/reference-gate.json"], destination, log)
    run([str(python), "scripts/generate_lifecycle_fixtures.py", "--output", "artifacts/lifecycle",
         "--public-manifest", "examples/squeezenet11-torchvision.manifest.json",
         "--public-inputs", "artifacts/squeezenet11-fixture.npy",
         "--public-reference", "artifacts/squeezenet11-reference.npy"], destination, log)
    run([str(python), "scripts/run_lifecycle_experiment.py", "--fixtures", "artifacts/lifecycle/fixtures.json",
         "--output", "artifacts/lifecycle-report.json"], destination, log, allowed_codes=(0, 1))
    run([str(python), "-c", "import json; from jsonschema import Draft202012Validator; "
         "schema=json.load(open('schemas/lifecycle-report.schema.json',encoding='utf-8')); "
         "Draft202012Validator.check_schema(schema); "
         "Draft202012Validator(schema).validate(json.load(open('artifacts/lifecycle-report.json',encoding='utf-8')))"], destination, log)
    run([str(python), "scripts/render_lifecycle_report.py", "artifacts/lifecycle-report.json",
         "--output", "artifacts/lifecycle-report.md"], destination, log)
    report = json.loads((artifacts / "lifecycle-report.json").read_text(encoding="utf-8"))
    checks = {check["id"]: check["status"] for check in report["checks"]}
    external = {"PH2-AC02", "PH2-AC06", "PH2-AC11"}
    expected = {f"PH2-AC{i:02}" for i in range(1, 12)}
    if set(checks) != expected or any(checks[key] != "pass" for key in expected - external):
        raise RuntimeError("required runtime evidence did not pass in the committed clone")
    if any(checks[key] not in {"pass", "not_verified"} for key in external):
        raise RuntimeError("external acceptance checks reported a failure or blocked evidence")
    suites = ET.parse(artifacts / "lifecycle-tests.xml").getroot()
    if suites.findall(".//failure") or suites.findall(".//error"):
        raise RuntimeError("the cloned test suite reported failures")
    deterministic = [case for case in suites.findall(".//testcase") if case.get("classname", "").endswith("test_lifecycle")]
    if not deterministic or any(case.find("skipped") is not None for case in deterministic):
        raise RuntimeError("deterministic lifecycle acceptance tests were absent or skipped")
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=destination, text=True)
    if not clean:
        raise RuntimeError("reproduction modified tracked or nonignored files")
    receipt = {"schema_version": "lifecycle-reproduction/v1", "reproduction_status": "pass",
               "source_revision": revision, "clone_revision": clone_revision, "clone_clean_after": clean,
               "fresh_virtual_environment": True, "deterministic_test_count": len(deterministic),
               "test_count": len(suites.findall('.//testcase')), "invocation_status": report["status"],
               "invocation_checks": checks,
               "limitations": ["Invocation status retains checks requiring independent acceptance evidence.",
                               "This receipt is local reproduction evidence, not signed provenance or production certification."]}
    (artifacts / "lifecycle-reproduction.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"clean lifecycle reproduction PASS code_commit={revision}")
    print("Generated clone and local evidence retained at the selected destination.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
