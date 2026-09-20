"""Integrity and contract checks for Model Release Gate reports."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping

from .models import CheckResult, GateReport, GateStatus, Manifest, ManifestError, TensorSpec


def canonical_manifest_digest(manifest: Manifest) -> str:
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"unable to read manifest {manifest_path}: {exc}") from exc
    return Manifest.from_dict(value)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integrity_checks(manifest: Manifest, artifact_path: Path) -> list[CheckResult]:
    expected = manifest.artifact
    if not artifact_path.exists():
        return [CheckResult("artifact.exists", GateStatus.FAIL, "artifact file does not exist", {"path": str(artifact_path)})]
    if not artifact_path.is_file():
        return [CheckResult("artifact.exists", GateStatus.FAIL, "artifact path is not a regular file", {"path": str(artifact_path)})]
    observed_size = artifact_path.stat().st_size
    checks = [CheckResult("artifact.exists", GateStatus.PASS, "artifact file exists", {"path": str(artifact_path)})]
    if expected.size_bytes is not None and observed_size != expected.size_bytes:
        checks.append(CheckResult("artifact.size", GateStatus.FAIL, "artifact size does not match manifest", {"expected": expected.size_bytes, "observed": observed_size}))
    else:
        checks.append(CheckResult("artifact.size", GateStatus.PASS, "artifact size matches manifest", {"observed": observed_size}))
    observed_digest = sha256_file(artifact_path)
    if observed_digest != expected.sha256:
        checks.append(CheckResult("artifact.sha256", GateStatus.FAIL, "artifact SHA-256 does not match manifest", {"expected": expected.sha256, "observed": observed_digest}))
    else:
        checks.append(CheckResult("artifact.sha256", GateStatus.PASS, "artifact SHA-256 matches manifest", {"sha256": observed_digest}))
    return checks


def _compare_tensor_specs(expected: tuple[TensorSpec, ...], observed: Any, direction: str) -> CheckResult:
    if observed is None:
        return CheckResult(f"contract.{direction}", GateStatus.BLOCKED, f"observed artifact {direction} contract is unavailable", {})
    if not isinstance(observed, list):
        return CheckResult(f"contract.{direction}", GateStatus.FAIL, f"observed artifact {direction} contract must be a list", {})
    actual = []
    try:
        actual = [TensorSpec.from_dict(item, f"observed.{direction}[{i}]") for i, item in enumerate(observed)]
    except ManifestError as exc:
        return CheckResult(f"contract.{direction}", GateStatus.FAIL, str(exc), {})
    if len(actual) != len(expected):
        return CheckResult(f"contract.{direction}", GateStatus.FAIL, f"artifact {direction} count does not match manifest", {"expected": len(expected), "observed": len(actual)})
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            return CheckResult(f"contract.{direction}", GateStatus.FAIL, f"artifact {direction} entry {index} does not match manifest", {"expected": want.to_dict(), "observed": got.to_dict()})
    return CheckResult(f"contract.{direction}", GateStatus.PASS, f"artifact {direction} match manifest", {"count": len(expected)})


def _environment_check(manifest: Manifest, observed: Mapping[str, Any] | None) -> CheckResult:
    if not manifest.environment:
        return CheckResult("environment.compatibility", GateStatus.NOT_VERIFIED, "manifest declares no environment compatibility scope", {})
    if observed is None:
        return CheckResult("environment.compatibility", GateStatus.BLOCKED, "declared environment was not tested", {"required": dict(manifest.environment)})
    mismatches = {key: {"expected": value, "observed": observed.get(key)} for key, value in manifest.environment.items() if observed.get(key) != value}
    if mismatches:
        return CheckResult("environment.compatibility", GateStatus.FAIL, "observed environment does not satisfy manifest", {"mismatches": mismatches})
    return CheckResult("environment.compatibility", GateStatus.PASS, "observed environment satisfies manifest", {"observed": dict(observed)})


def _profiles_check(manifest: Manifest, observed_contract: Mapping[str, Any]) -> CheckResult | None:
    if not manifest.profiles:
        return None
    observed = observed_contract.get("profiles")
    if observed is None:
        return CheckResult("contract.profiles", GateStatus.BLOCKED, "observed artifact profiles are unavailable", {"required": list(manifest.profiles)})
    if not isinstance(observed, list) or any(not isinstance(item, str) for item in observed):
        return CheckResult("contract.profiles", GateStatus.FAIL, "observed artifact profiles must be a list of strings", {})
    if set(observed) != set(manifest.profiles):
        return CheckResult("contract.profiles", GateStatus.FAIL, "artifact profiles do not match manifest", {"expected": list(manifest.profiles), "observed": observed})
    return CheckResult("contract.profiles", GateStatus.PASS, "artifact profiles match manifest", {"profiles": observed})


def current_environment() -> dict[str, str]:
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()}


def _aggregate(checks: list[CheckResult]) -> GateStatus:
    statuses = {check.status for check in checks}
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.BLOCKED in statuses:
        return GateStatus.BLOCKED
    if GateStatus.NOT_VERIFIED in statuses:
        return GateStatus.NOT_VERIFIED
    return GateStatus.PASS


def run_gate(
    manifest: Manifest,
    artifact_path: str | Path | None = None,
    *,
    observed_contract: Mapping[str, Any] | None = None,
    observed_environment: Mapping[str, Any] | None = None,
) -> GateReport:
    """Run deterministic core checks and return a serializable report.

    Runtime numerical checks and benchmarks are deliberately represented by
    later checks; this core has no runtime dependency and reports unavailable
    runtime evidence as ``blocked`` rather than guessing.
    """
    path = Path(artifact_path) if artifact_path is not None else Path(manifest.artifact.path)
    checks = _integrity_checks(manifest, path)
    observed_contract = observed_contract or {}
    checks.extend((_compare_tensor_specs(manifest.inputs, observed_contract.get("inputs"), "inputs"), _compare_tensor_specs(manifest.outputs, observed_contract.get("outputs"), "outputs")))
    profiles_check = _profiles_check(manifest, observed_contract)
    if profiles_check is not None:
        checks.append(profiles_check)
    checks.append(_environment_check(manifest, observed_environment))
    return GateReport(_aggregate(checks), canonical_manifest_digest(manifest), {**manifest.artifact.to_dict(), "path": str(path)}, tuple(checks), ("Runtime numerical equivalence and benchmarks are not implemented by the core gate.",))


def run_gate_from_file(manifest_path: str | Path, artifact_path: str | Path | None = None, **kwargs: Any) -> GateReport:
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    if artifact_path is None:
        candidate = Path(manifest.artifact.path)
        if not candidate.is_absolute():
            candidate = manifest_file.parent / candidate
        artifact_path = candidate
    return run_gate(manifest, artifact_path, **kwargs)
