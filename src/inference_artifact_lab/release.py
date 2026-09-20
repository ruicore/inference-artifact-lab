"""Full Phase 1 release decision composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .gate import run_gate
from .models import CheckResult, GateReport, GateStatus, Manifest
from .runtime import benchmark, compare_outputs


def _aggregate(checks: list[CheckResult]) -> GateStatus:
    statuses = {check.status for check in checks}
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.BLOCKED in statuses:
        return GateStatus.BLOCKED
    if GateStatus.NOT_VERIFIED in statuses:
        return GateStatus.NOT_VERIFIED
    return GateStatus.PASS


def run_release_gate(
    manifest: Manifest,
    artifact_path: str | Path | None = None,
    *,
    observed_contract: Mapping[str, Any] | None = None,
    observed_environment: Mapping[str, Any] | None = None,
    reference_outputs: Any = None,
    target_outputs: Any = None,
    benchmark_call: Callable[[], Any] | None = None,
    benchmark_evidence: Mapping[str, Any] | None = None,
) -> GateReport:
    """Run the complete Phase 1 decision for supplied adapter observations.

    Runtime adapters remain outside the core package. They provide plain output
    values and a callable workload; this function applies the same evidence and
    status rules to those observations.
    """

    core = run_gate(
        manifest,
        artifact_path,
        observed_contract=observed_contract,
        observed_environment=observed_environment,
    )
    checks = list(core.checks)
    if reference_outputs is None or target_outputs is None:
        checks.append(CheckResult("runtime.equivalence", GateStatus.BLOCKED, "reference and target runtime outputs are required", {}))
    else:
        absolute = float(manifest.tolerances.get("absolute", 0.0))
        relative = float(manifest.tolerances.get("relative", 0.0))
        checks.append(compare_outputs(reference_outputs, target_outputs, absolute_tolerance=absolute, relative_tolerance=relative))
    if benchmark_evidence is not None:
        checks.append(CheckResult("benchmark.workload", GateStatus.PASS, "benchmark workload evidence supplied by the declared runtime harness", dict(benchmark_evidence)))
    elif benchmark_call is None:
        checks.append(CheckResult("benchmark.workload", GateStatus.BLOCKED, "benchmark workload evidence is required", {}))
    else:
        result = benchmark(benchmark_call)
        checks.append(CheckResult("benchmark.workload", GateStatus.PASS, "benchmark workload completed", result.to_dict()))
    limitations = tuple(item for item in core.limitations if "not implemented" not in item)
    return GateReport(_aggregate(checks), core.manifest_digest, core.artifact, tuple(checks), limitations)
