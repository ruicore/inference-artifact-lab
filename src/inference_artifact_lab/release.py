"""Full Phase 1 release decision composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import io
import math
from pathlib import Path
from typing import Any

from .gate import run_gate
from .models import CheckResult, GateReport, GateStatus, Manifest
from .runtime import benchmark, compare_outputs


_BOUND_OBSERVER = object()


def _aggregate(checks: list[CheckResult]) -> GateStatus:
    statuses = {check.status for check in checks}
    if GateStatus.FAIL in statuses:
        return GateStatus.FAIL
    if GateStatus.BLOCKED in statuses:
        return GateStatus.BLOCKED
    if GateStatus.NOT_VERIFIED in statuses:
        return GateStatus.NOT_VERIFIED
    return GateStatus.PASS


def _fixture_check(manifest: Manifest, fixture_bytes: bytes | None, reference_output_bytes: bytes | None) -> CheckResult:
    if not manifest.fixture:
        return CheckResult("fixture.binding", GateStatus.BLOCKED, "manifest does not pin fixture and reference output digests", {})
    if fixture_bytes is None or reference_output_bytes is None:
        return CheckResult("fixture.binding", GateStatus.BLOCKED, "fixture and reference output bytes are required", {})
    observed = {
        "input_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "reference_output_sha256": hashlib.sha256(reference_output_bytes).hexdigest(),
    }
    mismatches = {key: {"expected": wanted, "observed": observed[key]} for key, wanted in manifest.fixture.items() if observed[key] != wanted}
    if mismatches:
        return CheckResult("fixture.binding", GateStatus.FAIL, "fixture or reference output differs from pinned evidence", {"mismatches": mismatches})
    return CheckResult("fixture.binding", GateStatus.PASS, "fixture and reference output match pinned digests", observed)


def _benchmark_check(evidence: Mapping[str, Any]) -> CheckResult:
    if not isinstance(evidence, Mapping):
        return CheckResult("benchmark.workload", GateStatus.FAIL, "benchmark evidence must be an object", {})
    required = ("measured_runs", "latency_mean_seconds", "throughput_runs_per_second")
    missing = [key for key in required if key not in evidence]
    if missing:
        return CheckResult("benchmark.workload", GateStatus.FAIL, "benchmark evidence is incomplete", {"missing": missing})
    runs, mean, throughput = (evidence[key] for key in required)
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        return CheckResult("benchmark.workload", GateStatus.FAIL, "measured_runs must be positive", {})
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0 for value in (mean, throughput)):
        return CheckResult("benchmark.workload", GateStatus.FAIL, "benchmark latency and throughput must be finite positive numbers", {})
    if "warmup_runs" not in evidence and "warmup_duration_ms" not in evidence:
        return CheckResult("benchmark.workload", GateStatus.NOT_VERIFIED, "benchmark warm-up scope is not recorded", dict(evidence))
    peak = evidence.get("peak_bytes")
    scope = evidence.get("peak_memory_scope")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0 or not isinstance(scope, str) or not scope:
        return CheckResult("benchmark.workload", GateStatus.NOT_VERIFIED, "peak memory is not observed for the declared benchmark scope", dict(evidence))
    return CheckResult("benchmark.workload", GateStatus.PASS, "bounded benchmark evidence was supplied", dict(evidence))


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
    fixture_bytes: bytes | None = None,
    reference_output_bytes: bytes | None = None,
    _runtime_observer: object | None = None,
) -> GateReport:
    """Compose checks from supplied observations, without claiming their provenance.

    Runtime adapters remain outside the core package. They provide plain output
    values and a callable workload; this function applies the same evidence and
    status rules to those observations. Only ``run_bound_adapter_gate`` can
    verify that reference bytes were decoded and the adapter ran the same
    pinned fixture; caller-supplied observations remain ``not_verified``.
    """

    core = run_gate(
        manifest,
        artifact_path,
        observed_contract=observed_contract,
        observed_environment=observed_environment,
    )
    checks = list(core.checks)
    checks.append(_fixture_check(manifest, fixture_bytes, reference_output_bytes))
    core_failed = any(check.status is GateStatus.FAIL for check in checks)
    if core_failed:
        checks.append(CheckResult("runtime.equivalence", GateStatus.BLOCKED, "runtime verification skipped because an integrity, contract, or environment check failed", {}))
        checks.append(CheckResult("benchmark.workload", GateStatus.BLOCKED, "benchmark skipped because an integrity, contract, or environment check failed", {}))
        limitations = tuple(item for item in core.limitations if "not implemented" not in item)
        return GateReport(_aggregate(checks), core.manifest_digest, core.artifact, tuple(checks), limitations, scope=core.scope)
    if reference_outputs is None or target_outputs is None:
        checks.append(CheckResult("runtime.equivalence", GateStatus.BLOCKED, "reference and target runtime outputs are required", {}))
    else:
        absolute = float(manifest.tolerances.get("absolute", 0.0))
        relative = float(manifest.tolerances.get("relative", 0.0))
        checks.append(compare_outputs(reference_outputs, target_outputs, absolute_tolerance=absolute, relative_tolerance=relative))
    if benchmark_evidence is not None:
        checks.append(_benchmark_check(benchmark_evidence))
    elif benchmark_call is None:
        checks.append(CheckResult("benchmark.workload", GateStatus.BLOCKED, "benchmark workload evidence is required", {}))
    else:
        result = benchmark(benchmark_call)
        checks.append(CheckResult("benchmark.workload", GateStatus.PASS, "benchmark workload completed", result.to_dict()))
    if _runtime_observer is _BOUND_OBSERVER:
        checks.append(CheckResult("runtime.provenance", GateStatus.PASS, "package runner decoded pinned reference bytes and executed adapter on pinned fixture", {}))
    else:
        checks.append(CheckResult("runtime.provenance", GateStatus.NOT_VERIFIED, "supplied runtime observations are not independently bound to fixture and reference bytes", {}))
    limitations = tuple(item for item in core.limitations if "not implemented" not in item)
    return GateReport(_aggregate(checks), core.manifest_digest, core.artifact, tuple(checks), limitations, scope=core.scope)


def run_bound_adapter_gate(
    manifest: Manifest,
    artifact_path: str | Path,
    *,
    adapter: Any,
    fixture_path: str | Path,
    reference_output_path: str | Path,
    observed_environment: Mapping[str, Any],
) -> GateReport:
    """Run a one-input/one-output adapter from pinned serialized NumPy evidence.

    The package decodes the exact bytes it hashes, then supplies the decoded
    fixture to the adapter. This is a bounded runner, not a generic serving API.
    """
    import numpy as np  # optional runtime dependency

    fixture_bytes = Path(fixture_path).read_bytes()
    reference_bytes = Path(reference_output_path).read_bytes()
    binding = _fixture_check(manifest, fixture_bytes, reference_bytes)
    contract = adapter.contract()
    if binding.status is not GateStatus.PASS:
        return run_release_gate(
            manifest, artifact_path, observed_contract=contract,
            observed_environment=observed_environment, fixture_bytes=fixture_bytes,
            reference_output_bytes=reference_bytes,
        )
    if len(contract.get("inputs", [])) != 1 or len(contract.get("outputs", [])) != 1:
        raise ValueError("bound adapter runner requires one input and one output")
    fixture = np.load(io.BytesIO(fixture_bytes), allow_pickle=False)
    reference = np.load(io.BytesIO(reference_bytes), allow_pickle=False)
    input_name = contract["inputs"][0]["name"]
    output_name = contract["outputs"][0]["name"]
    core = run_gate(manifest, artifact_path, observed_contract=contract, observed_environment=observed_environment)
    if core.status is not GateStatus.PASS:
        return run_release_gate(
            manifest, artifact_path, observed_contract=contract,
            observed_environment=observed_environment, fixture_bytes=fixture_bytes,
            reference_output_bytes=reference_bytes,
        )
    target = adapter.run({input_name: fixture})[output_name]
    return run_release_gate(
        manifest, artifact_path, observed_contract=contract,
        observed_environment=observed_environment,
        reference_outputs=reference, target_outputs=target,
        benchmark_call=lambda: adapter.run({input_name: fixture}),
        fixture_bytes=fixture_bytes, reference_output_bytes=reference_bytes,
        _runtime_observer=_BOUND_OBSERVER,
    )
