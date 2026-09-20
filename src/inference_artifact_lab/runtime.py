"""Runtime-output comparison and bounded benchmark helpers.

The gate core stays independent of ONNX/TensorRT. Adapters provide plain
JSON-compatible outputs, allowing the same evidence rules to be tested without
making a runtime dependency part of the package core.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .models import CheckResult, GateStatus


def _json_value(value: Any) -> Any:
    """Convert common array-like values without requiring NumPy."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _numbers(value: Any, path: str = "output") -> Iterator[tuple[str, float]]:
    value = _json_value(value)
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _numbers(item, f"{path}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} contains a non-numeric value")
    yield path, float(value)


def compare_outputs(
    expected: Any,
    observed: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> CheckResult:
    """Compare finite numeric outputs under explicit absolute/relative tolerances."""

    try:
        expected_values = list(_numbers(expected))
        observed_values = list(_numbers(observed))
    except TypeError as exc:
        return CheckResult("runtime.equivalence", GateStatus.FAIL, str(exc), {})
    if len(expected_values) != len(observed_values):
        return CheckResult(
            "runtime.equivalence",
            GateStatus.FAIL,
            "reference and target output sizes differ",
            {"expected_values": len(expected_values), "observed_values": len(observed_values)},
        )
    max_error = 0.0
    mismatches = 0
    for (expected_path, expected_number), (_, observed_number) in zip(expected_values, observed_values):
        if not math.isfinite(expected_number) or not math.isfinite(observed_number):
            return CheckResult("runtime.equivalence", GateStatus.FAIL, "reference or target output contains a non-finite value", {"path": expected_path})
        error = abs(expected_number - observed_number)
        max_error = max(max_error, error)
        if error > absolute_tolerance + relative_tolerance * abs(expected_number):
            mismatches += 1
    if mismatches:
        return CheckResult("runtime.equivalence", GateStatus.FAIL, "target output exceeds declared tolerance", {"mismatches": mismatches, "max_absolute_error": max_error})
    return CheckResult("runtime.equivalence", GateStatus.PASS, "target output is within declared tolerance", {"compared_values": len(expected_values), "max_absolute_error": max_error})


@dataclass(frozen=True)
class BenchmarkResult:
    warmup_runs: int
    measured_runs: int
    latency_seconds: tuple[float, ...]
    peak_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "latency_seconds": list(self.latency_seconds),
            "peak_bytes": self.peak_bytes,
            "latency_min_seconds": min(self.latency_seconds),
            "latency_max_seconds": max(self.latency_seconds),
        }


def benchmark(call: Callable[[], Any], *, warmup_runs: int = 1, measured_runs: int = 5) -> BenchmarkResult:
    """Measure a bounded callable workload with an explicit sample count."""

    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs must be positive")
    for _ in range(warmup_runs):
        call()
    tracemalloc.start()
    samples: list[float] = []
    try:
        for _ in range(measured_runs):
            started = time.perf_counter()
            call()
            samples.append(time.perf_counter() - started)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return BenchmarkResult(warmup_runs, measured_runs, tuple(samples), peak_bytes)
