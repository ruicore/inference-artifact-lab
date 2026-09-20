"""Runtime-output comparison and bounded benchmark helpers.

The gate core stays independent of ONNX/TensorRT. Adapters provide plain
JSON-compatible outputs, allowing the same evidence rules to be tested without
making a runtime dependency part of the package core.
"""

from __future__ import annotations

import math
import statistics
import time
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import CheckResult, GateStatus


def _json_value(value: Any) -> Any:
    """Convert common array-like values without requiring NumPy."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _structure(value: Any, path: str = "output") -> tuple[tuple[Any, ...], list[tuple[str, float]]]:
    """Return a topology signature and numeric leaves for an output value.

    Keeping the topology separate from flattened values prevents a ragged list,
    reordered mapping, or scalar/list mismatch from passing merely because the
    number of leaves happens to match.
    """

    value = _json_value(value)
    if isinstance(value, Mapping):
        keys = list(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise TypeError(f"{path} mapping keys must be strings")
        if len(keys) != len(set(keys)):
            raise TypeError(f"{path} mapping keys must be unique")
        children = []
        leaves: list[tuple[str, float]] = []
        for key in sorted(keys):
            child_shape, child_leaves = _structure(value[key], f"{path}.{key}")
            children.append((key, child_shape))
            leaves.extend(child_leaves)
        return ("mapping", tuple(children)), leaves
    if isinstance(value, (list, tuple)):
        children = []
        leaves = []
        for index, item in enumerate(value):
            child_shape, child_leaves = _structure(item, f"{path}[{index}]")
            children.append(child_shape)
            leaves.extend(child_leaves)
        if children and any(child != children[0] for child in children[1:]):
            raise TypeError(f"{path} contains a ragged sequence")
        return ("sequence", tuple(children)), leaves
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} contains a non-numeric value")
    return ("scalar",), [(path, float(value))]


def compare_outputs(
    expected: Any,
    observed: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> CheckResult:
    """Compare finite numeric outputs under explicit absolute/relative tolerances."""

    if (
        isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not math.isfinite(float(absolute_tolerance))
        or not math.isfinite(float(relative_tolerance))
        or absolute_tolerance < 0
        or relative_tolerance < 0
    ):
        return CheckResult("runtime.equivalence", GateStatus.FAIL, "tolerances must be finite non-negative numbers", {})
    try:
        expected_shape, expected_values = _structure(expected)
        observed_shape, observed_values = _structure(observed)
    except TypeError as exc:
        return CheckResult("runtime.equivalence", GateStatus.FAIL, str(exc), {})
    if expected_shape != observed_shape:
        return CheckResult(
            "runtime.equivalence",
            GateStatus.FAIL,
            "reference and target output shapes differ",
            {"expected_shape": repr(expected_shape), "observed_shape": repr(observed_shape)},
        )
    if not expected_values:
        return CheckResult("runtime.equivalence", GateStatus.FAIL, "reference and target outputs are empty", {})
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
    memory_scope: str = "python_tracemalloc"

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "latency_seconds": list(self.latency_seconds),
            "peak_bytes": self.peak_bytes,
            "latency_min_seconds": min(self.latency_seconds),
            "latency_max_seconds": max(self.latency_seconds),
            "latency_mean_seconds": statistics.fmean(self.latency_seconds),
            "latency_median_seconds": statistics.median(self.latency_seconds),
            "latency_p95_seconds": _percentile(self.latency_seconds, 0.95),
            # Samples are sequential calls. Counting the run count again would
            # overstate throughput by the number of samples.
            "throughput_runs_per_second": 1.0 / statistics.fmean(self.latency_seconds),
            "peak_memory_scope": self.memory_scope,
        }


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    """Linear-interpolated percentile with no external statistics dependency."""

    if not values:
        raise ValueError("cannot compute a percentile for no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark(call: Callable[[], Any], *, warmup_runs: int = 1, measured_runs: int = 5) -> BenchmarkResult:
    """Measure a bounded callable workload with an explicit sample count."""

    if isinstance(warmup_runs, bool) or isinstance(measured_runs, bool) or warmup_runs < 0 or measured_runs <= 0:
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
