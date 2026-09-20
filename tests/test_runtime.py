from inference_artifact_lab import GateStatus
from inference_artifact_lab.runtime import benchmark, compare_outputs


def test_compare_outputs_accepts_declared_tolerance() -> None:
    result = compare_outputs([1.0, 2.0], [1.001, 1.999], absolute_tolerance=0.002)
    assert result.status is GateStatus.PASS


def test_compare_outputs_rejects_non_finite_value() -> None:
    result = compare_outputs([1.0], [float("nan")], absolute_tolerance=0.1)
    assert result.status is GateStatus.FAIL


def test_benchmark_records_bounded_samples() -> None:
    result = benchmark(lambda: sum(range(10)), warmup_runs=1, measured_runs=2)
    assert result.measured_runs == 2
    assert len(result.latency_seconds) == 2
