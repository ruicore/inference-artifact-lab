from inference_artifact_lab import GateStatus
from inference_artifact_lab.runtime import benchmark, compare_outputs


def test_compare_outputs_accepts_declared_tolerance() -> None:
    result = compare_outputs([1.0, 2.0], [1.001, 1.999], absolute_tolerance=0.002)
    assert result.status is GateStatus.PASS


def test_compare_outputs_rejects_non_finite_value() -> None:
    result = compare_outputs([1.0], [float("nan")], absolute_tolerance=0.1)
    assert result.status is GateStatus.FAIL


def test_compare_outputs_rejects_topology_mismatch() -> None:
    result = compare_outputs([[1.0, 2.0]], [1.0, 2.0], absolute_tolerance=0.1)
    assert result.status is GateStatus.FAIL


def test_compare_outputs_rejects_ragged_and_empty_values() -> None:
    ragged = compare_outputs([[1.0], [2.0, 3.0]], [[1.0], [2.0, 3.0]], absolute_tolerance=0.1)
    empty = compare_outputs([], [], absolute_tolerance=0.1)
    assert ragged.status is GateStatus.FAIL
    assert empty.status is GateStatus.FAIL


def test_compare_outputs_preserves_mapping_keys() -> None:
    result = compare_outputs({"a": [1.0], "b": [2.0]}, {"a": [1.0], "c": [2.0]}, absolute_tolerance=0.1)
    assert result.status is GateStatus.FAIL


def test_benchmark_records_bounded_samples() -> None:
    result = benchmark(lambda: sum(range(10)), warmup_runs=1, measured_runs=2)
    assert result.measured_runs == 2
    assert len(result.latency_seconds) == 2
    evidence = result.to_dict()
    assert evidence["throughput_runs_per_second"] > 0
    assert evidence["peak_memory_scope"] == "python_tracemalloc"
