import hashlib
import numpy as np

from inference_artifact_lab import GateStatus, Manifest
from inference_artifact_lab.release import run_bound_adapter_gate, run_release_gate


def test_release_gate_requires_runtime_and_benchmark_evidence(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    manifest = Manifest.from_dict(
        {
            "schema_version": "1",
            "model": {"name": "m", "version": "1", "source": "public"},
            "artifact": {"path": str(artifact), "format": "fixture", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
            "contract": {
                "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}],
                "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}],
            },
            "runtime": {"profile": "generated-preview", "acceptance_role": "optional_preview"},
            "tolerances": {"absolute": 0.01},
            "environment": {"test": "local"},
            "fixture": {"input_sha256": hashlib.sha256(b"fixture").hexdigest(), "reference_output_sha256": hashlib.sha256(b"reference").hexdigest()},
        }
    )
    common = {
        "observed_contract": {
            "artifact_format": "fixture",
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}],
            "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}],
        },
        "observed_environment": {"test": "local"},
    }
    blocked = run_release_gate(manifest, **common)
    assert blocked.status is GateStatus.BLOCKED
    assert blocked.to_dict()["scope"] == {"runtime": dict(manifest.runtime), "environment": dict(manifest.environment)}
    passed = run_release_gate(manifest, reference_outputs=[1.0], target_outputs=[1.0], benchmark_call=lambda: None, fixture_bytes=b"fixture", reference_output_bytes=b"reference", **common)
    assert passed.status is GateStatus.NOT_VERIFIED
    assert passed.to_dict()["scope"] == blocked.to_dict()["scope"]
    assert next(check for check in passed.checks if check.check_id == "runtime.provenance").status is GateStatus.NOT_VERIFIED


def test_release_gate_accepts_declared_external_benchmark_evidence(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    manifest = Manifest.from_dict(
        {
            "schema_version": "1",
            "model": {"name": "m", "version": "1", "source": "public"},
            "artifact": {"path": str(artifact), "format": "fixture", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
            "contract": {"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
            "tolerances": {"absolute": 0.01},
            "environment": {"test": "local"},
            "fixture": {"input_sha256": hashlib.sha256(b"fixture").hexdigest(), "reference_output_sha256": hashlib.sha256(b"reference").hexdigest()},
        }
    )
    report = run_release_gate(
        manifest,
        observed_contract={"artifact_format": "fixture", "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        observed_environment={"test": "local"},
        reference_outputs=[1.0],
        target_outputs=[1.0],
        benchmark_evidence={"source": "external", "warmup_runs": 1, "measured_runs": 5, "latency_mean_seconds": 0.1, "throughput_runs_per_second": 10.0, "peak_bytes": 128, "peak_memory_scope": "test_process"},
        fixture_bytes=b"fixture",
        reference_output_bytes=b"reference",
    )
    assert report.status is GateStatus.NOT_VERIFIED


def test_bound_adapter_decodes_exact_pinned_bytes_and_runs_same_fixture(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    fixture = tmp_path / "fixture.npy"
    reference = tmp_path / "reference.npy"
    np.save(fixture, np.array([1.0], dtype=np.float32))
    np.save(reference, np.array([1.0], dtype=np.float32))
    manifest = Manifest.from_dict({
        "schema_version": "1", "model": {"name": "m", "version": "1", "source": "public"},
        "artifact": {"path": str(artifact), "format": "fixture", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "contract": {"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        "runtime": {"profile": "generated-cpu", "acceptance_role": "cpu_baseline"},
        "environment": {"test": "local"},
        "fixture": {"input_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "reference_output_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()},
    })
    class Adapter:
        def __init__(self) -> None:
            self.observed = []
        def contract(self):
            return {"artifact_format": "fixture", "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]}
        def run(self, inputs):
            self.observed.append(inputs["x"].tolist())
            return {"y": inputs["x"]}
    adapter = Adapter()
    report = run_bound_adapter_gate(manifest, artifact, adapter=adapter, fixture_path=fixture, reference_output_path=reference, observed_environment={"test": "local"})
    assert report.status is GateStatus.PASS
    assert report.to_dict()["scope"]["runtime"] == manifest.runtime
    assert adapter.observed and all(item == [1.0] for item in adapter.observed)
    assert next(check for check in report.checks if check.check_id == "runtime.provenance").status is GateStatus.PASS
    reference.write_bytes(b"tampered")
    changed = run_bound_adapter_gate(manifest, artifact, adapter=adapter, fixture_path=fixture, reference_output_path=reference, observed_environment={"test": "local"})
    assert changed.status is GateStatus.FAIL
    assert changed.to_dict()["scope"] == report.to_dict()["scope"]


def test_release_gate_rejects_changed_fixture_before_runtime(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    manifest = Manifest.from_dict({
        "schema_version": "1", "model": {"name": "m", "version": "1", "source": "public"},
        "artifact": {"path": str(artifact), "format": "fixture", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "contract": {"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        "environment": {"test": "local"},
        "fixture": {"input_sha256": hashlib.sha256(b"fixture").hexdigest(), "reference_output_sha256": hashlib.sha256(b"reference").hexdigest()},
    })
    called = False
    def workload():
        nonlocal called
        called = True
    report = run_release_gate(
        manifest, observed_contract={"artifact_format": "fixture", "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        observed_environment={"test": "local"}, reference_outputs=[1.0], target_outputs=[1.0], benchmark_call=workload,
        fixture_bytes=b"changed", reference_output_bytes=b"reference",
    )
    assert report.status is GateStatus.FAIL
    assert called is False
    assert next(check for check in report.checks if check.check_id == "fixture.binding").status is GateStatus.FAIL


def test_release_gate_rejects_incomplete_benchmark_evidence(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    manifest = Manifest.from_dict({
        "schema_version": "1", "model": {"name": "m", "version": "1", "source": "public"},
        "artifact": {"path": str(artifact), "format": "fixture", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "contract": {"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        "environment": {"test": "local"},
        "fixture": {"input_sha256": hashlib.sha256(b"fixture").hexdigest(), "reference_output_sha256": hashlib.sha256(b"reference").hexdigest()},
    })
    report = run_release_gate(
        manifest, observed_contract={"artifact_format": "fixture", "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        observed_environment={"test": "local"}, reference_outputs=[1.0], target_outputs=[1.0],
        fixture_bytes=b"fixture", reference_output_bytes=b"reference", benchmark_evidence={"source": "external"},
    )
    assert report.status is GateStatus.FAIL
    assert next(check for check in report.checks if check.check_id == "benchmark.workload").status is GateStatus.FAIL


def test_release_gate_skips_runtime_after_core_failure(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    manifest = Manifest.from_dict(
        {
            "schema_version": "1",
            "model": {"name": "m", "version": "1", "source": "public"},
            "artifact": {"path": str(artifact), "format": "fixture", "sha256": "0" * 64},
            "contract": {"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
            "environment": {"test": "local"},
        }
    )
    called = False
    def workload():
        nonlocal called
        called = True
    report = run_release_gate(manifest, observed_contract={"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]}, observed_environment={"test": "local"}, reference_outputs=[1.0], target_outputs=[1.0], benchmark_call=workload)
    assert report.status is GateStatus.FAIL
    assert called is False
