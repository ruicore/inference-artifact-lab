import hashlib

from inference_artifact_lab import GateStatus, Manifest
from inference_artifact_lab.release import run_release_gate


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
            "tolerances": {"absolute": 0.01},
            "environment": {"test": "local"},
        }
    )
    common = {
        "observed_contract": {
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1]}],
            "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}],
        },
        "observed_environment": {"test": "local"},
    }
    blocked = run_release_gate(manifest, **common)
    assert blocked.status is GateStatus.BLOCKED
    passed = run_release_gate(manifest, reference_outputs=[1.0], target_outputs=[1.0], benchmark_call=lambda: None, **common)
    assert passed.status is GateStatus.PASS


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
        }
    )
    report = run_release_gate(
        manifest,
        observed_contract={"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
        observed_environment={"test": "local"},
        reference_outputs=[1.0],
        target_outputs=[1.0],
        benchmark_evidence={"source": "external", "throughput_runs_per_second": 10.0},
    )
    assert report.status is GateStatus.PASS


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
