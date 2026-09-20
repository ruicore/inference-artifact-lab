import hashlib
import json

import pytest

from inference_artifact_lab import GateStatus, Manifest, ManifestError, canonical_manifest_digest, run_gate, run_gate_from_file


def make_manifest(path, digest, size):
    return Manifest.from_dict(
        {
            "schema_version": "1",
            "model": {"name": "fixture-model", "version": "1", "source": "https://example.invalid/model"},
            "artifact": {"path": str(path), "format": "fixture", "sha256": digest, "size_bytes": size},
            "contract": {
                "inputs": [{"name": "input", "dtype": "float32", "shape": [1, "features"]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 2]}],
            },
            "environment": {"test": "local"},
        }
    )


def observed_contract():
    return {
        "inputs": [{"name": "input", "dtype": "float32", "shape": [1, "features"]}],
        "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 2]}],
    }


def test_valid_artifact_and_contract_pass(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"public fixture")
    manifest = make_manifest(artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size)

    report = run_gate(manifest, observed_contract=observed_contract(), observed_environment={"test": "local"})

    assert report.status is GateStatus.PASS
    assert report.manifest_digest == canonical_manifest_digest(manifest)
    assert {check.check_id for check in report.checks} >= {"artifact.sha256", "contract.inputs", "contract.outputs"}


def test_missing_artifact_fails(tmp_path):
    path = tmp_path / "missing.bin"
    manifest = make_manifest(path, "0" * 64, 1)

    report = run_gate(manifest, observed_contract=observed_contract(), observed_environment={"test": "local"})

    assert report.status is GateStatus.FAIL
    assert next(check for check in report.checks if check.check_id == "artifact.exists").status is GateStatus.FAIL


def test_modified_artifact_fails_sha256(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"original")
    manifest = make_manifest(artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size)
    artifact.write_bytes(b"modified")

    report = run_gate(manifest, observed_contract=observed_contract(), observed_environment={"test": "local"})

    assert report.status is GateStatus.FAIL
    check = next(check for check in report.checks if check.check_id == "artifact.sha256")
    assert check.status is GateStatus.FAIL
    assert check.evidence["expected"] != check.evidence["observed"]


def test_unavailable_contract_is_blocked(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"fixture")
    manifest = make_manifest(artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size)

    report = run_gate(manifest, observed_environment={"test": "local"})

    assert report.status is GateStatus.BLOCKED
    assert all(check.status is not GateStatus.FAIL for check in report.checks)


def test_environment_mismatch_fails(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"fixture")
    manifest = make_manifest(artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size)

    report = run_gate(manifest, observed_contract=observed_contract(), observed_environment={"test": "different"})

    check = next(check for check in report.checks if check.check_id == "environment.compatibility")
    assert check.status is GateStatus.FAIL
    assert report.status is GateStatus.FAIL


def test_manifest_rejects_invalid_digest():
    with pytest.raises(ManifestError, match="64-character"):
        Manifest.from_dict(
            {
                "schema_version": "1",
                "model": {"name": "m", "version": "1", "source": "public"},
                "artifact": {"path": "model.bin", "format": "onnx", "sha256": "bad"},
                "contract": {"inputs": [{"name": "x", "dtype": "float32", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]},
            }
        )


def test_report_is_json_serializable(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"fixture")
    manifest = make_manifest(artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size)
    report = run_gate(manifest, observed_contract=observed_contract(), observed_environment={"test": "local"})

    encoded = json.dumps(report.to_dict())
    assert '"status": "pass"' in encoded


def test_manifest_loader_resolves_relative_artifact_path(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"fixture")
    manifest = make_manifest(artifact, hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_size)
    manifest_data = manifest.to_dict()
    manifest_data["artifact"]["path"] = "model.bin"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    report = run_gate_from_file(manifest_path, observed_contract=observed_contract(), observed_environment={"test": "local"})

    assert report.status is GateStatus.PASS
    assert report.artifact["path"] == str(artifact)
