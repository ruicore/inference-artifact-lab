"""Acceptance-oriented tests for the bounded package runtime launcher."""

from __future__ import annotations

import hashlib
import json
import platform

from inference_artifact_lab.__main__ import main


def test_ac08_incompatible_environment_fails_before_loading_fixture_or_adapter(tmp_path, capsys) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"public test artifact")
    manifest = {
        "schema_version": "1",
        "model": {"name": "fixture", "version": "1", "source": "public"},
        "artifact": {"path": "model.onnx", "format": "onnx", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "contract": {"inputs": [{"name": "data", "dtype": "tensor(float)", "shape": [1]}], "outputs": [{"name": "output", "dtype": "tensor(float)", "shape": [1]}]},
        "environment": {"python": "0.0-impossible", "provider": "CPUExecutionProvider"},
        "fixture": {"input_sha256": "0" * 64, "reference_output_sha256": "0" * 64},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code = main([str(manifest_path), "--runtime", "onnx-cpu", "--inputs-npy", str(tmp_path / "missing-fixture.npy"), "--reference-npy", str(tmp_path / "missing-reference.npy")])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["status"] == "fail"
    environment_check = next(check for check in report["checks"] if check["id"] == "environment.compatibility")
    assert environment_check["status"] == "fail"
    assert "python" in environment_check["evidence"]["mismatches"]


def test_runtime_launcher_rejects_supplied_environment_and_outputs(tmp_path, capsys) -> None:
    code = main([str(tmp_path / "manifest.json"), "--runtime", "onnx-cpu", "--inputs-npy", "fixture.npy", "--reference-npy", "reference.npy", "--environment", "supplied.json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert "supplied observation files are not allowed" in payload["error"]


def test_changed_fixture_fails_before_loading_onnx_adapter(tmp_path, capsys) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"public test artifact")
    fixture = tmp_path / "fixture.npy"
    fixture.write_bytes(b"changed fixture; intentionally not valid npy")
    reference = tmp_path / "reference.npy"
    reference.write_bytes(b"reference; intentionally not valid npy")
    manifest = {
        "schema_version": "1", "model": {"name": "fixture", "version": "1", "source": "public"},
        "artifact": {"path": "model.onnx", "format": "onnx", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "contract": {"inputs": [{"name": "data", "dtype": "tensor(float)", "shape": [1]}], "outputs": [{"name": "output", "dtype": "tensor(float)", "shape": [1]}]},
        "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "provider": "CPUExecutionProvider"},
        "fixture": {"input_sha256": hashlib.sha256(b"original fixture").hexdigest(), "reference_output_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code = main([str(manifest_path), "--runtime", "onnx-cpu", "--inputs-npy", str(fixture), "--reference-npy", str(reference)])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["status"] == "fail"
    assert next(check for check in report["checks"] if check["id"] == "fixture.binding")["status"] == "fail"
