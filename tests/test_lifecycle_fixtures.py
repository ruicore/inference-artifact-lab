from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from inference_artifact_lab.lifecycle import LoadSpec
from inference_artifact_lab.lifecycle_fixtures import generate_fixtures
from inference_artifact_lab.lifecycle_onnx import OnnxLifecycleAdapter


def test_generated_fixtures_are_reproducible_and_run_on_real_cpu(tmp_path):
    first = generate_fixtures(tmp_path / "first")
    second = generate_fixtures(tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text())
    digests = []
    for name, entry in manifest["artifacts"].items():
        spec = LoadSpec.from_dict(entry["load_spec"], base_dir=first.parent)
        assert spec.artifact_path.read_bytes() == (second.parent / f"{name}.onnx").read_bytes()
        assert hashlib.sha256(spec.artifact_path.read_bytes()).hexdigest() == spec.artifact_sha256
        fixture_bytes = (first.parent / entry["fixture_path"]).read_bytes()
        assert hashlib.sha256(fixture_bytes).hexdigest() == entry["fixture_sha256"]
        case = json.loads(fixture_bytes)["cases"][0]
        values = case["inputs"]["input"]
        inputs = np.asarray(values["values"], dtype=values["dtype"]).reshape(values["shape"])
        expected = case["expected_outputs"]["output"]
        adapter = OnnxLifecycleAdapter(spec)
        np.testing.assert_allclose(adapter.run({"input": inputs})["output"],
                                   np.asarray(expected["values"], dtype=expected["dtype"]).reshape(expected["shape"]),
                                   rtol=case["rtol"], atol=case["atol"])
        adapter.close()
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.run({"input": inputs})
        digests.append(spec.artifact_sha256)
    assert len(set(digests)) == 3


def test_incomplete_public_fixture_arguments_reject(tmp_path):
    with pytest.raises(ValueError, match="all three"):
        generate_fixtures(tmp_path, public_manifest=tmp_path / "unused.json")


def test_public_artifact_digest_rejects_before_runtime_construction(tmp_path, monkeypatch):
    from inference_artifact_lab import adapters

    original = Path(__file__).resolve().parents[1] / "examples/squeezenet11-torchvision.manifest.json"
    manifest = json.loads(original.read_text())
    (tmp_path / "invalid.onnx").write_bytes(b"modified artifact")
    manifest["artifact"]["path"] = "invalid.onnx"
    source = tmp_path / "public.json"
    source.write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        pytest.fail("runtime must not load an artifact that failed digest verification")

    monkeypatch.setattr(adapters, "OnnxRuntimeAdapter", forbidden)
    with pytest.raises(ValueError, match="artifact differs"):
        generate_fixtures(tmp_path / "out", public_manifest=source,
                          public_inputs=tmp_path / "unused.npy", public_reference=tmp_path / "unused-reference.npy")


def test_real_cpu_adapter_rejects_unobserved_runtime_version(tmp_path):
    path = generate_fixtures(tmp_path)
    value = json.loads(path.read_text())["artifacts"]["A1"]["load_spec"]
    value["runtime_version"] = "unobserved-version"
    spec = LoadSpec.from_dict(value, base_dir=path.parent)
    with pytest.raises(ValueError, match="version does not match"):
        OnnxLifecycleAdapter(spec)


def test_public_gate_failure_reports_the_failed_check(tmp_path, monkeypatch):
    from inference_artifact_lab import adapters, release
    from inference_artifact_lab.models import CheckResult, GateReport, GateStatus

    original = Path(__file__).resolve().parents[1] / "examples/squeezenet11-torchvision.manifest.json"
    manifest = json.loads(original.read_text())
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"isolated failure-path fixture")
    data, reference = tmp_path / "input.npy", tmp_path / "reference.npy"
    np.save(data, np.zeros((1,), dtype=np.float32))
    np.save(reference, np.ones((1,), dtype=np.float32))
    manifest["artifact"].update(path=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    manifest["fixture"] = {"input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                           "reference_output_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    manifest["environment"] = {}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    monkeypatch.setattr(adapters, "OnnxRuntimeAdapter", lambda *a, **kw: object())
    result = GateReport(GateStatus.FAIL, None, {}, (CheckResult("environment.compatibility", GateStatus.FAIL, "controlled failure"),))
    monkeypatch.setattr(release, "run_bound_adapter_gate", lambda *a, **kw: result)
    with pytest.raises(ValueError, match="environment.compatibility"):
        generate_fixtures(tmp_path / "out", public_manifest=source, public_inputs=data, public_reference=reference)
