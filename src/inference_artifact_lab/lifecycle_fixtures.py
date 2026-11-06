"""Independent ONNX fixtures and pinned public inputs for PH2-R09."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

from .gate import load_manifest, sha256_file
from .lifecycle_onnx import ADAPTER_VERSION


def _write_json(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def _tensor(value: Any) -> dict[str, Any]:
    return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.reshape(-1).tolist()}


def _entry(output: Path, artifact: Path, fixture: Path) -> dict[str, Any]:
    return {
        "load_spec": {
            "artifact_path": Path(os.path.relpath(artifact, output)).as_posix(),
            "artifact_sha256": sha256_file(artifact),
            "adapter_version": ADAPTER_VERSION,
            "runtime_version": importlib.metadata.version("onnxruntime"),
            "providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "session_options": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        },
        "fixture_path": fixture.name,
        "fixture_sha256": sha256_file(fixture),
    }


def _public_fixture(output: Path, manifest_path: Path, inputs_path: Path, reference_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import onnxruntime as ort

    from .adapters import OnnxRuntimeAdapter
    from .release import run_bound_adapter_gate

    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    artifact = (manifest_path.parent / manifest.artifact.path).resolve()
    if not manifest.source_sha256 or not manifest.fixture:
        raise ValueError("public lifecycle fixture requires pinned source and fixture identity")
    if sha256_file(artifact) != manifest.artifact.sha256:
        raise ValueError("public model artifact differs from its pinned manifest")
    if sha256_file(inputs_path) != manifest.fixture["input_sha256"] or sha256_file(reference_path) != manifest.fixture["reference_output_sha256"]:
        raise ValueError("public model fixture/reference bytes differ from the pinned manifest")
    environment = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                   "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider"}
    for package in ("torch", "torchvision"):
        if package in manifest.environment:
            environment[package] = importlib.import_module(package).__version__
    adapter = OnnxRuntimeAdapter(artifact, providers=["CPUExecutionProvider"])
    report = run_bound_adapter_gate(manifest, artifact, adapter=adapter, fixture_path=inputs_path,
                                    reference_output_path=reference_path, observed_environment=environment)
    if report.status.value != "pass":
        failed = [check.check_id for check in report.checks if check.status.value != "pass"]
        raise ValueError("public model gate did not pass: " + ", ".join(failed))
    report_data = report.to_dict()
    # Phase 1 records may include the input artifact location. The PH2 sidecar
    # retains only a portable basename; the manifest digest still binds identity.
    report_data["artifact"]["path"] = artifact.name
    for check in report_data["checks"]:
        check.get("evidence", {}).pop("path", None)
    report_path = output / "public-model-gate.json"
    report_digest = _write_json(report_path, report_data)
    fixture = output / "M.fixture.json"
    _write_json(fixture, {"cases": [{"name": "public-reference", "inputs": {manifest.inputs[0].name: _tensor(np.load(inputs_path, allow_pickle=False))},
                                    "expected_outputs": {manifest.outputs[0].name: _tensor(np.load(reference_path, allow_pickle=False))},
                                    "rtol": manifest.tolerances.get("relative", 0), "atol": manifest.tolerances.get("absolute", 0)}]})
    gate = {"status": "pass", "manifest_path": Path(os.path.relpath(manifest_path, output)).as_posix(),
            "manifest_sha256": sha256_file(manifest_path), "report_path": report_path.name, "report_sha256": report_digest,
            "artifact_sha256": manifest.artifact.sha256, "input_sha256": manifest.fixture["input_sha256"],
            "reference_output_sha256": manifest.fixture["reference_output_sha256"]}
    return _entry(output, artifact, fixture), gate


def generate_fixtures(output: str | Path, *, public_manifest: str | Path | None = None,
                      public_inputs: str | Path | None = None, public_reference: str | Path | None = None) -> Path:
    """Write generated local artifacts, optionally binding the actual public gate."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    public = (public_manifest, public_inputs, public_reference)
    if any(value is not None for value in public) and not all(value is not None for value in public):
        raise ValueError("all three public manifest/input/reference paths are required together")
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inputs = np.random.default_rng(20261106).normal(size=(1, 4)).astype(np.float32)
    manifest: dict[str, Any] = {"schema_version": "lifecycle-fixtures/v1", "artifacts": {}}
    for name, offset in (("A1", 1.0), ("A2", 2.0), ("B", 3.0)):
        constant = np.full((1, 4), offset, dtype=np.float32)
        graph = helper.make_graph([helper.make_node("Add", ["input", "offset"], ["output"])], "lifecycle-generated",
                                  [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
                                  [numpy_helper.from_array(constant, "offset")])
        model = helper.make_model(graph, producer_name="inference-artifact-lab", opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.checker.check_model(model)
        artifact = directory / f"{name}.onnx"
        onnx.save_model(model, artifact)
        fixture = directory / f"{name}.fixture.json"
        _write_json(fixture, {"cases": [{"name": "generated-reference", "inputs": {"input": _tensor(inputs)},
                                       "expected_outputs": {"output": _tensor(inputs + constant)}, "rtol": 1e-5, "atol": 1e-6}]})
        manifest["artifacts"][name] = _entry(directory, artifact, fixture)
    if all(value is not None for value in public):
        entry, gate = _public_fixture(directory, Path(public_manifest), Path(public_inputs), Path(public_reference))
        manifest["artifacts"]["M"] = entry
        manifest["public_model_gate"] = gate
    path = directory / "fixtures.json"
    _write_json(path, manifest)
    return path
