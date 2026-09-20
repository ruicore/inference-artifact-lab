import hashlib

import pytest

from inference_artifact_lab import Manifest, ManifestError


def _manifest(**overrides):
    data = {
        "schema_version": "1",
        "model": {"name": "fixture", "version": "1", "source": "public"},
        "artifact": {
            "path": "model.onnx",
            "format": "onnx",
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
        },
        "contract": {
            "inputs": [{"name": "x", "dtype": "float32", "shape": [1, "batch"]}],
            "outputs": [{"name": "y", "dtype": "tensor(float)", "shape": [1]}],
        },
        "tolerances": {"absolute": 0.001, "relative": 0.01},
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("schema", ["0", "1.0", "v1", ""])
def test_manifest_rejects_unknown_schema_version(schema):
    with pytest.raises(ManifestError, match="schema_version"):
        Manifest.from_dict(_manifest(schema_version=schema))


def test_manifest_rejects_unknown_artifact_format():
    with pytest.raises(ManifestError, match="artifact.format"):
        Manifest.from_dict(_manifest(artifact={"path": "m", "format": "pickle", "sha256": "0" * 64}))


def test_manifest_rejects_unknown_dtype_and_malformed_tensor_entry():
    with pytest.raises(ManifestError, match="dtype"):
        Manifest.from_dict(_manifest(contract={"inputs": [{"name": "x", "dtype": "made-up", "shape": [1]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]}))
    with pytest.raises(ManifestError, match="must be an object"):
        Manifest.from_dict(_manifest(contract={"inputs": [None], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]}))


def test_manifest_rejects_non_finite_tolerance():
    with pytest.raises(ManifestError, match="finite"):
        Manifest.from_dict(_manifest(tolerances={"absolute": float("nan")}))


def test_manifest_rejects_empty_dynamic_dimension_name():
    with pytest.raises(ManifestError, match="dimension name"):
        Manifest.from_dict(_manifest(contract={"inputs": [{"name": "x", "dtype": "float32", "shape": [""]}], "outputs": [{"name": "y", "dtype": "float32", "shape": [1]}]}))
