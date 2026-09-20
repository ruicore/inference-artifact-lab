"""Public data structures for the Model Release Gate.

The structures intentionally use plain JSON-compatible values.  This keeps a
manifest and its resulting report portable between the CLI and other tools
without making a model runtime a dependency of the gate core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Mapping


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_VERIFIED = "not_verified"


class ManifestError(ValueError):
    """Raised when a manifest is malformed or incomplete."""


# These are the formats understood by the public P1 adapters and fixture
# harness.  Rejecting arbitrary labels matters because ``format`` participates
# in the release contract; a typo must not silently become a new format.
SUPPORTED_ARTIFACT_FORMATS = frozenset({"fixture", "onnx", "tensorrt", "tensorrt-engine"})
SUPPORTED_TENSOR_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "string",
        "tensor(bool)",
        "tensor(bfloat16)",
        "tensor(float16)",
        "tensor(float)",
        "tensor(double)",
        "tensor(int8)",
        "tensor(int16)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(uint8)",
        "tensor(uint16)",
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(string)",
    }
)


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ArtifactSpec:
    """Identity and integrity expectations for one artifact file."""

    path: str
    format: str
    sha256: str
    size_bytes: int | None = None
    build_inputs: Mapping[str, Any] = field(default_factory=dict)
    tools: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactSpec":
        if not isinstance(value, Mapping):
            raise ManifestError("artifact must be an object")
        path = _required_string(value.get("path"), "artifact.path")
        fmt = _required_string(value.get("format"), "artifact.format")
        if fmt not in SUPPORTED_ARTIFACT_FORMATS:
            raise ManifestError(
                f"artifact.format must be one of {sorted(SUPPORTED_ARTIFACT_FORMATS)}"
            )
        digest = _required_string(value.get("sha256"), "artifact.sha256").lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestError("artifact.sha256 must be a 64-character hexadecimal digest")
        size = value.get("size_bytes")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ManifestError("artifact.size_bytes must be a non-negative integer")
        build_inputs = value.get("build_inputs", {})
        tools = value.get("tools", {})
        if not isinstance(build_inputs, Mapping) or not isinstance(tools, Mapping):
            raise ManifestError("artifact.build_inputs and artifact.tools must be objects")
        return cls(path=path, format=fmt, sha256=digest, size_bytes=size, build_inputs=dict(build_inputs), tools=dict(tools))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path, "format": self.format, "sha256": self.sha256}
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.build_inputs:
            result["build_inputs"] = dict(self.build_inputs)
        if self.tools:
            result["tools"] = dict(self.tools)
        return result


@dataclass(frozen=True)
class TensorSpec:
    """A single input or output contract entry."""

    name: str
    dtype: str
    shape: tuple[int | str | None, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], field_name: str) -> "TensorSpec":
        if not isinstance(value, Mapping):
            raise ManifestError(f"{field_name} must be an object")
        name = _required_string(value.get("name"), f"{field_name}.name")
        dtype = _required_string(value.get("dtype"), f"{field_name}.dtype")
        if dtype not in SUPPORTED_TENSOR_DTYPES:
            raise ManifestError(
                f"{field_name}.dtype must be a supported tensor dtype"
            )
        shape_value = value.get("shape")
        if not isinstance(shape_value, list):
            raise ManifestError(f"{field_name}.shape must be a list")
        shape: list[int | str | None] = []
        for index, dimension in enumerate(shape_value):
            if dimension is None or isinstance(dimension, str):
                if isinstance(dimension, str) and not dimension.strip():
                    raise ManifestError(
                        f"{field_name}.shape[{index}] must not be an empty dimension name"
                    )
                shape.append(dimension)
            elif isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0:
                shape.append(dimension)
            else:
                raise ManifestError(f"{field_name}.shape[{index}] must be a non-negative integer, string, or null")
        return cls(name=name, dtype=dtype, shape=tuple(shape))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class Manifest:
    """Versioned release contract used as the gate's source of truth."""

    schema_version: str
    model_name: str
    model_version: str
    source: str
    artifact: ArtifactSpec
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    runtime: Mapping[str, Any] = field(default_factory=dict)
    tolerances: Mapping[str, float] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    profiles: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Manifest":
        if not isinstance(value, Mapping):
            raise ManifestError("manifest must be a JSON object")
        schema = _required_string(value.get("schema_version"), "schema_version")
        if schema != "1":
            raise ManifestError("schema_version must be '1'")
        model = value.get("model")
        if not isinstance(model, Mapping):
            raise ManifestError("model must be an object")
        model_name = _required_string(model.get("name"), "model.name")
        model_version = _required_string(model.get("version"), "model.version")
        source = _required_string(model.get("source"), "model.source")
        artifact_value = value.get("artifact")
        if not isinstance(artifact_value, Mapping):
            raise ManifestError("artifact must be an object")
        contract = value.get("contract")
        if not isinstance(contract, Mapping):
            raise ManifestError("contract must be an object")
        inputs_value = contract.get("inputs")
        outputs_value = contract.get("outputs")
        if not isinstance(inputs_value, list) or not inputs_value:
            raise ManifestError("contract.inputs must be a non-empty list")
        if not isinstance(outputs_value, list) or not outputs_value:
            raise ManifestError("contract.outputs must be a non-empty list")
        inputs = tuple(TensorSpec.from_dict(item, f"contract.inputs[{i}]") for i, item in enumerate(inputs_value))
        outputs = tuple(TensorSpec.from_dict(item, f"contract.outputs[{i}]") for i, item in enumerate(outputs_value))
        for direction, tensors in (("inputs", inputs), ("outputs", outputs)):
            names = [tensor.name for tensor in tensors]
            if len(names) != len(set(names)):
                raise ManifestError(f"contract.{direction} tensor names must be unique")
        runtime = value.get("runtime", {})
        tolerances = value.get("tolerances", {})
        environment = value.get("environment", {})
        profiles_value = contract.get("profiles", [])
        for name, section in (("runtime", runtime), ("tolerances", tolerances), ("environment", environment)):
            if not isinstance(section, Mapping):
                raise ManifestError(f"{name} must be an object")
        if not isinstance(profiles_value, list) or any(not isinstance(item, str) or not item.strip() for item in profiles_value):
            raise ManifestError("contract.profiles must be a list of non-empty strings")
        profiles = tuple(profiles_value)
        if len(profiles) != len(set(profiles)):
            raise ManifestError("contract.profiles must be unique")
        parsed_tolerances: dict[str, float] = {}
        for name, tolerance in tolerances.items():
            if not isinstance(name, str) or not name.strip():
                raise ManifestError("tolerance names must be non-empty strings")
            if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
                raise ManifestError(f"tolerances.{name} must be a non-negative number")
            if not math.isfinite(float(tolerance)):
                raise ManifestError(f"tolerances.{name} must be finite")
            parsed_tolerances[str(name)] = float(tolerance)
        return cls(schema, model_name, model_version, source, ArtifactSpec.from_dict(artifact_value), inputs, outputs, dict(runtime), parsed_tolerances, dict(environment), profiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": {"name": self.model_name, "version": self.model_version, "source": self.source},
            "artifact": self.artifact.to_dict(),
            "contract": {"inputs": [item.to_dict() for item in self.inputs], "outputs": [item.to_dict() for item in self.outputs], **({"profiles": list(self.profiles)} if self.profiles else {})},
            "runtime": dict(self.runtime),
            "tolerances": dict(self.tolerances),
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: GateStatus
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.check_id, "status": self.status.value, "message": self.message, "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class GateReport:
    status: GateStatus
    manifest_digest: str | None
    artifact: Mapping[str, Any]
    checks: tuple[CheckResult, ...]
    limitations: tuple[str, ...] = ()
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "manifest_digest": self.manifest_digest,
            "artifact": dict(self.artifact),
            "checks": [check.to_dict() for check in self.checks],
            "limitations": list(self.limitations),
        }
