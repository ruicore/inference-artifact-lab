"""Optional runtime adapters used by the Phase 1 evidence harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class RuntimeUnavailable(RuntimeError):
    """Raised when a declared runtime dependency is not installed."""


class OnnxRuntimeAdapter:
    """Small ONNX Runtime adapter with no dependency on the gate domain."""

    def __init__(self, artifact: str | Path, *, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeUnavailable("install the runtime-cpu extra for ONNX Runtime") from exc
        self._session = ort.InferenceSession(str(artifact), providers=providers)

    def contract(self) -> dict[str, list[dict[str, Any]]]:
        def describe(value: Any) -> dict[str, Any]:
            return {"name": value.name, "dtype": value.type, "shape": list(value.shape)}

        return {
            "inputs": [describe(value) for value in self._session.get_inputs()],
            "outputs": [describe(value) for value in self._session.get_outputs()],
        }

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        values = self._session.run(None, dict(inputs))
        return {spec["name"]: value.tolist() if hasattr(value, "tolist") else value for spec, value in zip(self.contract()["outputs"], values)}
