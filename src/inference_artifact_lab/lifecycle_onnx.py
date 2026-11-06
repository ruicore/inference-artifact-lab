"""CPU session ownership for the independent lifecycle experiment (PH2-R07)."""

from __future__ import annotations

from typing import Any, Mapping

from .lifecycle import LoadSpec


ADAPTER_VERSION = "onnx-cpu-lifecycle/v1"


class OnnxLifecycleAdapter:
    """Own one ORT session; close releases references, not an RSS guarantee."""

    def __init__(self, spec: LoadSpec) -> None:
        import onnxruntime as ort

        if spec.adapter_version != ADAPTER_VERSION or spec.runtime_version != ort.__version__:
            raise ValueError("declared lifecycle adapter/runtime version does not match")
        providers = [(provider.name, dict(provider.options)) for provider in spec.providers]
        if providers != [("CPUExecutionProvider", {})]:
            raise ValueError("the lifecycle CPU adapter requires the explicit CPU provider")
        options = ort.SessionOptions()
        for name, value in spec.session_options:
            setattr(options, name, value)
        self._session = ort.InferenceSession(str(spec.artifact_path), sess_options=options, providers=providers)
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("observed execution provider differs from the lifecycle CPU profile")

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("lifecycle adapter is closed")
        outputs = self._session.run(None, dict(inputs))
        return {spec.name: value.tolist() for spec, value in zip(self._session.get_outputs(), outputs)}

    def close(self) -> None:
        self._session = None
