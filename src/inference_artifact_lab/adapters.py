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


class TensorRTAdapter:
    """TensorRT engine adapter used inside the pinned NVIDIA container."""

    def __init__(self, artifact: str | Path) -> None:
        try:
            import tensorrt as trt
            from cuda import cudart
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on container runtime
            raise RuntimeUnavailable("TensorRTAdapter requires the pinned NVIDIA TensorRT container") from exc
        self._trt = trt
        self._cuda = cudart
        self._np = np
        logger = trt.Logger(trt.Logger.ERROR)
        with Path(artifact).open("rb") as handle:
            self._runtime = trt.Runtime(logger)
            self._engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self._context = self._engine.create_execution_context()

    @staticmethod
    def _dtype_name(dtype: Any) -> str:
        return f"tensor({str(dtype).split('.')[-1]})"

    def contract(self) -> dict[str, list[dict[str, Any]]]:
        inputs: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = [None if dimension < 0 else dimension for dimension in self._engine.get_tensor_shape(name)]
            value = {"name": name, "dtype": self._dtype_name(self._engine.get_tensor_dtype(name)), "shape": shape}
            target = inputs if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT else outputs
            target.append(value)
        return {"inputs": inputs, "outputs": outputs}

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        allocations: list[tuple[Any, Any]] = []
        stream = self._cuda.cudaStreamCreate()[1]
        try:
            for name in [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]:
                if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT:
                    value = self._np.asarray(inputs[name])
                    self._context.set_input_shape(name, tuple(value.shape))
                else:
                    shape = tuple(self._context.get_tensor_shape(name))
                    value = self._np.empty(shape, dtype=self._np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name))))
                device = self._cuda.cudaMalloc(value.nbytes)[1]
                self._context.set_tensor_address(name, int(device))
                allocations.append((device, value))
                if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT:
                    self._cuda.cudaMemcpyAsync(device, value.ctypes.data, value.nbytes, self._cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            if not self._context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("TensorRT execution failed")
            result: dict[str, Any] = {}
            for device, value in allocations:
                name = next(self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors) if self._context.get_tensor_address(self._engine.get_tensor_name(i)) == int(device))
                if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.OUTPUT:
                    self._cuda.cudaMemcpyAsync(value.ctypes.data, device, value.nbytes, self._cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
                    result[name] = value.tolist()
            self._cuda.cudaStreamSynchronize(stream)
            return result
        finally:
            for device, _ in allocations:
                self._cuda.cudaFree(device)
            self._cuda.cudaStreamDestroy(stream)
