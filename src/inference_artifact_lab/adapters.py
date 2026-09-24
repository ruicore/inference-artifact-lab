"""Optional runtime adapters used by the Phase 1 evidence harness."""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Any, Mapping


class RuntimeUnavailable(RuntimeError):
    """Raised when a declared runtime dependency is not installed."""


class OnnxRuntimeAdapter:
    """Small ONNX Runtime adapter with no dependency on the gate domain."""

    def __init__(self, artifact: str | Path, *, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeUnavailable("install the runtime-cpu extra for ONNX Runtime") from exc
        self._session = ort.InferenceSession(str(artifact), providers=providers)
        self._ort_version = ort.__version__
        self._requested_providers = tuple(providers) if providers is not None else None

    def contract(self) -> dict[str, list[dict[str, Any]]]:
        def describe(value: Any) -> dict[str, Any]:
            return {"name": value.name, "dtype": value.type, "shape": list(value.shape)}

        return {
            "artifact_format": "onnx",
            "inputs": [describe(value) for value in self._session.get_inputs()],
            "outputs": [describe(value) for value in self._session.get_outputs()],
        }

    def runtime_info(self) -> dict[str, Any]:
        """Return selected providers, including an explicit fallback provider."""

        return {
            "runtime": "onnxruntime",
            "version": self._ort_version,
            "requested_providers": list(self._requested_providers) if self._requested_providers is not None else None,
            "providers": list(self._session.get_providers()),
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        }

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        values = self._session.run(None, dict(inputs))
        return {
            spec["name"]: value.tolist() if hasattr(value, "tolist") else value
            for spec, value in zip(self.contract()["outputs"], values)
        }


class _CtypesCudaRuntime:
    """Minimal libcudart binding used when the optional cuda-python is absent."""

    binding_name = "ctypes-libcudart"

    class _MemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    cudaMemcpyKind = _MemcpyKind

    def __init__(self) -> None:
        last_error: OSError | None = None
        for name in ("libcudart.so", "libcudart.so.12", "cudart64_*.dll"):
            try:
                self._lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeUnavailable("TensorRTAdapter requires libcudart or cuda-python") from last_error
        self._configure()

    def _configure(self) -> None:
        ptr, size = ctypes.c_void_p, ctypes.c_size_t
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ptr)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ptr]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ptr]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ptr), size]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ptr]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [ptr, ptr, size, ctypes.c_int, ptr]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int

    def cudaStreamCreate(self) -> tuple[int, int | None]:
        value = ctypes.c_void_p()
        return self._lib.cudaStreamCreate(ctypes.byref(value)), value.value

    def cudaStreamDestroy(self, stream: int | None) -> int:
        return self._lib.cudaStreamDestroy(ctypes.c_void_p(stream))

    def cudaStreamSynchronize(self, stream: int | None) -> int:
        return self._lib.cudaStreamSynchronize(ctypes.c_void_p(stream))

    def cudaMalloc(self, size: int) -> tuple[int, int | None]:
        value = ctypes.c_void_p()
        return self._lib.cudaMalloc(ctypes.byref(value), ctypes.c_size_t(size)), value.value

    def cudaFree(self, device: int | None) -> int:
        return self._lib.cudaFree(ctypes.c_void_p(device))

    def cudaMemcpyAsync(self, destination: int, source: int, size: int, kind: int, stream: int | None) -> int:
        return self._lib.cudaMemcpyAsync(ctypes.c_void_p(destination), ctypes.c_void_p(source), ctypes.c_size_t(size), int(kind), ctypes.c_void_p(stream))


def _load_cuda_runtime() -> Any:
    try:
        from cuda import cudart  # type: ignore[import-not-found]
    except ImportError:
        return _CtypesCudaRuntime()
    # Kept as evidence in runtime_info; provider selection is never implicit.
    try:
        cudart.binding_name = "cuda-python"  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return cudart


def _status_is_success(status: Any) -> bool:
    value = getattr(status, "value", status)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 0
    name = str(getattr(status, "name", status)).lower()
    return name in {"success", "cudasuccess", "cuda_success"}


def _check_cuda(result: Any, operation: str) -> Any:
    """Check cuda-python/libcudart status and return an optional payload."""

    if isinstance(result, tuple):
        status, payload = result[0], result[1] if len(result) > 1 else None
    else:
        status, payload = result, None
    if not _status_is_success(status):
        raise RuntimeError(f"{operation} failed with CUDA status {status!r}")
    return payload


class TensorRTAdapter:
    """TensorRT engine adapter with explicit CUDA ownership and validation."""

    def __init__(self, artifact: str | Path, *, trt_module: Any | None = None, cuda_runtime: Any | None = None, numpy_module: Any | None = None) -> None:
        try:
            trt = trt_module or __import__("tensorrt")
            np = numpy_module or __import__("numpy")
        except ImportError as exc:  # pragma: no cover - optional container dependency
            raise RuntimeUnavailable("TensorRTAdapter requires the pinned NVIDIA TensorRT container") from exc
        self._trt, self._np = trt, np
        self._cuda = cuda_runtime or _load_cuda_runtime()
        self._logger = trt.Logger(trt.Logger.ERROR)
        with Path(artifact).open("rb") as handle:
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT execution context creation failed")

    @staticmethod
    def _dtype_name(dtype: Any, *, np: Any, trt: Any) -> str:
        try:
            value = np.dtype(trt.nptype(dtype))
        except (TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError(f"unsupported TensorRT tensor dtype {dtype!r}") from exc
        names = {"float32": "float", "float16": "float16", "float64": "double", "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64", "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64", "bool": "bool"}
        try:
            return f"tensor({names[value.name]})"
        except KeyError as exc:
            raise RuntimeError(f"unsupported TensorRT tensor dtype {value}") from exc

    def _tensor_shape(self, name: str) -> list[int | None]:
        return [None if int(dimension) < 0 else int(dimension) for dimension in self._engine.get_tensor_shape(name)]

    def contract(self) -> dict[str, list[dict[str, Any]]]:
        inputs: list[dict[str, Any]] = []
        outputs: list[dict[str, Any]] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            value = {"name": name, "dtype": self._dtype_name(self._engine.get_tensor_dtype(name), np=self._np, trt=self._trt), "shape": self._tensor_shape(name)}
            target = inputs if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT else outputs
            target.append(value)
        return {"artifact_format": "tensorrt-engine", "inputs": inputs, "outputs": outputs, "optimization_profiles": self.optimization_profiles()}

    def optimization_profiles(self) -> list[dict[str, Any]]:
        """Return engine profile bounds without inventing a manifest schema."""
        count = int(getattr(self._engine, "num_optimization_profiles", 0) or 0)
        profiles: list[dict[str, Any]] = []
        for profile_index in range(count):
            tensors: dict[str, dict[str, list[int]]] = {}
            for index in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(index)
                if self._engine.get_tensor_mode(name) != self._trt.TensorIOMode.INPUT:
                    continue
                try:
                    minimum, optimum, maximum = self._engine.get_tensor_profile_shape(name, profile_index)
                except (AttributeError, RuntimeError):
                    continue
                tensors[name] = {"min": list(minimum), "opt": list(optimum), "max": list(maximum)}
            profiles.append({"index": profile_index, "inputs": tensors})
        return profiles

    def runtime_info(self) -> dict[str, Any]:
        return {"runtime": "tensorrt", "tensorrt_version": getattr(self._trt, "__version__", None), "cuda_binding": getattr(self._cuda, "binding_name", type(self._cuda).__name__), "optimization_profiles": self.optimization_profiles(), "python": platform.python_version(), "system": platform.system(), "machine": platform.machine()}

    def _names(self, mode: Any) -> list[str]:
        return [self._engine.get_tensor_name(index) for index in range(self._engine.num_io_tensors) if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == mode]

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        expected_inputs, expected_outputs = self._names(self._trt.TensorIOMode.INPUT), self._names(self._trt.TensorIOMode.OUTPUT)
        missing, extra = [name for name in expected_inputs if name not in inputs], [name for name in inputs if name not in expected_inputs]
        if missing or extra:
            raise ValueError(f"TensorRT inputs do not match engine (missing={missing}, extra={extra})")
        prepared: dict[str, Any] = {}
        for name in expected_inputs:
            value = self._np.asarray(inputs[name])
            expected_dtype = self._np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name)))
            if value.dtype != expected_dtype:
                raise TypeError(f"input {name!r} has dtype {value.dtype}, expected {expected_dtype}")
            if not value.flags.c_contiguous:
                raise ValueError(f"input {name!r} must be C-contiguous")
            engine_shape = self._engine.get_tensor_shape(name)
            if len(value.shape) != len(engine_shape) or any(int(want) >= 0 and int(want) != got for want, got in zip(engine_shape, value.shape)):
                raise ValueError(f"input {name!r} shape {tuple(value.shape)} does not satisfy engine shape {tuple(engine_shape)}")
            prepared[name] = value
            if self._context.set_input_shape(name, tuple(value.shape)) is False:
                raise ValueError(f"TensorRT rejected input shape for {name!r}: {tuple(value.shape)}")

        allocations: list[tuple[str, int, Any]] = []
        stream: Any = None
        synchronized, active_error = False, None
        try:
            stream = _check_cuda(self._cuda.cudaStreamCreate(), "cudaStreamCreate")
            if stream is None:
                raise RuntimeError("cudaStreamCreate returned a null stream")
            for name in expected_inputs + expected_outputs:
                if name in prepared:
                    value = prepared[name]
                else:
                    shape = tuple(int(dimension) for dimension in self._context.get_tensor_shape(name))
                    if any(dimension < 0 for dimension in shape):
                        raise ValueError(f"TensorRT output {name!r} has unresolved dynamic shape {shape}")
                    value = self._np.empty(shape, dtype=self._np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name))))
                if value.nbytes <= 0:
                    raise ValueError(f"TensorRT tensor {name!r} has an empty buffer")
                device = _check_cuda(self._cuda.cudaMalloc(value.nbytes), f"cudaMalloc({name})")
                if not device:
                    raise RuntimeError(f"cudaMalloc({name}) returned a null pointer")
                allocations.append((name, int(device), value))
                self._context.set_tensor_address(name, int(device))
                if name in prepared:
                    _check_cuda(self._cuda.cudaMemcpyAsync(int(device), int(value.ctypes.data), value.nbytes, self._cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream), f"cudaMemcpyAsync(HtoD:{name})")
            if self._context.execute_async_v3(stream_handle=stream) is False:
                raise RuntimeError("TensorRT execution failed")
            for name, device, value in allocations:
                if name not in expected_outputs:
                    continue
                _check_cuda(self._cuda.cudaMemcpyAsync(int(value.ctypes.data), device, value.nbytes, self._cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream), f"cudaMemcpyAsync(DtoH:{name})")
            _check_cuda(self._cuda.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            synchronized = True
            return {name: value.tolist() for name, _, value in allocations if name in expected_outputs}
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            cleanup_errors: list[str] = []
            if stream is not None and not synchronized:
                try:
                    _check_cuda(self._cuda.cudaStreamSynchronize(stream), "cudaStreamSynchronize(cleanup)")
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            for name, device, _ in reversed(allocations):
                try:
                    _check_cuda(self._cuda.cudaFree(device), f"cudaFree({name})")
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            if stream is not None:
                try:
                    _check_cuda(self._cuda.cudaStreamDestroy(stream), "cudaStreamDestroy")
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            if active_error is None and cleanup_errors:
                raise RuntimeError("TensorRT CUDA cleanup failed: " + "; ".join(cleanup_errors))
