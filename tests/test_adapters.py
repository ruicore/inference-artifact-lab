from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from inference_artifact_lab.adapters import TensorRTAdapter


class _FakeCuda:
    binding_name = "test-cuda"

    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self, *, fail_stream: bool = False) -> None:
        self.fail_stream = fail_stream
        self.buffers: dict[int, ctypes.Array[ctypes.c_char]] = {}
        self.pending: list[tuple[int, int, int]] = []
        self.sync_calls = 0
        self.destroy_calls = 0

    def cudaStreamCreate(self):
        return (9, None) if self.fail_stream else (0, 123)

    def cudaStreamDestroy(self, stream):
        self.destroy_calls += 1
        return 0

    def cudaStreamSynchronize(self, stream):
        self.sync_calls += 1
        for destination, source, size in self.pending:
            ctypes.memmove(destination, source, size)
        self.pending.clear()
        return 0

    def cudaMalloc(self, size):
        storage = ctypes.create_string_buffer(size)
        address = ctypes.addressof(storage)
        self.buffers[address] = storage
        return 0, address

    def cudaFree(self, device):
        self.buffers.pop(device, None)
        return 0

    def cudaMemcpyAsync(self, destination, source, size, kind, stream):
        if kind == self.cudaMemcpyKind.cudaMemcpyHostToDevice:
            ctypes.memmove(destination, source, size)
        else:
            self.pending.append((destination, source, size))
        return 0


class _FakeContext:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.addresses: dict[str, int] = {}
        self.input_shape = None

    def set_input_shape(self, name, shape):
        self.input_shape = tuple(shape)
        return True

    def get_tensor_shape(self, name):
        return (self.input_shape[0], 1)

    def set_tensor_address(self, name, address):
        self.addresses[name] = address

    def execute_async_v3(self, stream_handle):
        result = np.full((self.input_shape[0], 1), 42.0, dtype=np.float32)
        ctypes.memmove(self.addresses["out"], result.ctypes.data, result.nbytes)
        return True


class _FakeEngine:
    num_io_tensors = 2
    num_optimization_profiles = 1

    def __init__(self):
        self.context = _FakeContext(self)

    def get_tensor_name(self, index):
        return ("data", "out")[index]

    def get_tensor_mode(self, name):
        return _FakeTrt.TensorIOMode.INPUT if name == "data" else _FakeTrt.TensorIOMode.OUTPUT

    def get_tensor_dtype(self, name):
        return _FakeTrt.FLOAT

    def get_tensor_shape(self, name):
        return (-1, 3) if name == "data" else (-1, 1)

    def get_tensor_profile_shape(self, name, profile):
        return ((1, 3), (2, 3), (4, 3))

    def create_execution_context(self):
        return self.context


class _FakeTrt:
    __version__ = "10.8.test"
    FLOAT = object()

    class Logger:
        ERROR = 0

        def __init__(self, severity):
            self.severity = severity

    class TensorIOMode:
        INPUT = "input"
        OUTPUT = "output"

    @staticmethod
    def nptype(dtype):
        assert dtype is _FakeTrt.FLOAT
        return np.float32

    class Runtime:
        def __init__(self, logger):
            self.logger = logger

        def deserialize_cuda_engine(self, payload):
            assert payload == b"engine"
            return _FakeEngine()


def _adapter(tmp_path, cuda=None):
    artifact = tmp_path / "engine.plan"
    artifact.write_bytes(b"engine")
    return TensorRTAdapter(artifact, trt_module=_FakeTrt, cuda_runtime=cuda or _FakeCuda(), numpy_module=np)


def test_tensorrt_contract_and_profile_are_observable(tmp_path):
    adapter = _adapter(tmp_path)
    assert adapter.contract() == {
        "artifact_format": "tensorrt-engine",
        "inputs": [{"name": "data", "dtype": "tensor(float)", "shape": [None, 3]}],
        "outputs": [{"name": "out", "dtype": "tensor(float)", "shape": [None, 1]}],
    }
    assert adapter.optimization_profiles()[0]["inputs"]["data"]["max"] == [4, 3]
    assert adapter.runtime_info()["cuda_binding"] == "test-cuda"


def test_tensorrt_synchronizes_before_reading_output(tmp_path):
    cuda = _FakeCuda()
    adapter = _adapter(tmp_path, cuda)
    result = adapter.run({"data": np.ones((2, 3), dtype=np.float32)})
    assert result == {"out": [[42.0], [42.0]]}
    assert cuda.sync_calls == 1
    assert cuda.destroy_calls == 1


def test_tensorrt_rejects_dtype_and_shape_mismatch(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(TypeError, match="dtype"):
        adapter.run({"data": np.ones((2, 3), dtype=np.float64)})
    with pytest.raises(ValueError, match="shape"):
        adapter.run({"data": np.ones((2, 4), dtype=np.float32)})


def test_tensorrt_reports_cuda_errors(tmp_path):
    adapter = _adapter(tmp_path, _FakeCuda(fail_stream=True))
    with pytest.raises(RuntimeError, match="cudaStreamCreate"):
        adapter.run({"data": np.ones((2, 3), dtype=np.float32)})
