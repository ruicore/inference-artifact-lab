---
audience: codex
document_role: environment_record
phase_id: PH1
status: blocked_on_execution
---

# TensorRT environment record

The project does not use any TensorRT executable, engine, or library from
`private project directories` or another private project environment.

The reproducible target is the public NVIDIA container
`nvcr.io/nvidia/tensorrt:25.02-py3`, run with Docker's NVIDIA runtime and the
public ONNX artifact mounted read-only. NVIDIA documents this container workflow
as `docker run --gpus all ...`; the exact image digest must be recorded after the
pull completes.

Required evidence:

- image digest and TensorRT version;
- GPU UUID, compute capability, driver, CUDA, and TensorRT versions;
- engine build command and engine SHA-256;
- engine load and inference output compared with the PyTorch reference;
- benchmark report and limitations.

Until those observations are collected, the TensorRT profile is `blocked` and the
overall Phase 1 release candidate cannot claim TensorRT compatibility.
