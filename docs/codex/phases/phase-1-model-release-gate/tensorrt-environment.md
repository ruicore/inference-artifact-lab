---
audience: codex
document_role: environment_record
phase_id: PH1
status: optional_preview
---

# TensorRT environment record

This is independent optional preview evidence. No TensorRT platform or engine is
required for CPU-only Phase 1 acceptance. This recorded platform is one evaluated
sample; future GPU/container profiles can carry their own manifests and evidence.

The project does not use any TensorRT executable, engine, or library from
`private project directories` or another private project environment.

The reproducible target is the public NVIDIA container
`nvcr.io/nvidia/tensorrt:25.02-py3`, run with Docker's NVIDIA runtime and the
public ONNX artifact mounted read-only. The observed image digest is
`sha256:814325e2b8a653f354c30bbcf5ecc8d4c780cf878a88a320ae648fbfdd9dd82d`.

Required evidence:

- image digest and TensorRT version;
- GPU UUID, compute capability, driver, CUDA, and TensorRT versions;
- engine build command and engine SHA-256;
- engine load and inference output compared with the PyTorch reference;
- benchmark report and limitations.

The first build command is prepared at
`scripts/build_tensorrt_engine.ps1`. It mounts only this public project directory
and the generated public ONNX artifact into the container.

Those observations are recorded in
`reports/phase-1/squeezenet11-tensorrt.json`. Local load/contract/numerical checks
passed for the pinned image digest and observed GPU/driver fingerprint. The
portable report remains `not_verified` for runtime provenance and peak GPU memory;
exact-engine fresh-checkout reproduction remains `blocked`. It does not claim
general TensorRT or hardware compatibility, and a CPU pass changes none of these
preview statuses.
