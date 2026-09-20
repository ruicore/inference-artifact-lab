---
audience: codex
document_role: phase_fixture_record
phase_id: PH1
status: pinned
---

# Phase 1 model and fixture record

## Model

- Name: SqueezeNet 1.1
- Source: ONNX Model Zoo public artifact
- Revision: `squeezenet1.1-7.onnx` at pinned commit `b1eeaa1ac722dcc1cd1a8284bde34393dab61c3d`
- SHA-256: `1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088`
- Size: `4,956,208` bytes
- Input: `data`, `tensor(float)`, `[1, 3, 224, 224]`
- Output: `squeezenet0_flatten0_reshape0`, `tensor(float)`, `[1, 1000]`

The source artifact is downloaded into the ignored `artifacts/` directory by the
fetch script. The binary model is not committed to Git.

## Deterministic fixture

The first fixture is generated locally with NumPy's PCG64 generator using seed
`20260920`, shape `[1, 3, 224, 224]`, dtype `float32`, and values uniformly sampled
from `[-1.0, 1.0)`. The fixture digest and generated reference output are recorded
in the validation report rather than committed as private or opaque evidence.

## Environment profiles

- `onnx-cpu-windows-py311`: Python 3.11.5, Windows AMD64, ONNX Runtime 1.30.0,
  `CPUExecutionProvider`.
- `tensorrt-gpu`: declared target profile; remains `not_verified` until a clean
  public-project environment supplies TensorRT and a compiled engine.
