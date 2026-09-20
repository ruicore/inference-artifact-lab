---
audience: codex
document_role: phase_fixture_record
phase_id: PH1
status: pinned
---

# Phase 1 model and fixture record

## Model

- Name: SqueezeNet 1.1
- Source: public torchvision `SqueezeNet1_1_Weights.IMAGENET1K_V1`
- Weights URL: `https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth`
- Weights SHA-256: `b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b`
- Exported ONNX SHA-256: `4ed9006c47bb1eb8521eed0f8b22aa49003a8013c8e4adf5db414f7c8e927534`
- Exported ONNX size: `4,958,275` bytes
- Input: `data`, `tensor(float)`, `[1, 3, 224, 224]`
- Output: `squeezenet0_flatten0_reshape0`, `tensor(float)`, `[1, 1000]`

The public weights are downloaded into the local cache and the exported ONNX
artifact is written to the ignored `artifacts/` directory. Neither binary is
committed to Git.

## Deterministic fixture

The fixture is generated locally with `torch.Generator().manual_seed(20260920)`
and `torch.rand`, shape `[1, 3, 224, 224]`, dtype `float32`, and values in
`[0.0, 1.0)`. The generated fixture and reference output remain ignored binary
evidence; their SHA-256 digests are recorded in the validation reports.

## Environment profiles

- `torch-reference-windows-py311`: Python 3.11.5, Windows AMD64, PyTorch
  2.14.0+cpu, torchvision 0.29.0+cpu.
- `onnx-cpu-windows-py311`: Python 3.11.5, Windows AMD64, ONNX Runtime 1.30.0,
  `CPUExecutionProvider`.
- `tensorrt-gpu`: TensorRT 10.8.0 in `nvcr.io/nvidia/tensorrt:25.02-py3`,
  observed on RTX 5050 / compute capability 12.0; evidence is scoped to this
  declared container and host fingerprint.

The reference smoke command uses a temporary public CPU environment and does not
add PyTorch to the gate package's default dependency set:

```text
uv run --with torch --with torchvision --with onnx --with onnxruntime \
  python scripts/run_torchvision_gate.py
```
