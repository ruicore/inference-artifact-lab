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
- Output: `output`, `tensor(float)`, `[1, 1000]`

The public weights are downloaded into the local cache and the exported ONNX
artifact is written to the ignored `artifacts/` directory. Neither binary is
committed to Git. Both Phase 1 manifests pin the public weight URL and digest;
the exporter and reference runner verify the cached bytes against that pin.

## Deterministic fixture

The fixture is generated locally with `torch.Generator().manual_seed(20260920)`
and `torch.rand`, shape `[1, 3, 224, 224]`, dtype `float32`, and values in
`[0.0, 1.0)`. The generated fixture and reference output remain ignored binary
evidence. Their serialized `.npy` SHA-256 digests are pinned in the CPU and
TensorRT manifests and checked before target execution:

- Fixture: `3161e1f639bd8c78b14164b87c3fb0be8aeb821aa715fcc0cab2d93353242d89`
- Reference output: `9b9e35c2d6a1191685178087ce91bd352fadcebc97af24cb7b88ecf75fec0f06`

## Environment profiles

The first mandatory acceptance profile is `onnx-cpu-windows-py311`, compared
against `torch-reference-windows-py311`. `tensorrt-gpu` is an independent optional
preview, not a required CPU release environment. Its recorded host is one sample
platform, not a restriction on which future TensorRT platforms can be evaluated.

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
