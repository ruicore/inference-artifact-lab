---
audience: codex
document_role: validation_record
phase_id: PH1
stage_id: PH1-ST02
status: local_pass
---

# Stage 02 validation record

The public torchvision SqueezeNet 1.1 reference loads on CPU and is compared with
the exported ONNX artifact through ONNX Runtime. Integrity, contract, environment,
independent runtime equivalence, deterministic fixture execution, and bounded
benchmark checks pass locally. The maximum absolute error observed was
`2.384185791015625e-06`.

The optional TensorRT preview is recorded separately by the pinned-container report
`reports/phase-1/squeezenet11-tensorrt.json`; this record remains scoped to the
PyTorch-to-ONNX CPU path.

Stage 02 passes for the CPU baseline. TensorRT's local numerical comparison is
recorded, but portable AC-05 provenance remains `not_verified`; see the current
[per-profile validation record](stage-04-validation.md).
