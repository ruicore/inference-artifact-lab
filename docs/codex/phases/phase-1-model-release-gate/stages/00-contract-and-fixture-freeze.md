---
audience: codex
document_role: stage_plan
phase_id: PH1
stage_id: PH1-ST00
status: complete
---

# Stage 00 - Contract and fixture freeze

Freeze the manifest/report schemas, status vocabulary, acceptance cases, one public
model revision, deterministic fixture, required ONNX and optional TensorRT artifact paths, and declared
environment record. Implementation may proceed only from this contract.

The exact model and environment values are recorded in [fixture.md](../fixture.md)
and the example manifests. The first acceptance profile is ONNX Runtime CPU.
The optional TensorRT preview is scoped to its pinned public container and
recorded host fingerprint; other platform profiles can be evaluated separately.
