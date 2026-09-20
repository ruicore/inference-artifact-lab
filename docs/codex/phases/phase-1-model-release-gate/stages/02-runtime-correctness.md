---
audience: codex
document_role: stage_plan
phase_id: PH1
stage_id: PH1-ST02
status: implementation_in_progress
---

# Stage 02 - Runtime correctness

Add a torchvision/PyTorch CPU reference adapter and target runtime adapters for
ONNX and TensorRT, then compare deterministic fixture outputs under manifest tolerances. Invalid,
non-finite, and wrong-shape outputs must remain failures.

Exit: AC-05 and AC-06 pass; no general model-quality claim is made.
