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

The current smoke report deliberately records two limitations:

- TensorRT engine compatibility is not verified in the current clean project
  environment.

Stage 02 is locally passed for the PyTorch-to-ONNX CPU path. These limitations must
remain visible in any report until the TensorRT profile is validated.
