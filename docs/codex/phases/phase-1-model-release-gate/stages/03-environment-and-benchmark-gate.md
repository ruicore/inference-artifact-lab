---
audience: codex
document_role: stage_plan
phase_id: PH1
stage_id: PH1-ST03
status: blocked_on_environment
---

# Stage 03 - Environment and benchmark gate

Probe declared Python, OS, accelerator, driver, CUDA, TensorRT, and package
requirements. Add bounded warm-up, latency, throughput, and peak-memory
measurements for a declared workload.

Exit: AC-07 through AC-09 pass for the declared matrix; other environments remain
`not_verified`. The CPU ONNX profile has local evidence. The TensorRT GPU profile
is blocked until the pinned public container is run with GPU access.
