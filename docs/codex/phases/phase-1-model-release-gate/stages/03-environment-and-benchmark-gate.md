---
audience: codex
document_role: stage_plan
phase_id: PH1
stage_id: PH1-ST03
status: complete
---

# Stage 03 - Environment and benchmark gate

Probe declared Python, OS, accelerator, driver, CUDA, TensorRT, and package
requirements. Add bounded warm-up, latency, throughput, and peak-memory
measurements for a declared workload.

Exit: AC-07 through AC-09 pass for the declared matrix; other environments remain
`not_verified`. Evidence is recorded for Windows CPU ONNX Runtime and the pinned
TensorRT 10.8 container on the observed RTX 5050 host.
