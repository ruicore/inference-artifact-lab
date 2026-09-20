---
audience: codex
document_role: validation_record
phase_id: PH1
stage_id: PH1-ST04
status: local_pass
---

# Phase 1 validation record

Phase 1 is locally validated for the declared public SqueezeNet profile.

| Acceptance | Result | Evidence |
|---|---|---|
| AC-01..04 | pass | CPU and TensorRT manifests, artifact digests, contract checks |
| AC-05..06 | pass | PyTorch→ONNX CPU report and TensorRT comparison, max abs error `2.86102294921875e-06` |
| AC-07..08 | pass | CPU fingerprint and pinned TensorRT container/GPU fingerprint |
| AC-09 | pass | CPU benchmark and TensorRT `trtexec` benchmark, 200 warmups/100 measurements |
| AC-10 | pass | Versioned JSON reports with limitations and redacted paths |
| AC-11 | pass for declared setup | public dependency pins and container reproduction commands |

Primary evidence:

- `reports/phase-1/squeezenet11-torchvision-onnx-cpu.json`
- `reports/phase-1/squeezenet11-tensorrt.json`
- `examples/squeezenet11-torchvision.manifest.json`
- `examples/squeezenet11-tensorrt.manifest.json`

The claim is limited to the declared model, fixture, Windows CPU profile, and
the pinned TensorRT container on the observed RTX 5050 / driver fingerprint.
