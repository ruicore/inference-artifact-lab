---
audience: codex
document_role: phase_requirements
phase_id: PH1
authority: normative
status: active
---

# Phase 1 requirements

All requirements below belong to Phase 1. Every implementation change must map to
one requirement and an acceptance case.

| ID | Requirement | Stage |
|---|---|---|
| PH1-R01 | Pin public source URL and weight digest in the manifest; verify source bytes at export and record artifact, format, build inputs, size, and digest | 01 |
| PH1-R02 | Parse and version a complete artifact manifest | 01 |
| PH1-R03 | Reject artifact/manifest identity, shape, dtype, or profile mismatches | 01 |
| PH1-R04 | Pin fixture/reference-output bytes and compare reference and target runtime outputs under declared tolerances | 02 |
| PH1-R05 | Probe declared environment requirements and fail fast on incompatibility before adapter execution | 03 |
| PH1-R06 | Measure declared workload latency, throughput, warm-up, and peak memory | 03 |
| PH1-R07 | Produce versioned JSON and human-readable release reports | 04 |
| PH1-R08 | Reproduce verification in a clean declared environment | 04 |

## Baseline constraints

The first implementation uses one pinned public non-robotics PyTorch model, a
generated deterministic fixture, and ONNX as the required interchange artifact.
The reference path is torchvision/PyTorch CPU; the first acceptance target is
`onnx-cpu-windows-py311`, the declared ONNX Runtime CPU environment. TensorRT is
an independent optional preview, with platform-specific engine and environment
evidence. No TensorRT environment or exact engine is required to close the CPU
baseline. Exact revisions and environment fingerprints belong to evidence records,
not to the general product contract. PH1-R06 / AC-09 apply to the explicitly
declared CPU benchmark scope; Python allocation peaks are not native process or
GPU peak-memory measurements.

## Requirement change rule

Changing model or tool versions does not change the product. Changing the gate
dimensions, evidence rules, status vocabulary, or release semantics requires an
updated product contract and acceptance review.
