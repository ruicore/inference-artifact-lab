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
| PH1-R01 | Record source, artifact, format, build inputs, size, and digest | 01 |
| PH1-R02 | Parse and version a complete artifact manifest | 01 |
| PH1-R03 | Reject artifact/manifest identity, shape, dtype, or profile mismatches | 01 |
| PH1-R04 | Compare reference and target runtime outputs under declared tolerances | 02 |
| PH1-R05 | Verify declared environment requirements and fail fast on incompatibility | 03 |
| PH1-R06 | Measure declared workload latency, throughput, warm-up, and peak memory | 03 |
| PH1-R07 | Produce versioned JSON and human-readable release reports | 04 |
| PH1-R08 | Reproduce verification in a clean declared environment | 04 |

## Baseline constraints

The first implementation uses one pinned public non-robotics model, a generated
deterministic fixture, ONNX as interchange artifact, and TensorRT as compiled
deployment artifact. The exact model revision and environment fingerprint belong
to the first evidence record, not to the general product contract.

## Requirement change rule

Changing model or tool versions does not change the product. Changing the gate
dimensions, evidence rules, status vocabulary, or release semantics requires an
updated product contract and acceptance review.
