---
audience: codex
document_role: phase_plan
phase_id: PH1
authority: delivery
status: cpu_complete_locally
---

# Phase 1 delivery plan

| Stage | Deliverable | Exit evidence |
|---|---|---|
| 00 | Freeze contract, fixture, model source, and environment record | Requirements and acceptance mapping approved |
| 01 | Manifest, artifact identity, integrity, and contract checks | AC-01..04 pass with negative fixtures |
| 02 | Reference/target runtime adapters and comparison | AC-05..06 pass with tolerance and invalid-output fixtures |
| 03 | Environment probe and bounded benchmark | AC-07..09 pass for declared CPU profile; optional TensorRT preview evidence separate |
| 04 | JSON report, human rendering, clean reproduction, package review | CPU AC-01..08, AC-10..11 pass; AC-09 required for declared performance; public-candidate checks |

Stages must preserve the product contract. A new runtime, model, or library does
not expand Phase 1 without a requirement and acceptance change.

The first Phase 1 exit is CPU-only. TensorRT preview limitations do not block it
and remain visible in separate reports and acceptance rows.
The declared CPU local exit was observed on 2026-09-24, including reproduction
from an independent clean clone of commit `d8936237db3321fad79c75627ae364668ff65e58`.
This does not authorize public push, package release, or TensorRT claims.
