---
audience: codex
document_role: phase_plan
phase_id: PH1
authority: delivery
status: active
---

# Phase 1 delivery plan

| Stage | Deliverable | Exit evidence |
|---|---|---|
| 00 | Freeze contract, fixture, model source, and environment record | Requirements and acceptance mapping approved |
| 01 | Manifest, artifact identity, integrity, and contract checks | AC-01..04 pass with negative fixtures |
| 02 | Reference/target runtime adapters and comparison | AC-05..06 pass with tolerance and invalid-output fixtures |
| 03 | Environment probe and bounded benchmark | AC-07..09 pass for declared matrix; TensorRT evidence recorded |
| 04 | JSON report, human rendering, clean reproduction, package review | AC-01..11 regression and public-candidate checks |

Stages must preserve the product contract. A new runtime, model, or library does
not expand Phase 1 without a requirement and acceptance change.
