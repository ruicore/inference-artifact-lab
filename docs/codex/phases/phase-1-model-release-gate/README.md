---
audience: codex
document_role: phase_index
phase_id: PH1
status: implementation_in_progress
---

# Phase 1 - Model Release Gate

Phase 1's first acceptance baseline is the declared ONNX Runtime CPU profile.
TensorRT is an independent optional preview. Phase 1 is delivered through five
internal stages; the stages control delivery order and evidence collection, not
separate products.

- [Requirements](requirements.md)
- [Acceptance contract](acceptance.md)
- [Model and fixture record](fixture.md)
- [Phase plan](phase-plan.md)
- [Stage 00 - contract and fixture freeze](stages/00-contract-and-fixture-freeze.md)
- [Stage 01 - identity and contract gate](stages/01-identity-and-contract-gate.md)
- [Stage 02 - runtime correctness](stages/02-runtime-correctness.md)
- [Stage 03 - environment and benchmark gate](stages/03-environment-and-benchmark-gate.md)
- [Stage 04 - reproducible delivery](stages/04-reproducible-delivery.md)
- [Validation record](stage-04-validation.md)

Current lifecycle: CPU acceptance candidate, awaiting committed-checkout AC-11.
TensorRT provenance/performance remain `not_verified` and clean-checkout
reproduction remains `blocked`; these preview gaps do not block CPU acceptance.
See [publication review](../../../publication-review.md).
