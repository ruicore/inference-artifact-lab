---
audience: codex
document_role: repository_document_index
authority: routing
---

# Inference Artifact Lab document index

Read documents in this order:

1. [Product contract](product-contract.md)
2. [Product phases](phases/README.md)
3. [Active Phase 1](phases/phase-1-model-release-gate/README.md)
4. Phase 1 [requirements](phases/phase-1-model-release-gate/requirements.md)
5. Phase 1 [acceptance contract](phases/phase-1-model-release-gate/acceptance.md)
6. Phase 1 [delivery plan](phases/phase-1-model-release-gate/phase-plan.md)
7. Stage plan and validation records
8. [Continuation guide](continuation-guide.md)

The catalog owns portfolio direction and clean-room provenance. This repository
owns product behavior, implementation contracts, and project evidence. Do not
copy private catalog context into this repository.

Planning, implementation, validation, public push, and release are separate facts.
Missing evidence remains `not_run`, `blocked`, or `not_verified`; it is never
inferred from a passing unit test.
