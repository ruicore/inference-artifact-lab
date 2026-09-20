---
audience: codex
document_role: stage_plan
phase_id: PH1
stage_id: PH1-ST01
status: complete
---

# Stage 01 - Identity and contract gate

Implement the manifest and report domain, file identity and SHA-256 checks, and
artifact/manifest contract validation. Negative fixtures must prove that missing,
modified, malformed, and mismatched inputs cannot pass.

Exit: AC-01 through AC-04 pass with machine-readable reasons. The standard-library
core and negative fixtures are implemented and tested locally.
