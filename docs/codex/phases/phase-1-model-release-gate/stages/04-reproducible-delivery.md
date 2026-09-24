---
audience: codex
document_role: stage_plan
phase_id: PH1
stage_id: PH1-ST04
status: cpu_complete_locally
---

# Stage 04 - Reproducible delivery

Produce versioned JSON and human-readable reports, clean-environment instructions,
redaction and package checks, and a complete acceptance regression. This stage
establishes a local release candidate; publication remains a separate decision.

Exit: AC-01 through AC-08 and AC-10 through AC-11 pass for the declared CPU scope;
AC-09 must also pass for its bounded performance claim. Require a fresh committed
checkout, fresh Python environment, and generated JSON and human reports.
Working-tree isolation is preliminary evidence until committed-checkout
reproduction passes. TensorRT preview gaps remain recorded independently and do
not block this exit.
The committed CPU checkout reproduced successfully on 2026-09-24; see the
[validation record](../stage-04-validation.md). This closes the local CPU exit
only. Publication and optional TensorRT preview acceptance remain separate.
