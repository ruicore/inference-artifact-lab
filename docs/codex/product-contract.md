---
audience: codex
document_role: product_contract
authority: normative
status: active
---

# Model Release Gate product contract

## Product boundary

Inference Artifact Lab validates whether a declared machine-learning deployment
artifact is complete, contract-compatible, behaviorally equivalent to a reference
path, and usable in a declared environment. It produces reviewable evidence for a
release decision.

## Product-owned entities

- `ModelSource`: pinned public source identity and logical model version.
- `ArtifactManifest`: versioned artifact, contract, runtime, and tolerance metadata.
- `ArtifactRecord`: file identity, digest, format, and lineage.
- `ValidationFixture`: deterministic inputs and expected reference observations.
- `EnvironmentFingerprint`: declared and observed runtime capabilities.
- `GateCheck`: one named check with evidence, status, and limitations.
- `ReleaseReport`: versioned aggregate decision and evidence references.

## Product invariants

- A file-exists check is never sufficient for release.
- The manifest is authoritative; directory scanning and implicit defaults cannot
  select an artifact or contract.
- Every release claim is scoped to the model, fixture, artifact, and environment
  recorded in the report.
- Missing evidence produces `blocked` or `not_verified`, never `pass`.
- A reference/runtime comparison must use declared tolerances and a reproducible
  fixture.
- Environment compatibility is explicit and fail-fast for declared requirements.
- A compiled TensorRT artifact is identified by its manifest digest, engine digest,
  build inputs, container digest, and observed hardware fingerprint. Rebuilding it
  may produce different bytes because TensorRT tactic timing is hardware- and run-
  dependent; byte-for-byte rebuild identity is outside the Phase 1 claim.

## Phase 1 boundary

Phase 1 includes integrity, manifest contract validation, reference/runtime
correctness, declared environment compatibility, bounded resource measurements,
and reproducible JSON reports. It does not include model training, general serving,
model leaderboards, registry upload, production deployment, or automatic release.

## Compatibility and status

Report and manifest formats are versioned. `pass`, `fail`, `blocked`, and
`not_verified` have distinct meanings. A future incompatible contract requires a
new version and an explicit migration or rejection path.
