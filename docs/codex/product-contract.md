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
- A public source weight digest is declared in the manifest and verified by the
  exporter; report source-pin evidence is a declaration, not a second download.
- Every release claim is scoped to the model, fixture, artifact, and environment
  recorded in the report.
- Missing evidence produces `blocked` or `not_verified`, never `pass`.
- A reference/runtime comparison must use declared tolerances and a reproducible
  fixture. The manifest pins SHA-256 digests of the serialized fixture and
  reference output; missing or mismatched bytes cannot pass the release gate.
- Caller-supplied runtime observations can be compared but cannot independently
  prove their input provenance. They remain `not_verified`; a bound runner must
  decode pinned bytes and execute the adapter to make a runtime pass claim.
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

The first acceptance and release scope is `onnx-cpu-windows-py311`: the pinned
public model and generated fixture on the declared ONNX Runtime CPU environment.
TensorRT is an independent optional preview because compiled engines depend on
their GPU, driver, runtime, and build platform. Preview evidence is evaluated per
declared platform; no single TensorRT platform is mandatory for CPU acceptance.
Missing or failed TensorRT evidence does not block the CPU-only Phase 1 exit.
A CPU pass never changes a TensorRT check to pass or claims GPU compatibility.
Future TensorRT support requires its own environment-specific acceptance.

The example manifests identify their profile and acceptance role in `runtime`;
these values participate in the canonical manifest digest. Reports carry the
manifest's declared `runtime` and `environment` in an additive `scope` object,
including failed and blocked decisions. This is declaration metadata, not proof
that the environment was observed. Environment checks retain the actual evidence.

## Compatibility and status

Report and manifest formats are versioned. `pass`, `fail`, `blocked`, and
`not_verified` have distinct meanings. A future incompatible contract requires a
new version and an explicit migration or rejection path.
