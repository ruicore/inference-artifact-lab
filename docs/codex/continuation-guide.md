---
audience: codex
document_role: continuation_guide
authority: routing
status: active
---

# Continuing development

This repository owns the reusable release-gate capability for declared machine-
learning inference artifacts. It records whether a specific artifact, fixture,
runtime, and environment have enough evidence for a scoped release decision.

## Before adding a capability

Answer these questions in order:

1. Does the capability validate artifact identity, contract, runtime behavior,
   environment compatibility, bounded resources, reproducibility, or release
   evidence? If not, it probably belongs in another repository.
2. Can it be designed from public specifications, public models, or generated
   fixtures? If not, stop and replace the evidence source before coding.
3. Does it change a product invariant, decision status, report field, manifest
   meaning, or release claim? If yes, update the product contract and obtain a
   product decision before implementation.
4. Is it a new runtime, model, or library within the existing Phase 1 boundary?
   Add a requirement and acceptance case only when the capability is part of the
   release-gate baseline; otherwise record it as a future phase or an optional
   adapter.

## Change sequence

For an accepted capability, use this order:

1. Update the relevant requirement, contract, manifest, or acceptance case.
2. Implement the smallest independent API or adapter and add meaningful tests,
   including malformed input and mismatch cases.
3. Collect runtime evidence for the declared model, fixture, and environment.
   Keep `blocked`, `fail`, and `not_verified` distinct from `pass`.
4. Update the JSON report schema, renderer, fixture record, and validation record
   when the evidence shape changes.
5. Run the repository tests, clean reproduction, report schema validation, and
   public disclosure review. Recheck all reachable history before publication.
6. Update the phase status and publication review so the remaining boundary is
   explicit. Commit, push, and publish a package only after the corresponding
   project-level authorization.

## Current implementation map

- Core entities and statuses: `src/inference_artifact_lab/models.py`
- Artifact and manifest checks: `src/inference_artifact_lab/gate.py`
- Runtime correctness and benchmark evidence: `src/inference_artifact_lab/runtime.py`
- Runtime adapters: `src/inference_artifact_lab/adapters.py`
- Release composition: `src/inference_artifact_lab/release.py`
- Package CLI: `src/inference_artifact_lab/__main__.py`
- Report schema and rendering: `schemas/` and `scripts/render_report.py`
- Public CPU reproduction: `scripts/clean_reproduction.ps1`
- Pinned TensorRT build and execution: `scripts/build_tensorrt_engine.ps1`,
  `scripts/benchmark_tensorrt_engine.ps1`, and
  `scripts/run_tensorrt_in_container.py`

The package CLI currently launches the ONNX CPU adapter. TensorRT validation is
performed through the pinned container scripts because the compiled engine claim
is scoped to its container and observed hardware fingerprint.

## Current release boundary

Phase 1's first acceptance and release scope is the declared ONNX Runtime CPU
profile. Committed-checkout AC-11 passed locally on 2026-09-24; future changes
to the acceptance path must be reproduced from their own committed revision.
TensorRT is an independent optional preview; its provenance, GPU peak-memory,
and exact-engine reproduction gaps do not block CPU acceptance and are never
satisfied by a CPU pass. Future TensorRT profiles remain platform-specific.
Keep both CPU acceptance and preview gaps explicit in `docs/publication-review.md`.
These paths do not establish universal hardware compatibility, production
deployment safety, or byte-identical TensorRT rebuilds.

The package release is a manually authorized distribution of the validation
library. It is separate from automatic model registry upload or production
deployment, both of which remain outside Phase 1.
