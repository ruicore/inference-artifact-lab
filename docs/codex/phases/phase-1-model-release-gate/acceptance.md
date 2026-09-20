---
audience: codex
document_role: phase_acceptance_contract
phase_id: PH1
authority: normative
status: active
case_default_status: not_run
---

# Phase 1 acceptance contract

Every case is `not_run` until implementation exists and all required observations
are collected. A unit test without the required evidence does not pass a case.

| ID | Case | Required observation |
|---|---|---|
| AC-01 | Complete source and artifact identity | Pinned source, version, digest, size, and build inputs |
| AC-02 | Missing or modified artifact | Controlled deletion/byte change is rejected with stable reason |
| AC-03 | Valid manifest | Canonical versioned manifest parses and is identified |
| AC-04 | Contract mismatch | Name, dtype, shape, profile, or format mismatch is rejected |
| AC-05 | Runtime equivalence | Fixed fixture, reference output, target output, tolerance decision |
| AC-06 | Invalid runtime output | Wrong shape, non-finite, or malformed output is rejected |
| AC-07 | Declared compatible environment | Fingerprint, load, and execution evidence |
| AC-08 | Incompatible environment | Fail-fast result with actionable reason |
| AC-09 | Reproducible benchmark | Workload, warm-up, repeated samples, resource observation |
| AC-10 | Complete report | JSON schema, human rendering, limitations, no secrets |
| AC-11 | Clean-environment reproduction | Fresh setup and matching gate decision |

## Exit rule

Phase 1 passes only when AC-01 through AC-08 and AC-10 pass, and AC-11 passes for
each environment the release claims to support. For compiled TensorRT artifacts,
AC-11 verifies the pinned engine and gate decision; it does not require a rebuild
to produce identical bytes. AC-09 is required for a performance claim. Any
unavailable required environment is `blocked`.

Numerical equivalence is bounded by the declared fixture and tolerances. A local
run does not prove compatibility with undeclared hardware, drivers, or runtimes.
