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
| AC-01 | Complete source and artifact identity | Machine-readable pinned source URL/weight digest verified at export, plus version, artifact digest, size, and build inputs |
| AC-02 | Missing or modified artifact | Controlled deletion/byte change is rejected with stable reason |
| AC-03 | Valid manifest | Canonical versioned manifest parses and is identified |
| AC-04 | Contract mismatch | Name, dtype, shape, profile, or format mismatch is rejected |
| AC-05 | Runtime equivalence | Pinned fixture and reference-output digests, same input to target adapter, target output, tolerance decision |
| AC-06 | Invalid runtime output | Wrong shape, non-finite, or malformed output is rejected |
| AC-07 | Declared compatible environment | Fingerprint, load, and execution evidence |
| AC-08 | Incompatible environment | Observed mismatch fails before loading fixture or adapter, with actionable reason |
| AC-09 | Reproducible benchmark | Workload, warm-up, repeated samples, resource observation |
| AC-10 | Complete report | JSON schema, human rendering, limitations, no secrets |
| AC-11 | Clean-environment reproduction | Fresh setup and matching gate decision |

## Exit rule

Phase 1 passes only when AC-01 through AC-08 and AC-10 pass, and AC-11 passes for
the first release scope: `onnx-cpu-windows-py311`, the declared ONNX Runtime CPU
environment. AC-11 requires a fresh checkout of a committed revision, a fresh
Python environment, public/generated inputs, and matching gate decisions.
Copying an uncommitted working tree is useful preliminary evidence but does not
close this case. AC-09 is required for a performance claim; the current CPU claim
is limited to the recorded workload and Python allocation peak. Any unavailable
required CPU evidence is `blocked` or `not_verified`, never inferred as pass.

TensorRT is an independent optional preview. Its AC-05 provenance, AC-09 GPU
memory, and AC-11 exact-engine gaps remain separately `not_verified` / `blocked`
and do not enter the CPU-only exit decision. For a future claimed TensorRT
platform, AC-11 verifies the pinned engine and gate decision; it does not require
a rebuild to produce identical bytes. No universal TensorRT platform is required,
and a CPU pass cannot satisfy any missing TensorRT evidence.

Numerical equivalence is bounded by the declared fixture and tolerances. A local
run does not prove compatibility with undeclared hardware, drivers, or runtimes.
