# Phase 1 Acceptance Contract

This is a concise human-readable summary. The normative acceptance authority is
[docs/codex/phases/phase-1-model-release-gate/acceptance.md](codex/phases/phase-1-model-release-gate/acceptance.md).

Every case starts as `not_run`. A case can become `passed` only when the required
observation and evidence are present in the report.

| ID | Acceptance case | Required evidence |
|---|---|---|
| AC-01 | Complete source and artifact identity | Pinned source, version, digest, size, and build inputs |
| AC-02 | Missing or modified artifact is rejected | Controlled deletion or byte modification and stable failure reason |
| AC-03 | Valid manifest is accepted | Versioned manifest parses and canonical digest is recorded |
| AC-04 | Manifest/artifact contract mismatch is rejected | Name, dtype, shape, or profile mismatch fixture |
| AC-05 | Reference/runtime outputs are equivalent | Fixed fixture, output summary, tolerance decision, and runtime versions |
| AC-06 | Invalid runtime output is rejected | Malformed, NaN/Inf, or wrong-shape output fixture |
| AC-07 | Declared compatible environment loads and runs | Environment fingerprint and successful execution evidence |
| AC-08 | Incompatible environment fails fast | Controlled version or capability mismatch and actionable reason |
| AC-09 | Benchmark is reproducible | Workload, warm-up policy, repeated measurements, and resource sample |
| AC-10 | Report is complete and redacted | JSON schema validation, human rendering, no secret/private input |
| AC-11 | Clean-environment reproduction succeeds | Fresh environment instructions and matching gate decision |

## Gate rule

The release decision is `pass` only when AC-01 through AC-08 and AC-10 pass, and
AC-11 passes from a fresh committed checkout and fresh Python environment for
`onnx-cpu-windows-py311`, the first release's only supported profile. AC-09 is required
for a performance claim; otherwise the report must state that performance is not
verified. Any unavailable required environment yields `blocked` rather than pass.

TensorRT is an independent optional preview with platform-specific evidence.
Its missing provenance, peak GPU memory, or exact-engine reproduction evidence
stays `not_verified` / `blocked` without blocking the CPU-only exit. CPU success
does not validate TensorRT. CPU performance is bounded to the recorded workload
and Python allocation peak; native process and GPU peaks remain unverified.

## Evidence boundaries

Numerical equivalence is bounded by the declared fixture and tolerances. A local
load test does not prove compatibility with undeclared hardware or arbitrary
drivers. Benchmark results describe the recorded environment and workload only.
