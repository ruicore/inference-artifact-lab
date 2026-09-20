---
audience: codex
document_role: validation_record
phase_id: PH1
stage_id: PH1-ST04
status: incomplete
---

# Phase 1 validation record

Supersedes the earlier blanket pass declaration. Public SqueezeNet CPU and
TensorRT smoke execution were observed, but do not close Phase 1 acceptance.

| Cases | Current evidence boundary |
|---|---|
| AC-01..04 | Basic unit checks and example identity evidence; full format/profile validation remains open |
| AC-05..06 | CPU/TRT numerical smoke evidence and negative-output tests; complete evidence binding remains open |
| AC-07..08 | Local compatible execution observed; full incompatible-environment fail-fast path remains open |
| AC-09 | Historical timings only; throughput, warmup units and memory scope require correction |
| AC-10 | JSON snapshots exist; formal schema and human report rendering remain open |
| AC-11 | Not verified from a fresh environment and empty artifact directory |

See [publication review](../../../publication-review.md) for specific limitations.
This is a development preview, not a completed product acceptance or release.
