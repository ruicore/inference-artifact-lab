---
audience: codex
document_role: acceptance_contract
phase_id: PH2
authority: normative
status: not_run
---

# Phase 2 acceptance matrix

All rows are currently `not_run`. Codes below define future checks, not results.
Each required row needs its own recorded evidence; unit tests do not substitute
for the real-runtime and clean-checkout rows.

| Case | Requirement | Scenario and required evidence |
|---|---|---|
| PH2-AC01 | R01, R02 | Equal artifact bytes at different locators share under identical effective load options; a mismatched declared hash rejects before loading; changed bytes/options produce distinct keys. Independent loader records agree. |
| PH2-AC02 | R02, R03 | Two barrier-controlled callers overlap acquisition of one key; exactly one successful load and one generation serve both; releasing one handle cannot close the other's instance. |
| PH2-AC03 | R03 | Released handle rejects inference; repeated release does not underflow counts or dispose an instance used by another handle. |
| PH2-AC04 | R04, R07 | An alias changes from generated revision A1 to A2 with the same I/O shape but distinct expected output. Old handles still return A1, new handles return A2; traces identify each digest/generation. |
| PH2-AC05 | R05 | With two slots, loading a third key evicts the deterministically oldest idle instance only; an active instance remains usable. When both slots are active/loading, admission rejects before a third loader runs. |
| PH2-AC06 | R02, R06 | Controlled loader failure releases its reservation; a subsequent valid acquisition works. Controlled inference failure and cooperative stop release caller ownership. No claim of interrupting a native call. |
| PH2-AC07 | R06, R07 | Normal teardown reports no outstanding handles and releases runner references. Parent independently observes worker exit. Cleanup failure and timeout fixtures cannot pass teardown. |
| PH2-AC08 | R07 | Generated ONNX fixtures and pinned public model execute on the declared CPU runtime; outputs match before/after sharing, switching and reload. Fake-only observations cannot satisfy this row. |
| PH2-AC09 | R05, R08 | Compare release-on-idle and retain-idle under identical fixture/operation schedules. Report cold/warm/reload timings, logical counts, sampled RSS and missing samples with units and method. No speedup or RSS-return threshold is assumed. |
| PH2-AC10 | R08 | New JSON schema and renderer preserve scenario/artifact/environment identity, check-level status, dirty state and limitations. Missing runtime or required RSS observations cannot become a full-profile pass. Phase 1 reports remain compatible. |
| PH2-AC11 | R09 | Clean committed checkout and fresh environment regenerate fixtures, verify upstream digests, execute required cases and validate reports. Compare decisions and identities, not byte-identical timing/memory measurements. |

## Exit interpretation

The required profile is local ONNX Runtime CPU with real-runtime checks and
sampled process RSS. Unsupported memory collection may leave deterministic
ownership checks passing, but AC09 and the full profile remain `not_verified`
or `blocked` with the reason recorded. There is no fallback that calls Python
allocation tracking total native memory.

Passing AC09 means measurement completeness and honest attribution, not a
performance improvement or evidence that memory always returns to baseline.
No acceptance case requires crashing the host through deliberate OOM.
GPU/TensorRT checks are optional, separate platform claims.
