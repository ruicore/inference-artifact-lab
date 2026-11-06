---
audience: codex
document_role: validation_record
phase_id: PH2
status: pending_clean_reproduction
---

# Phase 2 validation record

Date: 2026-11-06. Scope: the declared local Windows/Python 3.11 ONNX Runtime CPU
experiment, generated A1/A2/B fixtures, and pinned public SqueezeNet artifact.
This record combines separate evidence sources; the raw invocation cannot
certify independent tests or its own fresh-checkout reproduction.

## Evidence collected before the implementation commit

- Existing Phase 1 regression baseline: 52 tests passed before implementation.
- Integrated suite after report-consistency fixes: 102 tests passed; the final
  clean-checkout reproduction evidence will be recorded after that procedure.
- A full CPU invocation used three repetitions per policy, five warmups,
  30 measured cycles, 50 ms sampling and one-second pre/post windows.
- Required runtime rows AC01, AC03, AC04, AC05, AC07, AC08, AC09 and AC10 passed
  in that invocation. AC02, AC06 and AC11 remain `not_verified` in the raw report
  because their independent evidence belongs to tests and checkout reproduction.
- Both policies executed the real public model and generated ONNX fixtures.
  The public artifact and both NPY fixture/reference digests matched the pinned
  Phase 1 manifest; the actual bound CPU gate passed before lifecycle use.

## Measurement correction and limits

An early exploratory run sampled the Windows virtual-environment launcher rather
than the actual interpreter. Its memory results are excluded from acceptance.
The corrected parent binds a private worker handshake to its own process subtree
and creation time, samples that interpreter, and independently observes its exit.
A regression uses a real 64 MiB interpreter allocation; timeout checks verify
that the owned worker exits. Public reports contain binding/exit booleans, not PIDs.

The corrected full run observed six verified worker identities and exits, no
missing RSS samples or collection errors, and complete baseline/post windows.
Its sampled RSS maxima were approximately 85-88 MiB. These observations do not
establish a native-memory ceiling, reclamation guarantee, or performance win.
Final committed-checkout values may differ and will be recorded separately.

## Current boundary

Clean committed-checkout AC11 has not yet run for this implementation. Do not
declare full Phase 2 acceptance from the working-tree run alone. Final review
also checks generated-report consistency, privacy and packaged schema availability.
Phase 1 historical evidence and TensorRT preview status remain separate.
