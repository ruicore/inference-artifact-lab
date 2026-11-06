---
audience: codex
document_role: validation_record
phase_id: PH2
status: cpu_local_pass
---

# Phase 2 validation record

Date: 2026-11-06. Scope: the declared local Windows/Python 3.11 ONNX Runtime CPU
experiment, generated A1/A2/B fixtures, and pinned public SqueezeNet artifact.
This record combines separate evidence sources; the raw invocation cannot
certify independent tests or its own fresh-checkout reproduction.

## Evidence collected before the implementation commit

- Existing Phase 1 regression baseline: 52 tests passed before implementation.
- Integrated suite after report-consistency fixes: 102 tests passed, followed by
  the same 102 tests in the independent clean-checkout reproduction below.
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

## Committed-checkout reproduction

Code revision: `18cfcea62e6fbcf25400dd4e5d4a2a29982c35dc`.
The independent clone ran the frozen project dependency set, pinned public
reference dependencies, generated/public artifact regeneration, all 102 tests,
the full experiment, actual JSON schema validation and semantic Markdown rendering.
The new environment used Python 3.11.5, NumPy 2.4.6, ONNX 1.20.0, ONNX Runtime
1.30.0, psutil 7.2.2 and jsonschema 4.26.0 on Windows AMD64. The clone was clean
afterward and the report recorded the same revision with `dirty: false`.

Two fresh-clone attempts were retained, using identical code, scenario settings,
artifact identities and fixture bindings:

1. The first refused a reproduction pass because one baseline window had an
   observed maximum sample gap of 0.3108341 seconds. AC09 and AC10 remained
   `not_verified`; no successful receipt was created. All tests and other runtime
   checks passed. This is retained negative coverage evidence, not discarded data.
2. The second passed the unchanged coverage criteria in all twelve baseline/post
   windows. Six actual interpreter identities and exits were verified; collection
   errors and missing samples were zero. Its receipt records 102 tests including
   25 deterministic lifecycle cases, a fresh environment, equal source/clone
   revisions, and clean tracked/nonignored files after reproduction.

The receipt's `reproduction_status` is `pass`. The raw invocation's status stays
`not_verified` for AC02, AC06 and AC11 because it cannot establish external test
or checkout facts by itself. The table below joins those independently observed
evidence sources; no raw invocation status was promoted or rewritten.

| Acceptance | Local result | Evidence |
|---|---|---|
| AC01 | pass | Independent loader observations, identity/mismatch tests, real CPU locator/options/byte checks |
| AC02 | pass | `test_ac02_barrier_controlled_acquisition_coalesces_one_load` and independent blocked-inference ownership probe |
| AC03 | pass | Repeated-release/use-after-release tests, serialized in-flight calls and real shared-handle outputs |
| AC04 | pass | Generated A1/A2 aliases preserve original keys and distinct expected real ONNX outputs |
| AC05 | pass | Idle LRU, busy/loading/cleanup capacity tests and real active-survives-pressure observations |
| AC06 | pass | Coalesced loader-failure/retry, inference-failure context cleanup and cooperative-stop tests |
| AC07 | pass | Failure/timeout negative tests, actual interpreter identity+exit observations and complete teardown reports |
| AC08 | pass | Generated A1/A2/B and pinned public M inference, shared use and reload comparisons; rehashed upstream/artifact/fixture bindings |
| AC09 | pass | Second full attempt: three repetitions per policy, complete twelve RSS windows, raw timing/RSS samples, declared coverage; first failed attempt retained |
| AC10 | pass | Actual generated JSON schema validation, semantic renderer, private-marker checks and contradictory-report regressions |
| AC11 | pass | Independent clean committed clone, fresh environment, regenerated inputs, 102 tests and validated/rendered actual report |

Scenario SHA-256: `71b89f0b570904a617c8bee5fcfe86ad29356d8a8137260442250ca14b1aeaf4`.
Successful raw report SHA-256:
`0ae729163188f4203cacf4c5590da68339ce51cb9ea589b8b36f643a5b405fc3`.
Full raw reports, Markdown and reproduction receipts remain in the ignored local
experiment destinations. The reproduction command regenerates their decisions
and identities; timing/memory bytes are observations, not golden-file targets.

## Bounded observations from the successful reproduction

| Repetition | Policy | RSS samples | Sampled maximum RSS (MiB) | Full-worker loads | Idle evictions |
|---|---|---|---|---|---|
| 1 | release_on_idle | 45 | 85.691 | 44 | 0 |
| 1 | retain_idle | 41 | 85.469 | 13 | 3 |
| 2 | release_on_idle | 45 | 85.848 | 44 | 0 |
| 2 | retain_idle | 40 | 85.676 | 13 | 3 |
| 3 | release_on_idle | 46 | 90.730 | 44 | 0 |
| 3 | retain_idle | 41 | 85.691 | 13 | 3 |

Counts cover all identity/sharing/revision/pressure/public-model sequences in
each worker, not just the 30 public-model cycles. The maximum admitted count was
two. Per-repetition median public cycle acquisition was approximately 11.32-12.72 ms
with release-on-idle and 2.47-2.51 ms with retention, including digest verification.
Warm inference medians were approximately 2.43-2.53 ms. These are observations of
this small fixed scenario, not a general performance comparison or capacity claim.

## Packaging, review and remaining boundaries

The wheel and source distribution build passed. Wheel module/schema bytes were
compared with the committed implementation. An isolated installed-wheel smoke
test exercised shipped-schema lookup, semantic validation/rendering and real
generated ONNX inference; it also rejected attribution to an enclosing unrelated
Git checkout. The installed-wheel smoke used the first clone's environment after
its failed sampling attempt and did not alter that raw report.

Independent review reproduced and verified rejection of false-pass labels,
invented RSS summaries, inconsistent instance keys, truncated measured cycles,
and nonzero actual-worker exit behind a successful launcher. Candidate source
and documentation passed targeted private-marker and detect-secrets checks.

This completes the declared local CPU acceptance through combined evidence.
It does not establish native-memory reclamation, universal thread/runtime safety,
GPU support, production serving or a public release. Phase 1 historical evidence
and TensorRT preview status remain separate. Both reproduction clones and local
evidence are retained for inspection; no public push was performed by this work.
