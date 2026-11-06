---
audience: codex
document_role: requirements
phase_id: PH2
authority: normative
status: planned
---

# Phase 2 requirements

## PH2-R01 - Runtime identity

An instance key consists of artifact SHA-256, adapter/runtime version, ordered
effective execution providers and provider options, and supported session-load
options. Paths and human aliases are locators, not identity. Effective values
must be explicit and canonicalized before loading. Unknown options reject at
the experiment boundary. Start with CPU provider and an allowlist of session
thread-count options; do not invent a universal adapter configuration framework.

Request inputs and fixture-selected output interpretation do not create a new
instance. They cannot change the model's I/O contract or preprocessing. The
runner verifies the declared digest before loading an immutable fixture artifact.
It does not watch or support concurrent mutation of artifact files.

## PH2-R02 - Compatible sharing

Acquiring the same key reuses the same live generation, including two callers
whose acquisitions overlap. Coalesce overlapping loads; failed loading leaves
no usable instance or occupied admission slot. Different artifact bytes or
effective load options must not share. Record loader observations independently
of the pool's counters in deterministic tests.

## PH2-R03 - Use ownership and release

Each acquisition returns one explicit use handle. Every inference requires an
open handle; use after release rejects. Releasing the same handle twice is
idempotent and cannot decrement another caller's ownership. Do not evict or
close an instance while any handle holds it, including during inference.
Serialize inference within one instance initially; parallel inference throughput
is not part of the claim. Use controlled barriers rather than timing sleeps to
test interleavings. Release handles with context managers or `finally` cleanup.

## PH2-R04 - Explicit revision changes

A scenario may replace an alias with a different immutable artifact declaration.
Existing handles retain their original generation and artifact identity; new
acquisitions resolve the new declaration. Updating an alias does not rewrite
an existing handle's identity or outputs. Record both generations in the trace.
No filesystem watcher, registry, or hot production rollout is required.

## PH2-R05 - Bounded admission and idle eviction

The scenario declares a positive `max_instances` including loading reservations,
active instances, and idle retained instances. For the initial experiments use
two slots. In `retain_idle` mode, evict the least-recently-used idle instance
when admission needs a slot; use a monotonic operation sequence for ordering.
If all slots are active/loading, reject the new acquisition with a stable
capacity reason before invoking another loader. No hidden wait queue or
unbounded retry is introduced. `release_on_idle` closes at the last release.

This is an instance-count bound. It does not estimate native-memory size from
artifact file size or assert an RSS/GPU-memory ceiling.

## PH2-R06 - Failure, cooperative stop, and teardown

Acquisition/load failures release their admission reservation. Inference failure
releases ownership through the runner's cleanup path. A scripted stop prevents
new operations; an in-flight synchronous native call is allowed to finish before
teardown. Do not claim native-call interruption. Runner shutdown reports
outstanding handles and cleanup errors; it cannot silently label them released.

Use a fresh worker process for each measurement repetition. The parent records
its actual exit and exit code. Termination after a declared timeout is a failed
or incomplete scenario, not a successful cleanup result. Only task-owned workers
are controlled. No interaction with unrelated processes or GPU users is needed.

## PH2-R07 - Runtime correctness and evidence layers

Real ONNX CPU inference must match fixture-specific expected outputs/tolerances
before and after sharing, switching, and eviction/reload. Deterministic fake
adapters validate ownership and operation ordering; they do not prove native
runtime cleanup. Report separately: pool membership, active handles, adapter
reference disposal, sampled process memory, and observed worker exit.

No generic Python `close()` promise is imposed on ONNX Runtime. Define adapter
teardown as releasing this runner's references and owned resources according to
the supported backend API. Python weak-reference observations alone cannot
prove that a native allocator returned memory to the OS.

## PH2-R08 - Measurement and reporting

Record cold load, warm inference, reload timing, sampled RSS, and resident/active
counts against operation markers. Declare units, clock, sampling interval,
warmup, repetitions, runtime/thread settings, fixture size and environment.
Sampled maximum RSS is a lower-bound observation of peak process RSS and includes
runtime, fixtures and allocator overhead; it is not per-model allocation.
Python allocation tracking, if present, is a separate optional metric.

Write a new `lifecycle-report/v1` contract/schema and renderer during implementation.
Do not change Phase 1 manifest/report meanings or combine their pass decisions.
Required evidence includes the scenario definition/digest, implementation revision
and dirty state, artifact/fixture digests, effective instance keys, environment,
operation trace, independent correctness checks, measurement availability and
limitations. Reports use `pass`, `fail`, `blocked`, and `not_verified` per check.
Timing/memory values are observed, not deterministic golden outputs.

## PH2-R09 - Reproduction and disclosure

Regenerate the small synthetic models and fixtures from tracked deterministic
generators. Reuse the public Phase 1 model only through its verified upstream
source and artifact pipeline. Track no weights, private logs or raw local machine
identifiers. Reproduce from a clean committed checkout and fresh environment.
Publish only independently generated evidence and supported platform claims.
