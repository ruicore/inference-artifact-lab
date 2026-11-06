---
audience: codex
document_role: phase_plan
phase_id: PH2
authority: delivery
status: planning
---

# Experiment and delivery plan

## Fixture and environment design

Use three independently generated ONNX graphs: A1 and A2 share an I/O contract
but have different constant weights/expected outputs; B is a third identity for
capacity and eviction checks. Generate input bytes and reference outputs with a
fixed seed and declared dtype/shape; hash the serialized bytes. No application
data or private model exporter is involved.

Use the existing pinned public SqueezeNet artifact and fixture for representative
CPU measurement, with a verified Phase 1 artifact gate before lifecycle use.
Record its current exported artifact hash; do not assume a fresh exporter must
produce the old bytes. An unexpected hash requires explicit manifest review,
not silent replacement. Tiny generated graphs remain correctness fixtures and
cannot establish realistic model-memory savings.

First environment: local Windows, Python 3.11, ONNX Runtime CPU, with exact
resolved dependency versions recorded. Reuse the existing runtime-cpu extra;
add only the smallest optional experiment dependencies needed for generation
and process RSS collection during implementation. Fix one intra-op and one
inter-op thread initially and explicitly select CPUExecutionProvider. Additional
platforms get separate profiles rather than inferred compatibility.

## Minimal implementation shape

- A small lifecycle module with instance identity, handles, a two-slot admission
  limit, and idle eviction. No server, database, plugin registry or scheduler.
- A deterministic fake adapter and independent loader/disposal ledger for
  interleaving tests. The ledger must not reuse the pool's accounting code.
- An ONNX CPU experiment adapter around the existing adapter boundary. Keep
  Phase 1 CLI behavior and report schema intact; fix backend lifecycle seams
  only where the new scenarios demonstrate a need.
- A local script that reads a validated scenario, runs one owned child worker,
  and writes a distinct lifecycle report. The parent collects timestamped RSS
  samples and exit facts. A lost worker connection cannot mean successful exit.
- A schema and Markdown renderer for the new report. Public records retain
  environment characteristics and content identities, not absolute private paths,
  hostnames, usernames, credentials, or unrelated process details.

## Measurement schedule

Use these fixed operation sequences before adding more scenarios:

| Sequence | Operations | Main observation |
|---|---|---|
| Shared use | acquire A1 twice; run both handles; release first; run second; release second | One generation, independent ownership |
| Revision switch | acquire alias=A1; change alias=A2; acquire alias; run both; release both | Correct output and identity for each revision |
| Idle pressure | acquire/release A1; acquire B and keep it active; acquire A2 | Only idle A1 can be evicted in retain-idle mode |
| Busy pressure | hold A1 and B; acquire A2 | Rejection without a third load |
| Public-model retention | hold public model M for warmup, release it; repeat acquire/run/release M 30 times | Cold/warm/reload costs by policy |
| Public-model reload | after the preceding cycles, acquire/release A1 then B, then acquire/run/release M | Retain-idle evicts the older idle M before its next load; output still matches |

The generated-model buffers are prepared before the memory baseline and held
identically across policies. Counts and expected generation transitions are
specified by each sequence, while native allocation and latency are observations.

Keep correctness/interleaving tests separate from timings; test barriers are
not performance measurements. For each of `release_on_idle` and `retain_idle`,
run three fresh worker processes in alternating policy order. Within each worker:

1. Import runtime and prepare fixture arrays outside the timed load boundary.
2. Observe a pre-load RSS window; record sampling coverage, not a single point.
3. Measure first adapter/session construction, then five warmup inferences.
4. Run 30 measured use/release cycles on the public model with a fixed input.
   Record acquisition/load and inference timings separately. One warm inference
   group holds a handle; release-on-idle load costs are not hidden in that group.
5. Perform the public-model reload sequence above and final teardown.
6. Observe an idle RSS window and worker exit from the parent.

The initial sampling target is 50 ms with 1-second pre-load and post-teardown
windows. Record actual timestamps/gaps and the collector implementation/version.
These are synthetic experiment settings, not service-level objectives. Keep
generation, download, environment setup, process startup and warmup outside the
declared steady inference samples; report excluded costs separately where useful.

Report per-repetition raw samples and summaries: median/p95 of observed latency,
sampled RSS maximum and before/after distributions, load count, eviction count,
maximum active handles and maximum admitted instances. State sample count and
quantile method. Do not sum per-stage maxima into a fake total, extrapolate a
server-capacity promise, or claim native memory leaks from RSS growth alone.

## Delivery stages

| Stage | Work | Exit |
|---|---|---|
| 00 - Contract and fixtures | Finalize scenario representation, identity option allowlist, generated fixtures and report shape | R01-R09/AC01-AC11 traceability; public/generated inputs only |
| 01 - Deterministic ownership | Implement identity, handle ownership, admission, idle eviction and controlled failure/interleaving tests | AC01-AC06 with independent fake-adapter observations |
| 02 - Real CPU integration | Add explicit adapter teardown, generated ONNX runs, public-model worker and process-exit observations | AC07-AC08; no fake-to-native evidence substitution |
| 03 - Measurement and report | Add RSS collection, cold/warm/reload schedule, JSON schema and renderer | AC09-AC10 with unavailable observations explicit |
| 04 - Reproduction and review | Fresh committed-checkout reproduction, regression/disclosure review and limitation record | AC11 and all required rows; publication tracked separately |

All stages are currently planned. Do not create successful validation records
until the checks actually run. Preserve Phase 1's historical records. A useful
result may show that retention improves load latency while RSS stays elevated;
the project must report that result without inventing a memory-reclamation win.

## Deferred scope and source references

TensorRT lifecycle, timing-cache comparisons, precision studies, automatic
byte-budget estimation, live serving, distributed sharing, training and broad
performance tuning are outside this first experiment. Revisit them only with
their own product problem and independent acceptance evidence.

Public API and measurement references, checked during planning:

- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
  separates session construction/configuration from inference calls.
- [ONNX Runtime memory guidance](https://onnxruntime.ai/docs/performance/tune-performance/memory.html)
  discusses allocator configuration and multi-session memory behavior. This
  informs measurement design; it is not evidence about this experiment's runtime.
