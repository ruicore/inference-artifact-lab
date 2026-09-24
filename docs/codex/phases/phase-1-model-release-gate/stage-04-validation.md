---
audience: codex
document_role: validation_record
phase_id: PH1
stage_id: PH1-ST04
status: cpu_acceptance_candidate
---

# Phase 1 validation record

Local revalidation on 2026-09-24. The first acceptance and release scope is
`onnx-cpu-windows-py311`; TensorRT is an independent optional preview. A CPU pass
never satisfies missing TensorRT evidence. This record supersedes prior blanket
completion language. Every pass below is bounded by the pinned public model,
fixture, and declared environment.

## CPU-only acceptance

| Case | Status | Observation and boundary |
|---|---|---|
| AC-01 | pass | Public weight URL and SHA-256 are pinned in the manifest and checked at export/reference execution. ONNX size and SHA-256 match. This is not a general supply-chain attestation. |
| AC-02 | pass | Controlled missing/modified artifact tests reject with stable check IDs. |
| AC-03 | pass | Version-1 canonical manifest parses; unsupported versions/formats reject. Runtime profile and acceptance role participate in the manifest digest. |
| AC-04 | pass | Actual ONNX name, dtype, shape, and format match the CPU contract; negative format/profile fixtures reject. |
| AC-05 | pass | Bound CPU runner decodes the exact pinned fixture/reference `.npy` bytes and invokes the adapter on that fixture before comparison. Generic supplied-observation API calls remain `not_verified`. |
| AC-06 | pass | Malformed, wrong-topology, ragged, empty and non-finite outputs fail tests. |
| AC-07 | pass | Declared Windows/Python/ONNX Runtime CPU environment loaded and executed. Other environments remain unverified. |
| AC-08 | pass | CPU CLI rejects an impossible Python version before loading a missing fixture or adapter. |
| AC-09 | pass for bounded CPU benchmark | Warm-up, repeated samples, latency, throughput, and Python allocation peak are observed. Native process and GPU peak memory are not observed or claimed. |
| AC-10 | pass for generated reports | The version-1 Draft 2020-12 schema validates the common report shape; generator tests cross-check the explicit profile/role and environment scope against the manifest, and Markdown rendering and limited private-path/credential checks pass. The version-1 schema alone does not require or authenticate `scope` in caller-supplied JSON. Publication review is separate. |
| AC-11 | not_verified for committed checkout | A temporary copy of Git-tracked current working-tree files, a new Python 3.11.5 environment, and no copied binary artifacts reproduced ONNX SHA-256 `4ed9006c...` and reference/package CLI `pass`. This preliminary run is not a fresh checkout of a committed revision. |

CPU runtime report: `reports/phase-1/squeezenet11-torchvision-onnx-cpu.json`.
The report's manifest digest identifies the exact CPU declaration. Runtime
`pass` is separate from Phase 1 exit: the remaining CPU gate is committed-checkout
AC-11. This row must be updated only after that actual run.

September 24 verification of the CPU-only candidate:

- `uv run --extra test --extra runtime-cpu pytest -q`: 52 passed.
- `uv build`: wheel and source distribution built successfully.
- `git diff --check`: passed.
- `pwsh -NoProfile -File scripts/clean_reproduction.ps1`: isolated working-tree
  reproduction passed in a fresh Python 3.11.5 environment, regenerated ONNX
  SHA-256 `4ed9006c47bb1eb8521eed0f8b22aa49003a8013c8e4adf5db414f7c8e927534`,
  and matched reference/CLI manifest digest
  `95187924e351a0817a37db70dac9f73770d200f8e84409b7c214f8178cd2451d`.
  The temporary copy was removed; this does not close committed-checkout AC-11.
- CPU runtime report regenerated; TensorRT portable report recomposed from
  existing public-model container evidence and remains `not_verified`. No new
  TensorRT GPU execution was required or claimed for this CPU-only change.

## Independent TensorRT preview

| Case / claim | Status | Observation and boundary |
|---|---|---|
| Local identity / contract / environment | pass for observed platform | Existing pinned engine matches digest and contract; observed RTX 5050 / pinned TensorRT container loaded and executed. Container digest in payload remains launcher-supplied; separate host image inspection is not portable attestation. |
| AC-05 portable runtime provenance | not_verified | Composer receives supplied container JSON and cannot independently replay the adapter call. Local numerical comparison does not close portable provenance. |
| AC-09 performance | not_verified | Device-only `trtexec` timings omit observed peak GPU memory. |
| AC-11 exact-engine reproduction | blocked | Engine is an ignored local binary unavailable in a fresh checkout. Rebuilding different bytes is not exact-engine evidence. |
| Preview report | not_verified | `reports/phase-1/squeezenet11-tensorrt.json` retains independent status and `optional_preview` role. |

These preview gaps do not block CPU-only Phase 1 acceptance. The recorded GPU
is one evaluated platform, not a required platform for future TensorRT work.
No cross-platform compatibility or byte-identical TensorRT rebuild is claimed.
Publication/release approval remains separate; see
[publication review](../../../publication-review.md).
