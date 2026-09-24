# Product Requirements: Model Release Gate

This is the human-readable brief. The normative product contract and Phase 1
requirements are maintained under [docs/codex](codex/README.md).

**Product:** Inference Artifact Lab
**Increment:** Phase 1 - Model Release Gate
**Status:** approved for Phase 1 implementation
**Date:** 2026-09-20

## Problem

A model conversion command can finish successfully while the resulting file is
incomplete, violates its input/output contract, changes numerical behavior, or
cannot load in the declared deployment environment. Release decisions need a
repeatable evidence bundle instead of a file-exists check.

## Product outcome

Given a declared model source, artifact manifest, validation dataset, and target
environment, the gate produces a deterministic report stating whether the artifact
is releasable, blocked, failed, or not verified. Every decision includes the
observed evidence and its limits.

## Phase 1 scope

### R1 - Source and artifact identity

Record public source identity, logical model version, artifact format, build inputs,
tool versions, file size, and SHA-256 digest. A changed or missing artifact must
fail the gate before runtime execution.

### R2 - Artifact manifest

Define a versioned manifest for input names, output names, dtype, shape or dynamic
dimensions, preprocessing, postprocessing, runtime requirements, and validation
tolerances. Malformed or incomplete manifests must be rejected.

### R3 - Contract validation

Inspect the artifact and verify that its declared inputs, outputs, dimensions, data
types, and required profiles match the manifest. Report each mismatch with a stable
machine-readable reason.

### R4 - Runtime correctness

Run a fixed public or generated validation fixture through a reference path and the
target runtime. Compare outputs using declared absolute/relative tolerances and
domain-specific checks. A missing reference, invalid output, or exceeded tolerance
blocks release.

### R5 - Environment compatibility

Declare the supported Python, operating system, accelerator, driver, CUDA/runtime,
and package versions. Verify load and execution in the declared environment. An
untested environment remains `not_verified`; an incompatible environment fails
with an actionable reason.

### R6 - Resource benchmark

Measure warm-up, latency, throughput, and peak memory for a declared workload and
record the exact workload and environment. Phase 1 reports measurements; it does
not impose universal performance targets.

### R7 - Release report

Generate a versioned JSON report and a human-readable rendering. The report must
include artifact identity, manifest identity, environment, checks, evidence
references, final status, limitations, and unverified boundaries.

### R8 - Reproducibility

Provide a clean-environment command sequence that reconstructs an interchange
artifact or verifies a pinned compiled artifact and reproduces the gate report.
For TensorRT, the pinned engine digest is the release identity; a rebuild may be
validated for contract and numerical behavior without claiming identical engine
bytes. Inputs must be public or generated, and the result must not depend on
private services or credentials.

## Frozen implementation baseline

These choices constrain the first implementation without changing the product
boundary:

- Use one pinned public, non-robotics model and a deterministic generated
  validation fixture. Record the exact model revision and source URL before its
  first artifact is committed.
- The mandatory artifact is ONNX, validated against a CPU reference path on
  `onnx-cpu-windows-py311`. TensorRT engines are an independent optional preview;
  their GPU/platform-specific evidence does not block the CPU-only Phase 1 exit.
- The manifest is authoritative for names, shapes, dtypes, tolerances, runtime
  requirements, and supported environments. Directory discovery and implicit
  defaults must not select an artifact.
- The first supported environment is the explicitly recorded ONNX Runtime CPU
  profile. Other Python, driver, CUDA, GPU, and operating-system combinations are
  `not_verified` until independently executed.
- Phase 1 produces a local gate CLI and JSON report first. Registry upload,
  production deployment, and automatic publication are out of scope.

Changing the model fixture, tool version, or environment entry does not change the
product requirements. Changing the gate dimensions, decision vocabulary, evidence
rules, or release semantics requires a new product decision and acceptance review.

## Explicit non-goals

- Model training or fine-tuning.
- A general model-serving gateway or orchestration platform.
- A model leaderboard or broad model-quality evaluation service.
- Universal hardware compatibility or arbitrary exactly-once guarantees.
- Automatic publication to a registry, cloud, or production environment.

## Release decision vocabulary

`pass` means every required check passed for the declared scope. `fail` means a
required check contradicted the contract. `blocked` means required evidence or an
environment is unavailable. `not_verified` means the claim was outside the
executed validation scope. The gate must never infer `pass` from artifact creation
alone.

A CPU-only pass never implies TensorRT preview acceptance. The current CPU
performance claim includes Python allocation peaks only, not native process or
GPU peak memory. TensorRT may be evaluated on separately declared platforms;
there is no mandatory GPU platform for the first CPU acceptance.
