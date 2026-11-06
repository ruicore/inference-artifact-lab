---
audience: codex
document_role: phase_index
phase_id: PH2
status: implemented_pending_clean_reproduction
---

# Phase 2 - Model Lifecycle Experiment

Selected and implemented: 2026-11-06. Clean committed reproduction is pending;
current evidence is recorded separately below.

## Public problem

A valid model artifact can still be used incorrectly when a local application
shares, switches, retains, or releases runtime instances. This experiment asks:
when can requests safely share one instance, what remains resident after use,
and what evidence supports a resource-lifecycle claim?

The result is an offline, reproducible experiment and report, using independently
generated ONNX fixtures and a pinned public model. It is not a model-serving
platform. No private models, APIs, state machines, traces, or work-derived
performance targets may enter the design, fixtures, or report.

- [Requirements and identity/ownership rules](requirements.md)
- [Acceptance matrix](acceptance.md)
- [Fixtures, measurements, and delivery plan](phase-plan.md)
- [Current validation record](validation.md)

## Run locally

```powershell
uv sync --frozen --extra test --extra runtime-cpu --extra lifecycle
uv pip install -r scripts/requirements-reference-public.txt
uv run --no-sync python scripts/export_torchvision_onnx.py
uv run --no-sync python scripts/run_torchvision_gate.py --report artifacts/reference-gate.json
uv run --no-sync python scripts/generate_lifecycle_fixtures.py --output artifacts/lifecycle --public-manifest examples/squeezenet11-torchvision.manifest.json --public-inputs artifacts/squeezenet11-fixture.npy --public-reference artifacts/squeezenet11-reference.npy
uv run --no-sync python scripts/run_lifecycle_experiment.py --fixtures artifacts/lifecycle/fixtures.json --output artifacts/lifecycle-report.json
uv run --no-sync python scripts/render_lifecycle_report.py artifacts/lifecycle-report.json
uv run --no-sync python -m pytest -q
```

The optional public-reference packages are needed to regenerate and verify the
pinned public reference path. `--no-sync` preserves those explicitly installed
packages. For synthetic-only development omit the three `--public-*` arguments;
the report then withholds the full public-model/measurement acceptance.

The invocation report intentionally leaves AC02, AC06 and AC11 `not_verified`:
deterministic concurrency/fault tests and clean-checkout reproduction are separate
evidence. The validation record combines named observed evidence for acceptance;
it does not rewrite the invocation's statuses. JSON schema and semantic checks
reject contradictions, but cannot authenticate caller-invented raw observations.

After committing the implementation in a clean worktree, reproduce independently:

```powershell
uv run --no-sync python scripts/reproduce_lifecycle.py
```

This clones the committed revision, creates a fresh environment, regenerates
public/generated artifacts, runs tests and the full CPU experiment, validates and
renders the actual report, then writes a local reproduction receipt. The clone
and generated evidence are retained under the ignored reproduction destination.

## First acceptance scope

One local experiment worker, ONNX Runtime CPU, explicit artifact digests and load
options, and controlled two-caller interleavings. Residency is bounded by the
number of loading/resident instances. Actual RSS and timing are measured
separately. No claim that a Python reference release returns native memory to
the OS is allowed without independent evidence of that narrower observation.

TensorRT lifecycle and build-cache studies are later, environment-specific
extensions. They do not block this CPU experiment and do not inherit its result.
Existing Phase 1 CPU acceptance remains historical evidence for its own scope.

## Exit

All required PH2 acceptance rows pass in the declared CPU environment, generated
reports validate against their new schema, a clean committed checkout reproduces
the experiment, and limitations remain visible. Planning completion is not an
acceptance pass. Public push and release status must be recorded separately.
