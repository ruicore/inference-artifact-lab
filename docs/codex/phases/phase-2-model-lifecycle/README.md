---
audience: codex
document_role: phase_index
phase_id: PH2
status: planning
---

# Phase 2 - Model Lifecycle Experiment

Selected: 2026-11-06. Implementation: not started. Acceptance: not run.

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
