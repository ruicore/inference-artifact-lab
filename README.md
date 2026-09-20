# Inference Artifact Lab

Inference Artifact Lab is a clean-room, public-by-design project for validating
machine-learning deployment artifacts before release. Its first product increment
is the **Model Release Gate**: a reproducible gate for artifact integrity,
input/output contracts, runtime correctness, and environment compatibility.

The project uses public models, public datasets, and generated fixtures. It does
not train models, provide a general model-serving gateway, or claim model quality
beyond the declared validation evidence.

## Current status

Product Phase 1, requirements approved for implementation. The implementation
skeleton is present; gate behavior has not started.

## Planned flow

```text
public model
  -> source and artifact manifest
  -> export/build
  -> integrity and contract checks
  -> reference/runtime equivalence checks
  -> environment compatibility checks
  -> resource benchmark
  -> machine-readable release report
```

The authoritative development documentation follows the same phase/stage model
used by the other portfolio repositories. Start at the [Codex document index](docs/codex/README.md), then read the [product contract](docs/codex/product-contract.md) and [Phase 1 plan](docs/codex/phases/phase-1-model-release-gate/README.md).

For the human-readable brief and clean-room record, see [Product Requirements](docs/product-requirements.md), [Acceptance Contract](docs/acceptance-contract.md), and [Clean-room Record](docs/clean-room-record.md).
