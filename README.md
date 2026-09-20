# Inference Artifact Lab

Inference Artifact Lab is a clean-room, public-by-design project for validating
machine-learning deployment artifacts before release. Its first product increment
is the **Model Release Gate**: a reproducible gate for artifact integrity,
input/output contracts, runtime correctness, and environment compatibility.

The project uses public models, public datasets, and generated fixtures. It does
not train models, provide a general model-serving gateway, or claim model quality
beyond the declared validation evidence.

## Current status

Development preview: public SqueezeNet CPU and TensorRT smoke runs are recorded.
Phase 1 acceptance remains incomplete; report delivery and clean reproduction
need further work. See [publication review](docs/publication-review.md) for
known limitations. The commands below are development examples, not a verified
from-scratch reproduction procedure.

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

Run the public-reference smoke gate with:

```text
uv run --with torch --with torchvision --with onnx --with onnxruntime python scripts/run_torchvision_gate.py
```

It writes `reports/phase-1/squeezenet11-torchvision-onnx-cpu.json`.

Build and verify the TensorRT profile after pulling the pinned public image:

```text
pwsh scripts/build_tensorrt_engine.ps1
pwsh scripts/benchmark_tensorrt_engine.ps1
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  -e MODEL_RELEASE_GATE_CONTAINER_DIGEST=sha256:814325e2b8a653f354c30bbcf5ecc8d4c780cf878a88a320ae648fbfdd9dd82d `
  nvcr.io/nvidia/tensorrt:25.02-py3 bash -lc `
  "python -m pip install --quiet --index-url https://pypi.org/simple cuda-python==12.8.0; `
   PYTHONPATH=/workspace/src python scripts/run_tensorrt_in_container.py `
   --engine artifacts/squeezenet1.1-fp32.engine `
   --fixture artifacts/squeezenet11-fixture.npy `
   --output artifacts/squeezenet11-tensorrt-output.json"
uv run --with numpy==2.4.6 python scripts/compose_tensorrt_report.py `
  --trtexec-log reports/phase-1/tensorrt-trtexec-benchmark.log
```

The generated TensorRT report includes the contract, engine digest, fixture
equivalence, declared GPU/container fingerprint, and `trtexec` benchmark scope.

Render any JSON report for review with:

```text
uv run python scripts/render_report.py reports/phase-1/squeezenet11-tensorrt.json
```

The report contract is defined by
`schemas/release-report.schema.json`. A clean CPU reproduction starts with
`pwsh scripts/clean_reproduction.ps1` in a fresh checkout.

The authoritative development documentation follows the same phase/stage model
used by the other portfolio repositories. Start at the [Codex document index](docs/codex/README.md), then read the [product contract](docs/codex/product-contract.md) and [Phase 1 plan](docs/codex/phases/phase-1-model-release-gate/README.md).

For the human-readable brief and clean-room record, see [Product Requirements](docs/product-requirements.md), [Acceptance Contract](docs/acceptance-contract.md), and [Clean-room Record](docs/clean-room-record.md).
