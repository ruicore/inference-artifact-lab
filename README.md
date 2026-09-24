# Inference Artifact Lab

Inference Artifact Lab is a clean-room, public-by-design project for validating
machine-learning deployment artifacts before release. Its first product increment
is the **Model Release Gate**: a reproducible gate for artifact integrity,
input/output contracts, runtime correctness, and environment compatibility.

The project uses public models, public datasets, and generated fixtures. It does
not train models, provide a general model-serving gateway, or claim model quality
beyond the declared validation evidence.

## Current status

The first Phase 1 acceptance and release scope is the declared ONNX Runtime CPU
profile (`onnx-cpu-windows-py311`). Its local acceptance passed, including an
independent clean clone of a committed revision and a fresh Python environment.
CPU performance evidence covers Python allocation peaks only. This is not a
published release or a claim about other environments.

TensorRT is an independent optional preview, evaluated per GPU/runtime platform.
Its portable provenance and peak GPU memory remain `not_verified`, and exact-
engine clean-checkout reproduction is `blocked`. These preview gaps do not block
CPU-only acceptance; CPU success does not validate TensorRT. See the separate
CPU and preview rows in the [validation record](docs/codex/phases/phase-1-model-release-gate/stage-04-validation.md).

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

Explore the optional TensorRT preview after pulling the pinned public image:

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

The generated TensorRT report includes the contract, optimization profile
bounds, engine digest, fixture equivalence, declared GPU/container fingerprint,
and `trtexec` benchmark scope. It currently exits nonzero with
`not_verified`: `trtexec` does not provide an observed peak GPU memory value
for the declared performance claim.

Render any JSON report for review with:

```text
uv run python scripts/render_report.py reports/phase-1/squeezenet11-tensorrt.json
```

The report contract is defined by
`schemas/release-report.schema.json`. A clean CPU reproduction starts with
`pwsh scripts/clean_reproduction.ps1` in a fresh checkout.

After generating the public fixture and reference output, the package CLI probes
the current environment, checks artifact and pinned fixture digests before
adapter execution, then runs the ONNX CPU adapter directly:

```text
python -m inference_artifact_lab examples/squeezenet11-torchvision.manifest.json `
  --runtime onnx-cpu --inputs-npy artifacts/squeezenet11-fixture.npy `
  --reference-npy artifacts/squeezenet11-reference.npy `
  --report reports/phase-1/cli-onnx-cpu.json
```

`pwsh scripts/clean_reproduction.ps1` copies only Git-tracked working-tree files to an
isolated temporary directory, creates a new Python environment, regenerates the
model and fixture without copied binary artifacts, and asserts matching CPU
gate decisions. This is not an independent Git checkout while local changes
remain uncommitted; it does not establish TensorRT clean-checkout reproduction.

The authoritative development documentation follows the same phase/stage model
used by the other portfolio repositories. Start at the [Codex document index](docs/codex/README.md), then read the [product contract](docs/codex/product-contract.md) and [Phase 1 plan](docs/codex/phases/phase-1-model-release-gate/README.md).

For the human-readable brief and clean-room record, see [Product Requirements](docs/product-requirements.md), [Acceptance Contract](docs/acceptance-contract.md), and [Clean-room Record](docs/clean-room-record.md).
