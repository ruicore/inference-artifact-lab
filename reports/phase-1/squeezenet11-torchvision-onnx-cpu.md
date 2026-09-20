# Model Release Gate report

- Status: **pass**
- Schema: `1`
- Manifest digest: `7edba69f960c634785642371f657a680e700dafd234861cc440a8a0815860e3b`
- Artifact: `artifacts\squeezenet1.1-torchvision.onnx`

## Checks

| Check | Status | Message |
|---|---|---|
| `artifact.exists` | **pass** | artifact file exists |
| `artifact.size` | **pass** | artifact size matches manifest |
| `artifact.sha256` | **pass** | artifact SHA-256 matches manifest |
| `contract.inputs` | **pass** | artifact inputs match manifest |
| `contract.outputs` | **pass** | artifact outputs match manifest |
| `artifact.format` | **pass** | observed artifact format matches manifest |
| `environment.compatibility` | **pass** | observed environment satisfies manifest |
| `runtime.equivalence` | **pass** | target output is within declared tolerance |
| `benchmark.workload` | **pass** | benchmark workload completed |

## Limitations

- TensorRT engine evidence is recorded in the separate pinned-container report; this report covers the CPU profile.
- The benchmark peak memory scope is Python allocations only; native and GPU memory are covered by the TensorRT container report.
