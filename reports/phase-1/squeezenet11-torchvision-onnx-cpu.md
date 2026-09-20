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

## Evidence

```json
{
  "fixture_sha256": "3161e1f639bd8c78b14164b87c3fb0be8aeb821aa715fcc0cab2d93353242d89",
  "reference_output_sha256": "9b9e35c2d6a1191685178087ce91bd352fadcebc97af24cb7b88ecf75fec0f06"
}
```
