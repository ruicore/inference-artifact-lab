# Model Release Gate report

- Status: **pass**
- Schema: `1`
- Manifest digest: `95187924e351a0817a37db70dac9f73770d200f8e84409b7c214f8178cd2451d`
- Artifact: `artifacts\squeezenet1.1-torchvision.onnx`
- Runtime profile: `onnx-cpu-windows-py311`
- Acceptance role: `cpu_baseline`

## Checks

| Check | Status | Message |
|---|---|---|
| `artifact.exists` | **pass** | artifact file exists |
| `artifact.size` | **pass** | artifact size matches manifest |
| `artifact.sha256` | **pass** | artifact SHA-256 matches manifest |
| `model.source_pin` | **pass** | public source identity and digest are declared; weight bytes are verified during export |
| `contract.inputs` | **pass** | artifact inputs match manifest |
| `contract.outputs` | **pass** | artifact outputs match manifest |
| `artifact.format` | **pass** | observed artifact format matches manifest |
| `environment.compatibility` | **pass** | observed environment satisfies manifest |
| `fixture.binding` | **pass** | fixture and reference output match pinned digests |
| `runtime.equivalence` | **pass** | target output is within declared tolerance |
| `benchmark.workload` | **pass** | benchmark workload completed |
| `runtime.provenance` | **pass** | package runner decoded pinned reference bytes and executed adapter on pinned fixture |

## Limitations

- This report covers only the declared ONNX Runtime CPU baseline; TensorRT is an independent optional preview and is not validated by a CPU pass.
- The benchmark peak memory scope is Python allocations only; native process and GPU peak memory are not observed or claimed.
- A passing runtime report does not itself establish committed-checkout AC-11 or authorize publication.

## Evidence

```json
{
  "fixture_sha256": "3161e1f639bd8c78b14164b87c3fb0be8aeb821aa715fcc0cab2d93353242d89",
  "reference_output_sha256": "9b9e35c2d6a1191685178087ce91bd352fadcebc97af24cb7b88ecf75fec0f06",
  "source_weight_sha256": "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b"
}
```
