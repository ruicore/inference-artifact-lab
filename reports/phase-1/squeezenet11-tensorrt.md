# Model Release Gate report

- Status: **pass**
- Schema: `1`
- Manifest digest: `e5a33c0b09061afb6c54001008d97fa93f3ada9496c8337e4a720bed0dcf9dbe`
- Artifact: `artifacts\squeezenet1.1-fp32.engine`

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
| `benchmark.workload` | **pass** | benchmark workload evidence supplied by the declared runtime harness |

## Limitations

- None recorded.

## Evidence

```json
{
  "container_runtime": {
    "cuda_binding": "cuda-python",
    "machine": "x86_64",
    "optimization_profiles": [
      {
        "index": 0,
        "inputs": {
          "data": {
            "max": [
              1,
              3,
              224,
              224
            ],
            "min": [
              1,
              3,
              224,
              224
            ],
            "opt": [
              1,
              3,
              224,
              224
            ]
          }
        }
      }
    ],
    "python": "3.12.3",
    "runtime": "tensorrt",
    "system": "Linux",
    "tensorrt_version": "10.8.0.43"
  },
  "fixture_sha256": "3161e1f639bd8c78b14164b87c3fb0be8aeb821aa715fcc0cab2d93353242d89",
  "reference_output_sha256": "9b9e35c2d6a1191685178087ce91bd352fadcebc97af24cb7b88ecf75fec0f06"
}
```
