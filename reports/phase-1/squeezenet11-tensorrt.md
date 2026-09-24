# Model Release Gate report

- Status: **not_verified**
- Schema: `1`
- Manifest digest: `63c04235b47b3408832e0554fcf4fef48a8c00ba7bd1a70205d6804c24d0ea74`
- Artifact: `artifacts\squeezenet1.1-fp32.engine`
- Runtime profile: `tensorrt-gpu`
- Acceptance role: `optional_preview`

## Checks

| Check | Status | Message |
|---|---|---|
| `artifact.exists` | **pass** | artifact file exists |
| `artifact.size` | **pass** | artifact size matches manifest |
| `artifact.sha256` | **pass** | artifact SHA-256 matches manifest |
| `model.source_pin` | **pass** | public source identity and digest are declared; weight bytes are verified during export |
| `contract.inputs` | **pass** | artifact inputs match manifest |
| `contract.outputs` | **pass** | artifact outputs match manifest |
| `contract.optimization_profiles` | **pass** | engine optimization profile bounds match manifest |
| `artifact.format` | **pass** | observed artifact format matches manifest |
| `environment.compatibility` | **pass** | observed environment satisfies manifest |
| `fixture.binding` | **pass** | fixture and reference output match pinned digests |
| `runtime.equivalence` | **pass** | target output is within declared tolerance |
| `benchmark.workload` | **not_verified** | peak memory is not observed for the declared benchmark scope |
| `runtime.provenance` | **not_verified** | supplied runtime observations are not independently bound to fixture and reference bytes |

## Limitations

- TensorRT is an independent optional preview for this declared GPU/platform; its missing evidence does not block CPU-only Phase 1 acceptance, and CPU success does not validate this profile.
- The portable composer receives container output as supplied JSON; it cannot independently replay the adapter call, so runtime provenance remains not_verified despite the local observed run.
- trtexec timings are device-only and do not include an observed peak GPU memory value; AC-09 is not verified for a TensorRT performance claim.
- The pinned engine is an ignored local binary; a fresh checkout cannot verify this TensorRT decision without the exact pinned bytes.
- The container digest in the runtime payload is launcher-supplied; host-side image digest inspection is a separate observation.

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
