---
audience: codex
document_role: validation_record
phase_id: PH1
stage_id: PH1-ST01
status: local_pass
---

# Stage 01 validation record

Commands:

```text
uv run --extra test pytest -q
uv build
git diff --check
```

Observed result: 13 tests passed, package build succeeded, and the diff check was
clean. The evidence covers manifest parsing, unique tensor names, artifact
existence/size/SHA-256, contract comparison, environment mismatch handling, and
release-status aggregation.

Stage 01 does not claim ONNX graph execution, TensorRT engine compatibility, or
runtime numerical equivalence. Those remain Stage 02 and Stage 03 evidence.
