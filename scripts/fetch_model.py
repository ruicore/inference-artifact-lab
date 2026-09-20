"""Fetch the pinned public Phase 1 model and verify its digest."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = "https://github.com/onnx/models/raw/b1eeaa1ac722dcc1cd1a8284bde34393dab61c3d/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
SHA256 = "1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/squeezenet1.1-7.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"digest mismatch: expected {SHA256}, observed {digest}")
    args.output.write_bytes(data)
    print(f"wrote {args.output} ({len(data)} bytes; sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
