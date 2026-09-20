"""Download the pinned public torchvision weights and export the ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


EXPECTED_WEIGHTS_SHA256 = "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/squeezenet1.1-torchvision.onnx"))
    args = parser.parse_args()
    weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
    cache_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    model = squeezenet1_1(weights=weights).eval()
    observed = hashlib.sha256(cache_file.read_bytes()).hexdigest()
    if observed != EXPECTED_WEIGHTS_SHA256:
        raise SystemExit(f"public weights digest mismatch: expected {EXPECTED_WEIGHTS_SHA256}, observed {observed}")
    example = torch.rand((1, 3, 224, 224), generator=torch.Generator().manual_seed(20260920))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, example, args.output, input_names=["data"], output_names=["output"], opset_version=18, dynamo=False)
    print(f"wrote {args.output} sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
