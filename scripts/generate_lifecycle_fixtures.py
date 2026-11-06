"""Generate independently designed lifecycle fixtures; binaries remain local."""

import argparse
from pathlib import Path

from inference_artifact_lab.lifecycle_fixtures import generate_fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/lifecycle"))
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--public-inputs", type=Path)
    parser.add_argument("--public-reference", type=Path)
    args = parser.parse_args()
    result = generate_fixtures(args.output, public_manifest=args.public_manifest,
                               public_inputs=args.public_inputs, public_reference=args.public_reference)
    print(f"lifecycle fixtures generated: {result.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
