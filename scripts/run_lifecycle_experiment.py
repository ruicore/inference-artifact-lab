"""Run the independent Phase 2 owned-process experiment from a checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_artifact_lab.lifecycle_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
