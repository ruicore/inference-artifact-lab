"""Inference Artifact Lab package."""

from .gate import canonical_manifest_digest, current_environment, load_manifest, run_gate, run_gate_from_file, sha256_file
from .adapters import OnnxRuntimeAdapter, RuntimeUnavailable, TensorRTAdapter
from .models import ArtifactSpec, CheckResult, GateReport, GateStatus, Manifest, ManifestError, TensorSpec
from .release import run_release_gate
from .runtime import BenchmarkResult, benchmark, compare_outputs

__all__ = [
    "__version__", "ArtifactSpec", "CheckResult", "GateReport", "GateStatus", "Manifest", "ManifestError", "TensorSpec",
    "canonical_manifest_digest", "current_environment", "load_manifest", "run_gate", "run_gate_from_file", "run_release_gate", "sha256_file",
    "BenchmarkResult", "benchmark", "compare_outputs", "OnnxRuntimeAdapter", "TensorRTAdapter", "RuntimeUnavailable",
]
__version__ = "0.1.0.dev0"
