"""Owned-process CPU lifecycle measurements, separate from the Phase 1 gate.

The parent owns process exit and RSS observations. Child results establish
logical ownership and output correctness, never native-memory reclamation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


STATUSES = {"pass", "fail", "blocked", "not_verified"}
POLICIES = ("release_on_idle", "retain_idle")
LIMITATIONS = [
    "Sampled maximum RSS is a lower bound on process peak RSS, including runtime, fixtures and allocator overhead.",
    "Adapter disposal releases runner references; it does not prove native memory returned to the OS.",
    "Results apply only to the declared artifacts, fixtures, CPU runtime and environment.",
    "Synchronous native inference is allowed to finish; cooperative stop does not interrupt it.",
    "Deterministic interleaving and injected-failure tests and clean-checkout reproduction require separate evidence.",
]


class ExperimentInputError(ValueError):
    """Invalid experiment input at the public boundary."""


def _object(value: Any, fields: set[str], required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) - fields or required - set(value):
        raise ExperimentInputError(f"invalid {label} fields")
    return value


def _positive(value: Any, label: str, *, integer: bool = False, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExperimentInputError(f"{label} must be finite numeric")
    if value < 0 or (not zero and value == 0) or (integer and not isinstance(value, int)):
        raise ExperimentInputError(f"invalid {label}")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ExperimentInputError("expected lowercase SHA-256")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ExperimentInputError("nonfinite JSON")))


def _tensor(value: Any) -> dict:
    value = _object(value, {"dtype", "shape", "values"}, {"dtype", "shape", "values"}, "tensor")
    if value["dtype"] not in {"float32", "float64", "int32", "int64", "bool"}:
        raise ExperimentInputError("unsupported fixture dtype")
    if not isinstance(value["shape"], list) or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in value["shape"]):
        raise ExperimentInputError("fixture shape must contain positive dimensions")
    if not isinstance(value["values"], list) or len(value["values"]) != math.prod(value["shape"]):
        raise ExperimentInputError("fixture values do not match shape")
    for item in value["values"]:
        if value["dtype"] == "bool":
            if not isinstance(item, bool):
                raise ExperimentInputError("boolean tensor requires booleans")
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ExperimentInputError("fixture values must be finite numbers")
        elif value["dtype"].startswith("int") and not isinstance(item, int):
            raise ExperimentInputError("integer tensor requires integers")
    return value


def _fixture(value: Any) -> dict:
    value = _object(value, {"cases"}, {"cases"}, "fixture")
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise ExperimentInputError("fixture must have cases")
    names = set()
    for case in value["cases"]:
        _object(case, {"name", "inputs", "expected_outputs", "rtol", "atol"}, {"name", "inputs", "expected_outputs", "rtol", "atol"}, "fixture case")
        if not isinstance(case["name"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", case["name"]) or case["name"] in names:
            raise ExperimentInputError("fixture case names must be unique")
        names.add(case["name"])
        for field in ("inputs", "expected_outputs"):
            if not isinstance(case[field], dict) or not case[field] or any(not isinstance(k, str) or not k for k in case[field]):
                raise ExperimentInputError("fixture inputs and outputs must be named")
            for tensor in case[field].values():
                _tensor(tensor)
        _positive(case["rtol"], "rtol", zero=True)
        _positive(case["atol"], "atol", zero=True)
    return value


def load_fixtures(path: Path) -> tuple[dict, dict]:
    """Validate manifest, artifact and fixture bytes before any runtime loading."""
    from .lifecycle import LoadSpec

    manifest = _object(_read_json(path), {"schema_version", "artifacts", "public_model_gate"}, {"schema_version", "artifacts"}, "fixture manifest")
    if manifest["schema_version"] != "lifecycle-fixtures/v1":
        raise ExperimentInputError("unsupported fixture manifest version")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) - {"A1", "A2", "B", "M"} or not {"A1", "A2", "B"} <= set(artifacts):
        raise ExperimentInputError("A1, A2 and B artifacts are required; only M is optional")
    resolved, public = {}, {}
    for alias, entry in artifacts.items():
        _object(entry, {"load_spec", "fixture_path", "fixture_sha256"}, {"load_spec", "fixture_path", "fixture_sha256"}, "artifact entry")
        spec = LoadSpec.from_dict(entry["load_spec"], base_dir=path.parent)
        spec.verify_artifact()
        if not re.fullmatch(r"[A-Za-z0-9_.+/-]{1,80}", spec.adapter_version) or not re.fullmatch(r"[A-Za-z0-9_.+-]{1,80}", spec.runtime_version):
            raise ExperimentInputError("runtime/adapter versions must be portable identifiers")
        if not isinstance(entry["fixture_path"], str) or not entry["fixture_path"]:
            raise ExperimentInputError("fixture_path must be a nonempty string")
        fixture_path = path.parent / entry["fixture_path"]
        if _hash(fixture_path) != _digest(entry["fixture_sha256"]):
            raise ExperimentInputError("fixture digest mismatch")
        fixture = _fixture(_read_json(fixture_path))
        resolved[alias] = {"load_spec": {**entry["load_spec"], "artifact_path": str(spec.artifact_path)}, "fixture": fixture}
        public[alias] = {
            "artifact_sha256": spec.artifact_sha256,
            "artifact_size_bytes": spec.artifact_path.stat().st_size,
            "fixture_sha256": entry["fixture_sha256"],
            "fixture_size_bytes": fixture_path.stat().st_size,
            "effective_key": spec.identity.key,
            "effective_options": spec.identity.to_dict(),
            "case_count": len(fixture["cases"]),
        }
    gate = manifest.get("public_model_gate")
    if gate is not None:
        fields = {"status", "manifest_path", "manifest_sha256", "report_path", "report_sha256", "artifact_sha256", "input_sha256", "reference_output_sha256"}
        _object(gate, fields, fields, "public model gate")
        if gate["status"] != "pass" or "M" not in artifacts:
            raise ExperimentInputError("public model gate must pass and bind M")
        for field in ("manifest_sha256", "report_sha256", "artifact_sha256", "input_sha256", "reference_output_sha256"):
            _digest(gate[field])
        for field in ("manifest_path", "report_path"):
            if not isinstance(gate[field], str) or not gate[field]:
                raise ExperimentInputError("invalid public gate locator")
        manifest_path, report_path = path.parent / gate["manifest_path"], path.parent / gate["report_path"]
        if _hash(manifest_path) != gate["manifest_sha256"] or _hash(report_path) != gate["report_sha256"]:
            raise ExperimentInputError("public gate evidence digest mismatch")
        phase1_manifest, phase1_report = _read_json(manifest_path), _read_json(report_path)
        from .gate import canonical_manifest_digest, load_manifest
        if phase1_report.get("manifest_digest") != canonical_manifest_digest(load_manifest(manifest_path)):
            raise ExperimentInputError("public gate report manifest binding mismatch")
        if phase1_report.get("status") != "pass" or not phase1_report.get("checks") or any(c.get("status") != "pass" for c in phase1_report["checks"]):
            raise ExperimentInputError("public gate report did not pass")
        required_checks = {"artifact.exists", "artifact.sha256", "fixture.binding", "runtime.equivalence", "runtime.provenance", "contract.inputs", "contract.outputs", "environment.compatibility"}
        if not required_checks <= {c.get("id") for c in phase1_report["checks"]}:
            raise ExperimentInputError("public gate report missing required checks")
        if phase1_manifest.get("runtime", {}).get("provider") != "CPUExecutionProvider":
            raise ExperimentInputError("public gate must declare CPU provider")
        artifact_sha = public["M"]["artifact_sha256"]
        if any(value != artifact_sha for value in (gate["artifact_sha256"], phase1_manifest.get("artifact", {}).get("sha256"), phase1_report.get("artifact", {}).get("sha256"))):
            raise ExperimentInputError("public gate artifact binding mismatch")
        for field in ("input_sha256", "reference_output_sha256"):
            if phase1_manifest.get("fixture", {}).get(field) != gate[field]:
                raise ExperimentInputError("public gate fixture binding mismatch")
        # Reconstruct the pinned NPY serialization from the exact JSON tensors.
        # This binds lifecycle expected values to Phase 1's reference bytes.
        import io
        import numpy as np
        public_cases = resolved["M"]["fixture"]["cases"]
        if len(public_cases) != 1:
            raise ExperimentInputError("public fixture must have exactly one reference case")
        for field, tensors in (("input_sha256", public_cases[0]["inputs"]), ("reference_output_sha256", public_cases[0]["expected_outputs"])):
            if len(tensors) != 1:
                raise ExperimentInputError("public fixture expects one input and output")
            tensor = next(iter(tensors.values()))
            buffer = io.BytesIO()
            np.save(buffer, np.array(tensor["values"], dtype=tensor["dtype"]).reshape(tensor["shape"]), allow_pickle=False)
            if hashlib.sha256(buffer.getvalue()).hexdigest() != gate[field]:
                raise ExperimentInputError("public JSON values differ from pinned NPY bytes")
        gate = {key: value for key, value in gate.items() if key not in {"manifest_path", "report_path"}}
    return resolved, {"manifest_sha256": _hash(path), "artifacts": public, "public_model_gate": gate}


def _environment() -> dict:
    dependencies = {}
    for name in ("numpy", "onnxruntime", "onnx", "psutil", "jsonschema", "inference-artifact-lab"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"python": platform.python_version(), "system": platform.system(), "release": platform.release(), "machine": platform.machine(), "logical_cpus": os.cpu_count(), "dependencies": dependencies}


def _revision() -> dict:
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or (root / "src/inference_artifact_lab/lifecycle_experiment.py").resolve() != Path(__file__).resolve():
        return {"revision": None, "dirty": None, "source": "unavailable"}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "source": "unavailable"}
    return {"revision": revision, "dirty": dirty, "source": "git"}


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"count": len(values), "median": statistics.median(values) if values else None, "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if values else None, "maximum": max(values) if values else None}


def _worker_summary(timings: list[dict], loads: list[dict], events: list[dict]) -> dict:
    return {
        "latency_ms": {stage: _summary([t["milliseconds"] for t in timings if t["stage"] == stage]) for stage in sorted({t["stage"] for t in timings})},
        "load_count": len(loads),
        "eviction_count": sum(e["event"] == "eviction_started" for e in events),
        "maximum_active_handles": max((e["active_handles"] for e in events), default=0),
        "maximum_admitted_instances": max((e["admitted_instances"] for e in events), default=0),
    }


def _worker(request: dict) -> dict:
    """Child boundary: materialize arrays before the baseline, then own a pool."""
    _object(request, {"artifacts", "settings", "policy", "origin"}, {"artifacts", "settings", "policy", "origin"}, "worker request")
    settings = validate_settings(request["settings"])
    if request["policy"] not in POLICIES:
        raise ExperimentInputError("unknown worker policy")
    _positive(request["origin"], "origin", zero=True)
    from .lifecycle import ArtifactDigestError, CapacityError, LifecyclePool, LoadSpec
    from .lifecycle_onnx import OnnxLifecycleAdapter
    import numpy as np
    # Import the optional native runtime before baseline and timed construction.
    __import__("onnxruntime")

    specs, cases = {}, {}
    if not isinstance(request["artifacts"], dict) or not {"A1", "A2", "B"} <= set(request["artifacts"]) or set(request["artifacts"]) - {"A1", "A2", "B", "M"}:
        raise ExperimentInputError("invalid worker artifacts")
    for alias, entry in request["artifacts"].items():
        _object(entry, {"load_spec", "fixture"}, {"load_spec", "fixture"}, "worker artifact")
        specs[alias] = LoadSpec.from_dict(entry["load_spec"])
        specs[alias].verify_artifact()
        cases[alias] = []
        for case in _fixture(entry["fixture"])["cases"]:
            cases[alias].append({
                "name": case["name"],
                "inputs": {name: np.array(t["values"], dtype=t["dtype"]).reshape(t["shape"]) for name, t in case["inputs"].items()},
                "expected": {name: np.array(t["values"], dtype=t["dtype"]).reshape(t["shape"]) for name, t in case["expected_outputs"].items()},
                "rtol": case["rtol"], "atol": case["atol"],
            })
    ledger, operations, correctness, timings, pool_events, shutdowns = [], [], [], [], [], []
    handles = []

    def loader(spec):
        start = time.perf_counter()
        adapter = OnnxLifecycleAdapter(spec)
        ledger.append({"key": spec.identity.key, "load_ms": (time.perf_counter() - start) * 1000})
        return adapter

    pool = LifecyclePool(loader, max_instances=2, policy=request["policy"])

    def reset_pool():
        nonlocal pool
        pool_index = len(shutdowns)
        shutdowns.append(pool.close())
        pool_events.extend({**event, "pool_index": pool_index} for event in pool.events)
        pool = LifecyclePool(loader, max_instances=2, policy=request["policy"])

    def mark(stage, **details):
        operations.append({"stage": stage, "elapsed_seconds": time.perf_counter() - request["origin"], "snapshot": pool.snapshot(), "pool_index": len(shutdowns), **details})

    def acquire(alias, stage):
        before = len(ledger)
        start = time.perf_counter()
        handle = pool.acquire(specs[alias])
        handles.append(handle)
        timings.append({"stage": stage, "kind": "acquire", "alias": alias, "milliseconds": (time.perf_counter() - start) * 1000, "loaded": len(ledger) > before})
        mark(stage, alias=alias, generation=handle.generation, key=handle.identity.key)
        return handle

    def infer(handle, alias, stage):
        for case in cases[alias]:
            start = time.perf_counter()
            actual = handle.run(case["inputs"])
            timings.append({"stage": stage, "kind": "inference", "alias": alias, "milliseconds": (time.perf_counter() - start) * 1000, "loaded": False})
            matched = set(actual) == set(case["expected"])
            for name, expected in case["expected"].items():
                if name not in actual:
                    matched = False
                    continue
                output = np.asarray(actual[name])
                matched = matched and output.shape == expected.shape and bool(np.isfinite(output).all()) and bool(np.allclose(output, expected, rtol=case["rtol"], atol=case["atol"]))
            correctness.append({"alias": alias, "stage": stage, "case": case["name"], "generation": handle.generation, "key": handle.identity.key, "status": "pass" if matched else "fail"})
        mark(stage, alias=alias, generation=handle.generation, key=handle.identity.key)

    failure = None
    observations = {}
    mark("baseline_start")
    time.sleep(settings["baseline_seconds"])
    mark("baseline_end")
    try:
        with tempfile.TemporaryDirectory(prefix="lifecycle-locator-") as directory:
            locator = Path(directory) / "same-artifact.onnx"
            locator.write_bytes(specs["A1"].artifact_path.read_bytes())
            original = acquire("A1", "identity_original")
            count = len(ledger)
            equivalent = pool.acquire(replace(specs["A1"], artifact_path=locator))
            handles.append(equivalent)
            observations["locator_independent_sharing"] = equivalent.generation == original.generation and len(ledger) == count
            equivalent.release()
            try:
                pool.acquire(replace(specs["A1"], artifact_sha256="0" * 64))
            except ArtifactDigestError:
                observations["digest_rejected_before_loader"] = len(ledger) == count
            else:
                observations["digest_rejected_before_loader"] = False
            changed_spec = LoadSpec.from_dict({**request["artifacts"]["A1"]["load_spec"], "session_options": {"intra_op_num_threads": 2, "inter_op_num_threads": 1}})
            changed = pool.acquire(changed_spec)
            handles.append(changed)
            observations["changed_options_distinct"] = changed.identity.key != original.identity.key and changed.generation != original.generation and len(ledger) == count + 1
            changed.release()
            original.release()
        observations["changed_bytes_distinct"] = specs["A1"].identity.key != specs["A2"].identity.key
        reset_pool()
        first = acquire("A1", "shared_first")
        second = acquire("A1", "shared_second")
        observations["shared_generation"] = first.generation == second.generation
        infer(first, "A1", "shared_before_release")
        first.release()
        first.release()
        try:
            first.run(cases["A1"][0]["inputs"])
        except Exception as exc:
            from .lifecycle import HandleReleasedError
            observations["released_handle_rejected"] = isinstance(exc, HandleReleasedError)
        else:
            observations["released_handle_rejected"] = False
        infer(second, "A1", "shared_after_release")
        second.release()
        old = acquire("A1", "revision_old")
        new = acquire("A2", "revision_new")
        observations["revision_identity_changed"] = old.identity.key != new.identity.key
        infer(old, "A1", "revision_old_output")
        infer(new, "A2", "revision_new_output")
        old.release()
        new.release()
        reset_pool()
        idle = acquire("A1", "idle_first")
        idle.release()
        active = acquire("B", "idle_active")
        newest = acquire("A2", "idle_pressure")
        evictions = [e for e in pool.events if e["event"] == "eviction_started"]
        observations["idle_eviction_correct"] = (len(evictions) == 1 and evictions[0]["instance_key"] == specs["A1"].identity.key) if request["policy"] == "retain_idle" else not evictions
        infer(active, "B", "active_survives_pressure")
        newest.release()
        active.release()
        reset_pool()
        one = acquire("A1", "busy_one")
        two = acquire("B", "busy_two")
        before = len(ledger)
        try:
            pool.acquire(specs["A2"])
        except CapacityError:
            observations["capacity_before_loader"] = len(ledger) == before
        else:
            observations["capacity_before_loader"] = False
        one.release()
        two.release()
        reset_pool()
        target = "M" if "M" in specs else "A1"
        held = acquire(target, "cold_load")
        shared = acquire(target, "public_shared_acquire")
        infer(shared, target, "public_shared_before_release")
        shared.release()
        infer(held, target, "public_shared_after_release")
        for _ in range(settings["warmup"]):
            infer(held, target, "warmup")
        # Warm samples hold the same handle, independent of retention policy.
        for _ in range(settings["cycles"]):
            infer(held, target, "warm_inference")
        held.release()
        for cycle in range(settings["cycles"]):
            current = acquire(target, "cycle_acquire")
            infer(current, target, "cycle_inference")
            current.release()
            mark("cycle_release", cycle=cycle)
        for alias in ("A1", "B"):
            pressure = acquire(alias, "reload_pressure")
            pressure.release()
        reloaded = acquire(target, "reload_acquire")
        infer(reloaded, target, "reload_inference")
        reloaded.release()
        pool.stop()
        observations["stop_requested"] = True
    except Exception as exc:
        # Child boundary translates errors into portable codes, never local paths.
        failure = type(exc).__name__
    finally:
        release_errors = []
        for handle in handles:
            try:
                handle.release()
            except Exception as exc:
                release_errors.append(type(exc).__name__)
        mark("teardown_start")
        cleanup = pool.close()
        shutdowns.append(cleanup.copy())
        pool_events.extend({**event, "pool_index": len(shutdowns) - 1} for event in pool.events)
        cleanup["release_errors"] = release_errors
        cleanup["prior_shutdowns"] = shutdowns
        mark("post_teardown_start")
        time.sleep(settings["post_teardown_seconds"])
        mark("post_teardown_end")
    summary = _worker_summary(timings, ledger, pool_events)
    return {"schema_version": "lifecycle-worker/v1", "policy": request["policy"], "runtime": "onnxruntime", "error": failure, "observations": observations, "correctness": correctness, "timings": timings, "operations": operations, "pool_events": pool_events, "loads": ledger, "cleanup": cleanup, "summary": summary}


def validate_settings(settings: Any) -> dict:
    fields = {"repetitions", "cycles", "warmup", "sample_interval_ms", "baseline_seconds", "post_teardown_seconds", "timeout_seconds"}
    _object(settings, fields, fields, "settings")
    for name in fields:
        _positive(settings[name], name, integer=name in {"repetitions", "cycles", "warmup"})
    return settings


def _validate_worker(value: Any, policy: str) -> dict:
    fields = {"schema_version", "policy", "runtime", "error", "observations", "correctness", "timings", "operations", "pool_events", "loads", "cleanup", "summary"}
    _object(value, fields, fields, "worker result")
    if value["schema_version"] != "lifecycle-worker/v1" or value["policy"] != policy or value["runtime"] != "onnxruntime":
        raise ExperimentInputError("worker identity mismatch")
    if value["error"] is not None and (not isinstance(value["error"], str) or not value["error"].isidentifier()):
        raise ExperimentInputError("invalid worker error code")
    if not isinstance(value["observations"], dict) or any(not isinstance(v, bool) for v in value["observations"].values()):
        raise ExperimentInputError("invalid worker observations")
    for name in ("correctness", "timings", "operations", "pool_events", "loads"):
        if not isinstance(value[name], list):
            raise ExperimentInputError(f"invalid worker {name}")
    for item in value["correctness"]:
        _object(item, {"alias", "stage", "case", "generation", "key", "status"}, {"alias", "stage", "case", "generation", "key", "status"}, "correctness")
        if item["status"] not in {"pass", "fail"} or item["alias"] not in {"A1", "A2", "B", "M"}:
            raise ExperimentInputError("invalid correctness status or alias")
        _digest(item["key"])
        if not isinstance(item["stage"], str) or not isinstance(item["case"], str):
            raise ExperimentInputError("invalid correctness labels")
        _positive(item["generation"], "generation", integer=True)
    for item in value["timings"]:
        _object(item, {"stage", "kind", "alias", "milliseconds", "loaded"}, {"stage", "kind", "alias", "milliseconds", "loaded"}, "timing")
        _positive(item["milliseconds"], "timing", zero=True)
        if item["kind"] not in {"acquire", "inference"} or not isinstance(item["loaded"], bool):
            raise ExperimentInputError("invalid timing kind")
    previous = -1.0
    for item in value["operations"]:
        if not isinstance(item, dict) or not {"stage", "elapsed_seconds", "snapshot"} <= set(item):
            raise ExperimentInputError("invalid operation marker")
        _positive(item["elapsed_seconds"], "operation timestamp", zero=True)
        if item["elapsed_seconds"] < previous or not isinstance(item["snapshot"], dict):
            raise ExperimentInputError("unordered operation marker")
        previous = item["elapsed_seconds"]
    cleanup = value["cleanup"]
    if not isinstance(cleanup, dict) or not {"complete", "outstanding_handles", "loading_instances", "cleanup_errors", "release_errors"} <= set(cleanup):
        raise ExperimentInputError("missing cleanup evidence")
    if not isinstance(cleanup["complete"], bool) or any(isinstance(cleanup[k], bool) or not isinstance(cleanup[k], int) or cleanup[k] < 0 for k in ("outstanding_handles", "loading_instances")):
        raise ExperimentInputError("invalid cleanup counts")
    if not isinstance(cleanup["cleanup_errors"], list) or not isinstance(cleanup["release_errors"], list):
        raise ExperimentInputError("invalid cleanup errors")
    if not isinstance(cleanup.get("prior_shutdowns"), list) or not cleanup["prior_shutdowns"]:
        raise ExperimentInputError("missing prior pool shutdown evidence")
    for shutdown in cleanup["prior_shutdowns"]:
        if not isinstance(shutdown, dict) or not {"complete", "outstanding_handles", "loading_instances", "cleanup_errors"} <= set(shutdown) or not isinstance(shutdown["complete"], bool):
            raise ExperimentInputError("invalid prior pool shutdown")
    if value["error"] is None and (not value["correctness"] or not value["timings"] or not value["operations"] or not value["loads"] or not value["pool_events"]):
        raise ExperimentInputError("successful worker omitted required evidence")
    summary = value["summary"]
    if not isinstance(summary, dict) or not {"latency_ms", "load_count", "eviction_count", "maximum_active_handles", "maximum_admitted_instances"} <= set(summary):
        raise ExperimentInputError("missing worker summary")
    for field in ("load_count", "eviction_count", "maximum_active_handles", "maximum_admitted_instances"):
        _positive(summary[field], field, integer=True, zero=True)
    for load in value["loads"]:
        _object(load, {"key", "load_ms"}, {"key", "load_ms"}, "load observation")
        _digest(load["key"])
        _positive(load["load_ms"], "load_ms", zero=True)
    for event in value["pool_events"]:
        if not isinstance(event, dict) or not {"sequence", "event", "active_handles", "admitted_instances"} <= set(event):
            raise ExperimentInputError("missing pool event counts")
        for field in ("active_handles", "admitted_instances"):
            _positive(event[field], field, integer=True, zero=True)
    if summary != _worker_summary(value["timings"], value["loads"], value["pool_events"]):
        raise ExperimentInputError("worker summaries contradict raw observations")
    return value


def _coverage(samples: list[dict], operations: list[dict], settings: dict) -> dict:
    windows = {}
    for name, start_stage, end_stage, required in (
        ("baseline", "baseline_start", "baseline_end", settings["baseline_seconds"]),
        ("post_teardown", "post_teardown_start", "post_teardown_end", settings["post_teardown_seconds"]),
    ):
        starts = [o["elapsed_seconds"] for o in operations if o["stage"] == start_stage]
        ends = [o["elapsed_seconds"] for o in operations if o["stage"] == end_stage]
        if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
            windows[name] = {"status": "not_verified", "reason": "missing_window_markers", "sample_count": 0}
            continue
        observed = [s for s in samples if starts[0] <= s["elapsed_seconds"] <= ends[0] and s["rss_bytes"] is not None]
        times = [starts[0], *(s["elapsed_seconds"] for s in observed), ends[0]]
        maximum_gap = max(b - a for a, b in zip(times, times[1:]))
        complete = len(observed) >= 2 and ends[0] - starts[0] >= required and maximum_gap <= max(0.25, settings["sample_interval_ms"] / 1000 * 4)
        windows[name] = {"status": "pass" if complete else "not_verified", "start_seconds": starts[0], "end_seconds": ends[0], "duration_seconds": ends[0] - starts[0], "sample_count": len(observed), "maximum_gap_seconds": maximum_gap, "rss_bytes": _summary([s["rss_bytes"] for s in observed])}
    return windows


def _rss_summaries(samples: list[dict], operations: list[dict], elapsed: float) -> dict:
    rss_values = [s["rss_bytes"] for s in samples if s["rss_bytes"] is not None]
    gaps = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])]
    stage_coverage = []
    for index, operation in enumerate(operations):
        end = operations[index + 1]["elapsed_seconds"] if index + 1 < len(operations) else elapsed
        observed = [s["rss_bytes"] for s in samples if operation["elapsed_seconds"] <= s["elapsed_seconds"] < end and s["rss_bytes"] is not None]
        stage_coverage.append({"stage": operation["stage"], "start_seconds": operation["elapsed_seconds"], "end_seconds": end, "sample_count": len(observed), "rss_bytes": _summary(observed), "marker_active_handles": operation["snapshot"]["active_handles"], "marker_admitted_instances": operation["snapshot"]["admitted_instances"]})
    return {"maximum_sample_gap_seconds": max(gaps) if gaps else None, "stage_coverage": stage_coverage, "summary_bytes": _summary(rss_values), "missing_sample_count": len(samples) - len(rss_values)}


def run_repetition(artifacts: dict, settings: dict, policy: str, *, worker_command: list[str] | None = None) -> dict:
    """Launch and observe one task-owned child; never trust its exit assertion."""
    validate_settings(settings)
    if policy not in POLICIES:
        raise ExperimentInputError("unknown policy")
    try:
        import psutil
    except ImportError:
        psutil = None
    origin = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lifecycle-") as directory:
        request_path, result_path = Path(directory) / "request.json", Path(directory) / "result.json"
        handshake_path = request_path.with_suffix(".worker.json")
        request_path.write_text(json.dumps({"artifacts": artifacts, "settings": settings, "policy": policy, "origin": origin}, allow_nan=False), encoding="utf-8")
        command = worker_command or [sys.executable, "-m", "inference_artifact_lab.lifecycle_experiment", "--worker"]
        environment = os.environ.copy()
        # Editable source checkout and installed-package execution use the same worker module.
        environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")])
        process = subprocess.Popen([*command, str(request_path), str(result_path)], env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        samples, errors = [], []
        timed_out = False
        monitored, worker_created = None, None
        worker_identity_verified, worker_exit_observed = False, False
        worker_exit_code = None
        while True:
            elapsed = time.perf_counter() - origin
            launcher_running = process.poll() is None
            if monitored is None and psutil is not None and handshake_path.exists():
                try:
                    handshake = _read_json(handshake_path)
                    _object(handshake, {"pid"}, {"pid"}, "private worker handshake")
                    if isinstance(handshake["pid"], bool) or not isinstance(handshake["pid"], int) or handshake["pid"] <= 0:
                        raise ExperimentInputError("invalid private worker identity")
                    launcher = psutil.Process(process.pid)
                    owned = [launcher, *launcher.children(recursive=True)]
                    monitored = next((candidate for candidate in owned if candidate.pid == handshake["pid"]), None)
                    if monitored is not None:
                        worker_created = monitored.create_time()
                        worker_identity_verified = True
                except (psutil.Error, ExperimentInputError, ValueError):
                    errors.append("worker_identity_unavailable")
            worker_running = False
            if monitored is not None:
                try:
                    worker_running = monitored.is_running() and monitored.create_time() == worker_created and monitored.status() != psutil.STATUS_ZOMBIE
                except psutil.Error:
                    worker_running = False
                worker_exit_observed = not worker_running
                if worker_exit_observed and worker_exit_code is None:
                    try:
                        worker_exit_code = monitored.wait(timeout=0)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
            if not launcher_running and not worker_running:
                break
            rss = None
            if monitored is not None and worker_running:
                try:
                    rss = monitored.memory_info().rss
                except psutil.NoSuchProcess:
                    worker_exit_observed = True
                    continue
                except psutil.Error as exc:
                    errors.append(type(exc).__name__)
                samples.append({"elapsed_seconds": elapsed, "rss_bytes": rss})
            if elapsed >= settings["timeout_seconds"]:
                timed_out = True
                if monitored is not None and worker_running:
                    try:
                        monitored.kill()
                        worker_exit_code = monitored.wait(timeout=10)
                        worker_exit_observed = True
                    except psutil.NoSuchProcess:
                        worker_exit_observed = True
                if process.poll() is None:
                    process.kill()
                break
            time.sleep(min(settings["sample_interval_ms"] / 1000, max(0, settings["timeout_seconds"] - elapsed)))
        exit_code = process.wait(timeout=10)
        if monitored is not None and monitored.pid == process.pid:
            worker_exit_code = exit_code
        worker, protocol_error, startup_error = None, None, None
        if result_path.exists():
            try:
                result = _read_json(result_path)
                if isinstance(result, dict) and set(result) == {"startup_error"} and isinstance(result["startup_error"], str) and result["startup_error"].isidentifier():
                    startup_error = result["startup_error"]
                else:
                    worker = _validate_worker(result, policy)
            except (ValueError, KeyError, TypeError) as exc:
                protocol_error = type(exc).__name__
        else:
            protocol_error = "missing_worker_result"
    operations = worker["operations"] if worker else []
    elapsed = time.perf_counter() - origin
    return {"policy": policy, "parent_exit": {"observed": True, "exit_code": exit_code, "exit_code_scope": "owned_launcher", "worker_exit_code": worker_exit_code, "timed_out": timed_out, "elapsed_seconds": elapsed, "worker_identity_verified": worker_identity_verified, "worker_exit_observed": worker_exit_observed}, "protocol_error": protocol_error, "startup_error": startup_error, "worker": worker, "rss": {"collector": "psutil" if psutil else "unavailable", "collector_version": psutil.__version__ if psutil else None, "sample_interval_ms": settings["sample_interval_ms"], "samples": samples, "collection_errors": sorted(set(errors)), **_rss_summaries(samples, operations, elapsed), "windows": _coverage(samples, operations, settings)}}


def evaluate_checks(repetitions: list[dict], identity: dict, settings: dict) -> list[dict]:
    checks = []

    def add(number, status, message):
        checks.append({"id": f"PH2-AC{number:02}", "status": status, "message": message})

    successful = len(repetitions) == settings["repetitions"] * 2 and all(r["parent_exit"]["observed"] and r["parent_exit"]["exit_code"] == 0 and r["parent_exit"].get("worker_exit_code") in (0, None) and not r["parent_exit"]["timed_out"] and r["protocol_error"] is None and r["worker"] is not None and r["worker"]["error"] is None for r in repetitions)
    workers = [r["worker"] for r in repetitions if r["worker"] is not None]
    correctness = [c for w in workers for c in w["correctness"]]
    correct = successful and bool(correctness) and all(c["status"] == "pass" for c in correctness)
    identities = successful and all(all(w["observations"].get(name) for name in ("locator_independent_sharing", "digest_rejected_before_loader", "changed_options_distinct", "changed_bytes_distinct")) for w in workers)
    add(1, "pass" if identities else "fail", "Real loader ledger verifies equal bytes at different locators share; mismatched digest rejects before loading; changed bytes/options yield distinct keys.")
    add(2, "not_verified", "Real shared-handle outputs are recorded; barrier-controlled overlapping load evidence is separate.")
    released = successful and all(w["observations"].get("released_handle_rejected") and w["observations"].get("shared_generation") for w in workers)
    add(3, "pass" if released and correct else "fail", "Repeated release, released-handle rejection and surviving shared-handle outputs observed in real workers.")
    revisions = correct and all(w["observations"].get("revision_identity_changed") for w in workers)
    add(4, "pass" if revisions else "fail", "Original A1 and replacement A2 handles retain distinct keys and fixture-specific outputs.")
    capacity = correct and all(w["observations"].get("idle_eviction_correct") and w["observations"].get("capacity_before_loader") and w["summary"]["maximum_admitted_instances"] <= 2 for w in workers)
    add(5, "pass" if capacity else "fail", "Real idle-pressure events evict only A1, active B remains correct, and two active slots reject a third loader; loading-reservation interleaving is separately unit tested.")
    add(6, "not_verified", "Normal cooperative stop is recorded; controlled loader/inference failures require separate deterministic evidence.")
    clean = successful and all(r["parent_exit"]["worker_identity_verified"] and r["parent_exit"]["worker_exit_observed"] for r in repetitions) and all(w["cleanup"]["complete"] and w["cleanup"]["outstanding_handles"] == 0 and w["cleanup"]["loading_instances"] == 0 and not w["cleanup"]["cleanup_errors"] and not w["cleanup"]["release_errors"] and all(s["complete"] and s["outstanding_handles"] == 0 and s["loading_instances"] == 0 and not s["cleanup_errors"] for s in w["cleanup"]["prior_shutdowns"]) for w in workers)
    add(7, "pass" if clean else "fail", "Parent observed owned-worker exit; child teardown reports are checked for incomplete ownership and cleanup errors. Negative fixtures are separately tested.")
    gate = identity.get("public_model_gate")
    full_model = "M" in identity["artifacts"] and gate is not None and gate["status"] == "pass"
    needed_stages = {"shared_before_release", "shared_after_release", "revision_old_output", "revision_new_output", "active_survives_pressure", "public_shared_before_release", "public_shared_after_release", "warmup", "warm_inference", "cycle_inference", "reload_inference"}
    target = "M" if "M" in identity["artifacts"] else "A1"
    expected_runs = {
        "shared_before_release": ("A1", 1), "shared_after_release": ("A1", 1),
        "revision_old_output": ("A1", 1), "revision_new_output": ("A2", 1),
        "active_survives_pressure": ("B", 1),
        "public_shared_before_release": (target, 1), "public_shared_after_release": (target, 1),
        "warmup": (target, settings["warmup"]), "warm_inference": (target, settings["cycles"]),
        "cycle_inference": (target, settings["cycles"]), "reload_inference": (target, 1),
    }
    stages = all(needed_stages <= {c["stage"] for c in w["correctness"]} and all(
        sum(c["stage"] == stage and c["alias"] == alias for c in w["correctness"]) == count * identity["artifacts"][alias]["case_count"]
        for stage, (alias, count) in expected_runs.items()
    ) for w in workers)
    models = {c["alias"] for c in correctness}
    runtime_status = "pass" if correct and stages and {"A1", "A2", "B", "M"} <= models and full_model else "not_verified" if correct else "fail"
    add(8, runtime_status, "Real ONNX fixture comparisons cover generated revisions and the public model only when M and its Phase 1 gate evidence are present.")
    memory = successful and all(r["parent_exit"]["worker_identity_verified"] and r["parent_exit"]["worker_exit_observed"] and r["rss"]["missing_sample_count"] == 0 and not r["rss"]["collection_errors"] and all(w["status"] == "pass" for w in r["rss"]["windows"].values()) for r in repetitions)
    timing_complete = all(all(any(t["stage"] == stage for t in w["timings"]) for stage in ("cold_load", "warm_inference", "cycle_acquire", "cycle_inference", "reload_acquire", "reload_inference")) and sum(t["stage"] == "cycle_acquire" for t in w["timings"]) == settings["cycles"] and all(
        sum(t["stage"] == stage and t["alias"] == alias for t in w["timings"]) == count * identity["artifacts"][alias]["case_count"]
        for stage, (alias, count) in expected_runs.items()
    ) for w in workers)
    required_schedule = settings["repetitions"] >= 3 and settings["cycles"] >= 30 and settings["warmup"] >= 5 and settings["baseline_seconds"] >= 1 and settings["post_teardown_seconds"] >= 1 and settings["sample_interval_ms"] <= 50
    add(9, "pass" if memory and timing_complete and full_model and required_schedule else "not_verified" if successful else "fail", "Alternating policies, required repetition/cycle/warmup/window settings, raw latency/RSS samples and coverage; no performance or native-reclamation threshold.")
    add(10, "pass" if successful and memory and runtime_status == "pass" else "not_verified" if successful else "fail", "Separate lifecycle contract records check status, identities, revision, environment, trace and missing observations; schema validation is separate.")
    add(11, "not_verified", "This invocation cannot establish clean committed-checkout/fresh-environment reproduction; attach separate evidence.")
    unavailable = any(r.get("startup_error") in {"ImportError", "ModuleNotFoundError", "RuntimeUnavailable"} for r in repetitions)
    if unavailable:
        for check in checks:
            if check["id"] in {"PH2-AC01", "PH2-AC03", "PH2-AC04", "PH2-AC05", "PH2-AC07", "PH2-AC08", "PH2-AC09", "PH2-AC10"}:
                check.update(status="blocked", message="Required optional runtime dependency unavailable; owned-worker exit is recorded separately.")
    return checks


def run_experiment(fixture_path: Path, *, settings: dict | None = None) -> dict:
    settings = validate_settings(settings or {"repetitions": 3, "cycles": 30, "warmup": 5, "sample_interval_ms": 50, "baseline_seconds": 1, "post_teardown_seconds": 1, "timeout_seconds": 300})
    artifacts, identity = load_fixtures(fixture_path)
    scenario = {"id": "cpu-model-lifecycle/v1", "settings": settings, "max_instances": 2, "policy_order": [policy for _ in range(settings["repetitions"]) for policy in POLICIES], "operation_schedule": ["identity", "shared_use", "revision_switch", "idle_pressure", "busy_pressure", "cold_load", "public_shared_use", "warmup", "warm_inference", "public_model_retention", "public_model_reload", "teardown"], "artifact_keys": {alias: item["effective_key"] for alias, item in identity["artifacts"].items()}, "fixture_digests": {alias: item["fixture_sha256"] for alias, item in identity["artifacts"].items()}}
    repetitions = [run_repetition(artifacts, settings, policy) for policy in scenario["policy_order"]]
    checks = evaluate_checks(repetitions, identity, settings)
    statuses = {c["status"] for c in checks}
    status = next((s for s in ("fail", "blocked", "not_verified") if s in statuses), "pass")
    report = {"schema_version": "lifecycle-report/v1", "status": status, "scenario": scenario, "scenario_sha256": _canonical_digest(scenario), "identity": identity, "implementation": _revision(), "environment": _environment(), "measurement": {"clock": "time.perf_counter (monotonic)", "time_unit": "milliseconds", "rss_unit": "bytes", "quantile_method": "nearest-rank p95; standard median", "steady_samples_exclude": ["process startup", "runtime import", "fixture construction", "warmup", "artifact generation", "download"]}, "checks": checks, "repetitions": repetitions, "limitations": LIMITATIONS}
    validate_report(report)
    return report


def validate_report(report: dict) -> None:
    """Validate an actual report against the shipped contract, including in wheels."""
    import jsonschema

    source_schema = Path(__file__).resolve().parents[2] / "schemas/lifecycle-report.schema.json"
    installed_schema = Path(sys.prefix) / "share/inference-artifact-lab/lifecycle-report.schema.json"
    schema_path = source_schema if source_schema.is_file() else installed_schema
    jsonschema.Draft202012Validator(_read_json(schema_path)).validate(report)
    if report["scenario_sha256"] != _canonical_digest(report["scenario"]):
        raise ExperimentInputError("scenario digest mismatch")
    settings = validate_settings(report["scenario"]["settings"])
    if [r["policy"] for r in report["repetitions"]] != report["scenario"]["policy_order"]:
        raise ExperimentInputError("policy schedule mismatch")
    for alias, artifact in report["identity"]["artifacts"].items():
        if artifact["effective_key"] != _canonical_digest(artifact["effective_options"]) or artifact["effective_options"]["artifact_sha256"] != artifact["artifact_sha256"]:
            raise ExperimentInputError("effective identity digest mismatch")
        if report["scenario"]["artifact_keys"].get(alias) != artifact["effective_key"] or report["scenario"]["fixture_digests"].get(alias) != artifact["fixture_sha256"]:
            raise ExperimentInputError("scenario artifact binding mismatch")
    for repetition in report["repetitions"]:
        worker = repetition["worker"]
        if worker is not None:
            _validate_worker(worker, repetition["policy"])
            for comparison in worker["correctness"]:
                artifact = report["identity"]["artifacts"].get(comparison["alias"])
                if artifact is None or comparison["key"] != artifact["effective_key"]:
                    raise ExperimentInputError("correctness effective key contradicts artifact identity")
            for operation in worker["operations"]:
                if "alias" in operation and "key" in operation and operation["key"] != report["identity"]["artifacts"].get(operation["alias"], {}).get("effective_key"):
                    raise ExperimentInputError("operation effective key contradicts artifact identity")
        actual_windows = _coverage(repetition["rss"]["samples"], worker["operations"] if worker else [], settings)
        if repetition["rss"]["windows"] != actual_windows:
            raise ExperimentInputError("RSS window coverage contradicts raw samples")
        missing = sum(sample["rss_bytes"] is None for sample in repetition["rss"]["samples"])
        if repetition["rss"]["missing_sample_count"] != missing:
            raise ExperimentInputError("RSS missing count contradicts raw samples")
        coverage = repetition["rss"]["stage_coverage"]
        endpoint = coverage[-1]["end_seconds"] if coverage else repetition["parent_exit"]["elapsed_seconds"]
        if coverage and not coverage[-1]["start_seconds"] <= endpoint <= repetition["parent_exit"]["elapsed_seconds"]:
            raise ExperimentInputError("stage coverage endpoint exceeds observed exit")
        summaries = _rss_summaries(repetition["rss"]["samples"], worker["operations"] if worker else [], endpoint)
        if any(repetition["rss"][field] != value for field, value in summaries.items()):
            raise ExperimentInputError("RSS summaries contradict raw samples")
    expected_checks = evaluate_checks(report["repetitions"], report["identity"], settings)
    if [(c["id"], c["status"]) for c in report["checks"]] != [(c["id"], c["status"]) for c in expected_checks]:
        raise ExperimentInputError("check statuses contradict lifecycle evidence")
    statuses = {check["status"] for check in expected_checks}
    expected_status = next((status for status in ("fail", "blocked", "not_verified") if status in statuses), "pass")
    if report["status"] != expected_status:
        raise ExperimentInputError("aggregate status contradicts check evidence")


def render_report(report: dict) -> str:
    """Render the complete portable report without changing Phase 1 rendering."""
    if report.get("schema_version") != "lifecycle-report/v1" or report.get("status") not in STATUSES:
        raise ExperimentInputError("not a lifecycle-report/v1 report")
    validate_report(report)
    lines = ["# Model lifecycle experiment", "", f"- Status: **{report['status']}**", f"- Scenario SHA-256: `{report['scenario_sha256']}`", f"- Revision: `{report['implementation']['revision']}`; dirty: `{report['implementation']['dirty']}`", "", "## Checks", "", "| Case | Status | Evidence boundary |", "|---|---|---|"]
    for check in report["checks"]:
        if check["status"] not in STATUSES:
            raise ExperimentInputError("invalid report check status")
        message = check["message"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check['id']} | **{check['status']}** | {message} |")
    lines.extend(["", "## Repetition observations", "", "| Policy | Launcher exit | Worker exit code | Timeout | RSS samples | Sampled maximum bytes | Baseline | Post-teardown |", "|---|---|---|---|---|---|---|---|"])
    for rep in report["repetitions"]:
        rss, exit_observation = rep["rss"], rep["parent_exit"]
        worker_code = exit_observation.get("worker_exit_code")
        lines.append(f"| {rep['policy']} | {exit_observation['exit_code']} | {worker_code if worker_code is not None else 'unavailable'} | {exit_observation['timed_out']} | {rss['summary_bytes']['count']} | {rss['summary_bytes']['maximum']} | {rss['windows']['baseline']['status']} | {rss['windows']['post_teardown']['status']} |")
    lines.extend(["", "## Limitations", "", *(f"- {item}" for item in report["limitations"]), "", "## Complete evidence", "", "```json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False), "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        if len(argv) != 3:
            return 2
        result_path = Path(argv[2])
        # This private IPC value is verified against the owned launcher subtree;
        # it never enters a portable report. Windows venv launchers may proxy.
        handshake_path = Path(argv[1]).with_suffix(".worker.json")
        handshake_temporary = handshake_path.with_suffix(".tmp")
        handshake_temporary.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        handshake_temporary.replace(handshake_path)
        try:
            result = _worker(_read_json(Path(argv[1])))
        except Exception as exc:
            result_path.write_text(json.dumps({"startup_error": type(exc).__name__}), encoding="utf-8")
            return 1
        result_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        return 1 if result["error"] else 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-interval-ms", type=float, default=50)
    parser.add_argument("--baseline-seconds", type=float, default=1)
    parser.add_argument("--post-teardown-seconds", type=float, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    settings = {name: getattr(args, name) for name in ("repetitions", "cycles", "warmup", "sample_interval_ms", "baseline_seconds", "post_teardown_seconds", "timeout_seconds")}
    report = run_experiment(args.fixtures, settings=settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output.name}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
