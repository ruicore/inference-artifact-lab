"""PH2-R06/R08 negative evidence and owned-child contract checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from inference_artifact_lab import lifecycle_experiment as experiment


SETTINGS = {"repetitions": 1, "cycles": 2, "warmup": 1, "sample_interval_ms": 10, "baseline_seconds": 0.08, "post_teardown_seconds": 0.08, "timeout_seconds": 20}


@pytest.fixture(scope="module")
def generated_report(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("psutil")
    from inference_artifact_lab.lifecycle_fixtures import generate_fixtures
    path = generate_fixtures(tmp_path_factory.mktemp("lifecycle"))
    return experiment.run_experiment(path, settings=SETTINGS)


def checks(report):
    return {item["id"]: item["status"] for item in report["checks"]}


def test_real_generated_workers_preserve_evidence_without_claiming_public_profile(generated_report):
    report = generated_report
    assert report["status"] == "not_verified"
    assert checks(report)["PH2-AC08"] == "not_verified"
    assert checks(report)["PH2-AC09"] == "not_verified"
    assert checks(report)["PH2-AC11"] == "not_verified"
    assert checks(report)["PH2-AC01"] == "pass"
    assert checks(report)["PH2-AC03"] == "pass"
    assert checks(report)["PH2-AC04"] == "pass"
    assert checks(report)["PH2-AC05"] == "pass"
    assert checks(report)["PH2-AC07"] == "pass"
    assert [r["policy"] for r in report["repetitions"]] == list(experiment.POLICIES)
    for rep in report["repetitions"]:
        assert rep["parent_exit"]["observed"] and rep["parent_exit"]["exit_code"] == 0
        assert rep["worker"]["summary"]["maximum_admitted_instances"] == 2
        assert len(rep["worker"]["cleanup"]["prior_shutdowns"]) == 5
        assert any(c["stage"] == "revision_old_output" for c in rep["worker"]["correctness"])
        assert rep["worker"]["summary"]["latency_ms"]["warm_inference"]["count"] == SETTINGS["cycles"]
    serialized = json.dumps(report)
    assert str(Path.home()) not in serialized
    assert "artifact_path" not in serialized
    assert '"pid"' not in serialized
    assert all(r["parent_exit"]["worker_identity_verified"] and r["parent_exit"]["worker_exit_observed"] for r in report["repetitions"])


def test_schema_renderer_preserve_status_and_reject_false_pass(generated_report):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).parents[1] / "schemas/lifecycle-report.schema.json").read_text())
    jsonschema.validate(generated_report, schema)
    markdown = experiment.render_report(generated_report)
    assert "not_verified" in markdown and generated_report["scenario_sha256"] in markdown
    assert "native memory returned" in markdown
    malformed = deepcopy(generated_report)
    malformed["status"] = "pass"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(malformed, schema)
    del malformed["repetitions"][0]["rss"]["samples"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(malformed, schema)


@pytest.mark.parametrize("mutation", ["cleanup_error", "timeout", "missing_correctness", "nonzero_exit", "nonzero_worker_exit", "prior_cleanup_error"])
def test_negative_worker_evidence_cannot_pass_teardown_or_correctness(generated_report, mutation):
    report = deepcopy(generated_report)
    rep = report["repetitions"][0]
    if mutation == "cleanup_error":
        rep["worker"]["cleanup"]["cleanup_errors"] = [{"error_type": "RuntimeError"}]
    elif mutation == "prior_cleanup_error":
        rep["worker"]["cleanup"]["prior_shutdowns"][0]["complete"] = False
    elif mutation == "timeout":
        rep["parent_exit"]["timed_out"] = True
    elif mutation == "nonzero_exit":
        rep["parent_exit"]["exit_code"] = 1
    elif mutation == "nonzero_worker_exit":
        rep["parent_exit"]["worker_exit_code"] = 7
    else:
        rep["worker"]["correctness"] = []
        with pytest.raises(experiment.ExperimentInputError):
            experiment._validate_worker(rep["worker"], rep["policy"])
    updated = experiment.evaluate_checks(report["repetitions"], report["identity"], SETTINGS)
    statuses = {c["id"]: c["status"] for c in updated}
    if mutation != "missing_correctness":
        assert statuses["PH2-AC07"] == "fail"
    assert statuses["PH2-AC08"] != "pass"


def test_parent_observes_and_rejects_real_owned_worker_timeout():
    rep = experiment.run_repetition({}, {**SETTINGS, "timeout_seconds": 0.08}, "retain_idle", worker_command=[sys.executable, "-c", "import time; time.sleep(10)"])
    assert rep["parent_exit"]["observed"]
    assert rep["parent_exit"]["timed_out"]
    assert rep["parent_exit"]["exit_code"] != 0
    assert rep["worker"] is None
    assert rep["protocol_error"] == "missing_worker_result"


@pytest.mark.parametrize("timeout", [False, True])
def test_parent_samples_verified_interpreter_and_controls_it_on_timeout(timeout):
    pytest.importorskip("psutil")
    code = "import json,os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]).with_suffix('.worker.json'); p.write_text(json.dumps({'pid':os.getpid()})); owned_buffer=bytearray(64*1024*1024); time.sleep(10 if sys.argv[3]=='timeout' else 0.3)"
    # -c keeps its extra command arguments; the final marker follows the two IPC paths.
    rep = experiment.run_repetition({}, {**SETTINGS, "timeout_seconds": 0.4 if timeout else 5}, "retain_idle", worker_command=[sys.executable, "-c", code.replace("sys.argv[3]=='timeout'", "True" if timeout else "False")])
    assert rep["parent_exit"]["worker_identity_verified"]
    assert rep["parent_exit"]["worker_exit_observed"]
    assert rep["parent_exit"]["timed_out"] is timeout
    assert rep["rss"]["summary_bytes"]["maximum"] >= 64 * 1024 * 1024


def test_missing_rss_and_missing_stages_never_become_measurement_pass(generated_report):
    report = deepcopy(generated_report)
    for rep in report["repetitions"]:
        rep["rss"]["missing_sample_count"] = 1
        rep["rss"]["windows"]["baseline"]["status"] = "not_verified"
        rep["worker"]["correctness"] = [c for c in rep["worker"]["correctness"] if c["stage"] != "revision_old_output"]
    updated = experiment.evaluate_checks(report["repetitions"], report["identity"], SETTINGS)
    assert all(c["status"] != "pass" for c in updated if c["id"] in {"PH2-AC08", "PH2-AC09", "PH2-AC10"})
    coverage = experiment._coverage([], [{"stage": "baseline_start", "elapsed_seconds": 0}, {"stage": "baseline_end", "elapsed_seconds": 1}], SETTINGS)
    assert coverage["baseline"]["status"] == "not_verified"
    assert coverage["post_teardown"]["reason"] == "missing_window_markers"


def test_worker_boundary_rejects_malformed_result_and_nonfinite_fixture():
    with pytest.raises(experiment.ExperimentInputError):
        experiment._validate_worker({"policy": "retain_idle"}, "retain_idle")
    with pytest.raises(experiment.ExperimentInputError):
        experiment._tensor({"dtype": "float32", "shape": [1], "values": [float("nan")]})
    with pytest.raises(experiment.ExperimentInputError):
        experiment.validate_settings({**SETTINGS, "cycles": True})


def test_unavailable_runtime_is_blocked(generated_report):
    report = deepcopy(generated_report)
    rep = report["repetitions"][0]
    rep.update(worker=None, startup_error="ModuleNotFoundError")
    rep["parent_exit"]["exit_code"] = 1
    updated = experiment.evaluate_checks(report["repetitions"], report["identity"], SETTINGS)
    assert {c["status"] for c in updated if c["id"] == "PH2-AC08"} == {"blocked"}


def test_forged_pass_with_missing_worker_or_rss_is_rejected_by_schema_and_renderer(generated_report):
    jsonschema = pytest.importorskip("jsonschema")
    forged = deepcopy(generated_report)
    forged["status"] = "pass"
    for check in forged["checks"]:
        check["status"] = "pass"
    for repetition in forged["repetitions"]:
        repetition["worker"] = None
        repetition["rss"]["samples"] = []
    schema = json.loads((Path(__file__).parents[1] / "schemas/lifecycle-report.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(forged, schema)
    with pytest.raises((jsonschema.ValidationError, experiment.ExperimentInputError)):
        experiment.render_report(forged)


def test_semantic_validator_recomputes_check_decisions_and_raw_rss_coverage(generated_report):
    forged = deepcopy(generated_report)
    forged["checks"][0]["status"] = "fail"
    with pytest.raises(experiment.ExperimentInputError, match="check statuses"):
        experiment.validate_report(forged)
    forged = deepcopy(generated_report)
    forged["repetitions"][0]["rss"]["samples"] = []
    with pytest.raises(experiment.ExperimentInputError, match="coverage"):
        experiment.validate_report(forged)


@pytest.mark.parametrize("mutation", ["rss_maximum", "latency_median", "correctness_key", "cycle_count"])
def test_semantic_validator_rejects_contradictory_measurement_and_identity_summaries(generated_report, mutation):
    forged = deepcopy(generated_report)
    worker = forged["repetitions"][0]["worker"]
    if mutation == "rss_maximum":
        forged["repetitions"][0]["rss"]["summary_bytes"]["maximum"] = 999999999999
    elif mutation == "latency_median":
        worker["summary"]["latency_ms"]["cycle_inference"]["median"] = 999999999999
    elif mutation == "correctness_key":
        worker["correctness"][0]["key"] = "0" * 64
    else:
        worker["timings"] = [t for t in worker["timings"] if t["stage"] != "cycle_acquire"]
    with pytest.raises(experiment.ExperimentInputError):
        experiment.validate_report(forged)
