"""PH2-AC01..07 ownership evidence; native-runtime/worker evidence is separate."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
from threading import Condition, Event
import weakref

import pytest

from inference_artifact_lab.lifecycle import (
    ArtifactDigestError,
    CapacityError,
    HandleReleasedError,
    LifecyclePool,
    LoadSpec,
    LoadSpecError,
    PoolStoppedError,
)


def declaration(tmp_path: Path, name: str = "a", content: bytes = b"1") -> dict:
    (tmp_path / name).write_bytes(content)
    return {
        "artifact_path": name,
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "adapter_version": "independent-fake/v1",
        "runtime_version": "1",
        "providers": [{"name": "CPUExecutionProvider", "options": {}}],
        "session_options": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
    }


def spec(tmp_path: Path, name: str = "a", content: bytes = b"1") -> LoadSpec:
    return LoadSpec.from_dict(declaration(tmp_path, name, content), tmp_path)


class Ledger:
    """Records actual adapter calls/resources without reading pool state."""

    def __init__(self) -> None:
        self.condition = Condition()
        self.records: list[tuple[str, int, str]] = []
        self.live: set[int] = set()
        self.running: set[int] = set()
        self.load_gates: dict[str, Event] = {}
        self.run_gate: Event | None = None
        self.close_gate: Event | None = None
        self.fail_loads = 0
        self.fail_close = False
        self.serial = 0

    def record(self, operation: str, serial: int, digest: str) -> None:
        with self.condition:
            self.records.append((operation, serial, digest))
            self.condition.notify_all()

    def wait(self, operation: str, count: int = 1) -> None:
        with self.condition:
            assert self.condition.wait_for(
                lambda: sum(item[0] == operation for item in self.records) >= count,
                timeout=5,
            ), f"missing ledger event: {operation}"

    def calls(self, operation: str) -> list[tuple[str, int, str]]:
        with self.condition:
            return [item for item in self.records if item[0] == operation]

    def load(self, load_spec: LoadSpec):
        with self.condition:
            self.serial += 1
            serial = self.serial
        self.record("load_started", serial, load_spec.artifact_sha256)
        gate = self.load_gates.get(load_spec.artifact_sha256)
        if gate is not None:
            assert gate.wait(5), "loader gate was not released"
        with self.condition:
            if self.fail_loads:
                self.fail_loads -= 1
                self.record("load_failed", serial, load_spec.artifact_sha256)
                raise RuntimeError("controlled loader failure")
            self.live.add(serial)
        adapter = FakeAdapter(self, serial, load_spec)
        self.record("loaded", serial, load_spec.artifact_sha256)
        return adapter


class FakeAdapter:
    def __init__(self, ledger: Ledger, serial: int, load_spec: LoadSpec) -> None:
        self.ledger = ledger
        self.serial = serial
        self.digest = load_spec.artifact_sha256
        self.bias = int(load_spec.artifact_path.read_bytes())

    def run(self, inputs):
        with self.ledger.condition:
            assert self.serial in self.ledger.live
            assert self.serial not in self.ledger.running, "same-instance calls overlapped"
            self.ledger.running.add(self.serial)
            self.ledger.record("run_started", self.serial, self.digest)
        try:
            if self.ledger.run_gate is not None:
                assert self.ledger.run_gate.wait(5), "inference gate was not released"
            if inputs.get("fail"):
                raise ArithmeticError("controlled inference failure")
            return inputs["x"] + self.bias
        finally:
            with self.ledger.condition:
                self.ledger.running.remove(self.serial)
                self.ledger.record("run_finished", self.serial, self.digest)

    def close(self):
        with self.ledger.condition:
            assert self.serial not in self.ledger.running, "closed during inference"
            self.ledger.record("close_started", self.serial, self.digest)
        if self.ledger.close_gate is not None:
            assert self.ledger.close_gate.wait(5), "cleanup gate was not released"
        if self.ledger.fail_close:
            raise RuntimeError("controlled cleanup failure")
        with self.ledger.condition:
            assert self.serial in self.ledger.live, "double disposal"
            self.ledger.live.remove(self.serial)
            self.ledger.record("closed", self.serial, self.digest)


def wait_pool_event(pool: LifecyclePool, event: str, count: int = 1) -> None:
    # This only coordinates an interleaving; load/run/disposal assertions use Ledger.
    with pool._condition:
        assert pool._condition.wait_for(
            lambda: sum(item["event"] == event for item in pool.events) >= count,
            timeout=5,
        )


def test_ac01_identity_uses_content_and_effective_options(tmp_path):
    first = declaration(tmp_path)
    second = declaration(tmp_path, "other-location")
    second["session_options"] = {"inter_op_num_threads": 1, "intra_op_num_threads": 1}
    second["artifact_sha256"] = second["artifact_sha256"].upper()
    a = LoadSpec.from_dict(first, tmp_path)
    same = LoadSpec.from_dict(second, tmp_path)
    assert a.identity == same.identity
    assert a.identity.key == same.identity.key
    assert "artifact_path" not in a.identity.to_dict()
    for key, value in (
        ("adapter_version", "independent-fake/v2"),
        ("runtime_version", "2"),
        ("session_options", {"intra_op_num_threads": 2, "inter_op_num_threads": 1}),
    ):
        changed = {**first, key: value}
        assert LoadSpec.from_dict(changed, tmp_path).identity != a.identity
    assert spec(tmp_path, "changed", b"2").identity != a.identity
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    with pool.acquire(a) as one, pool.acquire(same) as two:
        assert one.generation == two.generation
        assert one.run({"x": 3}) == 4
        assert two.run({"x": 9}) == 10
    assert len(ledger.calls("loaded")) == 1
    first["session_options"]["intra_op_num_threads"] = 7
    assert dict(a.session_options)["intra_op_num_threads"] == 1
    assert pool.close()["complete"]


@pytest.mark.parametrize("change", [
    {"unknown": True},
    {"session_options": {"intra_op_num_threads": 1}},
    {"session_options": {"intra_op_num_threads": True, "inter_op_num_threads": 1}},
    {"session_options": {"intra_op_num_threads": 0, "inter_op_num_threads": 1}},
    {"providers": [{"name": "CPUExecutionProvider", "options": {"unknown": "1"}}]},
    {"providers": [{"name": "CUDAExecutionProvider", "options": {}}]},
    {"providers": []},
    {"artifact_sha256": "invalid"},
    {"runtime_version": ""},
])
def test_ac01_unknown_or_implicit_options_reject(tmp_path, change):
    with pytest.raises(LoadSpecError):
        LoadSpec.from_dict({**declaration(tmp_path), **change}, tmp_path)


def test_ac01_mismatch_rejects_even_when_identity_is_cached(tmp_path):
    valid = spec(tmp_path)
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    with pool.acquire(valid):
        wrong = declaration(tmp_path, "wrong", b"2")
        wrong["artifact_sha256"] = valid.artifact_sha256
        with pytest.raises(ArtifactDigestError):
            pool.acquire(LoadSpec.from_dict(wrong, tmp_path))
    assert len(ledger.calls("load_started")) == 1
    assert pool.close()["complete"]


def test_ac02_barrier_controlled_acquisition_coalesces_one_load(tmp_path):
    a = spec(tmp_path)
    ledger = Ledger()
    gate = ledger.load_gates[a.artifact_sha256] = Event()
    pool = LifecyclePool(ledger.load, policy="release_on_idle")
    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(pool.acquire, a)
        ledger.wait("load_started")
        second = workers.submit(pool.acquire, a)
        try:
            wait_pool_event(pool, "acquire_joined")
            assert len(ledger.calls("load_started")) == 1
        finally:
            gate.set()
        one, two = first.result(5), second.result(5)
    assert one.generation == two.generation
    one.release()
    one.release()
    assert not ledger.calls("closed")
    assert two.run({"x": 4}) == 5
    two.release()
    assert len(ledger.calls("loaded")) == len(ledger.calls("closed")) == 1
    assert pool.close()["complete"]


def test_ac03_repeated_release_cannot_drop_other_owner(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load, policy="release_on_idle")
    a = spec(tmp_path)
    one, two = pool.acquire(a), pool.acquire(a)
    one.release()
    one.release()
    with pytest.raises(HandleReleasedError):
        one.run({"x": 0})
    with pytest.raises(HandleReleasedError):
        with one:
            pass
    assert two.run({"x": 0}) == 1
    assert ledger.live == {1}
    two.release()
    assert ledger.live == set()
    assert pool.snapshot()["active_handles"] == 0


def test_ac03_release_during_run_defers_disposal_and_serializes_calls(tmp_path):
    ledger = Ledger()
    ledger.run_gate = Event()
    pool = LifecyclePool(ledger.load, policy="release_on_idle")
    a = spec(tmp_path)
    one, two = pool.acquire(a), pool.acquire(a)
    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(one.run, {"x": 3})
        ledger.wait("run_started")
        second = workers.submit(two.run, {"x": 4})
        try:
            wait_pool_event(pool, "run_started", 2)
            one.release()
            one.release()
            two.release()
            assert len(ledger.calls("run_started")) == 1
            assert not ledger.calls("close_started")
            assert pool.close()["outstanding_handles"] == 2
        finally:
            ledger.run_gate.set()
        assert first.result(5) == 4
        assert second.result(5) == 5
    assert len(ledger.calls("closed")) == 1
    assert pool.close()["complete"]


def test_ac04_revision_alias_keeps_original_handles_unchanged(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    alias = spec(tmp_path, "a1", b"1")
    with pool.acquire(alias) as old:
        alias = spec(tmp_path, "a2", b"2")
        with pool.acquire(alias) as new:
            assert old.run({"x": 10}) == 11
            assert new.run({"x": 10}) == 12
            assert old.identity != new.identity
            assert old.generation != new.generation
    loaded = [event for event in pool.events if event["event"] == "load_finished"]
    assert {event["identity"]["artifact_sha256"] for event in loaded} == {
        item[2] for item in ledger.calls("loaded")
    }
    assert pool.close()["complete"]


def test_ac05_lru_eviction_uses_last_idle_operation(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    a, b, c = (spec(tmp_path, str(i), str(i).encode()) for i in (1, 2, 3))
    for item in (a, b, a):
        with pool.acquire(item):
            pass
    with pool.acquire(c):
        assert [item[2] for item in ledger.calls("closed")] == [b.artifact_sha256]
        assert len(ledger.live) == 2
    assert pool.close()["complete"]


def test_ac05_active_instance_survives_idle_pressure_and_busy_rejects(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    a, b, c = (spec(tmp_path, str(i), str(i).encode()) for i in (1, 2, 3))
    with pool.acquire(a):
        pass
    with pool.acquire(b) as active, pool.acquire(c):
        assert [item[2] for item in ledger.calls("closed")] == [a.artifact_sha256]
        assert active.run({"x": 4}) == 6
        before = ledger.calls("load_started")
        with pytest.raises(CapacityError, match="capacity_exhausted"):
            pool.acquire(a)
        assert ledger.calls("load_started") == before
    assert pool.close()["complete"]


def test_ac05_loading_reservations_block_third_loader_without_blocking_shutdown(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    a, b, c = (spec(tmp_path, str(i), str(i).encode()) for i in (1, 2, 3))
    gates = [Event(), Event()]
    ledger.load_gates = dict(zip((a.artifact_sha256, b.artifact_sha256), gates))
    with ThreadPoolExecutor(2) as workers:
        futures = [workers.submit(pool.acquire, item) for item in (a, b)]
        try:
            ledger.wait("load_started", 2)
            with pytest.raises(CapacityError):
                pool.acquire(c)
            result = pool.close()
            assert result["loading_instances"] == 2
            assert not result["complete"]
            assert len(ledger.calls("load_started")) == 2
        finally:
            for gate in gates:
                gate.set()
        for future in futures:
            with pytest.raises(PoolStoppedError):
                future.result(5)
    assert len(ledger.calls("closed")) == 2
    assert pool.close()["complete"]


def test_ac05_cleanup_still_occupies_capacity(tmp_path):
    ledger = Ledger()
    ledger.close_gate = Event()
    pool = LifecyclePool(ledger.load, max_instances=1, policy="release_on_idle")
    a, b = spec(tmp_path), spec(tmp_path, "b", b"2")
    handle = pool.acquire(a)
    with ThreadPoolExecutor(1) as workers:
        releasing = workers.submit(handle.release)
        try:
            ledger.wait("close_started")
            with pytest.raises(CapacityError):
                pool.acquire(b)
            assert len(ledger.calls("load_started")) == 1
            assert pool.close()["closing_instances"] == 1
        finally:
            ledger.close_gate.set()
        releasing.result(5)
    assert pool.close()["complete"]


def test_ac06_failed_load_wakes_coalesced_callers_and_allows_retry(tmp_path):
    ledger = Ledger()
    ledger.fail_loads = 1
    a = spec(tmp_path)
    gate = ledger.load_gates[a.artifact_sha256] = Event()
    pool = LifecyclePool(ledger.load, max_instances=1)
    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(pool.acquire, a)
        ledger.wait("load_started")
        second = workers.submit(pool.acquire, a)
        try:
            wait_pool_event(pool, "acquire_joined")
        finally:
            gate.set()
        for future in (first, second):
            with pytest.raises(RuntimeError, match="controlled loader failure"):
                future.result(5)
    assert not ledger.live
    assert pool.snapshot()["admitted_instances"] == 0
    with pool.acquire(a) as handle:
        assert handle.run({"x": 1}) == 2
    assert len(ledger.calls("load_started")) == 2
    assert len(ledger.calls("loaded")) == 1
    assert pool.close()["complete"]


def test_ac06_inference_failure_context_releases_ownership(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load, policy="release_on_idle")
    with pytest.raises(ArithmeticError):
        with pool.acquire(spec(tmp_path)) as handle:
            handle.run({"fail": True})
    assert not ledger.live
    assert len(ledger.calls("closed")) == 1
    assert pool.close()["complete"]


def test_ac06_stop_rejects_new_work_and_allows_running_call_to_finish(tmp_path):
    ledger = Ledger()
    ledger.run_gate = Event()
    pool = LifecyclePool(ledger.load)
    a = spec(tmp_path)
    handle = pool.acquire(a)
    with ThreadPoolExecutor(1) as workers:
        running = workers.submit(handle.run, {"x": 2})
        try:
            ledger.wait("run_started")
            pool.stop()
            with pytest.raises(PoolStoppedError):
                pool.acquire(a)
            with pytest.raises(PoolStoppedError):
                handle.run({"x": 2})
            handle.release()
            result = pool.close()
            assert result["outstanding_handles"] == 1
            assert not result["complete"]
            assert not ledger.calls("closed")
        finally:
            ledger.run_gate.set()
        assert running.result(5) == 3
    assert len(ledger.calls("closed")) == 1
    assert pool.close()["complete"]


def test_ac07_teardown_reports_handles_and_drops_runner_references(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    handle = pool.acquire(spec(tmp_path))
    reference = weakref.ref(handle._entry.instance)
    result = pool.close()
    assert result["outstanding_handles"] == 1
    assert not result["complete"]
    assert not ledger.calls("closed")
    handle.release()
    gc.collect()
    assert reference() is None  # Python reference evidence only, never native RSS proof.
    assert pool.close()["complete"]
    assert len(ledger.calls("closed")) == 1


def test_ac07_cleanup_failure_cannot_report_success_or_free_capacity(tmp_path):
    ledger = Ledger()
    ledger.fail_close = True
    pool = LifecyclePool(ledger.load, max_instances=1, policy="release_on_idle")
    with pool.acquire(spec(tmp_path)):
        pass
    with pytest.raises(CapacityError):
        pool.acquire(spec(tmp_path, "b", b"2"))
    result = pool.close()
    assert not result["complete"]
    assert result["outstanding_handles"] == 0
    assert result["cleanup_errors"][0]["error_type"] == "RuntimeError"
    assert pool.snapshot()["admitted_instances"] == 1
    assert len(ledger.live) == 1
    assert len(ledger.calls("close_started")) == 1
    assert not ledger.calls("closed")
    assert pool.close() == result


def test_events_are_copied_and_have_no_locator(tmp_path):
    ledger = Ledger()
    pool = LifecyclePool(ledger.load)
    with pool.acquire(spec(tmp_path)):
        pass
    events = pool.events
    saved = deepcopy(events)
    events[0]["identity"]["providers"][0]["name"] = "mutated"
    assert pool.events == saved
    assert str(tmp_path) not in str(events)
    assert pool.close()["complete"]
