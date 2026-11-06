"""Bounded, process-local model ownership for the PH2 CPU experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from threading import Condition, Lock
from typing import Callable, Mapping, Protocol


class LoadSpecError(ValueError):
    pass


class ArtifactDigestError(ValueError):
    pass


class CapacityError(RuntimeError):
    reason = "capacity_exhausted"

    def __init__(self) -> None:
        super().__init__(self.reason)


class PoolStoppedError(RuntimeError):
    pass


class HandleReleasedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class InstanceIdentity:
    artifact_sha256: str
    adapter_version: str
    runtime_version: str
    providers: tuple[ProviderSpec, ...]
    session_options: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "adapter_version": self.adapter_version,
            "runtime_version": self.runtime_version,
            "providers": [
                {"name": provider.name, "options": dict(provider.options)}
                for provider in self.providers
            ],
            "session_options": dict(self.session_options),
        }

    @property
    def key(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoadSpec:
    artifact_path: Path
    artifact_sha256: str
    adapter_version: str
    runtime_version: str
    providers: tuple[ProviderSpec, ...]
    session_options: tuple[tuple[str, int], ...]

    @classmethod
    def from_dict(cls, value: object, base_dir: Path = Path(".")) -> LoadSpec:
        required = {
            "artifact_path", "artifact_sha256", "adapter_version",
            "runtime_version", "providers", "session_options",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise LoadSpecError("load specification requires exactly the supported fields")
        for name in ("artifact_path", "adapter_version", "runtime_version"):
            if not isinstance(value[name], str) or not value[name].strip():
                raise LoadSpecError(f"{name} must be a nonempty string")
        digest = value["artifact_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise LoadSpecError("artifact_sha256 must be a SHA-256 hex digest")
        providers = value["providers"]
        if providers != [{"name": "CPUExecutionProvider", "options": {}}]:
            raise LoadSpecError("exactly one explicit CPU provider with empty options is supported")
        options = value["session_options"]
        if not isinstance(options, dict) or set(options) != {
            "intra_op_num_threads", "inter_op_num_threads",
        }:
            raise LoadSpecError("both supported thread-count options must be explicit")
        if any(type(count) is not int or count < 1 for count in options.values()):
            raise LoadSpecError("thread counts must be positive integers")
        return cls(
            (Path(base_dir) / value["artifact_path"]).resolve(),
            digest.lower(), value["adapter_version"], value["runtime_version"],
            (ProviderSpec("CPUExecutionProvider"),), tuple(sorted(options.items())),
        )

    @property
    def identity(self) -> InstanceIdentity:
        return InstanceIdentity(
            self.artifact_sha256, self.adapter_version, self.runtime_version,
            self.providers, self.session_options,
        )

    def verify_artifact(self) -> None:
        with self.artifact_path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != self.artifact_sha256:
            raise ArtifactDigestError("artifact does not match its declared SHA-256")


class RuntimeInstance(Protocol):
    def run(self, inputs: Mapping[str, object]) -> object: ...

    def close(self) -> None: ...


@dataclass
class _Entry:
    identity: InstanceIdentity
    generation: int
    state: str = "loading"
    pending: int = 1
    handles: int = 0
    last_used: int = 0
    instance: RuntimeInstance | None = None
    error: BaseException | None = None
    run_lock: Lock = field(default_factory=Lock)


class UseHandle:
    def __init__(self, pool: LifecyclePool, entry: _Entry) -> None:
        self._pool = pool
        self._entry = entry
        self._released = False
        self._inflight = 0

    @property
    def identity(self) -> InstanceIdentity:
        return self._entry.identity

    @property
    def generation(self) -> int:
        return self._entry.generation

    def __enter__(self) -> UseHandle:
        with self._pool._condition:
            if self._released:
                raise HandleReleasedError("handle is released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    def run(self, inputs: Mapping[str, object]) -> object:
        with self._pool._condition:
            if self._released:
                raise HandleReleasedError("handle is released")
            if self._pool._stopped:
                raise PoolStoppedError("pool is stopped")
            self._inflight += 1
            self._pool._record("run_started", self._entry)
        try:
            # Admission owns the handle before waiting for per-instance serialization.
            with self._entry.run_lock:
                assert self._entry.instance is not None
                return self._entry.instance.run(inputs)
        finally:
            dispose = False
            with self._pool._condition:
                self._inflight -= 1
                self._pool._record("run_finished", self._entry)
                if self._released and self._inflight == 0:
                    dispose = self._pool._drop_handle(self._entry)
            if dispose:
                self._pool._dispose(self._entry)

    def release(self) -> None:
        with self._pool._condition:
            if self._released:
                return
            self._released = True
            self._pool._record("release_requested", self._entry)
            dispose = self._inflight == 0 and self._pool._drop_handle(self._entry)
        if dispose:
            self._pool._dispose(self._entry)


class LifecyclePool:
    def __init__(
        self, loader: Callable[[LoadSpec], RuntimeInstance], max_instances: int = 2,
        policy: str = "retain_idle",
    ) -> None:
        if type(max_instances) is not int or max_instances < 1:
            raise ValueError("max_instances must be a positive integer")
        if policy not in {"retain_idle", "release_on_idle"}:
            raise ValueError("unsupported retention policy")
        self._loader = loader
        self.max_instances = max_instances
        self.policy = policy
        self._condition = Condition()
        self._entries: dict[InstanceIdentity, _Entry] = {}
        self._events: list[dict[str, object]] = []
        self._cleanup_errors: list[dict[str, object]] = []
        self._generation = 0
        self._sequence = 0
        self._stopped = False
        self._closing = False

    def acquire(self, spec: LoadSpec) -> UseHandle:
        with self._condition:
            self._require_running()
        # Verify even on a cache hit: locators never authorize mismatched content.
        spec.verify_artifact()
        load_owner = False
        while True:
            with self._condition:
                self._require_running()
                entry = self._entries.get(spec.identity)
                if entry is not None:
                    if entry.state not in {"loading", "ready"}:
                        self._record("capacity_rejected", entry)
                        raise CapacityError()
                    entry.pending += 1
                    self._record("acquire_joined", entry)
                    while entry.state == "loading" and not self._stopped:
                        self._condition.wait()
                    entry.pending -= 1
                    if self._stopped:
                        dispose = self._idle_cleanup(entry)
                    elif entry.state == "failed":
                        assert entry.error is not None
                        raise entry.error
                    else:
                        return self._new_handle(entry)
                    break
                if len(self._entries) < self.max_instances:
                    self._generation += 1
                    entry = _Entry(spec.identity, self._generation)
                    self._entries[spec.identity] = entry
                    self._record("load_reserved", entry)
                    load_owner = True
                    break
                idle = [
                    item for item in self._entries.values()
                    if item.state == "ready" and item.handles == 0 and item.pending == 0
                ]
                if not idle:
                    self._record("capacity_rejected")
                    raise CapacityError()
                victim = min(idle, key=lambda item: item.last_used)
                victim.state = "closing"
                self._record("eviction_started", victim)
            self._dispose(victim)
        if not load_owner:
            if dispose:
                self._dispose(entry)
            raise PoolStoppedError("pool is stopped")
        try:
            instance = self._loader(spec)
        except BaseException as error:
            # Loader failures must release reservations and wake coalesced callers.
            with self._condition:
                entry.error = error
                entry.state = "failed"
                entry.pending -= 1
                del self._entries[entry.identity]
                self._record("load_failed", entry, error_type=type(error).__name__)
                self._condition.notify_all()
            raise
        with self._condition:
            entry.instance = instance
            entry.state = "ready"
            entry.pending -= 1
            self._record("load_finished", entry)
            self._condition.notify_all()
            if not self._stopped:
                return self._new_handle(entry)
            dispose = self._idle_cleanup(entry)
        if dispose:
            self._dispose(entry)
        raise PoolStoppedError("pool stopped during load")

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._record("stop")
            self._condition.notify_all()

    def close(self) -> dict[str, object]:
        with self._condition:
            self._stopped = True
            self._closing = True
            self._record("shutdown_requested")
            self._condition.notify_all()
            idle = [entry for entry in self._entries.values() if self._idle_cleanup(entry)]
        for entry in idle:
            self._dispose(entry)
        with self._condition:
            counts = self._counts()
            return {
                "complete": not self._entries and not self._cleanup_errors,
                "outstanding_handles": counts["active_handles"],
                "pending_acquisitions": counts["pending_acquisitions"],
                "loading_instances": counts["loading_instances"],
                "closing_instances": counts["closing_instances"],
                "cleanup_errors": [dict(error) for error in self._cleanup_errors],
            }

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                **self._counts(), "stopped": self._stopped,
                "max_instances": self.max_instances, "policy": self.policy,
                "entries": [
                    {
                        "identity": entry.identity.to_dict(),
                        "instance_key": entry.identity.key,
                        "generation": entry.generation, "state": entry.state,
                        "active_handles": entry.handles,
                        "pending_acquisitions": entry.pending,
                    }
                    for entry in self._entries.values()
                ],
            }

    @property
    def events(self) -> list[dict[str, object]]:
        with self._condition:
            return json.loads(json.dumps(self._events))

    def _require_running(self) -> None:
        if self._stopped:
            raise PoolStoppedError("pool is stopped")

    def _new_handle(self, entry: _Entry) -> UseHandle:
        entry.handles += 1
        self._record("acquired", entry)
        return UseHandle(self, entry)

    def _drop_handle(self, entry: _Entry) -> bool:
        entry.handles -= 1
        self._record("released", entry)
        entry.last_used = self._sequence
        return self._idle_cleanup(entry)

    def _idle_cleanup(self, entry: _Entry) -> bool:
        if (
            entry.state == "ready" and entry.handles == 0 and entry.pending == 0
            and (self._closing or self.policy == "release_on_idle")
        ):
            entry.state = "closing"
            self._record("cleanup_started", entry)
            return True
        return False

    def _dispose(self, entry: _Entry) -> None:
        assert entry.instance is not None
        try:
            entry.instance.close()
        except Exception as error:
            with self._condition:
                entry.state = "cleanup_failed"
                self._cleanup_errors.append({
                    "generation": entry.generation, "instance_key": entry.identity.key,
                    "error_type": type(error).__name__,
                })
                self._record("cleanup_failed", entry, error_type=type(error).__name__)
                self._condition.notify_all()
            return
        with self._condition:
            entry.instance = None
            entry.state = "closed"
            del self._entries[entry.identity]
            self._record("disposed", entry)
            self._condition.notify_all()

    def _counts(self) -> dict[str, int]:
        return {
            "admitted_instances": len(self._entries),
            "resident_instances": sum(item.instance is not None for item in self._entries.values()),
            "loading_instances": sum(item.state == "loading" for item in self._entries.values()),
            "closing_instances": sum(item.state == "closing" for item in self._entries.values()),
            "active_handles": sum(item.handles for item in self._entries.values()),
            "pending_acquisitions": sum(item.pending for item in self._entries.values()),
        }

    def _record(self, event: str, entry: _Entry | None = None, **details: object) -> None:
        self._sequence += 1
        item: dict[str, object] = {
            "sequence": self._sequence, "event": event, **self._counts(), **details,
        }
        if entry is not None:
            item.update({
                "generation": entry.generation, "instance_key": entry.identity.key,
                "identity": entry.identity.to_dict(),
            })
        self._events.append(item)
        self._condition.notify_all()
