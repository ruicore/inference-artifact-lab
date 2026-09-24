# Initial publication review

Date: 2026-09-21

Scope: tracked files, all reachable commits and refs, commit identity, URL
destinations, binary artifacts, and disclosure claims. This is a source
publication review, not a production release certification.

- Gitleaks scanned the initial nine-commit history without credential findings.
- All 95 historical file blobs were checked for private identifiers and paths.
  A local directory reference was removed throughout unpublished history.
- Only public model, package, and source URLs were found. The test-only domain
  `example.invalid` is intentional. Authors use a GitHub noreply address.
- No binary weights, engines, screenshots, logs, credentials, or datasets are
  tracked. Generated artifacts and local review backups remain ignored.
- The domain and implementation concern generic artifact contracts and generated
  fixtures; no employer code or model is included in the reviewed tree.

## Product evidence boundary

The first Phase 1 acceptance and release scope is ONNX Runtime CPU on the
declared `onnx-cpu-windows-py311` profile. Committed-checkout AC-11 passed locally
on 2026-09-24 for commit `d8936237db3321fad79c75627ae364668ff65e58`.
TensorRT is an independent optional platform-specific
preview; its limitations do not block the CPU-only exit. Earlier blanket
completion language is superseded by the current per-profile validation record.

The September 24 local implementation now gives the package CLI a bounded ONNX
CPU adapter path. It probes the actual environment, rejects supplied observation
files on that path, checks artifact and pinned fixture/reference-output digests
before execution, and has dedicated incompatible-environment and tampered-fixture
tests. The TensorRT container adapter remains separate because its engine claim
is bound to the pinned container and observed GPU profile. A new public-history
and disclosure review is required before these local edits are pushed.

The CPU-only local acceptance gate passed. An independent clean local clone of
the committed revision ran the public-input reproduction script in a new Python
3.11.5 environment, regenerated the pinned ONNX digest, and returned matching
reference/package CLI `pass` decisions. This is local validation, not a new
public-history or disclosure review of the changes after that commit.

Independent TensorRT preview gaps (not CPU blockers):

- `AC-05` portable TensorRT provenance: the composer receives container output
  as supplied JSON and cannot independently replay the adapter call. The
  generic supplied-observation API is `not_verified` even when values compare.
- `AC-09` for a TensorRT performance claim: the current `trtexec` evidence has
  latency and throughput but no observed peak GPU memory. The report is
  `not_verified`, not `pass`.
- `AC-11` for TensorRT: the exact pinned engine is an ignored local artifact;
  a clean checkout cannot verify its bytes or gate decision without a separate
  legitimate exact-byte input. Rebuilding an engine does not substitute for it.
- The container digest inside runtime evidence is launcher-supplied. Local
  host-side image inspection was performed for this run, but that observation
  is not yet bound into a portable signed report.

Completed since the initial review: benchmark throughput now uses inverse mean
latency, TensorRT benchmark values are parsed from a supplied `trtexec` log,
the report schema and Markdown renderer exist, and a public CPU clean-reproduction
script now exports the pinned model from its verified weight digest.

The pinned TensorRT build script now runs successfully with the public image
digest. TensorRT engine bytes can still change across rebuilds because tactic
timing selects an engine for the observed GPU; the current manifest therefore
pins the observed engine hash and does not claim byte-for-byte engine rebuild
reproducibility.

The checked-in CPU report is a runtime `pass` for its declared profile; the
separate acceptance record owns the CPU-only Phase 1 exit, including the locally
observed committed-checkout AC-11.
Its benchmark measures Python allocation peaks only; native process and GPU
peaks are unverified. TensorRT remains `not_verified` with `blocked` AC-11 and is
not validated by a CPU pass. Neither report authorizes automatic model release
or publication. The new profile/role declarations are manifest-digest-bound and
rendered explicitly in reports. The version-1 JSON schema permits `scope` to be
absent for compatibility; the generator and checked-in report tests, not the
schema alone, cross-check scope against the manifest. Caller-supplied JSON needs
the same cross-check before its scope can be trusted.
No model binaries or third-party weights are redistributed under this project's
MIT license. Upstream model/runtime licenses apply independently.
