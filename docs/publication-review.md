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

The project is a development preview. CPU and TensorRT smoke execution were
observed, but Phase 1 acceptance is not complete. Earlier completion language was
too broad and is superseded by this review and the current validation record.

Known gaps requiring implementation and revalidation:

- The CLI composes a release report from supplied runtime observations; the
  adapter-specific launch scripts remain separate from the package CLI.
- Fixture/output evidence binding and incompatible-environment coverage still
  need dedicated acceptance tests.

Completed since the initial review: benchmark throughput now uses inverse mean
latency, TensorRT benchmark values are parsed from a supplied `trtexec` log,
the report schema and Markdown renderer exist, and a public CPU clean-reproduction
script now exports the pinned model from its verified weight digest.

The pinned TensorRT build script now runs successfully with the public image
digest. TensorRT engine bytes can still change across rebuilds because tactic
timing selects an engine for the observed GPU; the current manifest therefore
pins the observed engine hash and does not claim byte-for-byte engine rebuild
reproducibility.

The checked-in reports are development evidence, not release approval until the
remaining acceptance cases and clean reproduction are executed.
No model binaries or third-party weights are redistributed under this project's
MIT license. Upstream model/runtime licenses apply independently.
