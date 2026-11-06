# Repository Instructions

This repository is public-by-design from its first commit.

- Use only public specifications, public models, public datasets, or generated fixtures.
- Do not add employer identifiers, private paths, internal terminology, private
  logs, screenshots, credentials, signed artifacts, or work-derived metrics.
- Keep the manifest, report schema, APIs, tests, and benchmarks independently designed.
- Every implementation change must map to a product requirement and acceptance case.
- A passing local run proves only the declared model, fixture, and environment.
- Do not claim arbitrary hardware compatibility, model quality, or production safety.
- Public remote creation, pushing, releases, and case studies require a separate
  project-level review and explicit authorization.

<!-- OPENCODE_STYLE_GUIDE_START -->
## Style Guide

Adapted from [OpenCode's Style Guide](https://github.com/anomalyco/opencode/blob/bcd713e2b71b433474d99fb1d29900c492c319a3/AGENTS.md#style-guide) at commit `bcd713e2b71b433474d99fb1d29900c492c319a3`. Existing repository-specific architecture, compatibility, formatting, and lint requirements take precedence where they are more specific; this guide does not authorize unrelated refactors or dependency changes. Language-specific sections apply only to the corresponding files.

### General Principles

- Keep simple logic in one function unless composition, reuse, or a clearly named domain concept makes extraction useful. Do not extract single-use helpers preemptively.
- Validate unknown input once at the boundary that owns it. Pass validated models or typed values inward; do not defensively revalidate values already guaranteed by a schema, constructor, or established internal contract.
- Before adding complexity for a speculative or vanishingly unlikely race or security edge case, explain the concrete failure mode, likelihood, and complexity cost to the user and get their buy-in. Do not silently expand scope for theoretical robustness.
- Avoid unnecessary exception handling. Catch only when recovery, cleanup, or boundary-level error translation is needed; keep the protected scope narrow and do not silently swallow errors.
- Reduce unnecessary temporary variables and unpacking. Inline a value used only once when that improves readability; retain names that clarify domain meaning, units, ownership, or a complex expression.
- Prefer guard clauses and early returns over deeply nested branches or unnecessary `else` blocks. Prefer assigning values once over repeated reassignment when the result remains clear.
- When a function has several validation branches or supporting details, make the main function read as the happy path and put the supporting details in small, nearby helpers. Extract real concepts, not trivial expressions; keep helpers below the main function when that improves readability.
- Keep synchronous parsing, validation, and option building synchronous. Add comments for non-obvious constraints and surprising behavior, not for obvious assignments or control flow.

### Python

These rules apply to Python files. Follow the repository's supported Python version, formatter, import sorter, type checker, and existing library conventions.

#### Imports (Python)

- Use `import module` when qualified access preserves useful context, or `from module import name` for clear direct imports. Both are normal Python; do not ban module imports as if they were JavaScript star imports.
- Avoid `from module import *` and unnecessary import aliases. Retain established ecosystem or repository aliases such as `np` and `pd`; use an alias when a real name collision requires it.
- Keep imports at module scope by default. Import a heavy optional dependency inside the narrow function or branch that needs it when lazy loading is required; do not make all imports lazy.
- Prefer the standard library and existing dependencies. Use `pathlib`, `json`, and context managers where appropriate; do not add a new dependency just to follow a style preference.

```python
import json
from pathlib import Path


def read_journal(directory: Path):
    with (directory / "journal.json").open(encoding="utf-8") as stream:
        return json.load(stream)
```

#### Types and Validation (Python)

- Keep useful type hints on public functions, domain boundaries, and non-obvious values as required by the repository's type checker. Straightforward local assignments usually do not need redundant annotations; do not remove useful annotations merely to reduce their count.
- Prefer precise types, protocols, typed dictionaries, or existing validated models over `typing.Any`. Use syntax supported by the repository's Python version.
- Validate untrusted input with the project's existing validation layer. After validation, do not repeat `isinstance`, `hasattr`, or dictionary-key checks for internal values whose invariants are already established. Type hints alone do not validate external input at runtime.

#### Variables, Iteration, and Unpacking (Python)

- Prefer readable comprehensions or generator expressions over `map` / `filter` chains for simple transformations. Use ordinary loops when side effects, branching, early exits, or clarity make them the better choice.
- Prefer direct attribute access when unpacking adds no clarity. Normal tuple unpacking, iteration unpacking, and direct named imports remain idiomatic.
- Prefer assigning values once and simple conditional expressions when clear. Do not force a complex expression just to eliminate a descriptive intermediate name.

```python
names = [item.name for item in items if item.enabled]
limit = configured_limit if configured_limit is not None else default_limit

for key, value in settings.items():
    apply_setting(key, value)
```

#### Control Flow and Helpers (Python)

- Prefer guard clauses and early returns over an unnecessary `else` after `return` or `raise`.
- Keep `try` / `except` blocks narrow and catch the specific expected exceptions. Prefer context managers or `try` / `finally` for resource cleanup; do not wrap successful-path logic in broad exception handlers without a concrete recovery policy.
- Use `async def` only when asynchronous work or an existing interface requires it. Keep synchronous helpers synchronous.

```python
def status_label(enabled: bool) -> str:
    if not enabled:
        return "disabled"
    return "enabled"


def load_thing(payload):
    config = require_config(payload)
    metadata = read_metadata(payload)
    return create_thing(config, metadata)
```

`require_config` and `read_metadata` represent real validation or parsing concepts; keep their implementations near the caller instead of extracting trivial expressions into more helpers.

#### Model and Database Fields (Python)

- Use `snake_case` for Python identifiers and new model or database fields according to the repository's conventions.
- Preserve existing external field aliases, persisted column names, public APIs, and compatibility contracts. A style change alone does not justify renaming an existing schema or adding a migration.

<details>
<summary>Upstream license notice (OpenCode, MIT)</summary>

```text
MIT License

Copyright (c) 2025 opencode

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>
<!-- OPENCODE_STYLE_GUIDE_END -->
