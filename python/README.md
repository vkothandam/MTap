# sourcing-py

Python sourcers for MTap. Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
cd python
uv sync --extra dev
```

## Run a source

```bash
uv run sourcing-py run example_api
# output -> ./out/example_api/dt=YYYY-MM-DD/example_api-<run_id>.jsonl
```

## Test (includes schema conformance)

```bash
uv run pytest
```

## Adding a source

Create `sourcing_py/sources/<name>/source.py` with a `Source` subclass that
implements `fetch()` and `extract()`. Register it in `../shared/config/sources.json`.
The base class handles the writer, provenance fields, and file naming per
`../schemas/conventions.md`.
