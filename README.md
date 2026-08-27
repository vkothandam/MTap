# MTap — Sourcing Module

A polyglot sourcing module. Each **source** consumes data from an API or a scraped
web page, extracts fields, and emits **parquet** or **jsonl**. Sources are written in
either **Python** or **Node.js**, chosen per-source based on the strongest tool for
the job (and reuse of existing libraries).

Alongside one-shot sources, a **service** is a long-running sourcer that holds a
persistent session and publishes onto a message bus instead of writing files — e.g.
the [E*TRADE scraper](node/src/services/etrade/README.md), which owns a headless
browser session (login + OTP) and streams live market data to MBin over Redis.

The [Python E*TRADE fundamentals collector](python/sourcing_py/etrade/README.md) reuses
that browser session (`state/etrade/session.json`) to pull per-stock fundamentals
(balance sheet, income statement, SEC filings) into a normalized local DuckDB store for
ML (TFT). Building the master symbol table needs no login; the fundamentals fetch is gated
on the headless login above being merged and producing a session.

## The contract, not the code, is shared

Python and Node sourcers do not share a runtime or share code. What they share is a
**language-neutral output contract**:

- `schemas/*.schema.json` — the field names, types, and required-ness of each source's output
- `schemas/conventions.md` — partitioning, file naming, and jsonl-vs-parquet rules

Both language stacks read those files and conform. A conformance check in each test
suite validates emitted files against the schema, so the two halves cannot drift.

## Layout

```
schemas/          language-neutral output contract (the heart of the repo)
python/           Python sources + shared common lib (uv)
node/             Node.js sources + shared common lib (TypeScript)
shared/config/    source registry, credential layout (no secrets)
```

## Adding a source

1. Add `schemas/<source>.schema.json`.
2. Register it in `shared/config/sources.json` (pick `python` or `node`).
3. Create `python/sourcing_py/sources/<source>/` OR `node/src/sources/<source>/`.
4. Implement `fetch → extract → write`, writing via the shared `writer`.
5. Ensure the conformance test passes.

## Development

- Python: see [python/README.md](python/README.md)
- Node: see [node/README.md](node/README.md)

CI runs the two stacks as independent, path-filtered jobs; changes under `schemas/`
run both.
